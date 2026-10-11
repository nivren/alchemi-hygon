# SPDX-License-Identifier: Apache-2.0
"""Registered product backend acceptance against frozen probes and ASE CPU steps."""

import argparse
import json
from pathlib import Path

import torch
from ase_bfgs_jacobi_candidate import load as probe_load
from ase_bfgs_jacobi_candidate_test import validate
from nvalchemiops import _bfgs_jacobi_hip as solver

BACKEND = "hip.bfgs-ase-jacobi-streams-v1"


def bfgs_gate(args):
    import io

    import numpy as np
    from ase import Atoms
    from ase.optimize import BFGS
    from nvalchemiops._bfgs_jacobi_hip import _eigh as eigh
    from nvalchemiops._bfgs_jacobi_hip import _jacobi_eigh as checked_raw
    from nvalchemiops.bfgs_ase import (
        ASEBFGSState,
        BFGSNumericalError,
        bfgs_ase_batch_step,
        bfgs_ase_prepare_state,
        bfgs_ase_step,
    )
    from nvalchemiops.bfgs_ase_batch_reference import (
        _step_batch_with_eigh as shared_math,
    )

    def _step_batch_with_eigh(q, f, states, callback):
        if callback.__name__ == "fail_eigh":
            return shared_math(q, f, states, callback)
        groups = {}
        for state in states:
            groups[state.num_scalar_dofs] = groups.get(state.num_scalar_dofs, 0) + 1
        shapes.extend([[count, n, n] for n, count in groups.items()])
        return bfgs_ase_batch_step(q, f, states, backend=BACKEND)

    widths = args.widths
    if len(widths) < 2 or any(n < 3 or n % 3 for n in widths):
        raise ValueError(
            "BFGS gate requires at least two positive Cartesian widths divisible by three"
        )
    qs = [
        torch.linspace(
            -0.4, 0.6 + i * 0.01, n, device="cuda", dtype=torch.float64
        ).reshape(-1, 3)
        for i, n in enumerate(widths)
    ]
    states = [
        bfgs_ase_prepare_state(
            n, device="cuda", system_id=i, alpha=53, maxstep=0.12, backend=BACKEND
        )
        for i, n in enumerate(widths)
    ]
    opts = [
        BFGS(
            Atoms("H" * (n // 3), positions=q.cpu().numpy()),
            logfile=None,
            alpha=53,
            maxstep=0.12,
        )
        for n, q in zip(widths, qs)
    ]

    def force(q):
        flat = q.flatten()
        k = torch.linspace(4, 14, flat.numel(), device=q.device, dtype=q.dtype)
        return -(k * flat + 0.7 * flat.sum() + 0.6 * flat**3).reshape_as(q)

    errors = {
        k: 0.0
        for k in ["coordinates", "hessian", "spectrum", "raw_step", "applied_step"]
    }
    shapes = []
    restart_qs, restart_states = None, None
    recorded_eigh = eigh
    for step in range(10):
        results = _step_batch_with_eigh(
            qs, [force(q) for q in qs], states, recorded_eigh
        )
        for q, state, opt, output in zip(qs, states, opts, results):
            old = opt.atoms.positions.copy()
            x = old.ravel()
            f = -(np.linspace(4, 14, x.size) * x + 0.7 * x.sum() + 0.6 * x**3)
            raw, lengths = opt.prepare_step(x, -f)
            applied = opt.determine_step(raw.copy(), lengths)
            opt.atoms.set_positions(old + applied.reshape(-1, 3))
            targets = [
                ("coordinates", q, opt.atoms.positions, 3e-12, 3e-12),
                ("hessian", state.hessian, opt.H, 3e-10, 3e-11),
                (
                    "spectrum",
                    output.eigenvalues,
                    np.linalg.eigvalsh(opt.H),
                    2e-10,
                    2e-11,
                ),
                ("raw_step", output.raw_step, raw.reshape(-1, 3), 1e-12, 1e-10),
                (
                    "applied_step",
                    output.applied_step,
                    applied.reshape(-1, 3),
                    1e-12,
                    1e-10,
                ),
            ]
            for key, value, target, atol, rtol in targets:
                actual = value.cpu().numpy()
                np.testing.assert_allclose(
                    actual, target, atol=atol, rtol=rtol, err_msg=f"{key} step{step}"
                )
                errors[key] = max(errors[key], float(np.abs(actual - target).max()))
        if restart_qs is not None:
            _step_batch_with_eigh(
                restart_qs, [force(q) for q in restart_qs], restart_states, eigh
            )
            for q, r, s, t in zip(qs, restart_qs, states, restart_states):
                assert torch.equal(q, r) and torch.equal(s.hessian, t.hessian)
                assert torch.equal(s.previous_positions, t.previous_positions)
                assert (
                    torch.equal(s.previous_forces, t.previous_forces)
                    and s.step_count == t.step_count
                )
        if step == 3:
            buffer = io.BytesIO()
            torch.save([s.state_dict() for s in states], buffer)
            buffer.seek(0)
            snapshots = torch.load(buffer, weights_only=True)
            restart_states = [ASEBFGSState.from_state_dict(s) for s in snapshots]
            restart_qs = [q.clone() for q in qs]
    # Same candidate on the original single-system math body.
    single_q = qs[0].clone()
    single_state = ASEBFGSState.from_state_dict(states[0].state_dict())
    reference_q = [q.clone() for q in qs]
    reference_states = [ASEBFGSState.from_state_dict(s.state_dict()) for s in states]
    bfgs_ase_step(single_q, force(single_q), single_state, backend=BACKEND)
    _step_batch_with_eigh(
        reference_q, [force(q) for q in reference_q], reference_states, eigh
    )
    torch.testing.assert_close(single_q, reference_q[0], atol=1e-12, rtol=1e-10)

    def assert_unchanged(q_before, snapshots):
        assert all(torch.equal(q, before) for q, before in zip(qs, q_before))
        for state, expected in zip(states, snapshots):
            actual = state.state_dict()
            for key, value in expected.items():
                assert (
                    torch.equal(actual[key], value)
                    if isinstance(value, torch.Tensor)
                    else actual[key] == value
                )

    failures = []
    for mode in ["zero_spectrum", "nonconvergence", "invalid_vectors"]:
        if mode == "zero_spectrum":
            states[1] = bfgs_ase_prepare_state(
                widths[1],
                device="cuda",
                system_id=1,
                alpha=53,
                maxstep=0.12,
                backend=BACKEND,
            )
            states[1].hessian.zero_()
        snapshots = [s.state_dict() for s in states]
        before = [q.clone() for q in qs]

        def fail_eigh(a, **kwargs):
            if mode == "nonconvergence":
                w, v, *_ = checked_raw(a, max_sweeps=1)
                return w, v
            if mode == "invalid_vectors":
                w, v = eigh(a, **kwargs)
                return w, v * 2 if a.shape[-1] == widths[1] else v
            return eigh(a, **kwargs)

        try:
            _step_batch_with_eigh(qs, [force(q) for q in qs], states, fail_eigh)
        except BFGSNumericalError as exc:
            failures.append(dict(mode=mode, error=str(exc)))
        else:
            raise AssertionError(f"{mode} not rejected")
        assert_unchanged(before, snapshots)
        states = [ASEBFGSState.from_state_dict(s.state_dict()) for s in restart_states]
        qs = [q.clone() for q in restart_qs]
    result = dict(
        status="passed",
        backend=BACKEND,
        widths=widths,
        steps=10,
        group_shapes=shapes,
        errors_max_abs=errors,
        restart_bitwise_equal=True,
        single_batch_gate=True,
        failures=failures,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def check_probe_parity(a, values, expected):
    # Preserve exact Jacobi parity; the zero-sweep diagonal correction is qualified
    # against original FP64 diagonal values rather than the probe's cancellation.
    if all(torch.equal(x, y) for x, y in zip(values, expected)):
        return
    assert all(torch.equal(x, y) for x, y in zip(values[1:], expected[1:]))
    changed = (values[0] != expected[0]).any(-1)
    assert bool((values[4][changed] == 0).all())
    original = a.diagonal(0, -2, -1).sort(stable=True).values
    assert torch.equal(values[0][changed], original[changed])
    assert torch.equal(values[0][~changed], expected[0][~changed])


def correctness(args):
    cases = torch.load(args.dataset, map_location="cpu", weights_only=True)
    real = torch.load(args.real_oracles, map_location="cpu", weights_only=True)
    inputs = [c["input"].unsqueeze(0).cuda() for c in cases]
    before = [a.view(torch.int64).clone() for a in inputs]
    scheduler = solver._GroupScheduler(inputs[0].device)
    try:
        values = scheduler.solve(inputs)
        rows = []
        original = probe_load()
        for a, v, ref, old in zip(inputs, values, cases, before):
            rows.append(dict(n=ref["n"], name=ref["name"], **validate(a, v, [ref])))
            assert torch.equal(a.view(torch.int64), old)
            expected = original.eigh(a, 100)
            check_probe_parity(a, v, expected)
        real_inputs = [r["input"].cuda() for r in real]
        real_rows = []
        for a, v, ref in zip(real_inputs, scheduler.solve(real_inputs), real):
            real_rows.append(
                dict(
                    scope=ref["scope"],
                    shape=list(a.shape),
                    **validate(a, v, ref["references"]),
                )
            )
            expected = original.eigh(a, 100)
            check_probe_parity(a, v, expected)
        batch_rows = []
        for n, count in ((147, 8), (189, 16), (199, 32), (285, 8)):
            refs = [c for c in cases if c["n"] == n and c["name"] != "upper_poison"]
            refs = [refs[i % len(refs)] for i in range(count)]
            a = torch.stack([c["input"] for c in refs]).cuda()
            v = solver._jacobi_eigh(a, 100)
            batch_rows.append(dict(n=n, batch=count, **validate(a, v, refs)))
            check_probe_parity(a, v, original.eigh(a, 100))
        args.output.write_text(
            json.dumps(
                dict(
                    status="passed",
                    backend=BACKEND,
                    cases=rows,
                    real_cases=real_rows,
                    batch_cases=batch_rows,
                    iterative_outputs_probe_bitwise=True,
                    zero_sweep_spectrum_original_diagonal_exact=True,
                ),
                indent=2,
            )
            + "\n"
        )
    finally:
        scheduler.close()
        solver._close_schedulers()
    print(
        json.dumps(
            dict(
                status="passed",
                mode="correctness",
                cases=len(cases),
                real_groups=len(real),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("correctness", "bfgs"), required=True)
    p.add_argument("--dataset", type=Path)
    p.add_argument("--real-oracles", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--widths",
        type=int,
        nargs="+",
        default=[15, 21, 30, 45, 57, 93, 201, 489, 189, 189, 561],
    )
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.mode == "correctness":
        correctness(args)
    else:
        bfgs_gate(args)
