# SPDX-FileCopyrightText: Copyright (c) 2026 alchemi-hygon contributors
# SPDX-License-Identifier: Apache-2.0
"""Product HIP Jacobi failure, lifecycle, stream and custom-op gates."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from nvalchemiops import _bfgs_jacobi_hip as solver
from nvalchemiops.bfgs_ase import (
    BFGSNumericalError,
    bfgs_ase_prepare_state,
    bfgs_ase_step,
)
from nvalchemiops.bfgs_ase_batch_reference import _step_batch_with_eigh

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.version.hip or not torch.cuda.is_available(),
        reason="requires allocated HIP device",
    ),
]


def inputs():
    """Independent matrices with different widths and true nontrivial eigenspaces."""
    gen = torch.Generator().manual_seed(7354)
    result = []
    for n, b in ((15, 8), (21, 16), (201, 1)):
        r = torch.randn(b, n, n, generator=gen, dtype=torch.float64)
        result.append(
            (70 * torch.eye(n, dtype=torch.float64) + (r + r.mT) * 0.1).cuda()
        )
    return result


def assert_same_state(qs, states, saved):
    """Compare full state after a rejected candidate calculation."""
    for q, s, (old, expected) in zip(qs, states, saved):
        assert torch.equal(q, old)
        for k, v in expected.items():
            actual = s.state_dict()[k]
            assert (
                torch.equal(actual, v) if isinstance(v, torch.Tensor) else actual == v
            )


def test_true_late_group_nonconvergence_drains_then_reuses_scheduler():
    """A real sweep-limited failure cannot commit, or poison the next submission."""
    hs = inputs()
    qs = [
        torch.full((n // 3, 3), 0.01, device="cuda", dtype=torch.float64)
        for n in (15, 21, 201)
    ]
    states = [
        bfgs_ase_prepare_state(q.numel(), device="cuda", system_id=i)
        for i, q in enumerate(qs)
    ]
    states[-1].hessian.copy_(hs[-1][0])
    saved = [(q.clone(), s.state_dict()) for q, s in zip(qs, states)]
    scheduler = solver._GroupScheduler(qs[0].device)
    try:
        with pytest.raises(BFGSNumericalError, match="non-converged"):
            _step_batch_with_eigh(
                qs,
                [-q for q in qs],
                states,
                solver._eigh,
                grouped_eigh=lambda groups: [v[:2] for v in scheduler.solve(groups, 1)],
            )
        assert_same_state(qs, states, saved)
        _step_batch_with_eigh(
            qs,
            [-q for q in qs],
            states,
            solver._eigh,
            grouped_eigh=lambda groups: [v[:2] for v in scheduler.solve(groups)],
        )
        assert all(s.step_count == 1 for s in states)
    finally:
        scheduler.close()
    with pytest.raises(RuntimeError, match="closed"):
        scheduler.solve(hs)


def test_producer_consumer_release_and_allocator_reuse(monkeypatch):
    """Unchecked native input prevents finite-check synchronization masking dependencies."""
    refs = inputs()
    expected = [solver._jacobi_eigh(a, 100) for a in refs]
    module = solver._load_extension(refs[0].device)
    scheduler = solver._GroupScheduler(refs[0].device)
    # Exercise identical worker/event/lifetime path, without a prior scalar input check.
    monkeypatch.setattr(
        solver, "_jacobi_eigh", lambda a, sweeps: module.eigh(a, sweeps)
    )
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    producer.wait_stream(torch.cuda.current_stream())
    try:
        for _ in range(5):
            with torch.cuda.stream(producer):
                temporary = [a.clone() for a in refs]
                out = scheduler.solve(temporary)
                done = torch.cuda.Event()
                done.record()
                del temporary
                junk = [torch.empty_like(a).fill_(1000) for a in refs]
            consumer.wait_event(done)
            with torch.cuda.stream(consumer):
                for group in out:
                    for t in group:
                        t.record_stream(consumer)
                copied = [tuple(t.clone() for t in group) for group in out]
                consumed = torch.cuda.Event()
                consumed.record()
            torch.cuda.current_stream().wait_event(consumed)
            assert all(
                torch.equal(x, y)
                for a, b in zip(expected, copied)
                for x, y in zip(a, b)
            )
            del junk
    finally:
        scheduler.close()


def test_cached_scheduler_closes_and_recreates_and_shared_calls_are_safe():
    """Two caller threads cannot mix groups; explicit close releases cached workers."""
    hs = inputs()
    expected = solver._grouped_eigh(hs)
    old = solver._SCHEDULERS[hs[0].device]

    def execute():
        with torch.cuda.device(hs[0].device):
            return solver._grouped_eigh(hs)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(execute) for _ in range(4)]
        for future in futures:
            out = future.result()
            assert all(
                torch.equal(x, y) for a, b in zip(expected, out) for x, y in zip(a, b)
            )
    solver._close_schedulers()
    assert old.closed and not solver._SCHEDULERS
    out = solver._grouped_eigh(hs)
    assert solver._SCHEDULERS[hs[0].device] is not old
    assert all(torch.equal(x, y) for a, b in zip(expected, out) for x, y in zip(a, b))
    solver._close_schedulers()


def test_custom_op_schema_and_fake():
    """Actual eager custom op respects declared schema, output ownership and fake metadata."""
    a = inputs()[0][:1].contiguous()
    torch.library.opcheck(
        solver._jacobi_eigh, (a, 100), test_utils=("test_schema", "test_faketensor")
    )


def test_partial_submit_failure_drains_accepted_work(monkeypatch):
    """An exception submitting worker two must wait for worker one before returning."""
    hs = inputs()
    scheduler = solver._GroupScheduler(hs[0].device)
    submit = scheduler.pool.submit
    accepted = []

    def fail_second(*args, **kwargs):
        if accepted:
            raise RuntimeError("injected submit failure")
        future = submit(*args, **kwargs)
        accepted.append(future)
        return future

    try:
        monkeypatch.setattr(scheduler.pool, "submit", fail_second)
        with pytest.raises(RuntimeError, match="submit failure"):
            scheduler.solve(hs)
        assert accepted[0].done()
        monkeypatch.setattr(scheduler.pool, "submit", submit)
        assert len(scheduler.solve(hs)) == len(hs)
    finally:
        scheduler.close()


def test_nonfinite_gradient_and_upper_are_rejected():
    """Invalid inputs fail explicitly while unselected upper NaNs remain legal."""
    a = inputs()[0]
    before = a.clone()
    a[:, torch.triu_indices(15, 15, 1)[0], torch.triu_indices(15, 15, 1)[1]] = float(
        "nan"
    )
    w, v = solver._eigh(a)
    expected = solver._eigh(before)
    assert torch.equal(w, expected[0]) and torch.equal(v, expected[1])
    a[0, 1, 0] = float("nan")
    with pytest.raises(BFGSNumericalError, match="LOWER"):
        solver._eigh(a)
    with pytest.raises(ValueError, match="forward-only"):
        solver._eigh(before.clone().requires_grad_())
    with pytest.raises(ValueError, match="LOWER"):
        solver._eigh(before, "U")


def test_zero_sweep_tiny_spectrum_signed_zero_and_public_step():
    """Centering must not erase tiny eigenvalues or change their stable column order."""
    diagonals = torch.tensor(
        [[2e-18, 1e-18, 2.0], [-0.0, 0.0, 2.0], [0.0, -0.0, 2.0], [-1.0, 1e-18, 2.0]],
        dtype=torch.float64,
    )
    h = torch.diag_embed(diagonals).cuda()
    w, v, info, residual, sweeps = solver._jacobi_eigh(h, 100)
    expected, permutation = diagonals.sort(stable=True)
    assert torch.equal(w.cpu().view(torch.int64), expected.view(torch.int64))
    assert bool((info == 0).all()) and bool((sweeps == 0).all())
    for i in range(len(h)):
        assert torch.equal(
            v[i].cpu(), torch.eye(3, dtype=torch.float64)[:, permutation[i]]
        )
    q = torch.zeros(1, 3, dtype=torch.float64)
    gpu = q.cuda()
    f = torch.tensor([[2e-19, 1e-19, 0.0]], dtype=torch.float64)
    cpu_state = bfgs_ase_prepare_state(3)
    cpu_state.hessian.copy_(torch.diag(diagonals[0]))
    state = bfgs_ase_prepare_state(
        3, device="cuda", backend="hip.bfgs-ase-jacobi-streams-v1"
    )
    state.hessian.copy_(h[0])
    expected_step = bfgs_ase_step(q, f, cpu_state)
    actual = bfgs_ase_step(
        gpu, f.cuda(), state, backend="hip.bfgs-ase-jacobi-streams-v1"
    )
    assert torch.equal(gpu.cpu(), q) and torch.equal(
        actual.raw_step.cpu(), expected_step.raw_step
    )
