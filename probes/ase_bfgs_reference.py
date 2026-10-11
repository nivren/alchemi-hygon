# SPDX-FileCopyrightText: Copyright (c) 2026 alchemi-hygon contributors
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bounded CPU/HCU BFGS probe; ASE/NumPy and transfers occur only in the oracle.

Run under the project environment, with one explicitly selected idle HCU.
This probe is not a model/Framework/Batch/variable-cell or performance gate.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import ase
import numpy as np
import torch
from ase import Atoms
from ase.optimize import BFGS
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from nvalchemiops.backend import resolve_backend
from nvalchemiops.bfgs_ase import ASEBFGSState, bfgs_ase_prepare_state, bfgs_ase_step


class DeviceGuard(TorchDispatchMode):
    """Observe reference operations and reject host tensors or precision changes."""

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.calls = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.calls[str(func)] += 1
        output = func(*args, **(kwargs or {}))
        for value in tree_flatten(output)[0]:
            if isinstance(value, torch.Tensor):
                if value.device != self.device:
                    raise AssertionError(
                        f"host/device fallback: {func} -> {value.device}"
                    )
                if value.is_floating_point() and value.dtype != torch.float64:
                    raise AssertionError(f"precision change: {func} -> {value.dtype}")
        return output


def force(q):
    """Independent coupled quartic potential, evaluated on CPU for the oracle."""
    x = q.ravel()
    matrix = np.diag(np.linspace(4.0, 14.0, x.size)) + 0.7 * np.ones((x.size, x.size))
    return (-matrix @ x - 0.6 * x**3).reshape(q.shape)


def compare(actual, target, errors, name, *, rtol, atol):
    """Record absolute differences and enforce the declared per-device tolerance."""
    target = np.asarray(target)
    # Probe-only transfer, outside DeviceGuard and outside the production step.
    observed = actual.cpu().numpy()
    errors[name] = max(errors.get(name, 0.0), float(np.max(np.abs(observed - target))))
    np.testing.assert_allclose(observed, target, rtol=rtol, atol=atol, err_msg=name)


def run_case(natoms, device, rtol, atol, nondefault_stream):
    """Compare two independent trajectories, including an on-device restart."""
    initial = np.linspace(-1.2, 1.4, 3 * natoms).reshape(-1, 3)
    oracle = BFGS(
        Atoms("H" * natoms, positions=initial), logfile=None, maxstep=0.09, alpha=22.0
    )
    q = torch.tensor(initial, device=device)
    state = bfgs_ase_prepare_state(
        q.numel(), alpha=22.0, maxstep=0.09, device=device, system_id=natoms
    )
    twin_q = q.clone()
    twin = ASEBFGSState.from_state_dict(state.state_dict())
    selection = resolve_backend(
        "torch_reference",
        operation="bfgs_ase_compat",
        device=device,
        dtype=torch.float64,
        features={"fixed_cell", "single_system"},
    )
    errors = {}
    guard = DeviceGuard(q.device)
    residual_max = 0.0
    stream = torch.cuda.Stream(device=device) if nondefault_stream else None
    for step in range(8):
        before = oracle.atoms.get_positions()
        oracle_force = force(before)
        raw, lengths = oracle.prepare_step(before.ravel(), -oracle_force.ravel())
        raw = raw.copy()
        applied = oracle.determine_step(raw.copy(), lengths)
        oracle.atoms.set_positions(before + applied.reshape(-1, 3))
        f = torch.tensor(force(q.cpu().numpy()), device=device)
        twin_f = torch.tensor(force(twin_q.cpu().numpy()), device=device)
        if step == 4:
            saved = state.state_dict()
            state = ASEBFGSState.from_state_dict(saved, device=device)
            assert state.hessian.data_ptr() != saved["hessian"].data_ptr()
        if stream is None:
            with guard:
                result = bfgs_ase_step(
                    q, f, state, selection=selection, system_id=natoms
                )
        else:
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream), guard:
                result = bfgs_ase_step(
                    q, f, state, selection=selection, system_id=natoms
                )
            torch.cuda.current_stream(device).wait_stream(stream)
        bfgs_ase_step(twin_q, twin_f, twin)
        torch.testing.assert_close(q, twin_q, rtol=0, atol=0)
        torch.testing.assert_close(state.hessian, twin.hessian, rtol=0, atol=0)
        for name, actual, expected in [
            ("hessian", state.hessian, oracle.H),
            ("spectrum", result.eigenvalues, np.linalg.eigvalsh(oracle.H)),
            ("raw_step", result.raw_step, raw.reshape(-1, 3)),
            ("applied_step", result.applied_step, applied.reshape(-1, 3)),
            ("coordinates", q, oracle.atoms.positions),
            ("history_positions", state.previous_positions, before),
            ("history_forces", state.previous_forces, oracle_force),
            ("scale", result.scale, min(1.0, 0.09 / np.max(lengths))),
        ]:
            compare(actual, expected, errors, name, rtol=rtol, atol=atol)
        values, vectors = torch.linalg.eigh(state.hessian)
        residual = torch.linalg.matrix_norm(
            state.hessian @ vectors - vectors * values
        ) / torch.linalg.matrix_norm(state.hessian)
        residual_max = max(residual_max, residual.item())
        assert residual_max < 1e-12
    return {
        "natoms": natoms,
        "steps": state.step_count,
        "nondefault_stream": nondefault_stream,
        "restart_exact": True,
        "max_abs_errors": errors,
        "max_eigen_residual": residual_max,
        "device_guard_operation_count": sum(guard.calls.values()),
        "eigh_calls": guard.calls["aten._linalg_eigh.default"],
    }


def main():
    """Emit reproducible numerical evidence and optional GPU kernel activity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    assert ase.__version__ == "3.29.0"
    device = torch.device(args.device)
    if args.device == "cuda":
        assert torch.cuda.is_available()
        device = torch.device("cuda", torch.cuda.current_device())
    rtol, atol = (1e-8, 1e-10) if args.device == "cuda" else (1e-10, 1e-12)
    cases = [
        run_case(n, device, rtol, atol, args.device == "cuda" and n == 12)
        for n in (1, 3, 12)
    ]
    # Explicit CPU -> target restoration is a probe-only portability check.
    portable = ASEBFGSState.from_state_dict(
        bfgs_ase_prepare_state(3, device="cpu").state_dict(), device=device
    )
    q = torch.zeros((1, 3), device=device, dtype=torch.float64)
    bfgs_ase_step(q, torch.ones_like(q), portable)
    profile = {"requested": args.profile, "gpu_events": 0}
    if args.profile:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as recorded:
            bfgs_ase_step(q, -2.0 * q, portable)
            torch.cuda.synchronize(device)
        profile["gpu_events"] = sum(
            event.device_type == torch.autograd.DeviceType.CUDA
            for event in recorded.events()
        )
        trace = args.output.with_suffix(".trace.json")
        recorded.export_chrome_trace(str(trace))
        profile["trace"] = str(trace)
        assert profile["gpu_events"] > 0, "No GPU kernel evidence was recorded"
    source = Path(ase.__file__).parent / "optimize" / "bfgs.py"
    report = {
        "passed": True,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device)
        if args.device == "cuda"
        else "CPU",
        "ase": ase.__version__,
        "ase_bfgs_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "dtype": "float64",
        "rtol": rtol,
        "atol": atol,
        "cases": cases,
        "explicit_device_restore": True,
        "profile": profile,
        "scope": "M1 single-system fixed-cell Ops; no Framework/model/Batch/variable-cell/NVIDIA gate",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
