# SPDX-FileCopyrightText: Copyright (c) 2026 alchemi-hygon contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded fixed-cell MACE relaxation, ASE oracle, restart and timing evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import ase
import numpy as np
import torch
from ase import Atoms
from ase.io import read
from ase.optimize import BFGS
from ase_bfgs_reference import DeviceGuard
from mace.calculators import MACECalculator
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import ASECompatibleBFGS
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.hooks.neighbor_list import NeighborListHook
from nvalchemi.models.mace import MACEWrapper


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class TimedBFGS(ASECompatibleBFGS):
    """Probe-only synchronized timings, separate from profiler instrumentation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_times, self.update_times, self.step_times, self.sync_times = (
            [],
            [],
            [],
            [],
        )

    def compute(self, batch):
        synchronize(batch.positions.device)
        start = time.perf_counter()
        with torch.profiler.record_function("bfgs.model"):
            out = super().compute(batch)
        before_sync = time.perf_counter()
        synchronize(batch.positions.device)
        end = time.perf_counter()
        self.model_times.append((end - start) * 1000)
        self.sync_times.append((end - before_sync) * 1000)
        return out

    def pre_update(self, batch):
        synchronize(batch.positions.device)
        start = time.perf_counter()
        with DeviceGuard(batch.positions.device):
            super().pre_update(batch)
        synchronize(batch.positions.device)
        self.update_times.append((time.perf_counter() - start) * 1000)

    def step(self, batch):
        synchronize(batch.positions.device)
        start = time.perf_counter()
        with torch.profiler.record_function("bfgs.framework_step"):
            out = super().step(batch)
        synchronize(batch.positions.device)
        self.step_times.append((time.perf_counter() - start) * 1000)
        return out


def make_dynamics(model, fmax, cls=TimedBFGS, backend="torch_reference"):
    neighbor = NeighborListHook(
        model.model_config.neighbor_config,
        skin=0.0,
        stage=DynamicsStage.BEFORE_COMPUTE,
        backend="torch_reference",
    )
    return cls(model=model, fmax=fmax, hooks=[neighbor], backend=backend)


def run(args):
    device = torch.device(args.device)
    torch.set_default_dtype(torch.float64)
    checkpoint = args.checkpoint.expanduser().resolve()
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != "75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638":
        raise ValueError("probe expects the previously verified MPA-0 checkpoint")
    atoms = (
        read(args.structure)
        if args.structure
        else Atoms(
            "H2O",
            positions=[[0, 0, 0], [1.06, 0, 0], [-0.29, 0.88, 0]],
            cell=[10, 10, 10],
            pbc=False,
        )
    )
    model = MACEWrapper.from_checkpoint(
        checkpoint,
        device=device,
        dtype=torch.float64,
        enable_cueq=False,
        compile_model=False,
    )
    model.set_config("active_outputs", {"energy", "forces"})
    model.eval()
    batch = Batch.from_data_list(
        [AtomicData.from_atoms(atoms, dtype=torch.float64)]
    ).to(device)
    if getattr(batch, "cell", None) is None:
        batch.cell = torch.tensor(
            np.asarray(atoms.cell)[None], dtype=torch.float64, device=device
        )
        batch.pbc = torch.tensor(
            np.asarray(atoms.pbc)[None], dtype=torch.bool, device=device
        )
    batch.forces = torch.zeros_like(batch.positions)
    batch.energy = torch.zeros(1, 1, dtype=torch.float64, device=device)
    batch.system_id = torch.tensor([[0]], device=device)
    initial_cell = batch.cell.clone()
    oracle_atoms = atoms.copy()
    # Independent ASE neighbor construction/calculator on CPU; transfers are probe-only.
    oracle_atoms.calc = MACECalculator(
        model_paths=str(checkpoint), device="cpu", default_dtype="float64"
    )
    oracle = BFGS(oracle_atoms, logfile=None)
    oracle.fmax = args.fmax
    d = make_dynamics(model, args.fmax, backend=args.backend)
    d._ensure_state_initialized(batch)
    d._prime(batch)
    errors = {"positions": 0.0, "forces": 0.0, "energy": 0.0, "hessian": 0.0}
    history = []
    restart = None
    restart_calls = 0
    restart_errors = {"positions": 0.0, "hessian": 0.0}
    repeated_evaluation = {}

    def compare():
        expected = {
            "positions": oracle_atoms.positions,
            "forces": oracle_atoms.get_forces(),
            "energy": np.asarray(oracle_atoms.get_potential_energy()).reshape(1, 1),
        }
        if d.bfgs_state.initialized:
            expected["hessian"] = oracle.H
        for key, target in expected.items():
            actual = d.bfgs_state.hessian if key == "hessian" else getattr(batch, key)
            value = actual.cpu().numpy()
            errors[key] = max(errors[key], float(np.abs(value - target).max()))
            # Per-quantity tolerances for independent MLIP trajectories; no global relaxation.
            tolerance = 2e-7 if key == "hessian" else 2e-9
            np.testing.assert_allclose(
                value, target, rtol=tolerance, atol=tolerance, err_msg=key
            )

    compare()
    start = time.monotonic()
    for _ in range(args.steps):
        if oracle.converged() != (d.convergence_hook.evaluate(batch) is not None):
            raise AssertionError("ASE and framework disagree on convergence")
        if oracle.converged():
            break
        oracle.step()
        oracle.nsteps += 1
        _, converged = d.step(batch)
        compare()
        if not torch.equal(batch.cell, initial_cell):
            raise AssertionError("fixed-cell optimizer changed the cell")
        if restart is not None:
            clone, resumed = restart
            resumed.step(clone)
            restart_calls += 1
            for key, actual, expected in (
                ("positions", clone.positions, batch.positions),
                ("hessian", resumed.bfgs_state.hessian, d.bfgs_state.hessian),
            ):
                restart_errors[key] = max(
                    restart_errors[key], (actual - expected).abs().max().item()
                )
                # GPU MACE uses atomic reductions. Recomputed forces may differ
                # at roundoff; the measured same-geometry variation is recorded below.
                tolerance = (
                    (2e-9 if key == "positions" else 2e-7)
                    if device.type == "cuda"
                    else 0
                )
                torch.testing.assert_close(
                    actual, expected, rtol=tolerance, atol=tolerance
                )
        if d.bfgs_state.step_count == 2:
            clone = batch.clone()
            resumed = make_dynamics(model, args.fmax, backend=args.backend)
            resumed.load_state_dict(d.state_dict())
            resumed._ensure_state_initialized(clone)
            resumed._prime(clone)
            repeated_evaluation = {
                key: (getattr(clone, key) - getattr(batch, key)).abs().max().item()
                for key in ("forces", "energy")
            }
            print(
                "restart same-geometry model recompute:",
                json.dumps(repeated_evaluation),
                flush=True,
            )
            restart = clone, resumed
        history.append(
            {
                "update": d.bfgs_state.step_count,
                "energy_eV": batch.energy.item(),
                "fmax_eV_A": batch.forces.norm(dim=-1).max().item(),
            }
        )
        if converged is not None:
            break
    elapsed = time.monotonic() - start
    converged = d.convergence_hook.evaluate(batch) is not None
    if args.require_convergence and not converged:
        raise AssertionError("real MLIP did not converge in the bounded window")
    profiling = {}
    if args.profile:
        profile_batch = Batch.from_data_list(
            [AtomicData.from_atoms(atoms, dtype=torch.float64)]
        ).to(device)
        if getattr(profile_batch, "cell", None) is None:
            profile_batch.cell = torch.tensor(
                np.asarray(atoms.cell)[None], dtype=torch.float64, device=device
            )
            profile_batch.pbc = torch.tensor(
                np.asarray(atoms.pbc)[None], dtype=torch.bool, device=device
            )
        profile_batch.forces = torch.zeros_like(profile_batch.positions)
        profile_batch.energy = torch.zeros(1, 1, dtype=torch.float64, device=device)
        profile_d = make_dynamics(model, 1e-10, backend=args.backend)
        profile_d.step(profile_batch)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities) as profiler:
            profile_d.run(profile_batch, n_steps=3)
            synchronize(device)
        trace = args.output.with_suffix(".trace.json")
        profiler.export_chrome_trace(str(trace))
        for event in profiler.key_averages():
            if event.key.startswith("bfgs.") or "Synchronize" in event.key:
                profiling[event.key] = {
                    "calls": event.count,
                    "host_total_us": event.cpu_time_total,
                    "device_total_us": event.device_time_total,
                }
    return {
        "status": "passed",
        "device": str(device),
        "torch": torch.__version__,
        "torch_hcu": getattr(torch.version, "__hcu_version__", None),
        "hip": torch.version.hip,
        "ase": ase.__version__,
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "CPU",
        "checkpoint_sha256": digest,
        "natoms": len(atoms),
        "dof": batch.positions.numel(),
        "dtype": str(batch.positions.dtype),
        "implementation": d._backend_selection.implementation_id,
        "neighbor_backend": "torch_reference",
        "fmax": args.fmax,
        "converged": converged,
        "updates": d.bfgs_state.step_count,
        "model_evaluations": len(d.model_times),
        "restart_compared_updates": restart_calls,
        "restart_errors_max_abs": restart_errors,
        "same_geometry_recompute_max_abs": repeated_evaluation,
        "errors_max_abs": errors,
        "history": history,
        "oracle_inclusive_elapsed_s": elapsed,
        "timings_ms": {
            "model": d.model_times,
            "update": d.update_times,
            "step": d.step_times,
            "terminal_sync": d.sync_times,
        },
        "profile": profiling,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument(
        "--backend", choices=("torch_reference", "hip"), default="torch_reference"
    )
    parser.add_argument("--structure", type=Path)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--require-convergence", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = run(args)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("status", "natoms", "updates", "converged", "errors_max_abs")
            }
        )
    )


if __name__ == "__main__":
    main()
