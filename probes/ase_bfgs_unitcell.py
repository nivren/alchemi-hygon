# SPDX-License-Identifier: Apache-2.0
"""Bounded M3 UnitCellFilter oracle, HCU device guard, restart and MPA-0."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import ase
import numpy as np
import torch
from torch.utils._pytree import tree_flatten
from ase.build import bulk
from ase.filters import UnitCellFilter
from ase.optimize import BFGS
from ase_bfgs_reference import DeviceGuard
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import ASECompatibleBFGSVariableCell
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.hooks.neighbor_list import NeighborListHook
from test.compatibility.test_ase_bfgs_cell import CellCalculator, CellModel, make_batch


class LifecycleDeviceGuard(DeviceGuard):
    """Also cover Framework status callbacks, allowing only host scalar metadata.

    BaseDynamics creates a bool scalar for convergence-mask assignment. This
    exception accepts no floating CPU values or non-scalar host tensors.
    Numerical pre_update retains the original stricter DeviceGuard.
    """

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.calls[str(func)] += 1
        output = func(*args, **(kwargs or {}))
        for value in tree_flatten(output)[0]:
            if not isinstance(value, torch.Tensor):
                continue
            metadata = (
                value.device.type == "cpu"
                and value.ndim == 0
                and value.dtype in (torch.bool, torch.int32, torch.int64)
            )
            if value.device != self.device and not metadata:
                raise AssertionError(f"host numerical tensor: {func} -> {value.device}")
            if value.is_floating_point() and value.dtype != torch.float64:
                raise AssertionError(f"precision change: {func} -> {value.dtype}")
        return output


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class TimedBFGS(ASECompatibleBFGSVariableCell):
    """Synchronized probe timings; includes Filter/scratch copies in update."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.times = {"model": [], "update": [], "step": []}
        self.update_guard_calls = Counter()

    def compute(self, batch):
        sync(batch.positions.device)
        start = time.perf_counter()
        result = super().compute(batch)
        sync(batch.positions.device)
        self.times["model"].append((time.perf_counter() - start) * 1000)
        return result

    def pre_update(self, batch):
        sync(batch.positions.device)
        start = time.perf_counter()
        guard = DeviceGuard(batch.positions.device)
        with guard:
            super().pre_update(batch)
        self.update_guard_calls.update(guard.calls)
        sync(batch.positions.device)
        self.times["update"].append((time.perf_counter() - start) * 1000)

    def step(self, batch):
        sync(batch.positions.device)
        start = time.perf_counter()
        result = super().step(batch)
        sync(batch.positions.device)
        self.times["step"].append((time.perf_counter() - start) * 1000)
        return result


class ObservedASEBFGS(BFGS):
    """Capture raw generalized direction before ASE maxstep scaling."""

    def determine_step(self, dpos, steplengths):
        self.raw_step = dpos.copy().reshape(-1, 3)
        return super().determine_step(dpos, steplengths)


def run_case(device, backend, real=False, checkpoint=None, options=None):
    options = options or {}
    torch.set_default_dtype(torch.float64)
    checkpoint_hash = None
    if real:
        from mace.calculators import MACECalculator
        from nvalchemi.models.mace import MACEWrapper

        checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        assert (
            checkpoint_hash
            == "75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638"
        )
        atoms = bulk("Al", "fcc", a=4.1, cubic=True)
        atoms.set_cell(
            [[4.12, 0.08, 0.03], [0.04, 4.10, 0.05], [0.02, 0.06, 4.15]],
            scale_atoms=True,
        )
        atoms.positions += np.array(
            [
                [0.02, -0.01, 0.01],
                [-0.02, 0.01, 0.0],
                [0.01, 0.02, -0.01],
                [-0.01, -0.02, 0.0],
            ]
        )
        model = MACEWrapper.from_checkpoint(
            checkpoint,
            device=device,
            dtype=torch.float64,
            enable_cueq=False,
            compile_model=False,
        )
        model.set_config("active_outputs", {"energy", "forces", "stress"})
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        atoms.calc = MACECalculator(
            model_paths=str(checkpoint), device="cpu", default_dtype="float64"
        )
        data = Batch.from_data_list(
            [AtomicData.from_atoms(atoms, dtype=torch.float64)]
        ).to(device)
        data.forces = torch.zeros_like(data.positions)
        data.stress = torch.zeros(1, 3, 3, dtype=torch.float64, device=device)
        data.energy = torch.zeros(1, 1, dtype=torch.float64, device=device)
        data.system_id = torch.tensor([[17]], device=device)

        def make_dynamics():
            hook = NeighborListHook(
                model.model_config.neighbor_config,
                skin=0,
                stage=DynamicsStage.BEFORE_COMPUTE,
                backend="torch_reference",
            )
            return TimedBFGS(model, hooks=[hook], backend=backend, **options)

        steps = 60
    else:
        data = make_batch(device)
        from ase import Atoms

        atoms = Atoms(
            "H2",
            positions=data.positions.cpu().numpy(),
            cell=data.cell[0].cpu().numpy(),
            pbc=True,
        )
        atoms.calc = CellCalculator()
        model = CellModel()

        def make_dynamics():
            return TimedBFGS(model, backend=backend, **options)

        steps = 100
    initial_positions, initial_cell = data.positions.clone(), data.cell.clone()
    initial_reference = np.asarray(atoms.cell).copy()
    filter_ = UnitCellFilter(atoms, **options)
    oracle = ObservedASEBFGS(filter_, logfile=None)
    oracle.fmax = 0.05
    dynamics = make_dynamics()
    dynamics._ensure_state_initialized(data)
    dynamics._prime(data)
    errors, restart_errors, history = {}, {}, []
    resumed = None
    guard = LifecycleDeviceGuard(device)

    def compare():
        targets = dict(
            positions=atoms.positions,
            cell=atoms.cell.array[None],
            forces=atoms.get_forces(),
            stress=atoms.get_stress(voigt=False)[None],
            energy=np.asarray(atoms.get_potential_energy()).reshape(1, 1),
            generalized_forces=filter_.get_forces(),
            generalized_positions=filter_.get_positions(),
        )
        actual = {
            k: getattr(data, k)
            for k in ("positions", "cell", "forces", "stress", "energy")
        }
        actual["generalized_forces"] = dynamics._generalized_forces(data)
        actual["generalized_positions"] = dynamics.filter_state.pack(
            data.positions, data.cell[0]
        )
        if oracle.state is not None:
            targets.update(hessian=oracle.H, raw_step=oracle.raw_step)
            actual.update(
                hessian=dynamics.bfgs_state.hessian,
                raw_step=dynamics.last_step.raw_step,
            )
        for key, target in targets.items():
            value = actual[key].cpu().numpy()
            error = float(np.abs(value - target).max())
            errors[key] = max(errors.get(key, 0), error)
            tol = (
                (2e-7 if key == "hessian" else 2e-9)
                if real
                else (5e-9 if key == "hessian" else 2e-11)
            )
            np.testing.assert_allclose(value, target, atol=tol, rtol=tol, err_msg=key)

    compare()
    started = time.monotonic()
    for _ in range(steps):
        local_converged = dynamics.convergence_hook.evaluate(data) is not None
        assert oracle.converged() == local_converged
        if local_converged:
            break
        oracle.step()
        oracle.nsteps += 1
        if real:
            # Existing reference neighbor hook copies Batch ptr/PBC metadata to
            # host. Guard the complete numerical BFGS pre_update instead, as in
            # M2/E1; do not call this a host-free entire MLIP/neighbors workflow.
            dynamics.step(data)
        else:
            with guard:
                dynamics.step(data)
        compare()
        if resumed is not None:
            clone, continuation = resumed
            if real:
                continuation.step(clone)
            else:
                with guard:
                    continuation.step(clone)
            for key, actual, expected in (
                ("positions", clone.positions, data.positions),
                ("cell", clone.cell, data.cell),
                (
                    "hessian",
                    continuation.bfgs_state.hessian,
                    dynamics.bfgs_state.hessian,
                ),
            ):
                error = (actual - expected).abs().max().item()
                restart_errors[key] = max(restart_errors.get(key, 0), error)
                tol = 2e-7 if key == "hessian" else 2e-9
                torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
        if dynamics.bfgs_state.step_count == 2:
            clone, continuation = data.clone(), make_dynamics()
            continuation.load_state_dict(dynamics.state_dict())
            resumed = clone, continuation
        history.append(
            dict(
                update=dynamics.bfgs_state.step_count,
                energy_eV=data.energy.item(),
                generalized_fmax=dynamics._generalized_forces(data)
                .norm(dim=1)
                .max()
                .item(),
                physical_fmax=data.forces.norm(dim=1).max().item(),
                stress_max_abs=data.stress.abs().max().item(),
                volume_A3=torch.linalg.det(data.cell).abs().item(),
            )
        )
    assert dynamics.convergence_hook.evaluate(data) is not None, (
        "bounded joint convergence not reached"
    )
    assert dynamics.bfgs_state.step_count == oracle.nsteps
    assert resumed is not None and restart_errors
    np.testing.assert_array_equal(
        dynamics.filter_state.orig_cell.cpu().numpy(), initial_reference
    )
    assert all(
        getattr(data, k).device == device
        for k in ("positions", "cell", "forces", "stress")
    )
    return dict(
        status="passed",
        case="mpa0-al-fcc" if real else "analytic",
        options=options,
        device=str(device),
        backend=backend,
        implementation=dynamics._backend_selection.implementation_id,
        natoms=len(atoms),
        scalar_dofs=data.positions.numel() + 9,
        updates=dynamics.bfgs_state.step_count,
        model_evaluations=len(dynamics.times["model"]),
        checkpoint_sha256=checkpoint_hash,
        converged=True,
        errors_max_abs=errors,
        restart_errors_max_abs=restart_errors,
        movement_max_abs=dict(
            positions=(data.positions - initial_positions).abs().max().item(),
            cell=(data.cell - initial_cell).abs().max().item(),
        ),
        history=history,
        timings_ms=dynamics.times,
        device_guard_calls=dict(guard.calls),
        numerical_update_guard_calls=dict(dynamics.update_guard_calls),
        oracle_inclusive_elapsed_s=time.monotonic() - started,
    )


def finite_difference(device):
    from nvalchemiops.ase_unitcell_filter import ASEUnitCellFilter

    data = make_batch(device)
    adapter = ASEUnitCellFilter.prepare(
        data.cell[0], data.num_nodes, scalar_pressure=0.037, cell_factor=2.7
    )
    q = adapter.pack(data.positions, data.cell[0])
    model = CellModel()
    out = model(data)
    gf, _ = adapter.generalized_forces(
        out["forces"], out["stress"][0].contiguous(), data.cell[0]
    )
    finite = torch.empty_like(q)
    guard = DeviceGuard(device)
    with guard:
        for j in range(q.numel()):
            energies = []
            for sign in (1, -1):
                candidate = q.clone()
                candidate.view(-1)[j] += sign * 1e-6
                r, h = adapter.unpack(candidate)
                clone = data.clone()
                clone.positions.copy_(r)
                clone.cell[0].copy_(h)
                energies.append(adapter.enthalpy(model(clone)["energy"], h).reshape(()))
            finite.view(-1)[j] = -(energies[0] - energies[1]) / 2e-6
    torch.testing.assert_close(gf, finite, atol=4e-9, rtol=4e-8)
    return dict(
        status="passed",
        scalar_dofs=q.numel(),
        pressure=0.037,
        max_abs_error=(gf - finite).abs().max().item(),
        device_guard_calls=dict(guard.calls),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument(
        "--backend", choices=("torch_reference", "hip"), default="torch_reference"
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        device = torch.device("cuda", torch.cuda.current_device())
    cases = (
        [
            run_case(
                device,
                args.backend,
                real=True,
                checkpoint=args.checkpoint.expanduser().resolve(),
            )
        ]
        if args.real
        else [
            run_case(device, args.backend, options=options)
            for options in (
                {},
                {"scalar_pressure": 0.037, "cell_factor": 2.7},
                {"mask": [1, 0, 1, 0, 1, 0]},
            )
        ]
    )
    result = dict(
        status="passed",
        ase=ase.__version__,
        torch=torch.__version__,
        torch_hcu=getattr(torch.version, "__hcu_version__", None),
        hip=torch.version.hip,
        device_name=torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "CPU",
        cases=cases,
        finite_difference=None if args.real else finite_difference(device),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": "passed",
                "updates": [c["updates"] for c in cases],
                "errors": [c["errors_max_abs"] for c in cases],
                "finite_difference": result["finite_difference"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
