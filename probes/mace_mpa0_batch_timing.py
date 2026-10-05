# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
"""One FP64 native FIRE Batch: native force criterion and bounded wall timing.

Native raw atomic/cell forces drive the stopping observer. No new filter or
optimization algorithm is introduced, and no performance tuning is done here.
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROBE_STARTED = time.perf_counter()

import numpy as np
import torch
from ase import Atoms
from ase.io import read, write

from nvalchemi.dynamics import FIREVariableCell, FusedStage
from nvalchemi.dynamics._ops.stress import stress_to_cell_force
from nvalchemi.dynamics.base import ConvergenceHook, DynamicsStage
from nvalchemi.dynamics.sampler import SizeAwareSampler
from nvalchemi.dynamics.sinks import HostMemory
from nvalchemi.hooks import NeighborListHook
from nvalchemi.models.mace import MACEWrapper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.relax_mace_mpa0 import (
    CIFDataset,
    MODEL_SHA256,
    StepCounter,
    WallDeadline,
    sha256,
    validate_atoms,
)


def native_max(forces, stress, cell, batch_idx, natoms):
    """Maximum row norm of raw atomic F and native full 3x3 cell force."""
    lattice = stress_to_cell_force(
        stress,
        cell,
        torch.linalg.det(cell).abs(),
        keep_aligned=False,
        backend="torch_reference",
    )
    maxima = torch.zeros_like(natoms, dtype=forces.dtype)
    maxima.scatter_reduce_(0, batch_idx.long(), forces.norm(dim=-1), reduce="amax")
    return torch.maximum(maxima, lattice.norm(dim=-1).amax(dim=-1))


def criterion_oracle(device):
    """Independent NumPy raw cell-force result, triclinic/mixed atom counts."""
    rng = np.random.default_rng(24)
    h0 = np.array([[4.0, 0.0, 0.0], [0.4, 3.8, 0.0], [-0.2, 0.3, 4.4]])
    cells, stresses, forces, counts, expected = [], [], [], [], []
    for n in (3, 5, 7):
        cell = h0 @ (np.eye(3) + rng.normal(0, 0.08, (3, 3)))
        force = rng.normal(0, 0.03, (n, 3))
        stress = rng.normal(0, 0.001, (3, 3))
        stress = (stress + stress.T) / 2
        lattice = -abs(np.linalg.det(cell)) * np.linalg.inv(cell).T @ stress
        expected.append(
            max(
                np.linalg.norm(force, axis=1).max(),
                np.linalg.norm(lattice, axis=1).max(),
            )
        )
        cells.append(cell)
        stresses.append(stress)
        forces.append(force)
        counts.append(n)

    def tensor(value):
        return torch.as_tensor(np.array(value), dtype=torch.float64, device=device)

    idx = torch.repeat_interleave(
        torch.arange(3, device=device), torch.tensor(counts, device=device)
    )
    maxima = native_max(
        tensor(np.concatenate(forces)),
        tensor(stresses),
        tensor(cells),
        idx,
        torch.tensor(counts, device=device),
    ).cpu()
    torch.testing.assert_close(
        maxima,
        torch.tensor(expected),
        rtol=2e-12,
        atol=2e-12,
    )
    return {
        "device": str(device),
        "NumPy_cases": 3,
        "max_abs_error": float((maxima - torch.tensor(expected)).abs().max()),
    }


class TimingDataset(CIFDataset):
    def __getitem__(self, i):
        data, info = super().__getitem__(i)
        data.add_system_property(
            "native_fmax",
            torch.full((1, 1), float("inf"), dtype=self.dtype, device=self.device),
        )
        return data, info


class NativeObserver:
    stage = DynamicsStage.AFTER_COMPUTE
    frequency = 1

    def __init__(self):
        self.calls = 0

    def __call__(self, ctx, stage):
        b = ctx.batch
        values = native_max(
            b.forces,
            b.stress,
            b.cell,
            b.batch_idx,
            b.batch_ptr.diff(),
        )
        if not torch.isfinite(values).all():
            raise ValueError("nonfinite native forces")
        active = b.status.flatten() == 0
        b.native_fmax = torch.where(active[:, None], values[:, None], b.native_fmax)
        overflow = active & (values > 100)
        b.exit_reason.flatten()[overflow] = 3
        b.status.flatten()[overflow] = 1
        self.calls += 1


class ExitAndProgress:
    stage = DynamicsStage.AFTER_STEP
    frequency = 1

    def __init__(self, started):
        self.started, self.steps, self.trace = started, 0, []

    def __call__(self, ctx, stage):
        b = ctx.batch
        reason, status = b.exit_reason.flatten(), b.status.flatten()
        reason[(status == 1) & (reason == 0)] = 1
        exhausted = (status == 0) & (b.relax_steps.flatten() >= 500)
        reason[exhausted], status[exhausted] = 2, 1
        self.steps += 1
        if self.steps % 20 == 0:
            item = {
                "step": self.steps,
                "elapsed": time.perf_counter() - self.started,
                "graphs": b.num_graphs,
                "active": int((status == 0).sum()),
                "atoms": b.num_nodes,
            }
            self.trace.append(item)
            print(json.dumps(item), flush=True)


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    assert device.type == "cuda" and torch.version.hip
    assert sha256(args.checkpoint) == MODEL_SHA256
    # A process-local allocator cap protects the shared card. This can produce
    # an explicit allocation failure; it is never a precision/backend fallback.
    torch.cuda.set_per_process_memory_fraction(0.8, device)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    startup = time.perf_counter() - PROBE_STARTED
    oracle_started = time.perf_counter()
    oracle = [criterion_oracle(torch.device("cpu")), criterion_oracle(device)]
    torch.cuda.synchronize(device)
    oracle_seconds = time.perf_counter() - oracle_started
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    selection = json.loads(args.selection.read_text())
    paths = [Path(r["source"]) for r in selection["batch64"]]
    structures = [read(p, format="cif") for p in paths]
    model = MACEWrapper.from_checkpoint(
        args.checkpoint,
        device=device,
        dtype=torch.float64,
        enable_cueq=False,
        compile_model=False,
    )
    model.set_config("active_outputs", {"energy", "forces", "stress"})
    model.eval()
    for parameter in model.model.parameters():
        parameter.requires_grad_(False)
    assert model.model_config.neighbor_config.cutoff == 6
    supported = set(model.model.atomic_numbers.cpu().tolist())
    for atoms in structures:
        validate_atoms(atoms, supported, args.max_atoms)
    assert len(structures) == 64 and sum(map(len, structures)) <= args.max_atoms
    dataset = TimingDataset(structures, list(range(len(paths))), torch.float64, device)
    sampler = SizeAwareSampler(
        dataset, max_atoms=args.max_atoms, max_batch_size=len(paths)
    )
    sink = HostMemory(capacity=len(paths))
    convergence = ConvergenceHook(
        criteria={"key": "native_fmax", "threshold": 0.05},
        source_status=0,
        target_status=1,
    )
    dynamics = FIREVariableCell(
        model=model,
        dt=0.1,
        dt_max=1.0,
        dt_min=0.002,
        maxstep=0.2,
        n_min=5,
        f_inc=1.1,
        f_dec=0.5,
        alpha_start=0.1,
        f_alpha=0.99,
        backend="torch_reference",
        hooks=[StepCounter()],
        convergence_hook=convergence,
        device_type=device.type,
    )
    stage = FusedStage(
        sub_stages=[(0, dynamics)],
        sampler=sampler,
        sinks=[sink],
        refill_frequency=20,
        device_type=device.type,
    )
    stage.register_hook(
        NeighborListHook(
            model.model_config.neighbor_config,
            skin=0.0,
            backend="torch_reference",
            stage=DynamicsStage.BEFORE_COMPUTE,
        )
    )

    class CapacityCheck:
        stage = DynamicsStage.BEFORE_COMPUTE
        frequency = 1

        def __call__(self, ctx, stage):
            b = ctx.batch
            counts = torch.bincount(b.neighbor_list[:, 0].long(), minlength=b.num_nodes)
            if counts.max() > 512:
                raise ValueError(
                    "neighbor row count exceeds supplied 512 limit; no truncation"
                )

    stage.register_hook(CapacityCheck())
    observer, progress = NativeObserver(), ExitAndProgress(started)
    stage.register_hook(observer)
    stage.register_hook(progress)
    stage.register_hook(WallDeadline(started + args.wall_seconds))
    prepared = time.perf_counter()
    failure = None
    try:
        stage.run(n_steps=520)
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
    torch.cuda.synchronize(device)
    dynamics_end = time.perf_counter()
    complete = sink.read().to_data_list() if len(sink) else []
    pending = (
        []
        if stage.active_batch is None
        else stage.active_batch.to("cpu").to_data_list()
    )
    records, arrays = [], {}
    for data in complete + pending:
        source = int(data.source_index.item())
        reason = int(data.exit_reason.item())
        status = {1: "converged", 2: "max_steps", 3: "force_overflow"}.get(
            reason, "calculation_error" if failure else "incomplete"
        )
        atoms = Atoms(
            numbers=data.atomic_numbers.flatten().detach().numpy(),
            positions=data.positions.detach().numpy(),
            cell=data.cell[0].detach().numpy(),
            pbc=True,
        )
        target = args.output / f"{source:04d}_{paths[source].stem}.cif"
        write(target, atoms, format="cif")
        gmax = float(data.native_fmax.item())
        if status == "converged":
            assert gmax <= 0.05
        for key in (
            "positions",
            "cell",
            "forces",
            "stress",
            "energy",
        ):
            arrays[f"{source}_{key}"] = getattr(data, key).detach().numpy()
        records.append(
            {
                "source_index": source,
                "source": str(paths[source]),
                "source_sha256": sha256(paths[source]),
                "system_id": int(data.system_id.item()),
                "natoms": len(atoms),
                "status": status,
                "steps": int(data.relax_steps.item()),
                "native_fmax_eV_A": gmax if np.isfinite(gmax) else None,
                "energy_eV": float(data.energy.item()),
                "fmax_eV_A": float(data.forces.norm(dim=-1).max()),
                "max_abs_stress_GPa": float(data.stress.abs().max()) * 160.21766208,
                "final_cif": str(target),
                "final_cif_sha256": sha256(target),
            }
        )
    np.savez(args.output / "raw-final-states.npz", **arrays)
    records.sort(key=lambda r: r["source_index"])
    (args.output / "results.jsonl").write_text(
        "".join(json.dumps(r, allow_nan=False) + "\n" for r in records)
    )
    torch.cuda.synchronize(device)
    finished = time.perf_counter()
    metadata = {
        "status": "complete"
        if failure is None and len(records) == len(paths) and stage.active_batch is None
        else "incomplete",
        "failure": failure,
        "dtype": "torch.float64",
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "device_total_memory_bytes": torch.cuda.get_device_properties(
            device
        ).total_memory,
        "process_allocator_fraction": 0.8,
        "torch_version": str(torch.__version__),
        "hip_version": torch.version.hip,
        "model_sha256": MODEL_SHA256,
        "candidate_count": selection["candidate_count"],
        "seed": 24,
        "structures": len(paths),
        "total_atoms": sum(map(len, structures)),
        "max_atoms": args.max_atoms,
        "max_batch_size": 64,
        "refill_frequency": 20,
        "arrivals_after_initial_batch": 0,
        "force_tolerance_eV_A": 0.05,
        "force_overflow_eV_A": 100,
        "max_updates_per_structure": 500,
        "startup_seconds": startup,
        "oracle_seconds": oracle_seconds,
        "preparation_seconds": prepared - started,
        "dynamics_seconds": dynamics_end - prepared,
        "export_seconds": finished - dynamics_end,
        "application_seconds": finished - started,
        "probe_total_seconds": finished - PROBE_STARTED,
        "native_force_oracle": oracle,
        "optimizer_parameters": {
            "dt": 0.1,
            "dt_max": 1.0,
            "dt_min": 0.002,
            "maxstep": 0.2,
            "n_min": 5,
            "f_inc": 1.1,
            "f_dec": 0.5,
            "alpha_start": 0.1,
            "f_alpha": 0.99,
        },
        "compute_calls": observer.calls,
        "fused_steps": progress.steps,
        "trace": progress.trace,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    (args.output / "timing.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(metadata, allow_nan=False), flush=True)
    if metadata["status"] != "complete":
        raise RuntimeError(failure or "batch did not drain")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-atoms", type=int, default=2304)
    parser.add_argument("--wall-seconds", type=float, default=600)
    run(parser.parse_args())
