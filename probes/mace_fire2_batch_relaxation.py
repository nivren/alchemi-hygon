"""Batch FIRE2 relaxation of the 92-atom periodic MACE structures."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from ase.io import read

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import FIRE2
from nvalchemi.dynamics.base import ConvergenceHook, DynamicsStage
from nvalchemi.hooks import NeighborListHook
from nvalchemi.models.mace import MACEWrapper


class _ConvergenceRecorder:
    """Record the first step at which each graph satisfies the fmax limit."""

    stage = DynamicsStage.AFTER_STEP
    frequency = 1

    def __init__(self, num_graphs: int, device: torch.device) -> None:
        self.first_step = torch.zeros(num_graphs, dtype=torch.int32, device=device)

    def __call__(self, ctx, stage) -> None:
        converged = ctx.workflow.convergence_hook.evaluate(ctx.batch)
        if converged is None:
            return
        previous = self.first_step.index_select(0, converged)
        current = torch.full_like(previous, ctx.step_count + 1)
        self.first_step[converged] = torch.where(previous == 0, current, previous)


def _fmax_per_graph(batch: Batch) -> torch.Tensor:
    force_norm = torch.linalg.vector_norm(batch.forces, dim=-1)
    result = torch.full(
        (batch.num_graphs,), float("-inf"), dtype=force_norm.dtype, device=batch.device
    )
    result.scatter_reduce_(0, batch.batch_idx.long(), force_norm, reduce="amax", include_self=False)
    return result


class _ProgressLogger:
    """Emit bounded-interval progress and stop a pathological run."""

    stage = DynamicsStage.AFTER_STEP

    def __init__(self, interval: int, start_time: float, max_wall_seconds: float) -> None:
        self.frequency = interval
        self.start_time = start_time
        self.max_wall_seconds = max_wall_seconds

    def __call__(self, ctx, stage) -> None:
        fmax = _fmax_per_graph(ctx.batch)
        if not torch.isfinite(fmax).all():
            raise FloatingPointError(f"non-finite fmax at step {ctx.step_count + 1}")
        elapsed = time.monotonic() - self.start_time
        status = getattr(ctx.batch, "status", None)
        if status is not None:
            status = status.squeeze(-1) if status.dim() == 2 else status
            active = status < 1
            converged_count = int((~active).sum().item())
            active_fmax = float(fmax[active].max().item()) if active.any() else 0.0
        else:
            active_fmax = float(fmax.max().item())
            converged_count = 0
        print(
            json.dumps(
                {
                    "stage": "progress",
                    "step": ctx.step_count + 1,
                    "elapsed_s": elapsed,
                    "max_fmax": float(fmax.max().item()),
                    "active_max_fmax": active_fmax,
                    "converged_count": converged_count,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if elapsed >= self.max_wall_seconds:
            raise TimeoutError(
                f"relaxation exceeded max_wall_seconds={self.max_wall_seconds}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/data/csp_data/perf_v2_sorted/perf_v2_92"),
    )
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--fmax", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument(
        "--skin",
        type=float,
        default=0.5,
        help="Reference neighbor Verlet skin; 0 forces a full rebuild every step.",
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-wall-seconds", type=float, default=190.0)
    parser.add_argument(
        "--dt",
        type=float,
        default=0.01,
        help="Required FIRE2 timestep; optional FIRE2 hyperparameters stay upstream defaults.",
    )
    args = parser.parse_args()
    if args.count < 1 or args.start < 0:
        raise ValueError("--count must be positive and --start must be non-negative")
    if args.fmax <= 0 or args.max_steps < 1 or args.dt <= 0:
        raise ValueError("fmax, max-steps, and dt must be positive")
    if args.log_every < 1 or args.max_wall_seconds <= 0 or args.skin < 0:
        raise ValueError("log-every and max-wall-seconds must be positive; skin must be non-negative")
    if args.device == "cuda" and (not torch.version.hip or not torch.cuda.is_available()):
        raise RuntimeError("HCU device requested but the HIP Torch device is unavailable")

    root = args.data_root.expanduser().resolve()
    files = sorted(root.glob("*.cif"))
    selected = files[args.start : args.start + args.count]
    if len(selected) != args.count:
        raise ValueError(f"requested {args.count} CIFs from {root}, found {len(selected)}")
    structures = [read(path) for path in selected]
    atom_counts = [len(atoms) for atoms in structures]
    if set(atom_counts) != {92}:
        raise ValueError(f"expected 92 atoms per structure, got {sorted(set(atom_counts))}")
    if not all(bool(atoms.pbc.all()) for atoms in structures):
        raise ValueError("the 92-atom relaxation probe requires three-dimensional PBC")

    device = torch.device(args.device)
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model = MACEWrapper.from_checkpoint(str(checkpoint), device=device, dtype=torch.float32)
    model.eval()
    supported = {int(z) for z in model.model.atomic_numbers.detach().cpu().tolist()}
    observed = {int(z) for atoms in structures for z in atoms.numbers.tolist()}
    unknown = sorted(observed - supported)
    if unknown:
        raise ValueError(f"input contains unsupported atomic numbers: {unknown}")

    data_list = []
    for atoms in structures:
        data = AtomicData.from_atoms(atoms)
        data.add_node_property("velocities", torch.zeros(len(atoms), 3))
        data.forces = torch.zeros(len(atoms), 3)
        data.energy = torch.zeros(1, 1)
        data_list.append(data)
    batch = Batch.from_data_list(data_list).to(device)
    batch.status = torch.zeros(batch.num_graphs, 1, dtype=torch.long, device=device)
    print(
        json.dumps(
            {
                "stage": "batch_ready",
                "num_structures": batch.num_graphs,
                "num_nodes": batch.num_nodes,
                "batch_ptr": batch.batch_ptr.cpu().tolist(),
                "device": args.device,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    convergence = ConvergenceHook.from_fmax(
        threshold=args.fmax, source_status=0, target_status=1
    )
    recorder = _ConvergenceRecorder(batch.num_graphs, device)
    start_time = time.monotonic()
    progress = _ProgressLogger(args.log_every, start_time, args.max_wall_seconds)
    dynamics = FIRE2(
        model=model,
        dt=args.dt,
        n_steps=args.max_steps,
        convergence_hook=convergence,
        hooks=[recorder, convergence, progress],
        backend="torch_reference",
    )
    dynamics.register_hook(
        NeighborListHook(
            model.model_config.neighbor_config,
            skin=args.skin,
            backend="torch_reference",
            stage=DynamicsStage.BEFORE_COMPUTE,
        )
    )

    dynamics.run(batch)
    # BaseDynamics preserves the mutable coordinate state of graduated graphs
    # while still evaluating the model on the temporary pre-update state used
    # in that step.  Re-evaluate once at the final coordinates so the reported
    # force norm is aligned with the returned batch, rather than with a stale
    # temporary position.
    dynamics._call_hooks(DynamicsStage.BEFORE_COMPUTE, batch)
    dynamics.compute(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.monotonic() - start_time
    final_fmax = _fmax_per_graph(batch)
    converged_mask = batch.status.squeeze(-1) >= 1
    print(
        json.dumps(
            {
                "status": "passed",
                "device": args.device,
                "torch_device_name": (
                    torch.cuda.get_device_name() if args.device == "cuda" else "cpu"
                ),
                "checkpoint": str(checkpoint),
                "data_root": str(root),
                "files": [path.name for path in selected],
                "num_structures": batch.num_graphs,
                "natoms_per_structure": atom_counts,
                "batch_ptr": batch.batch_ptr.cpu().tolist(),
                "neighbor_edges": int(batch.neighbor_list.shape[0]),
                "fmax_threshold": args.fmax,
                "max_steps": args.max_steps,
                "dt": args.dt,
                "skin": args.skin,
                "optimizer": "FIRE2",
                "backend": "torch_reference",
                "step_count": dynamics.step_count,
                "converged_count": int(converged_mask.sum().item()),
                "first_converged_step": recorder.first_step.cpu().tolist(),
                "final_fmax": final_fmax.detach().cpu().tolist(),
                "max_final_fmax": float(final_fmax.max().item()),
                "elapsed_s": elapsed,
                "scope": "32 periodic 92-atom structures, fixed-cell FIRE2 relaxation",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
