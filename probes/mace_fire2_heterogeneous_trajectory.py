"""Short MACE/FIRE2 trajectory probe for a heterogeneous periodic Batch.

This probe deliberately uses one 46-atom and one 92-atom CIF in a single
Batch.  It exercises the public fixed-cell FIRE2 reference path together with
the reference neighbor hook and HostMemory-backed SnapshotHook.  It is a
small orchestration probe, not a performance or restart benchmark.
"""

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
from nvalchemi.dynamics.hooks import SnapshotHook
from nvalchemi.dynamics.sinks import HostMemory
from nvalchemi.models.mace import MACEWrapper
from nvalchemi.hooks import NeighborListHook


class _TrajectoryStep:
    stage = DynamicsStage.AFTER_STEP
    frequency = 1

    def __call__(self, ctx, stage) -> None:
        # Keep a graph-level frame label in the snapshot so frame ordering is
        # checked after HostMemory has rebuilt a heterogeneous Batch.
        ctx.batch.trajectory_step.fill_(ctx.step_count + 1)


def _fmax_per_graph(batch: Batch) -> torch.Tensor:
    norms = torch.linalg.vector_norm(batch.forces, dim=-1)
    result = torch.full(
        (batch.num_graphs,), float("-inf"), dtype=norms.dtype, device=batch.device
    )
    result.scatter_reduce_(0, batch.batch_idx.long(), norms, reduce="amax", include_self=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cif-46",
        type=Path,
        default=Path("/data/csp_data/perf_46/formal_c1_1_100_z1_46.cif"),
    )
    parser.add_argument(
        "--cif-92",
        type=Path,
        default=Path(
            "/data/csp_data/perf_v2_sorted/perf_v2_92/formal_c1_2_1000_z1_46.cif"
        ),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--skin", type=float, default=0.5)
    parser.add_argument("--max-wall-seconds", type=float, default=90.0)
    args = parser.parse_args()
    if args.steps < 1 or args.skin < 0 or args.max_wall_seconds <= 0:
        raise ValueError("steps and max-wall-seconds must be positive; skin must be non-negative")
    if args.device == "cuda" and (not torch.version.hip or not torch.cuda.is_available()):
        raise RuntimeError("HCU device requested but the HIP Torch device is unavailable")

    paths = [args.cif_46.expanduser().resolve(), args.cif_92.expanduser().resolve()]
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError(paths)
    structures = [read(path) for path in paths]
    atom_counts = [len(atoms) for atoms in structures]
    if atom_counts != [46, 92]:
        raise ValueError(f"expected [46, 92] atoms, got {atom_counts}")
    if not all(bool(atoms.pbc.all()) for atoms in structures):
        raise ValueError("heterogeneous trajectory probe requires three-dimensional PBC")

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
    batch.status = torch.zeros((2, 1), dtype=torch.long, device=device)
    batch.system_id = torch.tensor([[46], [92]], dtype=torch.long, device=device)
    batch.trajectory_step = torch.zeros((2, 1), dtype=torch.long, device=device)

    # Use a tiny positive threshold so this probe records all requested frames
    # instead of depending on whether either input happens to start converged.
    convergence = ConvergenceHook.from_fmax(
        threshold=1.0e-12, source_status=0, target_status=1
    )
    sink = HostMemory(capacity=args.steps * batch.num_graphs)
    snapshot = SnapshotHook(sink=sink, frequency=1)
    dynamics = FIRE2(
        model=model,
        dt=0.01,
        n_steps=args.steps,
        convergence_hook=convergence,
        hooks=[convergence, _TrajectoryStep(), snapshot],
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

    start = time.monotonic()
    dynamics.run(batch)
    if time.monotonic() - start > args.max_wall_seconds:
        raise TimeoutError(f"probe exceeded max_wall_seconds={args.max_wall_seconds}")
    if device.type == "cuda":
        torch.cuda.synchronize()

    stored = sink.read()
    if stored.num_graphs != args.steps * 2:
        raise AssertionError(f"expected {args.steps * 2} frames, got {stored.num_graphs}")
    expected_ptr = []
    for frame in range(args.steps):
        expected_ptr.extend([frame * 138, frame * 138 + 46])
    expected_ptr.append(args.steps * 138)
    if stored.batch_ptr.cpu().tolist() != expected_ptr:
        raise AssertionError(
            f"unexpected trajectory batch_ptr: {stored.batch_ptr.cpu().tolist()}"
        )
    if not torch.all(stored.batch_idx[1:] >= stored.batch_idx[:-1]):
        raise AssertionError("trajectory batch_idx is not non-decreasing")
    if stored.system_id.squeeze(-1).cpu().tolist() != [46, 92] * args.steps:
        raise AssertionError("trajectory system IDs are not frame-ordered")
    if stored.trajectory_step.squeeze(-1).cpu().tolist() != sum(
        ([frame, frame] for frame in range(1, args.steps + 1)), []
    ):
        raise AssertionError("trajectory frame labels are not preserved")

    edge_src = stored.neighbor_list[:, 0].long()
    edge_dst = stored.neighbor_list[:, 1].long()
    cross_system_edges = int(
        (stored.batch_idx[edge_src] != stored.batch_idx[edge_dst]).sum().item()
    )
    if cross_system_edges:
        raise AssertionError(f"trajectory neighbor list crossed systems: {cross_system_edges}")
    fmax = _fmax_per_graph(batch)
    print(
        json.dumps(
            {
                "status": "passed",
                "device": args.device,
                "torch_device_name": (
                    torch.cuda.get_device_name() if args.device == "cuda" else "cpu"
                ),
                "checkpoint": str(checkpoint),
                "files": [str(path) for path in paths],
                "atom_counts": atom_counts,
                "batch_ptr": batch.batch_ptr.cpu().tolist(),
                "stored_batch_ptr": stored.batch_ptr.cpu().tolist(),
                "stored_frames": args.steps,
                "neighbor_edges_last_frame": int(batch.neighbor_list.shape[0]),
                "cross_system_edges": cross_system_edges,
                "final_fmax": fmax.detach().cpu().tolist(),
                "elapsed_s": time.monotonic() - start,
                "backend": "torch_reference",
                "scope": "heterogeneous [46,92] periodic Batch, fixed-cell FIRE2, HostMemory trajectory",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
