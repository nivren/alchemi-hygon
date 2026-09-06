"""Small real-MACE FIRE2 inflight replacement probe.

Four periodic structures (46, 92, 46, 92 atoms) are sampled into a single
heterogeneous two-system batch.  A high convergence threshold makes the first
pair graduate, the sampler supplies the second pair, and HostMemory records
each graduated system exactly once.  This is an orchestration probe, not a
convergence or performance benchmark.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from ase import Atoms
from ase.io import read

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import FIRE2, FusedStage
from nvalchemi.dynamics.base import ConvergenceHook, DynamicsStage
from nvalchemi.dynamics.sampler import SizeAwareSampler
from nvalchemi.dynamics.sinks import HostMemory
from nvalchemi.hooks import NeighborListHook
from nvalchemi.models.mace import MACEWrapper


class _CIFDataset:
    def __init__(self, paths: list[Path], device: torch.device) -> None:
        self.structures: list[Atoms] = [read(path) for path in paths]
        self.atom_counts = [len(atoms) for atoms in self.structures]
        self.device = device

    def __len__(self) -> int:
        return len(self.structures)

    def get_metadata(self, index: int) -> tuple[int, int]:
        return self.atom_counts[index], 0

    def __getitem__(self, index: int) -> tuple[AtomicData, dict]:
        atoms = self.structures[index]
        data = AtomicData.from_atoms(atoms).to(self.device)
        data.add_node_property(
            "velocities", torch.zeros(len(atoms), 3, device=self.device)
        )
        data.forces = torch.zeros(len(atoms), 3, device=self.device)
        data.energy = torch.zeros(1, 1, device=self.device)
        data.add_system_property(
            "trajectory_step",
            torch.zeros(1, 1, dtype=torch.long, device=self.device),
        )
        return data, {}


class _FrameLabelHook:
    stage = DynamicsStage.AFTER_STEP
    frequency = 1

    def __call__(self, ctx, stage) -> None:
        ctx.batch.trajectory_step.fill_(ctx.step_count + 1)


def _initialize_batch(batch: Batch) -> None:
    # The sampler carries the fields needed by FIRE2.  Keep this callback as a
    # guard for the initial batch so the probe also checks the documented
    # FusedStage init hook boundary.
    if getattr(batch, "velocities", None) is None:
        batch.add_node_property("velocities", torch.zeros_like(batch.positions))
    if getattr(batch, "forces", None) is None:
        batch.forces = torch.zeros_like(batch.positions)
    if getattr(batch, "energy", None) is None:
        batch.energy = torch.zeros(batch.num_graphs, 1, device=batch.device)


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
    parser.add_argument("--max-wall-seconds", type=float, default=90.0)
    args = parser.parse_args()
    if args.max_wall_seconds <= 0:
        raise ValueError("max-wall-seconds must be positive")
    if args.device == "cuda" and (not torch.version.hip or not torch.cuda.is_available()):
        raise RuntimeError("HCU device requested but the HIP Torch device is unavailable")

    base_paths = [args.cif_46.expanduser().resolve(), args.cif_92.expanduser().resolve()]
    paths = base_paths + base_paths
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError(paths)
    device = torch.device(args.device)
    dataset = _CIFDataset(paths, device)
    if dataset.atom_counts != [46, 92, 46, 92]:
        raise ValueError(f"expected [46, 92, 46, 92], got {dataset.atom_counts}")
    if not all(bool(atoms.pbc.all()) for atoms in dataset.structures):
        raise ValueError("inflight probe requires three-dimensional PBC")

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model = MACEWrapper.from_checkpoint(str(checkpoint), device=device, dtype=torch.float32)
    model.eval()
    supported = {int(z) for z in model.model.atomic_numbers.detach().cpu().tolist()}
    observed = {int(z) for atoms in dataset.structures for z in atoms.numbers.tolist()}
    unknown = sorted(observed - supported)
    if unknown:
        raise ValueError(f"input contains unsupported atomic numbers: {unknown}")

    sampler = SizeAwareSampler(dataset, max_atoms=138, max_batch_size=2)
    sink = HostMemory(capacity=4)
    convergence = ConvergenceHook.from_fmax(
        threshold=10.0, source_status=0, target_status=1
    )
    fire2 = FIRE2(
        model=model,
        dt=0.01,
        n_steps=None,
        convergence_hook=convergence,
        backend="torch_reference",
        hooks=[convergence, _FrameLabelHook()],
        device_type=args.device,
    )
    fused = FusedStage(
        sub_stages=[(0, fire2)],
        sampler=sampler,
        sinks=[sink],
        refill_frequency=1,
        device_type=args.device,
        init_fn=_initialize_batch,
    )
    fused.register_hook(
        NeighborListHook(
            model.model_config.neighbor_config,
            skin=0.5,
            backend="torch_reference",
            stage=DynamicsStage.BEFORE_COMPUTE,
        )
    )

    start = time.monotonic()
    result = fused.run(n_steps=6)
    elapsed = time.monotonic() - start
    if elapsed > args.max_wall_seconds:
        raise TimeoutError(f"probe exceeded max_wall_seconds={args.max_wall_seconds}")
    if result is not None:
        raise AssertionError("expected sampler exhaustion after four graduated samples")
    if len(sink) != 4:
        raise AssertionError(f"expected four graduated samples, got {len(sink)}")
    stored = sink.read()
    # The sampler initially round-robins [46, 92], then its budgeted refill
    # scans bins largest-first and admits [92, 46] for the freed 138 atoms.
    expected_ptr = [0, 46, 138, 230, 276]
    if stored.batch_ptr.cpu().tolist() != expected_ptr:
        raise AssertionError(f"unexpected graduated batch_ptr: {stored.batch_ptr.cpu().tolist()}")
    if stored.system_id.squeeze(-1).cpu().tolist() != [0, 1, 2, 3]:
        raise AssertionError("system IDs were not written exactly once in sampler order")
    if stored.trajectory_step.squeeze(-1).cpu().tolist() != [1, 1, 2, 2]:
        raise AssertionError("graduated frame labels were not preserved")
    if stored.status.squeeze(-1).cpu().tolist() != [1, 1, 1, 1]:
        raise AssertionError("graduated samples did not carry target status")
    if not torch.all(stored.batch_idx[1:] >= stored.batch_idx[:-1]):
        raise AssertionError("graduated batch_idx is not non-decreasing")
    edge_src = stored.neighbor_list[:, 0].long()
    edge_dst = stored.neighbor_list[:, 1].long()
    cross_system_edges = int(
        (stored.batch_idx[edge_src] != stored.batch_idx[edge_dst]).sum().item()
    )
    if cross_system_edges:
        raise AssertionError(f"graduated neighbor list crossed systems: {cross_system_edges}")

    print(
        json.dumps(
            {
                "status": "passed",
                "device": args.device,
                "torch_device_name": (
                    torch.cuda.get_device_name() if args.device == "cuda" else "cpu"
                ),
                "checkpoint": str(checkpoint),
                "atom_counts": dataset.atom_counts,
                "initial_batch_ptr": [0, 46, 138],
                "graduated_batch_ptr": stored.batch_ptr.cpu().tolist(),
                "system_ids": stored.system_id.squeeze(-1).cpu().tolist(),
                "trajectory_steps": stored.trajectory_step.squeeze(-1).cpu().tolist(),
                "cross_system_edges": cross_system_edges,
                "neighbor_edges": int(stored.neighbor_list.shape[0]),
                "elapsed_s": elapsed,
                "backend": "torch_reference",
                "scope": "real MACE/FIRE2 heterogeneous [46,92] inflight replacement",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
