"""Run fixed-cell MACE dynamics through the public Torch-reference APIs.

This probe intentionally exercises a heterogeneous ``Batch`` (two systems
with different atom counts), the model-owned reference neighbor hook, and one
of the public dynamics classes.  It is a cross-device probe, not a replacement
for the upstream dynamics tests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from ase import Atoms
from ase.io import read

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import FIRE, FIRE2, NVE
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.models.mace import MACEWrapper


def _structures(cifs: list[Path] | None) -> list[Atoms]:
    if cifs:
        structures = [read(path.expanduser().resolve()) for path in cifs]
    else:
        structures = [
            Atoms(
                "H2O",
                positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.0, 0.96, 0.0]],
            ),
            Atoms(
                "CH4",
                positions=[
                    [0.0, 0.0, 0.0],
                    [0.63, 0.63, 0.63],
                    [-0.63, -0.63, 0.63],
                    [-0.63, 0.63, -0.63],
                    [0.63, -0.63, -0.63],
                ],
            ),
        ]
    if not structures:
        raise ValueError("at least one structure is required")
    if any(bool(atoms.pbc.any()) for atoms in structures):
        raise ValueError(
            "mace_dynamics_reference currently requires non-periodic structures; "
            "periodic reference neighbor dynamics is a separate feature slice"
        )
    return structures


def _batch(structures: list[Atoms], device: torch.device) -> Batch:
    data_list = []
    for atoms in structures:
        data = AtomicData.from_atoms(atoms)
        data.add_node_property("velocities", torch.zeros(len(atoms), 3))
        data.forces = torch.zeros(len(atoms), 3)
        data.energy = torch.zeros(1, 1)
        data_list.append(data)
    return Batch.from_data_list(data_list).to(device)


def _run(
    mode: str,
    model: MACEWrapper,
    structures: list[Atoms],
    device: torch.device,
    steps: int,
) -> dict[str, object]:
    batch = _batch(structures, device)
    dynamics_type = {"nve": NVE, "fire": FIRE, "fire2": FIRE2}[mode]
    dynamics = dynamics_type(
        model=model,
        dt=0.01,
        n_steps=steps,
        backend="torch_reference",
    )
    for hook in model.make_neighbor_hooks(neighbor_backend="torch_reference"):
        dynamics.register_hook(hook, stage=DynamicsStage.BEFORE_COMPUTE)
    dynamics.run(batch)

    edge_src = batch.neighbor_list[:, 0].long()
    edge_dst = batch.neighbor_list[:, 1].long()
    cross_system_edges = int(
        (batch.batch_idx[edge_src] != batch.batch_idx[edge_dst]).sum().item()
    )
    if cross_system_edges:
        raise AssertionError(f"neighbor list crossed systems: {cross_system_edges}")
    if not torch.isfinite(batch.positions).all() or not torch.isfinite(batch.forces).all():
        raise FloatingPointError(f"{mode} produced non-finite state")
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {
        "mode": mode,
        "steps": steps,
        "num_systems": batch.num_graphs,
        "natoms_per_system": [int(value) for value in batch.num_nodes_per_graph],
        "batch_ptr": batch.batch_ptr.cpu().tolist(),
        "neighbor_edges": int(batch.neighbor_list.shape[0]),
        "cross_system_edges": cross_system_edges,
        "energy": batch.energy.detach().cpu().tolist(),
        "force_norm": float(torch.linalg.vector_norm(batch.forces).item()),
        "position_norm": float(torch.linalg.vector_norm(batch.positions).item()),
        "finite": True,
        "backend": "torch_reference",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cif", type=Path, nargs="*")
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--mode", choices=("nve", "fire", "fire2", "all"), default="all")
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.device == "cuda" and (not torch.version.hip or not torch.cuda.is_available()):
        raise RuntimeError("HCU device requested but the HIP Torch device is unavailable")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    device = torch.device(args.device)
    structures = _structures(args.cif)
    model = MACEWrapper.from_checkpoint(str(checkpoint), device=device, dtype=torch.float32)
    model.eval()
    supported = {int(z) for z in model.model.atomic_numbers.detach().cpu().tolist()}
    observed = {int(z) for atoms in structures for z in atoms.numbers.tolist()}
    unknown = sorted(observed - supported)
    if unknown:
        raise ValueError(f"input contains unsupported atomic numbers: {unknown}")
    modes = ("nve", "fire", "fire2") if args.mode == "all" else (args.mode,)
    results = [_run(mode, model, structures, device, args.steps) for mode in modes]
    print(
        json.dumps(
            {
                "status": "passed",
                "device": args.device,
                "torch_device_name": (
                    torch.cuda.get_device_name() if args.device == "cuda" else "cpu"
                ),
                "checkpoint": str(checkpoint),
                "modes": results,
                "scope": (
                    "MACEWrapper + reference NeighborListHook + public NVE/FIRE/FIRE2; "
                    "fixed-cell, non-PBC, heterogeneous batch"
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
