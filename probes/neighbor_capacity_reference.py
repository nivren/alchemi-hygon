#!/usr/bin/env python3
"""Probe Torch-reference staging capacity grow/shrink semantics."""

from __future__ import annotations

import argparse
import json

import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.hooks import NeighborListHook
from nvalchemi.hooks._context import HookContext
from nvalchemi.models.base import NeighborConfig, NeighborListFormat


def _call(hook: NeighborListHook, batch: Batch) -> None:
    NeighborListHook.__call__.__wrapped__(hook, HookContext(batch=batch), None)


def _line_batch(num_atoms: int) -> Batch:
    positions = torch.zeros(num_atoms, 3)
    positions[:, 0] = torch.arange(num_atoms, dtype=torch.float32)
    return Batch.from_data_list(
        [
            AtomicData(
                positions=positions,
                atomic_numbers=torch.ones(num_atoms, dtype=torch.long),
            )
        ]
    )


def _make_sparse(batch: Batch) -> None:
    with torch.no_grad():
        batch.positions[0, 0] += 0.3
        batch.positions[2:, 0] = (
            torch.arange(2, batch.num_nodes, dtype=batch.positions.dtype) * 100
        )


def run(device: torch.device) -> dict[str, object]:
    grow_hook = NeighborListHook(
        NeighborConfig(cutoff=5.0, format=NeighborListFormat.MATRIX),
        backend="torch_reference",
        max_neighbors=1,
    )
    grow_batch = _line_batch(4).to(device)
    _call(grow_hook, grow_batch)
    assert grow_hook._max_neighbors == 16
    assert grow_batch.num_neighbors.tolist() == [3, 3, 3, 3]

    mixed_hook = NeighborListHook(
        NeighborConfig(cutoff=5.0, format=NeighborListFormat.MATRIX),
        backend="torch_reference",
        max_neighbors=1,
    )
    mixed_batch = Batch.from_data_list(
        [
            AtomicData(
                positions=torch.arange(12, dtype=torch.float32, device=device).reshape(4, 3)
                * 0.0
                + torch.tensor(
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
                    device=device,
                ),
                atomic_numbers=torch.ones(4, dtype=torch.long, device=device),
            ),
            AtomicData(
                positions=torch.tensor(
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device=device
                ),
                atomic_numbers=torch.ones(2, dtype=torch.long, device=device),
            ),
        ]
    )
    _call(mixed_hook, mixed_batch)
    assert mixed_hook._max_neighbors == 16
    assert mixed_batch.batch_ptr.tolist() == [0, 4, 6]
    assert mixed_batch.num_neighbors.tolist() == [3, 3, 3, 3, 1, 1]

    shrink_hook = NeighborListHook(
        NeighborConfig(cutoff=20.0, format=NeighborListFormat.MATRIX),
        backend="torch_reference",
        skin=0.5,
    )
    shrink_batch = _line_batch(17).to(device)
    _call(shrink_hook, shrink_batch)
    initial_capacity = shrink_hook._max_neighbors
    _make_sparse(shrink_batch)
    _call(shrink_hook, shrink_batch)
    assert initial_capacity == 32
    assert shrink_hook._max_neighbors == 16
    assert int(shrink_batch.num_neighbors.max()) == 1

    floor_hook = NeighborListHook(
        NeighborConfig(cutoff=20.0, format=NeighborListFormat.MATRIX),
        backend="torch_reference",
        skin=0.5,
        max_neighbors=24,
    )
    floor_batch = _line_batch(17).to(device)
    _call(floor_hook, floor_batch)
    _make_sparse(floor_batch)
    _call(floor_hook, floor_batch)
    assert floor_hook._max_neighbors == 24
    assert int(floor_batch.num_neighbors.max()) == 1

    trajectory_hook = NeighborListHook(
        NeighborConfig(cutoff=20.0, format=NeighborListFormat.MATRIX),
        backend="torch_reference",
        skin=0.5,
    )
    trajectory_batch = _line_batch(17).to(device)
    trajectory_capacities: list[int] = []
    for step in range(6):
        with torch.no_grad():
            if step % 2 == 0:
                trajectory_batch.positions[:, 0] = torch.arange(
                    trajectory_batch.num_nodes,
                    dtype=trajectory_batch.positions.dtype,
                    device=device,
                )
            else:
                trajectory_batch.positions[:, 0] = (
                    torch.arange(
                        trajectory_batch.num_nodes,
                        dtype=trajectory_batch.positions.dtype,
                        device=device,
                    )
                    * 100
                )
                trajectory_batch.positions[1, 0] = 1.0
            trajectory_batch.positions[0, 0] += 0.3
        _call(trajectory_hook, trajectory_batch)
        trajectory_capacities.append(trajectory_hook._max_neighbors)
    assert trajectory_capacities == [32, 16, 32, 16, 32, 16]

    return {
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "backend": "torch_reference",
        "grow": {
            "initial_override": 1,
            "final_capacity": grow_hook._max_neighbors,
            "counts": grow_batch.num_neighbors.detach().cpu().tolist(),
        },
        "mixed_batch": {
            "batch_ptr": mixed_batch.batch_ptr.detach().cpu().tolist(),
            "capacity": mixed_hook._max_neighbors,
            "counts": mixed_batch.num_neighbors.detach().cpu().tolist(),
        },
        "shrink": {
            "initial_capacity": initial_capacity,
            "final_capacity": shrink_hook._max_neighbors,
            "max_count": int(shrink_batch.num_neighbors.max()),
        },
        "floor": {
            "override": 24,
            "final_capacity": floor_hook._max_neighbors,
            "max_count": int(floor_batch.num_neighbors.max()),
        },
        "trajectory": {
            "capacities": trajectory_capacities,
            "max_count": int(trajectory_batch.num_neighbors.max()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested cuda but no HCU is visible")
    print(json.dumps(run(device), sort_keys=True))


if __name__ == "__main__":
    main()
