#!/usr/bin/env python3
"""Probe per-system Torch-reference skin rebuild semantics for a Batch."""

from __future__ import annotations

import argparse
import json

import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.hooks import NeighborListHook
from nvalchemi.hooks._context import HookContext
from nvalchemi.models.base import NeighborConfig, NeighborListFormat


def _call_eager(hook: NeighborListHook, batch: Batch) -> None:
    NeighborListHook.__call__.__wrapped__(hook, HookContext(batch=batch), None)


def run(device: torch.device) -> dict[str, object]:
    structures = [
        AtomicData(
            positions=torch.tensor(
                [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]], device=device
            ),
            atomic_numbers=torch.tensor([1, 1], device=device),
        ),
        AtomicData(
            positions=torch.tensor(
                [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], device=device
            ),
            atomic_numbers=torch.tensor([1, 1], device=device),
        ),
    ]
    batch = Batch.from_data_list(structures)
    hook = NeighborListHook(
        NeighborConfig(cutoff=2.0, format=NeighborListFormat.MATRIX),
        skin=0.5,
        backend="torch_reference",
    )
    _call_eager(hook, batch)
    first_reference = hook._ref_positions.detach().clone()
    first_matrix = batch.neighbor_matrix.detach().clone()

    with torch.no_grad():
        batch.positions[0, 0] += 0.3
    _call_eager(hook, batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    return {
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "backend": hook.backend,
        "batch_ptr": batch.batch_ptr.detach().cpu().tolist(),
        "moved_system": 0,
        "unchanged_reference_system": bool(
            torch.equal(hook._ref_positions[2:], first_reference[2:])
        ),
        "updated_reference_system": bool(
            torch.equal(hook._ref_positions[:2], batch.positions[:2])
        ),
        "unchanged_neighbor_matrix_system": bool(
            torch.equal(batch.neighbor_matrix[2:], first_matrix[2:])
        ),
        "neighbor_matrix": batch.neighbor_matrix.detach().cpu().tolist(),
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
