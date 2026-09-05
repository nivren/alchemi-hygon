#!/usr/bin/env python3
"""Run the Torch reference periodic neighbor Hook on CPU or HCU."""

from __future__ import annotations

import argparse
import json

import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.hooks import NeighborListHook
from nvalchemi.hooks._context import HookContext
from nvalchemi.models.base import NeighborConfig, NeighborListFormat


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device requested but unavailable")

    dtype = torch.float64
    cell = torch.diag(
        torch.tensor([2.0, 10.0, 10.0], device=device, dtype=dtype)
    ).unsqueeze(0)
    batch = Batch.from_data_list(
        [
            AtomicData(
                positions=torch.tensor(
                    [[0.1, 0.0, 0.0], [1.9, 0.0, 0.0]],
                    device=device,
                    dtype=dtype,
                ),
                atomic_numbers=torch.tensor([1, 1], device=device),
                cell=cell,
                pbc=torch.tensor([[True, False, False]], device=device),
            )
        ]
    )
    hook = NeighborListHook(
        NeighborConfig(cutoff=0.5, format=NeighborListFormat.MATRIX),
        backend="torch_reference",
    )
    hook(HookContext(batch=batch), hook.stage)
    torch.cuda.synchronize() if device.type == "cuda" else None

    assert batch.neighbor_matrix.tolist() == [[1], [0]]
    assert batch.neighbor_matrix_shifts.tolist() == [[[-1, 0, 0]], [[1, 0, 0]]]
    print(
        json.dumps(
            {
                "device": (
                    torch.cuda.get_device_name() if device.type == "cuda" else "cpu"
                ),
                "backend": hook.backend,
                "neighbor_matrix": batch.neighbor_matrix.tolist(),
                "neighbor_matrix_shifts": batch.neighbor_matrix_shifts.tolist(),
                "num_neighbors": batch.num_neighbors.tolist(),
            }
        )
    )


if __name__ == "__main__":
    main()
