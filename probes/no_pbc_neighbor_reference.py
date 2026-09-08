#!/usr/bin/env python3
"""Short CPU/HCU probe for no-PBC Torch-reference neighbor assembly."""

from __future__ import annotations

import argparse
import json
import time

import torch

from nvalchemiops.torch_backend import dispatch_neighbor_list


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _positions(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(20260908)
    sizes = torch.tensor([46, 92], device=device, dtype=torch.int64)
    first = torch.rand((46, 3), device=device, dtype=dtype, generator=generator) * 5.0
    second = (
        torch.rand((92, 3), device=device, dtype=dtype, generator=generator) * 5.0
        + torch.tensor([20.0, 0.0, 0.0], device=device, dtype=dtype)
    )
    return torch.cat((first, second)), torch.cat((torch.zeros(1, device=device, dtype=torch.int64), sizes.cumsum(0)))


def _run_once(
    positions: torch.Tensor,
    batch_ptr: torch.Tensor,
    *,
    half: bool,
) -> tuple[tuple[torch.Tensor, ...], float]:
    _synchronize(positions.device)
    started = time.perf_counter()
    output = dispatch_neighbor_list(
        positions,
        cutoff=1.5,
        batch_ptr=batch_ptr,
        half_fill=half,
        return_distances=True,
        return_vectors=True,
        backend="torch_reference",
    )
    _synchronize(positions.device)
    return output, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steady", type=int, default=2)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    positions, batch_ptr = _positions(device, dtype)
    for _ in range(args.warmup):
        _run_once(positions, batch_ptr, half=False)

    result: dict[str, object] = {
        "backend": "torch_reference",
        "device": str(device),
        "dtype": args.dtype,
        "batch_ptr": batch_ptr.cpu().tolist(),
        "cases": {},
    }
    for half in (False, True):
        samples: list[float] = []
        output: tuple[torch.Tensor, ...] | None = None
        for _ in range(args.steady):
            output, elapsed = _run_once(positions, batch_ptr, half=half)
            samples.append(elapsed)
        assert output is not None
        matrix, counts, distances, vectors = output
        result["cases"]["half" if half else "full"] = {
            "seconds": samples,
            "num_edges": int(counts.sum().item()),
            "max_neighbors": int(counts.max().item()),
            "matrix_shape": list(matrix.shape),
            "distance_shape": list(distances.shape),
            "vector_shape": list(vectors.shape),
        }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
