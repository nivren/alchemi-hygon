#!/usr/bin/env python3
"""Bounded CPU/HCU correctness probe for periodic Torch cell-list neighbors."""

from __future__ import annotations

import argparse
import json

import torch

from nvalchemiops.dispatch import dispatch_neighbor_list
from nvalchemiops.torch_reference_cell_list import (
    allocate_cell_list,
    build_cell_list,
    estimate_cell_list_sizes,
    query_cell_list,
)


def _tuples(output: tuple[torch.Tensor, ...]) -> list[tuple[int, ...]]:
    matrix, counts, shifts = (value.detach().cpu() for value in output[:3])
    return sorted(
        (row, int(matrix[row, column]), *(int(v) for v in shifts[row, column]))
        for row, count in enumerate(counts.tolist())
        for column in range(count)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device requested but unavailable")

    dtype = torch.float64
    positions = torch.tensor(
        [
            [0.1, 0.1, 0.1],
            [1.8, 0.2, 0.1],
            [0.4, 1.5, 0.2],
            [0.1, 0.0, 0.0],
            [1.9, 0.0, 0.0],
        ],
        dtype=dtype,
        device=device,
    )
    cells = torch.tensor(
        [
            [[2.0, 0.0, 0.0], [0.5, 1.8, 0.0], [0.2, 0.3, 2.1]],
            [[2.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
        ],
        dtype=dtype,
        device=device,
    )
    pbc = torch.tensor(
        [[True, True, True], [True, False, False]], device=device
    )
    batch_ptr = torch.tensor([0, 3, 5], dtype=torch.int64, device=device)
    common = dict(
        cutoff=0.65,
        cell=cells,
        pbc=pbc,
        batch_ptr=batch_ptr,
        max_neighbors=64,
        return_distances=True,
        return_vectors=True,
    )
    dense = dispatch_neighbor_list(positions, backend="torch_reference", **common)
    full = dispatch_neighbor_list(
        positions, backend="torch_reference", method="cell_list", **common
    )
    half = dispatch_neighbor_list(
        positions,
        backend="torch_reference",
        method="cell_list",
        half_fill=True,
        **common,
    )
    if _tuples(full) != _tuples(dense):
        raise AssertionError("periodic cell-list pair/shift set differs from dense oracle")
    if int(full[1].sum()) != 2 * int(half[1].sum()):
        raise AssertionError("periodic half-list does not contain one reverse representative")

    # Exercise the caller-owned single-system build/query surface.
    local_positions = positions[3:]
    max_cells, radius = estimate_cell_list_sizes(
        cells[1], pbc[1], 0.65, min_cells_per_dimension=1
    )
    scratch = allocate_cell_list(2, max_cells, radius, device)
    build_cell_list(
        local_positions,
        0.65,
        cells[1],
        pbc[1],
        *scratch,
        min_cells_per_dimension=1,
    )
    matrix = torch.full((2, 8), 2, dtype=torch.int32, device=device)
    shifts = torch.zeros((2, 8, 3), dtype=torch.int32, device=device)
    counts = torch.zeros(2, dtype=torch.int32, device=device)
    query_cell_list(
        local_positions,
        0.65,
        cells[1],
        pbc[1],
        *scratch,
        matrix,
        shifts,
        counts,
    )
    if counts.tolist() != [1, 1]:
        raise AssertionError(f"unexpected layered query counts: {counts.tolist()}")

    # Continuous outputs must retain the force-training gradient path.
    grad_positions = local_positions.detach().clone().requires_grad_()
    grad_cell = cells[1].detach().clone().requires_grad_()
    grad_output = dispatch_neighbor_list(
        grad_positions,
        cutoff=0.65,
        cell=grad_cell,
        pbc=pbc[1],
        max_neighbors=8,
        return_distances=True,
        return_vectors=True,
        backend="torch_reference",
        method="cell_list",
    )
    grad_value = grad_output[3].square().sum() + grad_output[4].square().sum()
    first = torch.autograd.grad(
        grad_value, (grad_positions, grad_cell), create_graph=True
    )
    second = torch.autograd.grad(
        first[0].square().sum() + first[1].square().sum(),
        (grad_positions, grad_cell),
    )
    if not all(bool(torch.isfinite(value).all()) for value in second):
        raise AssertionError("periodic cell-list second derivative is non-finite")
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    print(
        json.dumps(
            {
                "device": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else "cpu"
                ),
                "dtype": str(dtype),
                "full_edges": int(full[1].sum()),
                "half_edges": int(half[1].sum()),
                "layered_counts": counts.tolist(),
                "pair_shift_parity": True,
                "second_gradient_finite": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
