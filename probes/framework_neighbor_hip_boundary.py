#!/usr/bin/env python3
"""Exercise framework ``compute_neighbors(backend='hip')`` on a HCU.

The Torch cell-list framework path is built as the independent oracle.  The
HIP call goes through the public framework selection, ops dispatcher, generic
executor loader, and the native staged pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.base import NeighborListFormat
from nvalchemi.neighbors import compute_neighbors


def _make_data(
    atoms_per_system: int,
    batch_systems: int,
    device: torch.device,
    dtype: torch.dtype,
) -> list[AtomicData]:
    cells = torch.tensor(
        [
            [[16.0, 0.0, 0.0], [0.5, 16.0, 0.0], [0.3, 0.4, 16.0]],
            [[12.0, 0.0, 0.0], [0.4, 12.0, 0.0], [0.2, 0.3, 12.0]],
        ],
        dtype=dtype,
        device=device,
    )
    pbc = torch.tensor(
        [[True, False, True], [False, True, False]], device=device
    )
    generator = torch.Generator(device=device).manual_seed(20260920)
    data: list[AtomicData] = []
    for system in range(batch_systems):
        template = system % 2
        cell = cells[template]
        fractional = torch.rand(
            (atoms_per_system, 3), dtype=dtype, device=device, generator=generator
        )
        positions = fractional @ cell
        data.append(
            AtomicData(
                positions=positions,
                atomic_numbers=torch.ones(
                    atoms_per_system, dtype=torch.int64, device=device
                ),
                cell=cell.unsqueeze(0),
                pbc=pbc[template].unsqueeze(0),
            )
        )
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atoms-per-system", type=int, default=46)
    parser.add_argument("--batch-systems", type=int, default=32)
    parser.add_argument("--capacity", type=int, default=256)
    args = parser.parse_args()
    if min(args.atoms_per_system, args.batch_systems, args.capacity) <= 0:
        raise ValueError("atoms, batch, and capacity must be positive")
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("HIP PyTorch and a visible HCU are required")

    root = Path(__file__).resolve().parent
    for variable, directory in (
        ("NVALCHEMI_HIP_BATCH_CELL_KEY_COUNT_BUILD_DIR", "point38-cell-key-count"),
        ("NVALCHEMI_HIP_CELL_SCAN_BUILD_DIR", "point38-cell-scan"),
        ("NVALCHEMI_HIP_CELL_FILL_BUILD_DIR", "point38-cell-fill"),
        ("NVALCHEMI_HIP_BATCH_CELL_QUERY_BUILD_DIR", "point38-cell-query"),
        ("NVALCHEMI_HIP_BATCH_QUERY_MATERIALIZE_BUILD_DIR", "point38-topology"),
    ):
        os.environ.setdefault(variable, str(root.parent / "artifacts" / directory))

    device = torch.device("cuda")
    dtype = torch.float32
    data = _make_data(args.atoms_per_system, args.batch_systems, device, dtype)
    native = Batch.from_data_list(data)
    compute_neighbors(
        native,
        cutoff=0.6,
        format=NeighborListFormat.MATRIX,
        max_neighbors=args.capacity,
        backend="hip",
        method="cell_list",
    )

    # Build an independent layered Torch oracle.  Calling the high-level
    # reference cell-list allocator here would exercise an unrelated
    # min-cells-per-dimension allocation mismatch at large Batch counts; the
    # staged ABI below is the same oracle used by the native runtime boundary.
    from nvalchemiops._cell_list_abi import (  # noqa: PLC0415
        build_batch_cell_key_counts_reference_into,
        build_cell_atom_list_reference_into,
        build_cell_starts_reference_into,
    )
    from nvalchemiops.torch_reference_cell_list import (  # noqa: PLC0415
        _grid_spec,
        batch_query_cell_list,
    )

    dimensions = []
    radii = []
    for system in range(native.cell.shape[0]):
        dims, radius = _grid_spec(
            native.cell[system],
            0.6,
            max_nbins=8192,
            min_cells_per_dimension=1,
        )
        dimensions.append(dims)
        radii.append(radius)
    dimensions_tensor = torch.stack(dimensions).to(torch.int32)
    radius_tensor = torch.stack(radii).to(torch.int32)
    cell_counts = torch.prod(dimensions_tensor.to(torch.int64), dim=1)
    offsets = torch.zeros_like(cell_counts, dtype=torch.int32)
    if offsets.numel() > 1:
        offsets[1:] = cell_counts.cumsum(0)[:-1].to(torch.int32)
    total_cells = int(cell_counts.sum().item())
    build = (
        torch.empty((native.num_nodes, 3), dtype=torch.int32, device=native.device),
        torch.empty((native.num_nodes, 3), dtype=torch.int32, device=native.device),
        torch.empty(native.num_nodes, dtype=torch.int32, device=native.device),
        torch.empty(total_cells, dtype=torch.int32, device=native.device),
        torch.empty(total_cells, dtype=torch.int32, device=native.device),
        torch.empty(native.num_nodes, dtype=torch.int32, device=native.device),
        torch.empty(total_cells, dtype=torch.int32, device=native.device),
    )
    build_batch_cell_key_counts_reference_into(
        native.positions,
        torch.linalg.inv(native.cell),
        dimensions_tensor,
        native.pbc,
        native.batch_idx,
        offsets,
        *build[:4],
    )
    build_cell_starts_reference_into(build[3], build[4])
    build_cell_atom_list_reference_into(
        build[2], build[3], build[4], build[5], build[6]
    )
    oracle = (
        torch.full(
            (native.num_nodes, args.capacity),
            native.num_nodes,
            dtype=torch.int32,
            device=native.device,
        ),
        torch.zeros(
            (native.num_nodes, args.capacity, 3),
            dtype=torch.int32,
            device=native.device,
        ),
        torch.zeros(native.num_nodes, dtype=torch.int32, device=native.device),
    )
    batch_query_cell_list(
        native.positions,
        native.cell,
        native.pbc,
        0.6,
        native.batch_idx,
        dimensions_tensor,
        radius_tensor,
        build[0],
        build[1],
        build[3],
        build[4],
        build[5],
        *oracle,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(native.neighbor_matrix, oracle[0])
    torch.testing.assert_close(native.num_neighbors, oracle[2])
    torch.testing.assert_close(native.neighbor_matrix_shifts, oracle[1])
    print(
        json.dumps(
            {
                "status": "passed",
                "device": torch.cuda.get_device_name(),
                "torch_hip": torch.version.hip,
                "atoms_per_system": args.atoms_per_system,
                "batch_systems": args.batch_systems,
                "atoms": int(native.num_nodes),
                "capacity": args.capacity,
                "backend": "hip",
                "pipeline": "framework selection -> ops executor -> native build/query/topology",
                "oracle": "independent Torch layered cell-list build/query ABI",
                "scope": "Point38 explicit periodic full-list matrix boundary; no skin/rebuild/half/COO",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
