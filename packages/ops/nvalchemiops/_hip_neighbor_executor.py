# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public executor adapter for the narrow native-HIP Batch neighbor path.

The native stages intentionally expose caller-owned buffers.  This module is
the small public-ABI adapter that allocates those buffers, derives fixed-cell
metadata, and invokes the already validated staged pipeline.  It is selected
only by the central registry; it does not implement backend policy itself.

Point38 deliberately registers only periodic, full-list, dense matrix output.
The adapter therefore does not claim support for half lists, COO output,
target rows, skin/rebuild state, or variable-cell execution.
"""

from __future__ import annotations

from typing import Any

import torch

from nvalchemiops._hip_batch_cell_build import _TrustedBatchCellBuildPlan
from nvalchemiops._hip_batch_neighbor_runtime import (
    run_native_hip_batch_neighbor_into,
)
from nvalchemiops._hip_batch_query_materialize import (
    allocate_batch_query_topology_composite_workspace,
)
from nvalchemiops._hip_cell_scan import cell_starts_hip_workspace_size
from nvalchemiops.backend import BackendUnavailableError
from nvalchemiops.torch_reference import (
    NeighborOverflowError,
    _normalize_periodic_geometry,
    _prepare_batch_ptr,
    _validate_positions,
)
from nvalchemiops.torch_reference_cell_list import _grid_spec


_MAX_NBINS = 8192


def _unsupported(message: str) -> BackendUnavailableError:
    return BackendUnavailableError(
        "native HIP neighbor executor does not support this request: " + message
    )


def _validate_request(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor | None,
    pbc: torch.Tensor | None,
    half_fill: bool,
    return_neighbor_list: bool,
    target_indices: torch.Tensor | None,
    kwargs: dict[str, object],
) -> None:
    _validate_positions(positions)
    if positions.device.type != "cuda" or not torch.version.hip:
        raise RuntimeError(
            "native HIP neighbor executor requires a visible HIP Torch device"
        )
    if not isinstance(cutoff, (int, float)) or cutoff <= 0:
        raise ValueError("cutoff must be positive")
    if cell is None or pbc is None:
        raise _unsupported("a fixed periodic cell and pbc tensor are required")
    if half_fill:
        raise _unsupported("half-list requests are not admitted")
    if return_neighbor_list:
        raise _unsupported("COO/return_neighbor_list output is not admitted")
    if target_indices is not None:
        raise _unsupported("target-index queries are not admitted")
    if kwargs:
        raise _unsupported("unknown options: " + ", ".join(sorted(kwargs)))


def _fixed_cell_metadata(
    cells: torch.Tensor,
    cutoff: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Derive the same conservative grid metadata as the Torch cell-list path."""
    dimensions: list[torch.Tensor] = []
    radii: list[torch.Tensor] = []
    for system in range(cells.shape[0]):
        dims, radius = _grid_spec(
            cells[system],
            cutoff,
            max_nbins=_MAX_NBINS,
            min_cells_per_dimension=1,
        )
        dimensions.append(dims)
        radii.append(radius)
    if not dimensions:
        empty = torch.empty((0, 3), dtype=torch.int32, device=cells.device)
        return (
            empty,
            empty.clone(),
            torch.empty((0,), dtype=torch.int32, device=cells.device),
            torch.empty((0, 3, 3), dtype=cells.dtype, device=cells.device),
        )
    cells_per_dimension = torch.stack(dimensions).to(
        device=cells.device, dtype=torch.int32
    )
    neighbor_search_radius = torch.stack(radii).to(
        device=cells.device, dtype=torch.int32
    )
    cell_counts = torch.prod(cells_per_dimension.to(torch.int64), dim=1)
    max_int32 = torch.iinfo(torch.int32).max
    if bool(torch.any(cell_counts > max_int32)):
        raise ValueError("fixed-cell metadata exceeds int32 cell-offset capacity")
    if bool(cell_counts.sum() > max_int32):
        raise ValueError("Batch cell offsets exceed int32 cell-offset capacity")
    cell_offsets = torch.zeros(
        (cells.shape[0],), dtype=torch.int32, device=cells.device
    )
    if cells.shape[0] > 1:
        cell_offsets[1:] = cell_counts.cumsum(0)[:-1].to(torch.int32)
    inverse_cells = torch.linalg.inv(cells).contiguous()
    return (
        cells_per_dimension,
        neighbor_search_radius,
        cell_offsets,
        inverse_cells,
    )


def _capacity_bounds(
    ptr: torch.Tensor,
    neighbor_search_radius: torch.Tensor,
) -> tuple[int, int]:
    """Return an initial capacity and a finite conservative retry bound."""
    ptr_cpu = ptr.detach().cpu().tolist()
    radii_cpu = neighbor_search_radius.detach().cpu().tolist()
    system_bounds = []
    for system, radius in enumerate(radii_cpu):
        atom_count = int(ptr_cpu[system + 1]) - int(ptr_cpu[system])
        image_cells = 1
        for value in radius:
            image_cells *= 2 * int(value) + 1
        system_bounds.append(max(1, atom_count * image_cells))
    initial = max(
        1,
        max(
            (int(ptr_cpu[system + 1]) - int(ptr_cpu[system]) for system in range(len(ptr_cpu) - 1)),
            default=1,
        ),
    )
    return initial, max(system_bounds, default=1)


def _allocate_outputs(
    n_atoms: int,
    capacity: int,
    dtype: torch.dtype,
    device: torch.device,
    *,
    return_distances: bool,
    return_vectors: bool,
) -> tuple[torch.Tensor, ...]:
    candidate_matrix = torch.empty((n_atoms, capacity), dtype=torch.int32, device=device)
    candidate_shifts = torch.empty(
        (n_atoms, capacity, 3), dtype=torch.int32, device=device
    )
    candidate_counts = torch.empty((n_atoms,), dtype=torch.int32, device=device)
    public_matrix = torch.empty_like(candidate_matrix)
    public_shifts = torch.empty_like(candidate_shifts)
    public_counts = torch.empty_like(candidate_counts)
    distances = (
        torch.empty((n_atoms, capacity), dtype=dtype, device=device)
        if return_distances
        else None
    )
    vectors = (
        torch.empty((n_atoms, capacity, 3), dtype=dtype, device=device)
        if return_vectors
        else None
    )
    return (
        candidate_matrix,
        candidate_shifts,
        candidate_counts,
        public_matrix,
        public_shifts,
        public_counts,
        distances,
        vectors,
    )


def neighbor_list(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor | None = None,
    pbc: torch.Tensor | None = None,
    batch_idx: torch.Tensor | None = None,
    batch_ptr: torch.Tensor | None = None,
    max_neighbors: int | None = None,
    half_fill: bool = False,
    fill_value: int | None = None,
    return_neighbor_list: bool = False,
    *,
    return_distances: bool = False,
    return_vectors: bool = False,
    target_indices: torch.Tensor | None = None,
    **kwargs: object,
) -> tuple[torch.Tensor, ...]:
    """Execute the registered fixed-cell/full-list native-HIP ABI."""
    _validate_request(
        positions,
        cutoff,
        cell,
        pbc,
        half_fill,
        return_neighbor_list,
        target_indices,
        kwargs,
    )
    assert cell is not None and pbc is not None
    positions_work = positions.contiguous()
    ptr = _prepare_batch_ptr(positions_work, batch_idx, batch_ptr)
    num_systems = int(ptr.numel() - 1)
    cells, periodic = _normalize_periodic_geometry(
        cell,
        pbc,
        num_systems=num_systems,
        device=positions_work.device,
        dtype=positions_work.dtype,
    )
    if cells.shape[0] == 0:
        raise ValueError("native HIP neighbor executor requires at least one system")
    if max_neighbors is not None:
        if isinstance(max_neighbors, bool) or not isinstance(max_neighbors, int):
            raise TypeError("max_neighbors must be an int or None")
        if max_neighbors <= 0:
            raise ValueError("native HIP neighbor executor requires positive max_neighbors")
    resolved_fill = positions_work.shape[0] if fill_value is None else int(fill_value)
    if resolved_fill < positions_work.shape[0]:
        raise ValueError("fill_value must be >= num atoms so it cannot collide with an index")

    if positions_work.shape[0] == 0:
        capacity = 0 if max_neighbors is None else max_neighbors
        matrix = torch.empty(
            (0, capacity), dtype=torch.int32, device=positions_work.device
        )
        counts = torch.empty((0,), dtype=torch.int32, device=positions_work.device)
        shifts = torch.empty(
            (0, capacity, 3), dtype=torch.int32, device=positions_work.device
        )
        output: list[torch.Tensor] = [matrix, counts, shifts]
        if return_distances:
            output.append(torch.empty((0, capacity), dtype=positions_work.dtype, device=positions_work.device))
        if return_vectors:
            output.append(torch.empty((0, capacity, 3), dtype=positions_work.dtype, device=positions_work.device))
        return tuple(output)

    batch_idx_work = torch.repeat_interleave(
        torch.arange(num_systems, device=positions_work.device, dtype=torch.int32),
        ptr[1:] - ptr[:-1],
    ).contiguous()
    if batch_idx is not None:
        supplied_batch_idx = batch_idx.to(device=positions_work.device, dtype=torch.int32)
        if not torch.equal(supplied_batch_idx, batch_idx_work):
            raise ValueError("batch_idx and batch_ptr describe different layouts")
    cells_per_dimension, neighbor_search_radius, cell_offsets, inverse_cells = (
        _fixed_cell_metadata(cells, cutoff)
    )
    total_cells = int(torch.prod(cells_per_dimension.to(torch.int64), dim=1).sum().item())
    atom_periodic_shifts = torch.empty(
        (positions_work.shape[0], 3), dtype=torch.int32, device=positions_work.device
    )
    atom_to_cell_mapping = torch.empty_like(atom_periodic_shifts)
    cell_keys = torch.empty(
        (positions_work.shape[0],), dtype=torch.int32, device=positions_work.device
    )
    cell_counts = torch.empty((total_cells,), dtype=torch.int32, device=positions_work.device)
    cell_starts = torch.empty_like(cell_counts)
    cell_atom_list = torch.empty_like(cell_keys)
    cell_cursor = torch.empty_like(cell_counts)
    cell_counts.zero_()
    scan_bytes = cell_starts_hip_workspace_size(cell_counts, cell_starts)
    scan_workspace = torch.empty(
        (scan_bytes,), dtype=torch.uint8, device=positions_work.device
    )
    build_plan = _TrustedBatchCellBuildPlan.initialize(
        positions_work,
        inverse_cells,
        cells_per_dimension,
        periodic,
        batch_idx_work,
        cell_offsets,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        cell_keys,
        cell_counts,
        cell_starts,
        cell_atom_list,
        cell_cursor,
        scan_workspace,
    )

    initial, upper = _capacity_bounds(ptr, neighbor_search_radius)
    capacity = initial if max_neighbors is None else max_neighbors
    while True:
        outputs = _allocate_outputs(
            positions_work.shape[0],
            capacity,
            positions_work.dtype,
            positions_work.device,
            return_distances=return_distances,
            return_vectors=return_vectors,
        )
        (
            candidate_matrix,
            candidate_shifts,
            candidate_counts,
            public_matrix,
            public_shifts,
            public_counts,
            distances,
            vectors,
        ) = outputs
        # The topology workspace validator inspects candidate_counts before
        # the first query writes it.  This buffer is an output of the query,
        # but it must still have a deterministic pre-query value because the
        # executor allocates the workspace before entering the fused build /
        # query / materialization runtime.
        candidate_counts.zero_()
        topology_workspace = allocate_batch_query_topology_composite_workspace(
            candidate_matrix, candidate_counts
        )
        try:
            run_native_hip_batch_neighbor_into(
                build_plan,
                cells,
                cutoff,
                neighbor_search_radius,
                candidate_matrix,
                candidate_shifts,
                candidate_counts,
                public_matrix,
                public_shifts,
                public_counts,
                distances,
                vectors,
                topology_workspace,
                fill_value=resolved_fill,
                gradient_order=2 if (return_distances or return_vectors) else 0,
            )
        except RuntimeError as exc:
            if max_neighbors is not None or "capacity overflow" not in str(exc):
                raise
            if capacity >= upper:
                raise NeighborOverflowError(
                    "native HIP neighbor capacity exceeded conservative retry bound"
                ) from exc
            capacity = min(capacity * 2, upper)
            continue
        output = [public_matrix, public_counts, public_shifts]
        if distances is not None:
            output.append(distances)
        if vectors is not None:
            output.append(vectors)
        return tuple(output)


__all__ = ["neighbor_list"]
