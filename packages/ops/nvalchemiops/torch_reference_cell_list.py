# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Warp-independent Torch reference cell-list neighbor construction.

This module implements the core upstream build/query layering.  It intentionally
defers target rows, pair callbacks, pair-centric queries, and automatic strategy
selection.  Neighbor topology is discrete; requested vectors and distances are
ordinary differentiable Torch expressions.
"""

from __future__ import annotations

import itertools
import math

import torch

from nvalchemiops._cell_list_abi import (
    build_cell_atom_list_reference_into,
    build_cell_csr_reference_into,
    build_cell_keys_reference_into,
)
from nvalchemiops.torch_reference import (
    NeighborOverflowError,
    _normalize_periodic_geometry,
    _prepare_batch_ptr,
    _validate_positions,
)


PairData = tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]


def _empty_pairs(positions: torch.Tensor) -> PairData:
    return (
        torch.empty(0, dtype=torch.long, device=positions.device),
        torch.empty(0, dtype=torch.long, device=positions.device),
        torch.empty((0, 3), dtype=torch.int32, device=positions.device),
        torch.empty(0, dtype=positions.dtype, device=positions.device),
        torch.empty((0, 3), dtype=positions.dtype, device=positions.device),
    )


def _geometry(
    cell: torch.Tensor,
    pbc: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cells, periodic = _normalize_periodic_geometry(
        cell,
        pbc,
        num_systems=1,
        device=positions.device,
        dtype=positions.dtype,
    )
    determinant = torch.linalg.det(cells[0])
    if not bool(torch.isfinite(determinant)) or bool(determinant.abs() == 0):
        raise RuntimeError("Cell with volume == 0.0 detected and is not supported")
    return cells[0], periodic[0]


def _grid_spec(
    cell: torch.Tensor,
    cutoff: float,
    *,
    max_nbins: int,
    min_cells_per_dimension: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")
    if max_nbins <= 0 or min_cells_per_dimension <= 0:
        raise ValueError("cell-list allocation limits must be positive")
    try:
        inverse = torch.linalg.inv(cell)
    except RuntimeError as exc:
        raise RuntimeError(
            "Cell with volume == 0.0 detected and is not supported"
        ) from exc
    reciprocal_norm = torch.linalg.vector_norm(inverse, dim=0)
    dimensions = torch.floor(reciprocal_norm.reciprocal() / cutoff).to(torch.int64)
    dimensions.clamp_(min=min_cells_per_dimension)
    while math.prod(int(v.item()) for v in dimensions) > max_nbins:
        largest = int(torch.argmax(dimensions).item())
        if int(dimensions[largest].item()) <= 1:
            break
        dimensions[largest] = (dimensions[largest] + 1) // 2
    # If |dr| < cutoff then |df[d]| <= cutoff * ||inv(cell)[:, d]||.
    # The extra cell covers floor-bin endpoints conservatively.
    radius = torch.ceil(cutoff * reciprocal_norm * dimensions).to(torch.int64) + 1
    return dimensions.to(torch.int32), radius.to(torch.int32)


def allocate_query_sort_scratch(
    total_atoms: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.empty((int(total_atoms), 3), dtype=dtype, device=device),
        torch.empty((int(total_atoms), 3), dtype=torch.int32, device=device),
    )


def estimate_cell_list_sizes(
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: float,
    max_nbins: int = 524288,
    min_cells_per_dimension: int = 4,
) -> tuple[int, torch.Tensor]:
    probe = torch.empty((0, 3), dtype=cell.dtype, device=cell.device)
    geometry, _ = _geometry(cell, pbc, probe)
    dimensions, radius = _grid_spec(
        geometry,
        cutoff,
        max_nbins=max_nbins,
        min_cells_per_dimension=min_cells_per_dimension,
    )
    return math.prod(int(v.item()) for v in dimensions), radius


def estimate_batch_cell_list_sizes(
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: float,
    max_nbins: int = 8192,
    min_cells_per_dimension: int = 4,
) -> tuple[int, torch.Tensor]:
    if cell.ndim != 3 or cell.shape[1:] != (3, 3):
        raise ValueError("cell must have shape (B, 3, 3)")
    if pbc.shape != (cell.shape[0], 3):
        raise ValueError("pbc must have shape (B, 3)")
    total = 0
    radii: list[torch.Tensor] = []
    for system in range(cell.shape[0]):
        size, radius = estimate_cell_list_sizes(
            cell[system],
            pbc[system],
            cutoff,
            max_nbins=max_nbins,
            min_cells_per_dimension=min_cells_per_dimension,
        )
        total += size
        radii.append(radius)
    if not radii:
        return 0, torch.empty((0, 3), dtype=torch.int32, device=cell.device)
    return total, torch.stack(radii)


def allocate_cell_list(
    total_atoms: int,
    max_total_cells: int,
    neighbor_search_radius: torch.Tensor,
    device: torch.device | str,
) -> tuple[torch.Tensor, ...]:
    dimension_shape = (
        (neighbor_search_radius.shape[0], 3)
        if neighbor_search_radius.ndim == 2
        else (3,)
    )
    return (
        torch.zeros(dimension_shape, dtype=torch.int32, device=device),
        neighbor_search_radius.to(device=device, dtype=torch.int32).clone(),
        torch.zeros((total_atoms, 3), dtype=torch.int32, device=device),
        torch.zeros((total_atoms, 3), dtype=torch.int32, device=device),
        torch.zeros(max_total_cells, dtype=torch.int32, device=device),
        torch.zeros(max_total_cells, dtype=torch.int32, device=device),
        torch.empty(total_atoms, dtype=torch.int32, device=device),
    )


def _build_into(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cells_per_dimension: torch.Tensor,
    atom_periodic_shifts: torch.Tensor,
    atom_to_cell_mapping: torch.Tensor,
    atoms_per_cell_count: torch.Tensor,
    cell_atom_start_indices: torch.Tensor,
    cell_atom_list: torch.Tensor,
    *,
    min_cells_per_dimension: int,
    global_atom_offset: int = 0,
) -> int:
    _validate_positions(positions)
    n_atoms = positions.shape[0]
    if cells_per_dimension.shape != (3,):
        raise ValueError("cells_per_dimension must have shape (3,)")
    if atom_periodic_shifts.shape != (n_atoms, 3):
        raise ValueError("atom_periodic_shifts must have shape (N, 3)")
    if atom_to_cell_mapping.shape != (n_atoms, 3):
        raise ValueError("atom_to_cell_mapping must have shape (N, 3)")
    if cell_atom_list.shape != (n_atoms,):
        raise ValueError("cell_atom_list must have shape (N,)")
    geometry, periodic = _geometry(cell, pbc, positions)
    dimensions, _ = _grid_spec(
        geometry,
        cutoff,
        max_nbins=max(int(atoms_per_cell_count.numel()), 1),
        min_cells_per_dimension=min_cells_per_dimension,
    )
    total_cells = math.prod(int(v.item()) for v in dimensions)
    if atoms_per_cell_count.numel() < total_cells:
        raise ValueError(
            f"cell buffers have capacity {atoms_per_cell_count.numel()}, need {total_cells}"
        )
    cells_per_dimension.copy_(dimensions)
    atoms_per_cell_count.zero_()
    cell_atom_start_indices.zero_()
    if n_atoms == 0:
        return total_cells
    keys = torch.empty(n_atoms, dtype=torch.int32, device=positions.device)
    build_cell_keys_reference_into(
        positions,
        torch.linalg.inv(geometry),
        dimensions,
        periodic,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        keys,
    )
    build_cell_csr_reference_into(
        keys,
        atoms_per_cell_count[:total_cells],
        cell_atom_start_indices[:total_cells],
        global_atom_offset=global_atom_offset,
    )
    cell_cursor = torch.empty_like(atoms_per_cell_count[:total_cells])
    build_cell_atom_list_reference_into(
        keys,
        atoms_per_cell_count[:total_cells],
        cell_atom_start_indices[:total_cells],
        cell_atom_list,
        cell_cursor,
        global_atom_offset=global_atom_offset,
    )
    return total_cells


def build_cell_list(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cells_per_dimension: torch.Tensor,
    neighbor_search_radius: torch.Tensor,
    atom_periodic_shifts: torch.Tensor,
    atom_to_cell_mapping: torch.Tensor,
    atoms_per_cell_count: torch.Tensor,
    cell_atom_start_indices: torch.Tensor,
    cell_atom_list: torch.Tensor,
    min_cells_per_dimension: int = 4,
) -> None:
    del neighbor_search_radius
    _build_into(
        positions,
        cutoff,
        cell,
        pbc,
        cells_per_dimension,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        min_cells_per_dimension=min_cells_per_dimension,
    )


def _offsets(radius: torch.Tensor, device: torch.device) -> torch.Tensor:
    rx, ry, rz = (int(v.item()) for v in radius)
    return torch.tensor(
        tuple(
            itertools.product(
                range(-rx, rx + 1),
                range(-ry, ry + 1),
                range(-rz, rz + 1),
            )
        ),
        dtype=torch.int64,
        device=device,
    )


def _sort_pairs(rows: torch.Tensor, columns: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
    order = torch.arange(rows.numel(), device=rows.device)
    for values in (shifts[:, 2], shifts[:, 1], shifts[:, 0], columns, rows):
        order = order[torch.argsort(values[order], stable=True)]
    return order


def _query_pairs(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cells_per_dimension: torch.Tensor,
    neighbor_search_radius: torch.Tensor,
    atom_periodic_shifts: torch.Tensor,
    atoms_per_cell_count: torch.Tensor,
    cell_atom_start_indices: torch.Tensor,
    cell_atom_list: torch.Tensor,
    *,
    half_fill: bool,
    atom_start: int = 0,
    atom_end: int | None = None,
) -> PairData:
    atom_end = positions.shape[0] if atom_end is None else atom_end
    if atom_start == atom_end:
        return _empty_pairs(positions)
    geometry, periodic = _geometry(cell, pbc, positions)
    dimensions = cells_per_dimension.to(torch.int64)
    nx, ny, nz = (int(v.item()) for v in dimensions)
    total_cells = nx * ny * nz
    counts = atoms_per_cell_count[:total_cells].to(torch.int64)
    starts = cell_atom_start_indices[:total_cells].to(torch.int64)
    occupied = torch.nonzero(counts > 0, as_tuple=False).flatten()
    if occupied.numel() == 0:
        return _empty_pairs(positions)
    occupied_xyz = torch.stack(
        (occupied % nx, (occupied // nx) % ny, occupied // (nx * ny)), dim=1
    )
    raw_targets = occupied_xyz[:, None, :] + _offsets(
        neighbor_search_radius, positions.device
    )[None, :, :]
    dims = dimensions.reshape(1, 1, 3)
    image = torch.div(raw_targets, dims, rounding_mode="floor")
    wrapped_targets = raw_targets - image * dims
    valid = torch.ones(raw_targets.shape[:2], dtype=torch.bool, device=positions.device)
    for dim in range(3):
        if not bool(periodic[dim]):
            valid &= (raw_targets[..., dim] >= 0) & (
                raw_targets[..., dim] < dimensions[dim]
            )
            image[..., dim] = 0
            wrapped_targets[..., dim] = raw_targets[..., dim]
    safe_targets = torch.minimum(
        torch.maximum(wrapped_targets, torch.zeros_like(wrapped_targets)), dims - 1
    )
    target_keys = safe_targets[..., 0] + nx * (
        safe_targets[..., 1] + ny * safe_targets[..., 2]
    )
    target_counts_grid = counts[target_keys]
    active_cell = valid & (target_counts_grid > 0)
    source_cells = occupied[:, None].expand_as(target_keys)[active_cell]
    target_cells = target_keys[active_cell]
    cell_images = image[active_cell]
    source_counts = counts[source_cells]
    target_counts = counts[target_cells]
    pair_counts = source_counts * target_counts
    total_candidates = int(pair_counts.sum().item())
    if total_candidates == 0:
        return _empty_pairs(positions)
    pair_id = torch.repeat_interleave(
        torch.arange(pair_counts.numel(), device=positions.device),
        pair_counts,
        output_size=total_candidates,
    )
    pair_starts = pair_counts.cumsum(0) - pair_counts
    pair_rank = torch.arange(total_candidates, device=positions.device) - torch.repeat_interleave(
        pair_starts, pair_counts, output_size=total_candidates
    )
    source_cell = source_cells[pair_id]
    target_cell = target_cells[pair_id]
    target_width = target_counts[pair_id]
    rows = cell_atom_list[starts[source_cell] + pair_rank // target_width].to(torch.long)
    columns = cell_atom_list[starts[target_cell] + pair_rank % target_width].to(torch.long)
    shifts = (
        atom_periodic_shifts[rows].to(torch.int64)
        - atom_periodic_shifts[columns].to(torch.int64)
        + cell_images[pair_id]
    ).to(torch.int32)
    vectors = positions[rows] - positions[columns] - shifts.to(positions.dtype) @ geometry
    distance_sq = vectors.square().sum(dim=-1)
    active = ((rows != columns) | torch.any(shifts != 0, dim=1)) & (
        distance_sq < cutoff * cutoff
    )
    if bool(torch.any(active & (distance_sq == 0))):
        raise ValueError("periodic neighbor list contains an overlapping active pair")
    if half_fill:
        same = rows == columns
        positive_image = (shifts[:, 0] > 0) | (
            (shifts[:, 0] == 0)
            & ((shifts[:, 1] > 0) | ((shifts[:, 1] == 0) & (shifts[:, 2] > 0)))
        )
        active &= (rows < columns) | (same & positive_image)
    if not bool(torch.any(active)):
        return _empty_pairs(positions)
    rows, columns, shifts, distance_sq, vectors = (
        rows[active], columns[active], shifts[active], distance_sq[active], vectors[active]
    )
    order = _sort_pairs(rows, columns, shifts)
    return (
        rows[order],
        columns[order],
        shifts[order],
        distance_sq[order],
        vectors[order],
    )


def _validate_optional_features(
    *,
    strategy: str,
    atom_centric_path: str,
    target_indices: torch.Tensor | None,
    pair_fn: object | None,
    pair_params: torch.Tensor | None,
    pair_energies: torch.Tensor | None,
    pair_forces: torch.Tensor | None,
) -> None:
    if strategy not in {"auto", "atom_centric"}:
        raise NotImplementedError("Torch reference cell-list supports atom_centric only")
    if atom_centric_path not in {"auto", "direct"}:
        raise NotImplementedError("Torch reference cell-list supports direct query only")
    if target_indices is not None:
        raise NotImplementedError("Torch reference cell-list does not support target_indices yet")
    if any(value is not None for value in (pair_fn, pair_params, pair_energies, pair_forces)):
        raise NotImplementedError("Torch reference cell-list does not support pair_fn outputs")


def _capacity(rows: torch.Tensor, n_rows: int, maximum: int | None) -> int:
    counts = torch.bincount(rows, minlength=n_rows)
    found = int(counts.max().item()) if counts.numel() else 0
    result = found if maximum is None else int(maximum)
    if result < 0:
        raise ValueError("max_neighbors must be non-negative")
    if found > result:
        row = int(torch.nonzero(counts > result, as_tuple=False)[0].item())
        raise NeighborOverflowError(
            f"neighbor capacity {result} is smaller than row {row} count {int(counts[row].item())}"
        )
    return result


def _scatter(
    pairs: PairData,
    matrix: torch.Tensor,
    counts: torch.Tensor,
    shifts: torch.Tensor,
    *,
    fill_value: int,
    row_start: int = 0,
    row_end: int | None = None,
    distances: torch.Tensor | None = None,
    vectors: torch.Tensor | None = None,
) -> None:
    rows, columns, pair_shifts, distance_sq, pair_vectors = pairs
    row_end = matrix.shape[0] if row_end is None else row_end
    matrix[row_start:row_end].fill_(fill_value)
    shifts[row_start:row_end].zero_()
    counts[row_start:row_end].zero_()
    if distances is not None:
        distances[row_start:row_end].zero_()
    if vectors is not None:
        vectors[row_start:row_end].zero_()
    if rows.numel() == 0:
        return
    local_counts = torch.bincount(rows - row_start, minlength=row_end - row_start).to(torch.int32)
    if bool(torch.any(local_counts > matrix.shape[1])):
        local_row = int(torch.nonzero(local_counts > matrix.shape[1])[0].item())
        raise NeighborOverflowError(
            f"neighbor capacity {matrix.shape[1]} is smaller than row "
            f"{row_start + local_row} count {int(local_counts[local_row].item())}"
        )
    counts[row_start:row_end].copy_(local_counts)
    starts = local_counts.to(torch.long).cumsum(0) - local_counts.to(torch.long)
    ranks = torch.arange(rows.numel(), device=rows.device) - starts[rows - row_start]
    matrix[rows, ranks] = columns.to(torch.int32)
    shifts[rows, ranks] = pair_shifts
    if distances is not None:
        distances[rows, ranks] = distance_sq.sqrt()
    if vectors is not None:
        vectors[rows, ranks] = pair_vectors


def _result(
    matrix: torch.Tensor,
    counts: torch.Tensor,
    shifts: torch.Tensor | None,
    *,
    return_neighbor_list: bool,
    distances: torch.Tensor | None,
    vectors: torch.Tensor | None,
) -> tuple[torch.Tensor, ...]:
    if not return_neighbor_list:
        output: list[torch.Tensor] = [matrix, counts]
        if shifts is not None:
            output.append(shifts)
        if distances is not None:
            output.append(distances)
        if vectors is not None:
            output.append(vectors)
        return tuple(output)
    active = torch.arange(matrix.shape[1], device=matrix.device)[None, :] < counts.to(torch.long)[:, None]
    rows = torch.repeat_interleave(
        torch.arange(matrix.shape[0], dtype=torch.int32, device=matrix.device),
        counts.to(torch.long),
    )
    output = [
        torch.stack((rows, matrix[active].to(torch.int32)), dim=0),
        torch.cat([torch.zeros(1, dtype=torch.int32, device=matrix.device), counts.cumsum(0)]),
    ]
    if shifts is not None:
        output.append(shifts[active])
    if distances is not None:
        output.append(distances[active])
    if vectors is not None:
        output.append(vectors[active])
    return tuple(output)


def query_cell_list(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cells_per_dimension: torch.Tensor,
    neighbor_search_radius: torch.Tensor,
    atom_periodic_shifts: torch.Tensor,
    atom_to_cell_mapping: torch.Tensor,
    atoms_per_cell_count: torch.Tensor,
    cell_atom_start_indices: torch.Tensor,
    cell_atom_list: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    num_neighbors: torch.Tensor,
    half_fill: bool = False,
    rebuild_flags: torch.Tensor | None = None,
    fill_value: int | None = None,
    strategy: str = "auto",
    atom_centric_path: str = "auto",
    sorted_positions: torch.Tensor | None = None,
    sorted_shifts: torch.Tensor | None = None,
    target_indices: torch.Tensor | None = None,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: object | None = None,
    pair_params: torch.Tensor | None = None,
    neighbor_vectors: torch.Tensor | None = None,
    neighbor_distances: torch.Tensor | None = None,
    pair_energies: torch.Tensor | None = None,
    pair_forces: torch.Tensor | None = None,
) -> None:
    del atom_to_cell_mapping, sorted_positions, sorted_shifts
    _validate_optional_features(
        strategy=strategy,
        atom_centric_path=atom_centric_path,
        target_indices=target_indices,
        pair_fn=pair_fn,
        pair_params=pair_params,
        pair_energies=pair_energies,
        pair_forces=pair_forces,
    )
    if rebuild_flags is not None and not bool(torch.any(rebuild_flags)):
        return
    if return_vectors and neighbor_vectors is None:
        raise ValueError("neighbor_vectors is required when return_vectors=True")
    if return_distances and neighbor_distances is None:
        raise ValueError("neighbor_distances is required when return_distances=True")
    pairs = _query_pairs(
        positions,
        cutoff,
        cell,
        pbc,
        cells_per_dimension,
        neighbor_search_radius,
        atom_periodic_shifts,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        half_fill=half_fill,
    )
    _scatter(
        pairs,
        neighbor_matrix,
        num_neighbors,
        neighbor_matrix_shifts,
        fill_value=positions.shape[0] if fill_value is None else int(fill_value),
        distances=neighbor_distances if return_distances else None,
        vectors=neighbor_vectors if return_vectors else None,
    )


def cell_list(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    max_neighbors: int | None = None,
    half_fill: bool = False,
    fill_value: int | None = None,
    return_neighbor_list: bool = False,
    neighbor_matrix: torch.Tensor | None = None,
    neighbor_matrix_shifts: torch.Tensor | None = None,
    num_neighbors: torch.Tensor | None = None,
    cells_per_dimension: torch.Tensor | None = None,
    neighbor_search_radius: torch.Tensor | None = None,
    atom_periodic_shifts: torch.Tensor | None = None,
    atom_to_cell_mapping: torch.Tensor | None = None,
    atoms_per_cell_count: torch.Tensor | None = None,
    cell_atom_start_indices: torch.Tensor | None = None,
    cell_atom_list: torch.Tensor | None = None,
    rebuild_flags: torch.Tensor | None = None,
    strategy: str = "auto",
    atom_centric_path: str = "auto",
    sorted_positions: torch.Tensor | None = None,
    sorted_shifts: torch.Tensor | None = None,
    target_indices: torch.Tensor | None = None,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: object | None = None,
    pair_params: torch.Tensor | None = None,
    neighbor_vectors: torch.Tensor | None = None,
    neighbor_distances: torch.Tensor | None = None,
    pair_energies: torch.Tensor | None = None,
    pair_forces: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    _validate_positions(positions)
    _validate_optional_features(
        strategy=strategy,
        atom_centric_path=atom_centric_path,
        target_indices=target_indices,
        pair_fn=pair_fn,
        pair_params=pair_params,
        pair_energies=pair_energies,
        pair_forces=pair_forces,
    )
    del sorted_positions, sorted_shifts
    n_atoms = positions.shape[0]
    fill_value = n_atoms if fill_value is None else int(fill_value)
    if fill_value < n_atoms:
        raise ValueError("fill_value must be >= num atoms so it cannot collide with an index")
    if rebuild_flags is not None and not bool(torch.any(rebuild_flags)):
        if neighbor_matrix is None or neighbor_matrix_shifts is None or num_neighbors is None:
            raise ValueError("rebuild_flags=False requires preallocated outputs")
        return _result(
            neighbor_matrix,
            num_neighbors,
            neighbor_matrix_shifts,
            return_neighbor_list=return_neighbor_list,
            distances=neighbor_distances if return_distances else None,
            vectors=neighbor_vectors if return_vectors else None,
        )
    if cells_per_dimension is None or neighbor_search_radius is None:
        max_cells, radius = estimate_cell_list_sizes(
            cell, pbc, cutoff, min_cells_per_dimension=1
        )
        allocated = allocate_cell_list(n_atoms, max_cells, radius, positions.device)
        cells_per_dimension, neighbor_search_radius = allocated[:2]
        atom_periodic_shifts, atom_to_cell_mapping = allocated[2:4]
        atoms_per_cell_count, cell_atom_start_indices, cell_atom_list = allocated[4:]
    scratch = (
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
    )
    if any(value is None for value in scratch):
        raise ValueError("all cell-list scratch tensors must be provided together")
    assert atom_periodic_shifts is not None
    assert atom_to_cell_mapping is not None
    assert atoms_per_cell_count is not None
    assert cell_atom_start_indices is not None
    assert cell_atom_list is not None
    build_cell_list(
        positions,
        cutoff,
        cell,
        pbc,
        cells_per_dimension,
        neighbor_search_radius,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        min_cells_per_dimension=1,
    )
    pairs = _query_pairs(
        positions,
        cutoff,
        cell,
        pbc,
        cells_per_dimension,
        neighbor_search_radius,
        atom_periodic_shifts,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        half_fill=half_fill,
    )
    capacity = _capacity(pairs[0], n_atoms, max_neighbors)
    if neighbor_matrix is None:
        neighbor_matrix = torch.full(
            (n_atoms, capacity), fill_value, dtype=torch.int32, device=positions.device
        )
    if bool(torch.any(torch.bincount(pairs[0], minlength=n_atoms) > neighbor_matrix.shape[1])):
        raise NeighborOverflowError("preallocated neighbor matrix is too small")
    if num_neighbors is None:
        num_neighbors = torch.zeros(n_atoms, dtype=torch.int32, device=positions.device)
    if neighbor_matrix_shifts is None:
        neighbor_matrix_shifts = torch.zeros(
            (*neighbor_matrix.shape, 3), dtype=torch.int32, device=positions.device
        )
    if return_distances and neighbor_distances is None:
        neighbor_distances = torch.zeros(
            neighbor_matrix.shape, dtype=positions.dtype, device=positions.device
        )
    if return_vectors and neighbor_vectors is None:
        neighbor_vectors = torch.zeros(
            (*neighbor_matrix.shape, 3), dtype=positions.dtype, device=positions.device
        )
    _scatter(
        pairs,
        neighbor_matrix,
        num_neighbors,
        neighbor_matrix_shifts,
        fill_value=fill_value,
        distances=neighbor_distances if return_distances else None,
        vectors=neighbor_vectors if return_vectors else None,
    )
    return _result(
        neighbor_matrix,
        num_neighbors,
        neighbor_matrix_shifts,
        return_neighbor_list=return_neighbor_list,
        distances=neighbor_distances if return_distances else None,
        vectors=neighbor_vectors if return_vectors else None,
    )


def batch_build_cell_list(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    batch_idx: torch.Tensor,
    cells_per_dimension: torch.Tensor,
    neighbor_search_radius: torch.Tensor,
    atom_periodic_shifts: torch.Tensor,
    atom_to_cell_mapping: torch.Tensor,
    atoms_per_cell_count: torch.Tensor,
    cell_atom_start_indices: torch.Tensor,
    cell_atom_list: torch.Tensor,
    min_cells_per_dimension: int = 4,
    max_nbins: int = 8192,
) -> None:
    del neighbor_search_radius
    if max_nbins <= 0:
        raise ValueError("max_nbins must be positive")
    ptr = _prepare_batch_ptr(positions, batch_idx, None)
    if cell.shape != (ptr.numel() - 1, 3, 3) or pbc.shape != (ptr.numel() - 1, 3):
        raise ValueError("cell/pbc batch dimensions must match batch_idx")
    atoms_per_cell_count.zero_()
    cell_atom_start_indices.zero_()
    cell_offset = 0
    for system in range(ptr.numel() - 1):
        start, end = int(ptr[system].item()), int(ptr[system + 1].item())
        dims, _ = _grid_spec(
            cell[system],
            cutoff,
            max_nbins=min(
                max_nbins,
                max(atoms_per_cell_count.numel() - cell_offset, 1),
            ),
            min_cells_per_dimension=min_cells_per_dimension,
        )
        n_cells = math.prod(int(v.item()) for v in dims)
        _build_into(
            positions[start:end],
            cutoff,
            cell[system],
            pbc[system],
            cells_per_dimension[system],
            atom_periodic_shifts[start:end],
            atom_to_cell_mapping[start:end],
            atoms_per_cell_count[cell_offset : cell_offset + n_cells],
            cell_atom_start_indices[cell_offset : cell_offset + n_cells],
            cell_atom_list[start:end],
            min_cells_per_dimension=min_cells_per_dimension,
            global_atom_offset=start,
        )
        cell_offset += n_cells


def _batch_pairs(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    batch_idx: torch.Tensor,
    cells_per_dimension: torch.Tensor,
    neighbor_search_radius: torch.Tensor,
    atom_periodic_shifts: torch.Tensor,
    atoms_per_cell_count: torch.Tensor,
    cell_atom_start_indices: torch.Tensor,
    cell_atom_list: torch.Tensor,
    *,
    half_fill: bool,
    rebuild_flags: torch.Tensor | None,
) -> list[tuple[int, int, PairData]]:
    ptr = _prepare_batch_ptr(positions, batch_idx, None)
    output: list[tuple[int, int, PairData]] = []
    cell_offset = 0
    for system in range(ptr.numel() - 1):
        start, end = int(ptr[system].item()), int(ptr[system + 1].item())
        dims = cells_per_dimension[system]
        n_cells = math.prod(int(v.item()) for v in dims)
        if rebuild_flags is None or bool(rebuild_flags[system]):
            output.append(
                (
                    start,
                    end,
                    _query_pairs(
                        positions,
                        cutoff,
                        cell[system],
                        pbc[system],
                        dims,
                        neighbor_search_radius[system],
                        atom_periodic_shifts,
                        atoms_per_cell_count[cell_offset : cell_offset + n_cells],
                        cell_atom_start_indices[cell_offset : cell_offset + n_cells],
                        cell_atom_list,
                        half_fill=half_fill,
                        atom_start=start,
                        atom_end=end,
                    ),
                )
            )
        cell_offset += n_cells
    return output


def batch_query_cell_list(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: float,
    batch_idx: torch.Tensor,
    cells_per_dimension: torch.Tensor,
    neighbor_search_radius: torch.Tensor,
    atom_periodic_shifts: torch.Tensor,
    atom_to_cell_mapping: torch.Tensor,
    atoms_per_cell_count: torch.Tensor,
    cell_atom_start_indices: torch.Tensor,
    cell_atom_list: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    num_neighbors: torch.Tensor,
    half_fill: bool = False,
    rebuild_flags: torch.Tensor | None = None,
    fill_value: int | None = None,
    strategy: str = "auto",
    atom_centric_path: str = "auto",
    target_indices: torch.Tensor | None = None,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: object | None = None,
    pair_params: torch.Tensor | None = None,
    neighbor_vectors: torch.Tensor | None = None,
    neighbor_distances: torch.Tensor | None = None,
    pair_energies: torch.Tensor | None = None,
    pair_forces: torch.Tensor | None = None,
) -> None:
    del atom_to_cell_mapping
    _validate_optional_features(
        strategy=strategy,
        atom_centric_path=atom_centric_path,
        target_indices=target_indices,
        pair_fn=pair_fn,
        pair_params=pair_params,
        pair_energies=pair_energies,
        pair_forces=pair_forces,
    )
    if return_vectors and neighbor_vectors is None:
        raise ValueError("neighbor_vectors is required when return_vectors=True")
    if return_distances and neighbor_distances is None:
        raise ValueError("neighbor_distances is required when return_distances=True")
    for start, end, pairs in _batch_pairs(
        positions,
        cutoff,
        cell,
        pbc,
        batch_idx,
        cells_per_dimension,
        neighbor_search_radius,
        atom_periodic_shifts,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        half_fill=half_fill,
        rebuild_flags=rebuild_flags,
    ):
        _scatter(
            pairs,
            neighbor_matrix,
            num_neighbors,
            neighbor_matrix_shifts,
            fill_value=positions.shape[0] if fill_value is None else int(fill_value),
            row_start=start,
            row_end=end,
            distances=neighbor_distances if return_distances else None,
            vectors=neighbor_vectors if return_vectors else None,
        )


def batch_cell_list(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    batch_idx: torch.Tensor,
    max_neighbors: int | None = None,
    half_fill: bool = False,
    fill_value: int | None = None,
    return_neighbor_list: bool = False,
    neighbor_matrix: torch.Tensor | None = None,
    neighbor_matrix_shifts: torch.Tensor | None = None,
    num_neighbors: torch.Tensor | None = None,
    cells_per_dimension: torch.Tensor | None = None,
    neighbor_search_radius: torch.Tensor | None = None,
    cell_offsets: torch.Tensor | None = None,
    atom_periodic_shifts: torch.Tensor | None = None,
    atom_to_cell_mapping: torch.Tensor | None = None,
    atoms_per_cell_count: torch.Tensor | None = None,
    cell_atom_start_indices: torch.Tensor | None = None,
    cell_atom_list: torch.Tensor | None = None,
    rebuild_flags: torch.Tensor | None = None,
    strategy: str = "auto",
    atom_centric_path: str = "auto",
    target_indices: torch.Tensor | None = None,
    return_vectors: bool = False,
    return_distances: bool = False,
    pair_fn: object | None = None,
    pair_params: torch.Tensor | None = None,
    neighbor_vectors: torch.Tensor | None = None,
    neighbor_distances: torch.Tensor | None = None,
    pair_energies: torch.Tensor | None = None,
    pair_forces: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    del cell_offsets
    _validate_positions(positions)
    _validate_optional_features(
        strategy=strategy,
        atom_centric_path=atom_centric_path,
        target_indices=target_indices,
        pair_fn=pair_fn,
        pair_params=pair_params,
        pair_energies=pair_energies,
        pair_forces=pair_forces,
    )
    ptr = _prepare_batch_ptr(positions, batch_idx, None)
    if cell.shape != (ptr.numel() - 1, 3, 3) or pbc.shape != (ptr.numel() - 1, 3):
        raise ValueError("cell/pbc batch dimensions must match batch_idx")
    fill_value = positions.shape[0] if fill_value is None else int(fill_value)
    if cells_per_dimension is None or neighbor_search_radius is None:
        max_cells, radius = estimate_batch_cell_list_sizes(
            cell, pbc, cutoff, min_cells_per_dimension=1
        )
        allocated = allocate_cell_list(positions.shape[0], max_cells, radius, positions.device)
        cells_per_dimension, neighbor_search_radius = allocated[:2]
        atom_periodic_shifts, atom_to_cell_mapping = allocated[2:4]
        atoms_per_cell_count, cell_atom_start_indices, cell_atom_list = allocated[4:]
    scratch = (
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
    )
    if any(value is None for value in scratch):
        raise ValueError("all batch cell-list scratch tensors must be provided together")
    assert atom_periodic_shifts is not None
    assert atom_to_cell_mapping is not None
    assert atoms_per_cell_count is not None
    assert cell_atom_start_indices is not None
    assert cell_atom_list is not None
    batch_build_cell_list(
        positions,
        cutoff,
        cell,
        pbc,
        batch_idx,
        cells_per_dimension,
        neighbor_search_radius,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        min_cells_per_dimension=1,
        max_nbins=8192,
    )
    rebuilt = _batch_pairs(
        positions,
        cutoff,
        cell,
        pbc,
        batch_idx,
        cells_per_dimension,
        neighbor_search_radius,
        atom_periodic_shifts,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        half_fill=half_fill,
        rebuild_flags=rebuild_flags,
    )
    if neighbor_matrix is None:
        if rebuild_flags is not None and not bool(torch.all(rebuild_flags)):
            raise ValueError("selective rebuild requires preallocated outputs")
        all_rows = (
            torch.cat([entry[2][0] for entry in rebuilt])
            if rebuilt
            else torch.empty(0, dtype=torch.long, device=positions.device)
        )
        capacity = _capacity(all_rows, positions.shape[0], max_neighbors)
        neighbor_matrix = torch.full(
            (positions.shape[0], capacity), fill_value, dtype=torch.int32, device=positions.device
        )
    if num_neighbors is None:
        num_neighbors = torch.zeros(positions.shape[0], dtype=torch.int32, device=positions.device)
    if neighbor_matrix_shifts is None:
        neighbor_matrix_shifts = torch.zeros(
            (*neighbor_matrix.shape, 3), dtype=torch.int32, device=positions.device
        )
    if return_distances and neighbor_distances is None:
        neighbor_distances = torch.zeros(
            neighbor_matrix.shape, dtype=positions.dtype, device=positions.device
        )
    if return_vectors and neighbor_vectors is None:
        neighbor_vectors = torch.zeros(
            (*neighbor_matrix.shape, 3), dtype=positions.dtype, device=positions.device
        )
    for start, end, pairs in rebuilt:
        _scatter(
            pairs,
            neighbor_matrix,
            num_neighbors,
            neighbor_matrix_shifts,
            fill_value=fill_value,
            row_start=start,
            row_end=end,
            distances=neighbor_distances if return_distances else None,
            vectors=neighbor_vectors if return_vectors else None,
        )
    return _result(
        neighbor_matrix,
        num_neighbors,
        neighbor_matrix_shifts,
        return_neighbor_list=return_neighbor_list,
        distances=neighbor_distances if return_distances else None,
        vectors=neighbor_vectors if return_vectors else None,
    )


def _nonperiodic_cell(positions: torch.Tensor, cutoff: float) -> tuple[torch.Tensor, torch.Tensor]:
    if positions.shape[0] == 0:
        lengths = positions.new_full((3,), max(cutoff * 2, 1.0))
        origin = positions.new_zeros(3)
    else:
        low = positions.amin(dim=0)
        high = positions.amax(dim=0)
        lengths = (high - low + 2 * cutoff).clamp_min(cutoff)
        origin = low - cutoff
    return torch.diag(lengths), origin


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
    _validate_positions(positions)
    if target_indices is not None:
        raise NotImplementedError("Torch reference cell-list does not support target_indices yet")
    if kwargs:
        raise NotImplementedError(
            "Torch reference cell-list does not support: " + ", ".join(sorted(kwargs))
        )
    if positions.shape[0] == 0:
        capacity = 0 if max_neighbors is None else int(max_neighbors)
        matrix = torch.empty((0, capacity), dtype=torch.int32, device=positions.device)
        counts = torch.empty(0, dtype=torch.int32, device=positions.device)
        periodic_input = cell is not None or pbc is not None
        shifts = (
            torch.empty((0, capacity, 3), dtype=torch.int32, device=positions.device)
            if periodic_input
            else None
        )
        distances = (
            torch.empty((0, capacity), dtype=positions.dtype, device=positions.device)
            if return_distances
            else None
        )
        vectors = (
            torch.empty((0, capacity, 3), dtype=positions.dtype, device=positions.device)
            if return_vectors
            else None
        )
        return _result(
            matrix,
            counts,
            shifts,
            return_neighbor_list=return_neighbor_list,
            distances=distances,
            vectors=vectors,
        )
    ptr = _prepare_batch_ptr(positions, batch_idx, batch_ptr)
    cells: torch.Tensor
    periodic: torch.Tensor
    translated = positions
    no_pbc = cell is None and pbc is None
    if no_pbc:
        local_cells: list[torch.Tensor] = []
        local_positions: list[torch.Tensor] = []
        for system in range(ptr.numel() - 1):
            start, end = int(ptr[system].item()), int(ptr[system + 1].item())
            local_cell, origin = _nonperiodic_cell(positions[start:end], cutoff)
            local_cells.append(local_cell)
            local_positions.append(positions[start:end] - origin)
        cells = torch.stack(local_cells) if local_cells else positions.new_empty((0, 3, 3))
        periodic = torch.zeros((ptr.numel() - 1, 3), dtype=torch.bool, device=positions.device)
        translated = torch.cat(local_positions) if local_positions else positions
    else:
        cells, periodic = _normalize_periodic_geometry(
            cell,
            pbc,
            num_systems=ptr.numel() - 1,
            device=positions.device,
            dtype=positions.dtype,
        )
    atom_system = torch.repeat_interleave(
        torch.arange(ptr.numel() - 1, device=positions.device), ptr[1:] - ptr[:-1]
    ).to(torch.int32)
    output = batch_cell_list(
        translated,
        cutoff,
        cells,
        periodic,
        atom_system,
        max_neighbors=max_neighbors,
        half_fill=half_fill,
        fill_value=positions.shape[0] if fill_value is None else fill_value,
        return_neighbor_list=return_neighbor_list,
        return_vectors=return_vectors,
        return_distances=return_distances,
    )
    if no_pbc:
        # The dispatcher ABI historically omits an all-zero shift tensor for
        # no-PBC inputs, while the layered cell-list API always exposes it.
        output = (*output[:2], *output[3:])
    return output


__all__ = [
    "allocate_cell_list",
    "allocate_query_sort_scratch",
    "batch_build_cell_list",
    "batch_cell_list",
    "batch_query_cell_list",
    "build_cell_list",
    "cell_list",
    "estimate_batch_cell_list_sizes",
    "estimate_cell_list_sizes",
    "neighbor_list",
    "query_cell_list",
]
