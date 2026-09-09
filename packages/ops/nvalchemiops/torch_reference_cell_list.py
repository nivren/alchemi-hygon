# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit Torch reference cell-list neighbor construction.

This module is deliberately a separate, opt-in backend.  The existing
``torch_reference`` implementation remains the semantic oracle and the
default explicit Torch path; this implementation only replaces its no-PBC
candidate generation with a uniform cell list.
"""

from __future__ import annotations

import itertools

import torch

from nvalchemiops.torch_reference import (
    NeighborOverflowError,
    _prepare_batch_ptr,
    _validate_positions,
)


_CELL_OFFSETS = tuple(itertools.product((-1, 0, 1), repeat=3))


def _empty_pairs(
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return empty pair tensors on the positions device."""
    return (
        torch.empty(0, dtype=torch.long, device=positions.device),
        torch.empty(0, dtype=torch.long, device=positions.device),
        torch.empty(0, dtype=positions.dtype, device=positions.device),
        torch.empty(0, 3, dtype=positions.dtype, device=positions.device),
    )


def _cell_list_pairs(
    positions: torch.Tensor,
    cutoff: float,
    *,
    half_fill: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate one system's no-PBC candidate pairs with a uniform cell list."""
    n_atoms = positions.shape[0]
    if n_atoms == 0:
        return _empty_pairs(positions)

    # A cell edge equal to the cutoff is sufficient: every pair inside the
    # sphere differs by at most one cell along each Cartesian axis.
    origin = positions.amin(dim=0)
    cell_coords = torch.floor((positions - origin) / cutoff).to(torch.int64)
    extent = cell_coords.amax(dim=0) + 1
    nx, ny, nz = (int(value.item()) for value in extent)
    linear_keys = cell_coords[:, 0] + nx * (
        cell_coords[:, 1] + ny * cell_coords[:, 2]
    )

    sorted_keys, sorted_atoms = torch.sort(linear_keys, stable=True)
    unique_keys, cell_counts = torch.unique_consecutive(
        sorted_keys, return_counts=True
    )
    cell_starts = torch.cat(
        [
            torch.zeros(1, dtype=torch.long, device=positions.device),
            cell_counts.to(torch.long).cumsum(0),
        ]
    )
    n_cells = unique_keys.shape[0]

    # Decode occupied cell coordinates, then map all 27 neighboring cells to
    # their occupied-group IDs with one device-side searchsorted operation.
    cell_coords_occupied = torch.stack(
        (
            unique_keys % nx,
            (unique_keys // nx) % ny,
            unique_keys // (nx * ny),
        ),
        dim=1,
    )
    offsets = torch.tensor(
        _CELL_OFFSETS, dtype=torch.long, device=positions.device
    )
    target_coords = cell_coords_occupied[:, None, :] + offsets[None, :, :]
    valid_targets = ((target_coords >= 0) & (target_coords < extent)).all(dim=-1)
    target_keys = (
        target_coords[..., 0]
        + nx * (target_coords[..., 1] + ny * target_coords[..., 2])
    )
    target_groups = torch.searchsorted(
        unique_keys, target_keys.reshape(-1), right=False
    ).reshape(n_cells, len(_CELL_OFFSETS))
    safe_target_groups = target_groups.clamp(max=n_cells - 1)
    occupied_targets = valid_targets & (
        unique_keys.index_select(0, safe_target_groups.reshape(-1)).reshape_as(target_groups)
        == target_keys
    )
    source_groups = (
        torch.arange(n_cells, dtype=torch.long, device=positions.device)
        .unsqueeze(1)
        .expand_as(target_groups)
    )[occupied_targets]
    target_groups = safe_target_groups[occupied_targets]

    # Expand occupied cell-pairs into atom-pairs without a Python loop over
    # atoms.  The result is only a near-neighbor candidate set; the exact
    # cutoff test below retains the dense reference semantics.
    source_counts = cell_counts.to(torch.long).index_select(0, source_groups)
    target_counts = cell_counts.to(torch.long).index_select(0, target_groups)
    pair_counts = source_counts * target_counts
    total_candidates = int(pair_counts.sum().item())
    pair_ids = torch.repeat_interleave(
        torch.arange(pair_counts.numel(), dtype=torch.long, device=positions.device),
        pair_counts,
        output_size=total_candidates,
    )
    pair_starts = pair_counts.cumsum(0) - pair_counts
    pair_rank = torch.arange(
        total_candidates, dtype=torch.long, device=positions.device
    ) - torch.repeat_interleave(
        pair_starts, pair_counts, output_size=total_candidates
    )
    source_group_per_pair = source_groups.index_select(0, pair_ids)
    target_group_per_pair = target_groups.index_select(0, pair_ids)
    target_count_per_pair = target_counts.index_select(0, pair_ids)
    source_atoms = sorted_atoms.index_select(
        0,
        cell_starts.index_select(0, source_group_per_pair)
        + pair_rank // target_count_per_pair,
    )
    target_atoms = sorted_atoms.index_select(
        0,
        cell_starts.index_select(0, target_group_per_pair)
        + pair_rank % target_count_per_pair,
    )

    vectors = positions.index_select(0, source_atoms) - positions.index_select(
        0, target_atoms
    )
    distance_sq = vectors.square().sum(dim=-1)
    active = distance_sq < cutoff * cutoff
    active = active & (source_atoms < target_atoms if half_fill else source_atoms != target_atoms)
    if bool(torch.any(active & (distance_sq == 0))):
        raise ValueError("neighbor list contains an overlapping active pair")
    if not bool(torch.any(active)):
        return _empty_pairs(positions)

    source_atoms = source_atoms[active]
    target_atoms = target_atoms[active]
    distance_sq = distance_sq[active]
    vectors = vectors[active]
    # The cell-pair traversal order is not row-major.  Restore the exact
    # matrix/COO ordering used by the dense reference implementation.
    row_order = torch.argsort(source_atoms * n_atoms + target_atoms, stable=True)
    return (
        source_atoms.index_select(0, row_order),
        target_atoms.index_select(0, row_order),
        distance_sq.index_select(0, row_order),
        vectors.index_select(0, row_order),
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
    """Build a no-PBC neighbor list with a Torch uniform cell list.

    The explicit backend preserves the dense Torch reference contract for
    full/half lists, MATRIX/COO output, deterministic row-major ordering,
    distances, vectors, empty systems and capacity errors.  Periodic cells,
    target rows, pair callbacks and other unsupported options fail explicitly.
    """
    _validate_positions(positions)
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")
    if cell is not None or pbc is not None:
        raise NotImplementedError(
            "Torch reference cell-list neighbor_list supports no-PBC inputs only"
        )
    if target_indices is not None:
        raise NotImplementedError(
            "Torch reference cell-list neighbor_list does not support target_indices yet"
        )
    if kwargs:
        unsupported = ", ".join(sorted(kwargs))
        raise NotImplementedError(
            f"Torch reference cell-list neighbor_list does not support: {unsupported}"
        )

    n_atoms = positions.shape[0]
    if fill_value is None:
        fill_value = n_atoms
    if fill_value < n_atoms:
        raise ValueError("fill_value must be >= num atoms so it cannot collide with an index")

    ptr = _prepare_batch_ptr(positions, batch_idx, batch_ptr)
    ptr_cpu = ptr.detach().cpu().tolist()
    row_parts: list[torch.Tensor] = []
    col_parts: list[torch.Tensor] = []
    distance_parts: list[torch.Tensor] = []
    vector_parts: list[torch.Tensor] = []
    for system in range(len(ptr_cpu) - 1):
        start, end = int(ptr_cpu[system]), int(ptr_cpu[system + 1])
        rows, cols, distance_sq, vectors = _cell_list_pairs(
            positions[start:end], cutoff, half_fill=half_fill
        )
        if rows.numel():
            row_parts.append(rows + start)
            col_parts.append(cols + start)
            distance_parts.append(distance_sq)
            vector_parts.append(vectors)

    empty_rows, empty_cols, empty_distances, empty_vectors = _empty_pairs(positions)
    rows = torch.cat(row_parts) if row_parts else empty_rows
    cols = torch.cat(col_parts) if col_parts else empty_cols
    distance_sq = torch.cat(distance_parts) if distance_parts else empty_distances
    vectors = torch.cat(vector_parts) if vector_parts else empty_vectors
    counts = torch.bincount(rows, minlength=n_atoms).to(torch.int32)
    found_max = int(counts.max().item()) if counts.numel() else 0
    if max_neighbors is None:
        max_neighbors = found_max
    if max_neighbors < 0:
        raise ValueError("max_neighbors must be non-negative")
    if found_max > max_neighbors:
        overflow_row = int(torch.nonzero(counts > max_neighbors)[0].item())
        raise NeighborOverflowError(
            f"neighbor capacity {max_neighbors} is smaller than row "
            f"{overflow_row} count {int(counts[overflow_row].item())}"
        )

    matrix = torch.full(
        (n_atoms, max_neighbors),
        int(fill_value),
        dtype=torch.int32,
        device=positions.device,
    )
    distances = (
        torch.zeros(n_atoms, max_neighbors, dtype=positions.dtype, device=positions.device)
        if return_distances
        else None
    )
    output_vectors = (
        torch.zeros(
            n_atoms, max_neighbors, 3, dtype=positions.dtype, device=positions.device
        )
        if return_vectors
        else None
    )
    if rows.numel():
        row_starts = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=positions.device),
                counts.to(torch.long).cumsum(0)[:-1],
            ]
        )
        ranks = torch.arange(rows.numel(), dtype=torch.long, device=positions.device)
        ranks = ranks - row_starts.index_select(0, rows)
        matrix[rows, ranks] = cols.to(torch.int32)
        if distances is not None:
            distances[rows, ranks] = distance_sq.sqrt()
        if output_vectors is not None:
            output_vectors[rows, ranks] = vectors

    if return_neighbor_list:
        active = torch.arange(
            max_neighbors, dtype=torch.long, device=positions.device
        ).unsqueeze(0) < counts.to(torch.long).unsqueeze(1)
        row_ids = torch.arange(
            n_atoms, dtype=torch.int32, device=positions.device
        )
        row_ids = torch.repeat_interleave(row_ids, counts.to(torch.long))
        edges = torch.stack((row_ids, matrix[active].to(torch.int32)), dim=0)
        ptr_out = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=positions.device),
                counts.cumsum(0),
            ]
        )
        output: list[torch.Tensor] = [edges, ptr_out]
        if distances is not None:
            output.append(distances[active])
        if output_vectors is not None:
            output.append(output_vectors[active])
        return tuple(output)

    output = [matrix, counts]
    if distances is not None:
        output.append(distances)
    if output_vectors is not None:
        output.append(output_vectors)
    return tuple(output)


__all__ = ["neighbor_list"]
