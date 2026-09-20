# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Torch materialization for isolated native Batch cell-list candidates.

Native HIP query candidates contain only discrete topology: source rows,
neighbor columns, image shifts, and per-row counts.  This module owns the
public ordering/scatter contract and, when requested, computes differentiable
vectors and distances from the original positions and cells.

The helper is intentionally private and is not registered with the runtime
dispatcher.  It is a bridge for candidate validation and later backend
experiments, not a production backend selection policy.
"""

from __future__ import annotations

import torch

from nvalchemiops.torch_reference_cell_list import _sort_pairs


CanonicalCandidate = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


def _validate_inputs(
    positions: torch.Tensor,
    cells: torch.Tensor,
    batch_idx: torch.Tensor,
    candidate_matrix: torch.Tensor,
    candidate_shifts: torch.Tensor,
    candidate_counts: torch.Tensor,
    public_matrix: torch.Tensor,
    public_shifts: torch.Tensor,
    public_counts: torch.Tensor,
    distances: torch.Tensor | None,
    vectors: torch.Tensor | None,
) -> None:
    if positions.ndim != 2 or positions.shape[1:] != (3,):
        raise ValueError("positions must have shape (N, 3)")
    if positions.dtype not in (torch.float32, torch.float64):
        raise TypeError("positions must have dtype float32 or float64")
    if cells.ndim != 3 or cells.shape[1:] != (3, 3) or cells.dtype != positions.dtype:
        raise ValueError("cells must have shape (B, 3, 3) and match positions dtype")
    if batch_idx.shape != (positions.shape[0],) or batch_idx.dtype != torch.int32:
        raise ValueError("batch_idx must have shape (N,) and dtype int32")
    if candidate_matrix.ndim != 2 or candidate_matrix.shape[0] != positions.shape[0]:
        raise ValueError("candidate_matrix must have shape (N, K)")
    if candidate_matrix.shape[1] <= 0:
        raise ValueError("candidate_matrix must provide positive capacity K")
    if candidate_shifts.shape != (*candidate_matrix.shape, 3):
        raise ValueError("candidate_shifts must have shape (N, K, 3)")
    if candidate_counts.shape != (positions.shape[0],):
        raise ValueError("candidate_counts must have shape (N,)")
    if public_matrix.shape != candidate_matrix.shape:
        raise ValueError("public_matrix must have the candidate matrix shape")
    if public_shifts.shape != candidate_shifts.shape:
        raise ValueError("public_shifts must have the candidate shift shape")
    if public_counts.shape != candidate_counts.shape:
        raise ValueError("public_counts must have the candidate count shape")
    if distances is not None:
        if distances.shape != candidate_matrix.shape or distances.dtype != positions.dtype:
            raise ValueError("distances must have shape (N, K) and match positions dtype")
    if vectors is not None:
        if vectors.shape != candidate_shifts.shape or vectors.dtype != positions.dtype:
            raise ValueError("vectors must have shape (N, K, 3) and match positions dtype")
    values = (
        cells,
        batch_idx,
        candidate_matrix,
        candidate_shifts,
        candidate_counts,
        public_matrix,
        public_shifts,
        public_counts,
    )
    if distances is not None:
        values += (distances,)
    if vectors is not None:
        values += (vectors,)
    if any(value.device != positions.device for value in values):
        raise ValueError("candidate materialization tensors must share one device")
    if any(not value.is_contiguous() for value in values):
        raise ValueError("candidate materialization tensors must be contiguous")
    if cells.shape[0] <= 0:
        raise ValueError("candidate materialization requires at least one system")
    if bool(torch.any(batch_idx < 0)) or bool(torch.any(batch_idx >= cells.shape[0])):
        raise ValueError("batch_idx contains an invalid system index")
    if any(
        value.dtype != torch.int32
        for value in (
            candidate_matrix,
            candidate_shifts,
            candidate_counts,
            public_matrix,
            public_shifts,
            public_counts,
        )
    ):
        raise TypeError("candidate and public topology buffers must have dtype int32")
    if bool(torch.any(candidate_counts < 0)) or bool(
        torch.any(candidate_counts > candidate_matrix.shape[1])
    ):
        raise ValueError("candidate_counts must be within the candidate capacity")


def _canonicalize_candidate(
    candidate_matrix: torch.Tensor,
    candidate_shifts: torch.Tensor,
    candidate_counts: torch.Tensor,
) -> CanonicalCandidate:
    """Return stable active pairs and their destination row ranks."""

    atoms, capacity = candidate_matrix.shape
    active = torch.arange(capacity, device=candidate_matrix.device)[None, :] < candidate_counts[:, None]
    local_counts = torch.zeros(atoms, dtype=torch.int32, device=candidate_matrix.device)
    if not bool(torch.any(active)):
        empty_rows = torch.empty(0, dtype=torch.long, device=candidate_matrix.device)
        empty_shifts = torch.empty((0, 3), dtype=torch.int32, device=candidate_matrix.device)
        return empty_rows, empty_rows.clone(), empty_shifts, empty_rows.clone(), local_counts

    rows = torch.arange(atoms, device=candidate_matrix.device)[:, None].expand_as(candidate_matrix)[active]
    columns = candidate_matrix[active].to(torch.long)
    pair_shifts = candidate_shifts[active]
    if bool(torch.any(columns < 0)) or bool(torch.any(columns >= atoms)):
        raise ValueError("candidate_matrix contains an invalid neighbor index")
    order = _sort_pairs(rows, columns, pair_shifts)
    rows = rows[order]
    columns = columns[order]
    pair_shifts = pair_shifts[order]

    local_counts = torch.bincount(rows, minlength=atoms).to(torch.int32)
    starts = local_counts.to(torch.long).cumsum(0) - local_counts.to(torch.long)
    ranks = torch.arange(rows.numel(), device=candidate_matrix.device) - starts[rows]
    return rows, columns, pair_shifts, ranks, local_counts


def _reset_topology_outputs(
    public_matrix: torch.Tensor,
    public_shifts: torch.Tensor,
    public_counts: torch.Tensor,
    fill_value: int,
) -> None:
    public_matrix.fill_(fill_value)
    public_shifts.zero_()
    public_counts.zero_()


def _scatter_canonical_candidate(
    canonical: CanonicalCandidate,
    public_matrix: torch.Tensor,
    public_shifts: torch.Tensor,
    public_counts: torch.Tensor,
) -> None:
    rows, columns, pair_shifts, ranks, local_counts = canonical
    public_counts.copy_(local_counts)
    if rows.numel() == 0:
        return
    public_matrix[rows, ranks] = columns.to(torch.int32)
    public_shifts[rows, ranks] = pair_shifts


def _reset_geometry_outputs(
    distances: torch.Tensor | None,
    vectors: torch.Tensor | None,
) -> None:
    if distances is not None:
        distances.zero_()
    if vectors is not None:
        vectors.zero_()


def _validate_topology_geometry_inputs(
    positions: torch.Tensor,
    cells: torch.Tensor,
    batch_idx: torch.Tensor,
    public_matrix: torch.Tensor,
    public_shifts: torch.Tensor,
    public_counts: torch.Tensor,
    distances: torch.Tensor | None,
    vectors: torch.Tensor | None,
) -> None:
    if positions.ndim != 2 or positions.shape[1:] != (3,):
        raise ValueError("positions must have shape (N, 3)")
    if positions.dtype not in (torch.float32, torch.float64):
        raise TypeError("positions must have dtype float32 or float64")
    if cells.ndim != 3 or cells.shape[1:] != (3, 3) or cells.dtype != positions.dtype:
        raise ValueError("cells must have shape (B, 3, 3) and match positions dtype")
    if batch_idx.shape != (positions.shape[0],) or batch_idx.dtype != torch.int32:
        raise ValueError("batch_idx must have shape (N,) and dtype int32")
    if public_matrix.ndim != 2 or public_matrix.shape[0] != positions.shape[0]:
        raise ValueError("public_matrix must have shape (N, K)")
    if public_matrix.shape[1] <= 0:
        raise ValueError("public_matrix must provide positive capacity K")
    if public_shifts.shape != (*public_matrix.shape, 3):
        raise ValueError("public_shifts must have shape (N, K, 3)")
    if public_counts.shape != (positions.shape[0],):
        raise ValueError("public_counts must have shape (N,)")
    if distances is not None:
        if distances.shape != public_matrix.shape or distances.dtype != positions.dtype:
            raise ValueError("distances must have shape (N, K) and match positions dtype")
    if vectors is not None:
        if vectors.shape != public_shifts.shape or vectors.dtype != positions.dtype:
            raise ValueError("vectors must have shape (N, K, 3) and match positions dtype")
    values = (
        cells,
        batch_idx,
        public_matrix,
        public_shifts,
        public_counts,
    )
    if distances is not None:
        values += (distances,)
    if vectors is not None:
        values += (vectors,)
    if any(value.device != positions.device for value in values):
        raise ValueError("topology geometry tensors must share one device")
    if any(not value.is_contiguous() for value in values):
        raise ValueError("topology geometry tensors must be contiguous")
    if cells.shape[0] <= 0:
        raise ValueError("topology geometry requires at least one system")
    if bool(torch.any(batch_idx < 0)) or bool(torch.any(batch_idx >= cells.shape[0])):
        raise ValueError("batch_idx contains an invalid system index")
    if any(
        value.dtype != torch.int32
        for value in (public_matrix, public_shifts, public_counts)
    ):
        raise TypeError("public topology buffers must have dtype int32")
    if bool(torch.any(public_counts < 0)) or bool(
        torch.any(public_counts > public_matrix.shape[1])
    ):
        raise ValueError("public_counts must be within the public capacity")
    active = torch.arange(
        public_matrix.shape[1], device=public_matrix.device
    )[None, :] < public_counts[:, None]
    active_columns = public_matrix[active]
    if bool(torch.any(active_columns < 0)) or bool(
        torch.any(active_columns >= positions.shape[0])
    ):
        raise ValueError("public_matrix contains an invalid active neighbor index")


def _materialize_canonical_geometry(
    positions: torch.Tensor,
    cells: torch.Tensor,
    batch_idx: torch.Tensor,
    canonical: CanonicalCandidate,
    distances: torch.Tensor | None,
    vectors: torch.Tensor | None,
) -> None:
    rows, columns, pair_shifts, ranks, _local_counts = canonical
    if rows.numel() == 0 or (distances is None and vectors is None):
        return
    geometry = cells[batch_idx[rows].to(torch.long)]
    pair_vectors = positions[rows] - positions[columns] - torch.einsum(
        "pi,pij->pj", pair_shifts.to(positions.dtype), geometry
    )
    distance_sq = pair_vectors.square().sum(dim=-1)
    if distances is not None:
        distances[rows, ranks] = distance_sq.sqrt()
    if vectors is not None:
        vectors[rows, ranks] = pair_vectors


def materialize_batch_query_topology_geometry_into(
    positions: torch.Tensor,
    cells: torch.Tensor,
    batch_idx: torch.Tensor,
    public_matrix: torch.Tensor,
    public_shifts: torch.Tensor,
    public_counts: torch.Tensor,
    *,
    distances: torch.Tensor | None = None,
    vectors: torch.Tensor | None = None,
) -> None:
    """Materialize differentiable geometry from an already public topology.

    This is the hybrid bridge for native topology candidates: the discrete
    matrix, shifts, and counts are treated as fixed topology, while distance
    and vector outputs remain ordinary Torch expressions over ``positions``
    and ``cells`` so first- and higher-order geometry gradients are preserved.
    """

    if distances is None and vectors is None:
        raise ValueError("at least one geometry output must be requested")
    _validate_topology_geometry_inputs(
        positions,
        cells,
        batch_idx,
        public_matrix,
        public_shifts,
        public_counts,
        distances,
        vectors,
    )
    _reset_geometry_outputs(distances, vectors)
    active = torch.arange(
        public_matrix.shape[1], device=public_matrix.device
    )[None, :] < public_counts[:, None]
    rows_grid = torch.arange(
        positions.shape[0], device=positions.device
    )[:, None].expand_as(public_matrix)
    rows = rows_grid[active]
    if rows.numel() == 0:
        return
    columns = public_matrix[active].to(torch.long)
    pair_shifts = public_shifts[active]
    geometry = cells[batch_idx[rows].to(torch.long)]
    pair_vectors = positions[rows] - positions[columns] - torch.einsum(
        "pi,pij->pj", pair_shifts.to(positions.dtype), geometry
    )
    if distances is not None:
        distances[active] = pair_vectors.square().sum(dim=-1).sqrt()
    if vectors is not None:
        vectors[active] = pair_vectors


def materialize_batch_query_candidate_into(
    positions: torch.Tensor,
    cells: torch.Tensor,
    batch_idx: torch.Tensor,
    candidate_matrix: torch.Tensor,
    candidate_shifts: torch.Tensor,
    candidate_counts: torch.Tensor,
    public_matrix: torch.Tensor,
    public_shifts: torch.Tensor,
    public_counts: torch.Tensor,
    *,
    fill_value: int | None = None,
    distances: torch.Tensor | None = None,
    vectors: torch.Tensor | None = None,
) -> None:
    """Canonicalize a raw candidate and optionally materialize geometry.

    The candidate is read as an unordered row-major list.  Active pairs are
    sorted with the same stable ``(row, column, shift-x, shift-y, shift-z)``
    ordering as the Torch reference implementation, then scattered into the
    caller-owned public buffers.  ``distances`` and ``vectors`` are ordinary
    Torch expressions over ``positions`` and ``cells``; gradients therefore
    remain available even though candidate topology itself is discrete.
    """

    _validate_inputs(
        positions,
        cells,
        batch_idx,
        candidate_matrix,
        candidate_shifts,
        candidate_counts,
        public_matrix,
        public_shifts,
        public_counts,
        distances,
        vectors,
    )
    resolved_fill = positions.shape[0] if fill_value is None else int(fill_value)
    if resolved_fill < positions.shape[0]:
        raise ValueError("fill_value must be >= num atoms so it cannot collide with an index")

    _reset_topology_outputs(public_matrix, public_shifts, public_counts, resolved_fill)
    _reset_geometry_outputs(distances, vectors)
    canonical = _canonicalize_candidate(
        candidate_matrix, candidate_shifts, candidate_counts
    )
    _scatter_canonical_candidate(canonical, public_matrix, public_shifts, public_counts)
    _materialize_canonical_geometry(
        positions, cells, batch_idx, canonical, distances, vectors
    )


__all__ = [
    "materialize_batch_query_candidate_into",
    "materialize_batch_query_topology_geometry_into",
]
