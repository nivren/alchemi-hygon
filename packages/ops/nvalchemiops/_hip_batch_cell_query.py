# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional native HIP Batch cell-list pair enumeration candidate.

The candidate owns only forward pair enumeration.  It reads caller-owned CSR
cell storage and writes an unordered, row-major candidate matrix plus image
shifts.  Stable public ordering, distances, vectors, gradients and pair
callbacks remain in the Torch materialization/reference path.

There is intentionally no dispatcher registration or runtime backend
selection here.  Capacity overflow and periodic overlap are explicit errors;
they are never silently truncated or sent to a CPU fallback.
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import ModuleType

import torch
from nvalchemiops._hip_extension_loader import load_hip_extension


_EXTENSION: ModuleType | None = None
_EXTENSION_LOCK = threading.Lock()


def _validate_inputs(
    positions: torch.Tensor,
    cells: torch.Tensor,
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
    fill_value: int,
) -> None:
    if positions.ndim != 2 or positions.shape[1:] != (3,):
        raise ValueError("positions must have shape (N, 3)")
    if positions.dtype not in (torch.float32, torch.float64):
        raise TypeError("positions must have dtype float32 or float64")
    systems = cells.shape[0] if cells.ndim else -1
    if cells.shape != (systems, 3, 3) or cells.dtype != positions.dtype:
        raise ValueError("cells must have shape (B, 3, 3) and match positions dtype")
    if systems <= 0:
        raise ValueError("native HIP Batch query requires at least one system")
    if pbc.shape != (systems, 3) or pbc.dtype != torch.bool:
        raise ValueError("pbc must have shape (B, 3) and dtype bool")
    if not isinstance(cutoff, (int, float)) or cutoff <= 0:
        raise ValueError("cutoff must be positive")
    if batch_idx.shape != (positions.shape[0],) or batch_idx.dtype != torch.int32:
        raise ValueError("batch_idx must have shape (N,) and dtype int32")
    for value, name in (
        (cells_per_dimension, "cells_per_dimension"),
        (neighbor_search_radius, "neighbor_search_radius"),
    ):
        if value.shape != (systems, 3) or value.dtype != torch.int32:
            raise ValueError(f"{name} must have shape (B, 3) and dtype int32")
    if bool(torch.any(cells_per_dimension <= 0)):
        raise ValueError("cells_per_dimension must be positive")
    if bool(torch.any(neighbor_search_radius < 0)):
        raise ValueError("neighbor_search_radius must be non-negative")
    if atom_periodic_shifts.shape != (positions.shape[0], 3):
        raise ValueError("atom_periodic_shifts must have shape (N, 3)")
    if atom_to_cell_mapping.shape != (positions.shape[0], 3):
        raise ValueError("atom_to_cell_mapping must have shape (N, 3)")
    if any(
        value.dtype != torch.int32
        for value in (
            atom_periodic_shifts,
            atom_to_cell_mapping,
            atoms_per_cell_count,
            cell_atom_start_indices,
            cell_atom_list,
            neighbor_matrix,
            neighbor_matrix_shifts,
            num_neighbors,
        )
    ):
        raise TypeError("native HIP Batch query integer buffers must have dtype int32")
    if atoms_per_cell_count.ndim != 1 or cell_atom_start_indices.shape != atoms_per_cell_count.shape:
        raise ValueError("cell counts and starts must have shape (M,)")
    if cell_atom_list.ndim != 1:
        raise ValueError("cell_atom_list must have shape (capacity,)")
    if neighbor_matrix.shape[0] != positions.shape[0] or neighbor_matrix.ndim != 2:
        raise ValueError("neighbor_matrix must have shape (N, K)")
    if neighbor_matrix.shape[1] <= 0:
        raise ValueError("neighbor_matrix must provide positive capacity K")
    if neighbor_matrix_shifts.shape != (*neighbor_matrix.shape, 3):
        raise ValueError("neighbor_matrix_shifts must have shape (N, K, 3)")
    if num_neighbors.shape != (positions.shape[0],):
        raise ValueError("num_neighbors must have shape (N,)")
    if isinstance(fill_value, bool) or not isinstance(fill_value, int):
        raise TypeError("fill_value must be an int")
    if not torch.iinfo(torch.int32).min <= fill_value <= torch.iinfo(torch.int32).max:
        raise ValueError("fill_value must fit int32")
    values = (
        cells,
        pbc,
        batch_idx,
        cells_per_dimension,
        neighbor_search_radius,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
    )
    if any(value.device != positions.device for value in values):
        raise ValueError("native HIP Batch query tensors must share one device")
    if any(not value.is_contiguous() for value in values):
        raise ValueError("native HIP Batch query tensors must be contiguous")
    if positions.device.type != "cuda" or not torch.version.hip:
        raise RuntimeError("native HIP Batch query requires a visible HIP Torch device")
    output_ptrs = (
        neighbor_matrix.data_ptr(),
        neighbor_matrix_shifts.data_ptr(),
        num_neighbors.data_ptr(),
    )
    if any(value.numel() for value in (neighbor_matrix, neighbor_matrix_shifts, num_neighbors)) and len(
        set(output_ptrs)
    ) != len(output_ptrs):
        raise ValueError("native HIP Batch query output buffers must not alias")


def _load_jit_extension() -> ModuleType:
    """Compile/load the native HIP query candidate into an explicit cache."""

    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    with _EXTENSION_LOCK:
        if _EXTENSION is not None:
            return _EXTENSION
        _EXTENSION = load_hip_extension(
            source_root=Path(__file__).with_name("_native"),
            extension_name="nvalchemi_batch_cell_query_hip",
            source_names=("batch_cell_query.cpp", "batch_cell_query.cu"),
            build_env_var="NVALCHEMI_HIP_BATCH_CELL_QUERY_BUILD_DIR",
            verbose_env_var="NVALCHEMI_HIP_BATCH_CELL_QUERY_VERBOSE",
        )
    return _EXTENSION


@torch.library.custom_op(
    "nvalchemiops::_batch_cell_query_hip",
    mutates_args=("neighbor_matrix", "neighbor_matrix_shifts", "num_neighbors"),
)
def _batch_cell_query_hip(
    positions: torch.Tensor,
    cells: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: float,
    batch_idx: torch.Tensor,
    cells_per_dimension: torch.Tensor,
    neighbor_search_radius: torch.Tensor,
    cell_offsets: torch.Tensor,
    atom_periodic_shifts: torch.Tensor,
    atom_to_cell_mapping: torch.Tensor,
    atoms_per_cell_count: torch.Tensor,
    cell_atom_start_indices: torch.Tensor,
    cell_atom_list: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    num_neighbors: torch.Tensor,
    half_fill: bool,
    fill_value: int,
) -> None:
    _validate_inputs(
        positions,
        cells,
        pbc,
        cutoff,
        batch_idx,
        cells_per_dimension,
        neighbor_search_radius,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        fill_value,
    )
    if cell_offsets.shape != (cells.shape[0],) or cell_offsets.dtype != torch.int32:
        raise ValueError("cell_offsets must have shape (B,) and dtype int32")
    if not cell_offsets.is_contiguous() or cell_offsets.device != positions.device:
        raise ValueError("cell_offsets must be contiguous on the positions device")
    _load_jit_extension().batch_cell_query_into(
        positions,
        cells,
        pbc,
        float(cutoff),
        batch_idx,
        cells_per_dimension,
        neighbor_search_radius,
        cell_offsets,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        bool(half_fill),
        int(fill_value),
    )


@_batch_cell_query_hip.register_fake
def _batch_cell_query_hip_fake(*args: object, **kwargs: object) -> None:
    """Declare mutation-only fake/compile behavior for the candidate."""

    del args, kwargs
    return None


def batch_query_cell_list_hip_into(
    positions: torch.Tensor,
    cells: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: float,
    batch_idx: torch.Tensor,
    cells_per_dimension: torch.Tensor,
    neighbor_search_radius: torch.Tensor,
    cell_offsets: torch.Tensor,
    atom_periodic_shifts: torch.Tensor,
    atom_to_cell_mapping: torch.Tensor,
    atoms_per_cell_count: torch.Tensor,
    cell_atom_start_indices: torch.Tensor,
    cell_atom_list: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    neighbor_matrix_shifts: torch.Tensor,
    num_neighbors: torch.Tensor,
    *,
    half_fill: bool = False,
    fill_value: int | None = None,
) -> None:
    """Enumerate forward-only candidate pairs into caller-owned dense rows.

    Within-row order is intentionally unspecified.  Callers must canonicalize
    before exposing this data as the public neighbor matrix.  The candidate
    does not provide differentiable distances or vectors.
    """

    resolved_fill = positions.shape[0] if fill_value is None else fill_value
    _validate_inputs(
        positions,
        cells,
        pbc,
        cutoff,
        batch_idx,
        cells_per_dimension,
        neighbor_search_radius,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        resolved_fill,
    )
    _batch_cell_query_hip(
        positions,
        cells,
        pbc,
        float(cutoff),
        batch_idx,
        cells_per_dimension,
        neighbor_search_radius,
        cell_offsets,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        atoms_per_cell_count,
        cell_atom_start_indices,
        cell_atom_list,
        neighbor_matrix,
        neighbor_matrix_shifts,
        num_neighbors,
        bool(half_fill),
        int(resolved_fill),
    )


__all__ = ["batch_query_cell_list_hip_into"]
