# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared cell-list build ABI for Torch, HIP, and Triton paths.

The ABI currently has three composable phases: per-atom fractional-coordinate
work (``positions @ inverse_cell``, periodic shifts, cell coordinates, linear
keys), key-to-CSR count/start construction, and stable CSR atom-list fill.
Neighbor enumeration and output capacity policy remain separate modules.

The ``*_into`` form is the authoritative interface.  It makes all output
mutation explicit, avoids a hidden allocation contract, and can map directly
to a future ``torch.library`` custom op and native HIP implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CellKeyBuildResult:
    """Allocated convenience result for :func:`build_cell_keys_reference`."""

    atom_periodic_shifts: torch.Tensor
    atom_to_cell_mapping: torch.Tensor
    cell_keys: torch.Tensor


@dataclass(frozen=True)
class CellKeyCsrResult:
    """Allocated convenience result for :func:`build_cell_csr_reference`."""

    cell_counts: torch.Tensor
    cell_starts: torch.Tensor


def _validate_inputs(
    positions: torch.Tensor,
    inverse_cell: torch.Tensor,
    cells_per_dimension: torch.Tensor,
    pbc: torch.Tensor,
) -> None:
    if positions.ndim != 2 or positions.shape[1:] != (3,):
        raise ValueError("positions must have shape (N, 3)")
    if positions.dtype not in (torch.float32, torch.float64):
        raise TypeError("positions must have dtype float32 or float64")
    if inverse_cell.shape != (3, 3):
        raise ValueError("inverse_cell must have shape (3, 3)")
    if inverse_cell.dtype != positions.dtype:
        raise TypeError("inverse_cell dtype must match positions")
    if cells_per_dimension.shape != (3,) or cells_per_dimension.dtype != torch.int32:
        raise ValueError("cells_per_dimension must have shape (3,) and dtype int32")
    if pbc.shape != (3,) or pbc.dtype != torch.bool:
        raise ValueError("pbc must have shape (3,) and dtype bool")
    if any(
        value.device != positions.device
        for value in (inverse_cell, cells_per_dimension, pbc)
    ):
        raise ValueError("cell-key inputs must be on the same device")
    if bool(torch.any(cells_per_dimension <= 0)):
        raise ValueError("cells_per_dimension must be positive")
    total_cells = 1
    for value in cells_per_dimension:
        total_cells *= int(value.item())
        if total_cells > torch.iinfo(torch.int32).max + 1:
            raise ValueError("cell-key range exceeds int32")


def _validate_outputs(
    positions: torch.Tensor,
    atom_periodic_shifts: torch.Tensor,
    atom_to_cell_mapping: torch.Tensor,
    cell_keys: torch.Tensor,
) -> None:
    count = positions.shape[0]
    expected_matrix = (count, 3)
    if atom_periodic_shifts.shape != expected_matrix:
        raise ValueError("atom_periodic_shifts must have shape (N, 3)")
    if atom_to_cell_mapping.shape != expected_matrix:
        raise ValueError("atom_to_cell_mapping must have shape (N, 3)")
    if cell_keys.shape != (count,):
        raise ValueError("cell_keys must have shape (N,)")
    if any(
        value.dtype != torch.int32
        for value in (atom_periodic_shifts, atom_to_cell_mapping, cell_keys)
    ):
        raise TypeError("cell-key outputs must have dtype int32")
    if any(
        value.device != positions.device
        for value in (atom_periodic_shifts, atom_to_cell_mapping, cell_keys)
    ):
        raise ValueError("cell-key outputs must be on the same device as positions")
    if count:
        output_ptrs = tuple(
            value.data_ptr()
            for value in (atom_periodic_shifts, atom_to_cell_mapping, cell_keys)
        )
        if len(set(output_ptrs)) != len(output_ptrs):
            raise ValueError("cell-key output buffers must not alias")


def build_cell_keys_reference_into(
    positions: torch.Tensor,
    inverse_cell: torch.Tensor,
    cells_per_dimension: torch.Tensor,
    pbc: torch.Tensor,
    atom_periodic_shifts: torch.Tensor,
    atom_to_cell_mapping: torch.Tensor,
    cell_keys: torch.Tensor,
) -> None:
    """Write the shared cell-key ABI outputs using ordinary Torch tensors.

    ``cells_per_dimension`` is a positive int32 vector.  The product of its
    entries must fit in int32; the cell-list allocator enforces a much smaller
    configured bound before calling this function.
    """

    _validate_inputs(positions, inverse_cell, cells_per_dimension, pbc)
    _validate_outputs(
        positions,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        cell_keys,
    )
    if not positions.numel():
        return
    fractional = positions @ inverse_cell
    shifts = torch.where(pbc, torch.floor(fractional), torch.zeros_like(fractional))
    wrapped = fractional - shifts
    coordinates = torch.floor(
        wrapped * cells_per_dimension.to(positions.dtype)
    ).to(torch.int64)
    coordinates = torch.minimum(
        torch.maximum(coordinates, torch.zeros_like(coordinates)),
        cells_per_dimension.to(torch.int64) - 1,
    )
    dimensions = cells_per_dimension.to(torch.int64)
    keys = coordinates[:, 0] + dimensions[0] * (
        coordinates[:, 1] + dimensions[1] * coordinates[:, 2]
    )
    if bool(torch.any(keys > torch.iinfo(torch.int32).max)):
        raise ValueError("cell-key range exceeds int32")
    atom_periodic_shifts.copy_(shifts.to(torch.int32))
    atom_to_cell_mapping.copy_(coordinates.to(torch.int32))
    cell_keys.copy_(keys.to(torch.int32))


def build_cell_keys_reference(
    positions: torch.Tensor,
    inverse_cell: torch.Tensor,
    cells_per_dimension: torch.Tensor,
    pbc: torch.Tensor,
) -> CellKeyBuildResult:
    """Allocate and return the shared cell-key ABI reference outputs."""

    outputs = CellKeyBuildResult(
        atom_periodic_shifts=torch.empty(
            (positions.shape[0], 3), dtype=torch.int32, device=positions.device
        ),
        atom_to_cell_mapping=torch.empty(
            (positions.shape[0], 3), dtype=torch.int32, device=positions.device
        ),
        cell_keys=torch.empty(
            positions.shape[0], dtype=torch.int32, device=positions.device
        ),
    )
    build_cell_keys_reference_into(
        positions,
        inverse_cell,
        cells_per_dimension,
        pbc,
        outputs.atom_periodic_shifts,
        outputs.atom_to_cell_mapping,
        outputs.cell_keys,
    )
    return outputs


def _validate_cell_count_inputs(
    cell_keys: torch.Tensor,
    cell_counts: torch.Tensor,
) -> None:
    if cell_keys.ndim != 1 or cell_keys.dtype != torch.int32:
        raise ValueError("cell_keys must have shape (N,) and dtype int32")
    if cell_counts.ndim != 1 or cell_counts.dtype != torch.int32:
        raise ValueError("cell_counts must have shape (M,) and dtype int32")
    if cell_counts.device != cell_keys.device:
        raise ValueError("cell-key CSR tensors must be on the same device")
    if cell_keys.numel() > torch.iinfo(torch.int32).max:
        raise ValueError("cell-count range exceeds int32")
    if not cell_keys.numel():
        return
    if not cell_counts.numel() or bool(
        torch.any((cell_keys < 0) | (cell_keys >= cell_counts.numel()))
    ):
        raise ValueError("cell_keys must be in [0, cell_counts.numel())")


def build_cell_counts_reference_into(
    cell_keys: torch.Tensor,
    cell_counts: torch.Tensor,
) -> None:
    """Write exact per-cell counts from ``int32`` linear cell keys.

    This is the first CSR phase.  It overwrites all ``cell_counts`` entries,
    including for empty key input, and does not determine atom-list order.
    """

    _validate_cell_count_inputs(cell_keys, cell_counts)
    counts = torch.bincount(cell_keys.to(torch.long), minlength=cell_counts.numel())
    int32_max = torch.iinfo(torch.int32).max
    if bool(torch.any(counts > int32_max)):
        raise ValueError("cell-count range exceeds int32")
    cell_counts.copy_(counts.to(torch.int32))


def _validate_batch_cell_key_count_inputs(
    positions: torch.Tensor,
    inverse_cells: torch.Tensor,
    cells_per_dimension: torch.Tensor,
    pbc: torch.Tensor,
    batch_idx: torch.Tensor,
    cell_offsets: torch.Tensor,
    atom_periodic_shifts: torch.Tensor,
    atom_to_cell_mapping: torch.Tensor,
    cell_keys: torch.Tensor,
    cell_counts: torch.Tensor,
) -> None:
    """Validate the batch-aware fused geometry/PBC/key/count ABI."""

    if positions.ndim != 2 or positions.shape[1:] != (3,):
        raise ValueError("positions must have shape (N, 3)")
    if positions.dtype not in (torch.float32, torch.float64):
        raise TypeError("positions must have dtype float32 or float64")
    systems = inverse_cells.shape[0] if inverse_cells.ndim else -1
    if inverse_cells.shape != (systems, 3, 3):
        raise ValueError("inverse_cells must have shape (B, 3, 3)")
    if inverse_cells.dtype != positions.dtype:
        raise TypeError("inverse_cells dtype must match positions")
    if systems <= 0:
        raise ValueError("batch cell-key/count ABI requires at least one system")
    if cells_per_dimension.shape != (systems, 3) or cells_per_dimension.dtype != torch.int32:
        raise ValueError("cells_per_dimension must have shape (B, 3) and dtype int32")
    if pbc.shape != (systems, 3) or pbc.dtype != torch.bool:
        raise ValueError("pbc must have shape (B, 3) and dtype bool")
    if batch_idx.shape != (positions.shape[0],) or batch_idx.dtype != torch.int32:
        raise ValueError("batch_idx must have shape (N,) and dtype int32")
    if cell_offsets.shape != (systems,) or cell_offsets.dtype != torch.int32:
        raise ValueError("cell_offsets must have shape (B,) and dtype int32")
    if cell_counts.ndim != 1 or cell_counts.dtype != torch.int32:
        raise ValueError("cell_counts must have shape (M,) and dtype int32")
    _validate_outputs(
        positions,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        cell_keys,
    )
    if any(
        value.device != positions.device
        for value in (
            inverse_cells,
            cells_per_dimension,
            pbc,
            batch_idx,
            cell_offsets,
            cell_counts,
        )
    ):
        raise ValueError("batch cell-key/count tensors must be on the same device")
    if bool(torch.any(cells_per_dimension <= 0)):
        raise ValueError("cells_per_dimension must be positive")
    cell_sizes = cells_per_dimension.to(torch.int64).prod(dim=1)
    int32_max = torch.iinfo(torch.int32).max
    if bool(torch.any(cell_sizes > int32_max + 1)):
        raise ValueError("per-system cell-key range exceeds int32")
    expected_offsets = cell_sizes.cumsum(0) - cell_sizes
    total_cells = int(cell_sizes.sum().item())
    if total_cells > int32_max + 1:
        raise ValueError("global cell-key range exceeds int32")
    if cell_counts.numel() != total_cells:
        raise ValueError("cell_counts must exactly cover the concatenated cell slices")
    if not torch.equal(cell_offsets.to(torch.int64), expected_offsets):
        raise ValueError("cell_offsets must be contiguous global cell-slice offsets")
    if batch_idx.numel() and bool(
        torch.any((batch_idx < 0) | (batch_idx >= systems))
    ):
        raise ValueError("batch_idx must be in [0, B)")
    output_ptrs = (
        atom_periodic_shifts.data_ptr(),
        atom_to_cell_mapping.data_ptr(),
        cell_keys.data_ptr(),
        cell_counts.data_ptr(),
    )
    if all(value.numel() for value in (
        atom_periodic_shifts,
        atom_to_cell_mapping,
        cell_keys,
        cell_counts,
    )) and len(set(output_ptrs)) != len(output_ptrs):
        raise ValueError("batch cell-key/count output buffers must not alias")


def build_batch_cell_key_counts_reference_into(
    positions: torch.Tensor,
    inverse_cells: torch.Tensor,
    cells_per_dimension: torch.Tensor,
    pbc: torch.Tensor,
    batch_idx: torch.Tensor,
    cell_offsets: torch.Tensor,
    atom_periodic_shifts: torch.Tensor,
    atom_to_cell_mapping: torch.Tensor,
    cell_keys: torch.Tensor,
    cell_counts: torch.Tensor,
) -> None:
    """Write batch-aware geometry/PBC/key/count outputs with Torch.

    The global key is each system-local linear key plus its contiguous
    ``cell_offsets`` entry.  ``cell_counts`` is fully overwritten, including
    empty atom input.  This is the oracle for the fused native HIP candidate;
    it deliberately excludes scan, CSR fill, and neighbor querying.
    """

    _validate_batch_cell_key_count_inputs(
        positions,
        inverse_cells,
        cells_per_dimension,
        pbc,
        batch_idx,
        cell_offsets,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        cell_keys,
        cell_counts,
    )
    cell_counts.zero_()
    if not positions.numel():
        return
    inverse_per_atom = inverse_cells[batch_idx.to(torch.long)]
    fractional = torch.bmm(positions.unsqueeze(1), inverse_per_atom).squeeze(1)
    pbc_per_atom = pbc[batch_idx.to(torch.long)]
    shifts = torch.where(
        pbc_per_atom, torch.floor(fractional), torch.zeros_like(fractional)
    )
    dimensions = cells_per_dimension[batch_idx.to(torch.long)]
    coordinates = torch.floor(
        (fractional - shifts) * dimensions.to(positions.dtype)
    ).to(torch.int64)
    coordinates = torch.minimum(
        torch.maximum(coordinates, torch.zeros_like(coordinates)),
        dimensions.to(torch.int64) - 1,
    )
    local_keys = coordinates[:, 0] + dimensions[:, 0].to(torch.int64) * (
        coordinates[:, 1] + dimensions[:, 1].to(torch.int64) * coordinates[:, 2]
    )
    keys = local_keys + cell_offsets[batch_idx.to(torch.long)].to(torch.int64)
    atom_periodic_shifts.copy_(shifts.to(torch.int32))
    atom_to_cell_mapping.copy_(coordinates.to(torch.int32))
    cell_keys.copy_(keys.to(torch.int32))
    cell_counts.copy_(
        torch.bincount(keys, minlength=cell_counts.numel()).to(torch.int32)
    )


def _validate_cell_scan_structure(
    cell_counts: torch.Tensor,
    cell_starts: torch.Tensor,
    global_atom_offset: int,
) -> None:
    if cell_counts.ndim != 1 or cell_counts.dtype != torch.int32:
        raise ValueError("cell_counts must have shape (M,) and dtype int32")
    if cell_starts.shape != cell_counts.shape or cell_starts.dtype != torch.int32:
        raise ValueError("cell_starts must have shape (M,) and dtype int32")
    if cell_starts.device != cell_counts.device:
        raise ValueError("cell-count and cell-start tensors must be on the same device")
    if cell_counts.numel() and cell_counts.data_ptr() == cell_starts.data_ptr():
        raise ValueError("cell_counts and cell_starts must not alias")
    if isinstance(global_atom_offset, bool) or not isinstance(global_atom_offset, int):
        raise TypeError("global_atom_offset must be an int")
    int32_max = torch.iinfo(torch.int32).max
    if not 0 <= global_atom_offset <= int32_max:
        raise ValueError("global_atom_offset must fit int32")


def _validate_cell_scan_inputs(
    cell_counts: torch.Tensor,
    cell_starts: torch.Tensor,
    global_atom_offset: int,
) -> None:
    _validate_cell_scan_structure(cell_counts, cell_starts, global_atom_offset)
    if cell_counts.numel() and bool(torch.any(cell_counts < 0)):
        raise ValueError("cell_counts must be non-negative")
    total = int(cell_counts.to(torch.int64).sum().item())
    int32_max = torch.iinfo(torch.int32).max
    if global_atom_offset + total > int32_max:
        raise ValueError("cell-start range exceeds int32")


def build_cell_starts_reference_into(
    cell_counts: torch.Tensor,
    cell_starts: torch.Tensor,
    *,
    global_atom_offset: int = 0,
) -> None:
    """Write CSR exclusive starts without modifying the caller's counts.

    ``cell_counts`` may represent one system or a globally concatenated Batch
    cell slice.  ``global_atom_offset`` is the first atom-list index of that
    active slice; no batch-specific branch is required in this phase.
    """

    _validate_cell_scan_inputs(cell_counts, cell_starts, global_atom_offset)
    counts = cell_counts.to(torch.int64)
    starts = counts.cumsum(0) - counts + global_atom_offset
    cell_starts.copy_(starts.to(torch.int32))


def _validate_cell_atom_list_fill_inputs(
    cell_keys: torch.Tensor,
    cell_counts: torch.Tensor,
    cell_starts: torch.Tensor,
    cell_atom_list: torch.Tensor,
    cell_cursor: torch.Tensor,
    *,
    global_atom_offset: int = 0,
) -> None:
    """Validate shared CSR fill inputs without mutating any tensor."""

    _validate_cell_count_inputs(cell_keys, cell_counts)
    _validate_cell_scan_inputs(cell_counts, cell_starts, global_atom_offset)
    if cell_atom_list.ndim != 1 or cell_atom_list.dtype != torch.int32:
        raise ValueError("cell_atom_list must have shape (capacity,) and dtype int32")
    if cell_cursor.shape != cell_counts.shape or cell_cursor.dtype != torch.int32:
        raise ValueError("cell_cursor must match cell_counts shape and have dtype int32")
    if any(
        value.device != cell_counts.device
        for value in (cell_atom_list, cell_cursor)
    ):
        raise ValueError("CSR fill tensors must be on the same device")
    if not cell_atom_list.is_contiguous() or not cell_cursor.is_contiguous():
        raise ValueError("CSR fill output and cursor buffers must be contiguous")
    if cell_atom_list.numel() < cell_keys.numel():
        raise ValueError("cell_atom_list capacity is smaller than the active atom count")
    if cell_cursor.data_ptr() == cell_counts.data_ptr():
        raise ValueError("cell_cursor must not alias cell_counts")
    if cell_cursor.numel() and cell_cursor.data_ptr() == cell_starts.data_ptr():
        raise ValueError("cell_cursor must not alias cell_starts")
    if int(cell_counts.to(torch.int64).sum().item()) != cell_keys.numel():
        raise ValueError("cell_counts total must equal the active atom count")
    int32_max = torch.iinfo(torch.int32).max
    if global_atom_offset + cell_keys.numel() > int32_max:
        raise ValueError("global atom index range exceeds int32")


def build_cell_atom_list_reference_into(
    cell_keys: torch.Tensor,
    cell_counts: torch.Tensor,
    cell_starts: torch.Tensor,
    cell_atom_list: torch.Tensor,
    cell_cursor: torch.Tensor,
    *,
    global_atom_offset: int = 0,
) -> None:
    """Fill a CSR atom list and explicit insertion cursor with Torch.

    ``cell_keys`` are the linear keys for the active atom slice.  The output
    list is local storage for that slice; each stored atom index receives
    ``global_atom_offset`` so a Batch can retain global atom IDs.  The cursor
    is caller-owned scratch and is left equal to final counts.  Public counts
    and starts are read-only, unlike the upstream ``bin_atoms`` implementation
    which may reuse the count array as an internal cursor.

    The stable key sort preserves the current Torch cell-list ordering
    contract.  A future atomic HIP fill must either reproduce this order or
    add an explicit canonicalization step before exposing the list publicly.
    """

    _validate_cell_atom_list_fill_inputs(
        cell_keys,
        cell_counts,
        cell_starts,
        cell_atom_list,
        cell_cursor,
        global_atom_offset=global_atom_offset,
    )

    cell_cursor.copy_(cell_counts)
    if not cell_keys.numel():
        return
    order = torch.argsort(cell_keys, stable=True)
    cell_atom_list[: cell_keys.numel()].copy_(
        order.to(torch.int32) + global_atom_offset
    )


def _validate_csr_inputs(
    cell_keys: torch.Tensor,
    cell_counts: torch.Tensor,
    cell_starts: torch.Tensor,
    global_atom_offset: int,
) -> None:
    _validate_cell_count_inputs(cell_keys, cell_counts)
    # ``cell_counts`` is an output of the preceding count phase and may be
    # uninitialized on entry.  Validate its layout here; validate its values
    # only after count construction, in ``build_cell_starts_reference_into``.
    _validate_cell_scan_structure(cell_counts, cell_starts, global_atom_offset)


def build_cell_csr_reference_into(
    cell_keys: torch.Tensor,
    cell_counts: torch.Tensor,
    cell_starts: torch.Tensor,
    *,
    global_atom_offset: int = 0,
) -> None:
    """Write count and exclusive starts for a cell-key CSR layout.

    ``cell_counts`` and ``cell_starts`` define the number of active cells.  A
    start is the exclusive prefix sum of counts plus ``global_atom_offset``;
    the atom-list sort/fill phase is intentionally a separate operation.
    """

    _validate_csr_inputs(
        cell_keys, cell_counts, cell_starts, global_atom_offset
    )
    build_cell_counts_reference_into(cell_keys, cell_counts)
    build_cell_starts_reference_into(
        cell_counts,
        cell_starts,
        global_atom_offset=global_atom_offset,
    )


def build_cell_csr_reference(
    cell_keys: torch.Tensor,
    total_cells: int,
    *,
    global_atom_offset: int = 0,
) -> CellKeyCsrResult:
    """Allocate and return the shared cell-key CSR count/start outputs."""

    if isinstance(total_cells, bool) or not isinstance(total_cells, int):
        raise TypeError("total_cells must be an int")
    if not 0 <= total_cells <= torch.iinfo(torch.int32).max + 1:
        raise ValueError("total_cells must be in [0, int32_max + 1]")
    outputs = CellKeyCsrResult(
        cell_counts=torch.empty(total_cells, dtype=torch.int32, device=cell_keys.device),
        cell_starts=torch.empty(total_cells, dtype=torch.int32, device=cell_keys.device),
    )
    build_cell_csr_reference_into(
        cell_keys,
        outputs.cell_counts,
        outputs.cell_starts,
        global_atom_offset=global_atom_offset,
    )
    return outputs


__all__ = [
    "CellKeyBuildResult",
    "CellKeyCsrResult",
    "build_batch_cell_key_counts_reference_into",
    "build_cell_atom_list_reference_into",
    "build_cell_counts_reference_into",
    "build_cell_csr_reference",
    "build_cell_csr_reference_into",
    "build_cell_starts_reference_into",
    "build_cell_keys_reference",
    "build_cell_keys_reference_into",
]
