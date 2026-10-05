# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Device-local, forward-only equivalents of upstream cell_filter utilities.

Public signatures and output-buffer identities follow the Warp utilities;
vectors/matrices are Torch tensors with their trailing component dimensions.
See docs/TORCH_CELL_TOOLS_CONTRACT.md for the two distinct index layouts.
"""

from __future__ import annotations

import torch

__all__ = [
    "align_cell",
    "extend_batch_idx",
    "extend_atom_ptr",
    "pack_positions_with_cell",
    "unpack_positions_with_cell",
    "pack_velocities_with_cell",
    "unpack_velocities_with_cell",
    "pack_forces_with_cell",
    "pack_masses_with_cell",
]


def _tensors(*tensors, device=None):
    for t in tensors:
        if not isinstance(t, torch.Tensor):
            raise TypeError("cell tools require Torch tensors")
        if t.requires_grad:
            raise ValueError(
                "cell tools are forward-only; requires_grad is unsupported"
            )
        if t.device != tensors[0].device:
            raise ValueError("all tensors must be on the same device")
        if torch._debug_has_internal_overlap(t) == 1:
            raise ValueError("internally overlapping output/input is unsupported")
    if device is not None and torch.device(device) != tensors[0].device:
        raise ValueError("device must match the input device")


def _floats(*tensors, device=None):
    _tensors(*tensors, device=device)
    if tensors[0].dtype not in (torch.float32, torch.float64):
        raise TypeError("cell tools require float32 or float64")
    if any(t.dtype != tensors[0].dtype for t in tensors):
        raise TypeError("floating tensors must have the same dtype")


def _shape(t, shape):
    if t.shape != shape:
        raise ValueError(f"expected shape {shape}, got {tuple(t.shape)}")


def _finite(*inputs):
    if any(not bool(torch.isfinite(t).all()) for t in inputs):
        raise ValueError("inputs must be finite")


def _independent(output, *others):
    if any(torch._C._overlaps(output, other) for other in others):
        raise ValueError("output buffers must not share storage with other tensors")


def _integer(t, anchor):
    _tensors(anchor, t)
    if t.dtype not in (torch.int32, torch.int64):
        raise TypeError("indices/pointers require int32 or int64")


def _idx(idx, n, b, anchor):
    _integer(idx, anchor)
    _shape(idx, (n,))
    if bool(((idx < 0) | (idx >= b)).any()):
        raise ValueError("batch_idx is out of range")


def align_cell(positions, cell, transform, batch_idx=None, device=None):
    """Align to positive-diagonal lower form and preserve fractional positions."""
    _floats(positions, cell, transform, device=device)
    if positions.ndim != 2 or cell.ndim != 3:
        raise ValueError("positions/cell require [N,3]/[B,3,3]")
    n, b = positions.shape[0], cell.shape[0]
    _shape(positions, (n, 3))
    _shape(cell, (b, 3, 3))
    _shape(transform, cell.shape)
    _independent(positions, cell, transform)
    _independent(cell, transform)
    _finite(positions, cell)
    if batch_idx is None:
        if b != 1 and not (b == 0 and n == 0):
            raise ValueError("batch_idx is required for multiple systems")
        batch_idx = torch.zeros(n, device=positions.device, dtype=torch.int64)
    _idx(batch_idx, n, b, positions)
    if bool((torch.linalg.det(cell) == 0).any()):
        raise ValueError("cell must be invertible")
    # Same metric-to-lower-form equations as upstream, vectorized over cells.
    a, length_b, c = torch.linalg.vector_norm(cell, dim=-1).unbind(-1)
    alpha = (cell[:, 1] * cell[:, 2]).sum(-1) / (length_b * c)
    beta = (cell[:, 0] * cell[:, 2]).sum(-1) / (a * c)
    gamma = (cell[:, 0] * cell[:, 1]).sum(-1) / (a * length_b)
    sin_gamma = (1 - gamma.square()).clamp_min(0).sqrt()
    c1 = c * beta
    c2 = c * (alpha - beta * gamma) / sin_gamma
    c3 = (c.square() - c1.square() - c2.square()).clamp_min(0).sqrt()
    aligned = torch.zeros_like(cell)
    aligned[:, 0, 0] = a
    aligned[:, 1, 0] = length_b * gamma
    aligned[:, 1, 1] = length_b * sin_gamma
    aligned[:, 2, 0] = c1
    aligned[:, 2, 1] = c2
    aligned[:, 2, 2] = c3
    _finite(aligned)
    if bool((aligned.diagonal(dim1=-2, dim2=-1) <= 0).any()):
        raise ValueError("cell is numerically degenerate in lower form")
    # UBF-010: use original H even for left-handed cells. Negating H here
    # would negate the fractional coordinates after alignment.
    rotation = torch.linalg.solve(aligned, cell)
    rotated = torch.bmm(rotation[batch_idx.long()], positions.unsqueeze(-1)).squeeze(-1)
    _finite(rotation, rotated)
    positions.copy_(rotated)
    cell.copy_(aligned)
    transform.copy_(rotation)
    return positions, cell


def extend_batch_idx(
    batch_idx, num_atoms, num_systems, extended_batch_idx, device=None
):
    """Preserve upstream append layout, NOT batched pack's interleaved layout."""
    _tensors(batch_idx, extended_batch_idx, device=device)
    if num_atoms < 0 or num_systems < 0:
        raise ValueError("counts must be nonnegative")
    _idx(batch_idx, num_atoms, num_systems, batch_idx)
    _integer(extended_batch_idx, batch_idx)
    if extended_batch_idx.dtype != batch_idx.dtype:
        raise TypeError("index output dtype must match input")
    _shape(extended_batch_idx, (num_atoms + 2 * num_systems,))
    _independent(extended_batch_idx, batch_idx)
    cells = torch.arange(num_systems, device=batch_idx.device, dtype=batch_idx.dtype)
    extended_batch_idx.copy_(torch.cat((batch_idx, cells.repeat_interleave(2))))
    return extended_batch_idx


def extend_atom_ptr(atom_ptr, extended_atom_ptr, device=None):
    """Add two generalized vectors to each contiguous system block."""
    _tensors(atom_ptr, extended_atom_ptr, device=device)
    _integer(atom_ptr, atom_ptr)
    _integer(extended_atom_ptr, atom_ptr)
    if atom_ptr.ndim != 1 or atom_ptr.numel() == 0:
        raise ValueError("atom_ptr requires [B+1]")
    _shape(extended_atom_ptr, atom_ptr.shape)
    if extended_atom_ptr.dtype != atom_ptr.dtype:
        raise TypeError("pointer output dtype must match input")
    if atom_ptr[0] != 0 or bool((atom_ptr[1:] < atom_ptr[:-1]).any()):
        raise ValueError("atom_ptr must start at zero and be nondecreasing")
    if int(atom_ptr[-1]) + 2 * (atom_ptr.numel() - 1) > torch.iinfo(atom_ptr.dtype).max:
        raise ValueError("extended pointer integer overflow")
    _independent(extended_atom_ptr, atom_ptr)
    offsets = torch.arange(
        atom_ptr.numel(), device=atom_ptr.device, dtype=atom_ptr.dtype
    )
    extended_atom_ptr.copy_(atom_ptr + 2 * offsets)
    return extended_atom_ptr


def _layout(n, b, anchor, atom_ptr, ext_atom_ptr, batch_idx):
    if atom_ptr is None:
        if ext_atom_ptr is not None or b != 1:
            raise ValueError("single-system mode requires one cell and no ext_atom_ptr")
        if batch_idx is not None:
            _idx(batch_idx, n, b, anchor)
        return torch.arange(n, device=anchor.device), torch.tensor(
            [n], device=anchor.device
        )
    if ext_atom_ptr is None:
        raise ValueError("ext_atom_ptr is required in batched mode")
    _integer(atom_ptr, anchor)
    _integer(ext_atom_ptr, anchor)
    _shape(atom_ptr, (b + 1,))
    _shape(ext_atom_ptr, (b + 1,))
    if (
        atom_ptr[0] != 0
        or atom_ptr[-1] != n
        or bool((atom_ptr[1:] < atom_ptr[:-1]).any())
    ):
        raise ValueError("atom_ptr must cover exactly the sorted atom blocks")
    ptr = atom_ptr.long()
    expected = ptr + 2 * torch.arange(b + 1, device=anchor.device)
    if not torch.equal(ext_atom_ptr.long(), expected):
        raise ValueError("ext_atom_ptr does not match atom_ptr + 2*system")
    counts = ptr[1:] - ptr[:-1]
    idx = torch.repeat_interleave(
        torch.arange(b, device=anchor.device), counts, output_size=n
    )
    if batch_idx is not None:
        _idx(batch_idx, n, b, anchor)
        if not torch.equal(batch_idx.long(), idx):
            raise ValueError("batch_idx does not match atom_ptr")
    return torch.arange(n, device=anchor.device) + 2 * idx, expected[:-1] + counts


def _vector_buffers(atoms, cell, extended, device):
    _floats(atoms, cell, extended, device=device)
    if atoms.ndim != 2 or cell.ndim != 3:
        raise ValueError("atomic and cell buffers require [N,3]/[B,3,3]")
    n, b = atoms.shape[0], cell.shape[0]
    _shape(atoms, (n, 3))
    _shape(cell, (b, 3, 3))
    _shape(extended, (n + 2 * b, 3))
    return n, b


def pack_positions_with_cell(
    positions,
    cell,
    extended,
    atom_ptr=None,
    ext_atom_ptr=None,
    device=None,
    batch_idx=None,
):
    """Pack six lower components; all upper entries are intentionally ignored."""
    n, b = _vector_buffers(positions, cell, extended, device)
    _finite(positions, cell)
    _independent(extended, positions, cell)
    atom_rows, cell_rows = _layout(n, b, positions, atom_ptr, ext_atom_ptr, batch_idx)
    extended[atom_rows] = positions
    extended[cell_rows] = cell[:, :, 0]
    extended[cell_rows + 1] = cell[:, [1, 2, 2], [1, 1, 2]]
    return extended


def unpack_positions_with_cell(
    extended,
    positions,
    cell,
    num_atoms=None,
    atom_ptr=None,
    ext_atom_ptr=None,
    device=None,
    batch_idx=None,
):
    """Restore atom vectors and lower cell, zeroing the three upper entries."""
    n, b = _vector_buffers(positions, cell, extended, device)
    _finite(extended)
    _independent(positions, extended, cell)
    _independent(cell, extended)
    if atom_ptr is None and num_atoms != n:
        raise ValueError("num_atoms must match the output in single-system mode")
    atom_rows, cell_rows = _layout(n, b, positions, atom_ptr, ext_atom_ptr, batch_idx)
    positions.copy_(extended[atom_rows])
    cell.zero_()
    cell[:, :, 0] = extended[cell_rows]
    cell[:, [1, 2, 2], [1, 1, 2]] = extended[cell_rows + 1]
    return positions, cell


def pack_velocities_with_cell(
    velocities,
    cell_velocity,
    extended,
    atom_ptr=None,
    ext_atom_ptr=None,
    device=None,
    batch_idx=None,
):
    return pack_positions_with_cell(
        velocities, cell_velocity, extended, atom_ptr, ext_atom_ptr, device, batch_idx
    )


def unpack_velocities_with_cell(
    extended,
    velocities,
    cell_velocity,
    num_atoms=None,
    atom_ptr=None,
    ext_atom_ptr=None,
    device=None,
    batch_idx=None,
):
    return unpack_positions_with_cell(
        extended,
        velocities,
        cell_velocity,
        num_atoms,
        atom_ptr,
        ext_atom_ptr,
        device,
        batch_idx,
    )


def pack_forces_with_cell(
    forces,
    cell_force,
    extended,
    atom_ptr=None,
    ext_atom_ptr=None,
    device=None,
    batch_idx=None,
):
    return pack_positions_with_cell(
        forces, cell_force, extended, atom_ptr, ext_atom_ptr, device, batch_idx
    )


def pack_masses_with_cell(
    masses, cell_masses, extended, atom_ptr=None, ext_atom_ptr=None, device=None
):
    _floats(masses, cell_masses, extended, device=device)
    if masses.ndim != 1 or cell_masses.ndim != 1:
        raise ValueError("masses require [N] and [B]")
    n, b = masses.numel(), cell_masses.numel()
    _shape(extended, (n + 2 * b,))
    _finite(masses, cell_masses)
    _independent(extended, masses, cell_masses)
    atom_rows, cell_rows = _layout(n, b, masses, atom_ptr, ext_atom_ptr, None)
    extended[atom_rows] = masses
    extended[cell_rows] = cell_masses
    extended[cell_rows + 1] = cell_masses
    return extended
