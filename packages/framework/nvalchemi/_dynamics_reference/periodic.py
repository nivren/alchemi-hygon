# SPDX-License-Identifier: Apache-2.0
"""Warp-free periodic coordinate wrapping for the reference backend."""

from __future__ import annotations

import torch

__all__ = ["wrap_positions_into_cell"]


@torch.library.custom_op(
    "nvalchemi::reference_wrap_positions", mutates_args={"positions"}
)
def wrap_positions_into_cell(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    batch_idx: torch.Tensor,
) -> None:
    """Wrap periodic dimensions of ``positions`` in-place.

    ``cell`` stores lattice vectors as rows.  Fractional coordinates are
    therefore computed as ``positions @ inv(cell)`` for the cell selected by
    each atom's ``batch_idx``.  Non-periodic dimensions retain their original
    Cartesian coordinates.
    """
    if positions.ndim != 2 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape [N, 3]")
    if cell.ndim != 3 or cell.shape[-2:] != (3, 3):
        raise ValueError("cell must have shape [B, 3, 3]")
    if pbc.shape != (cell.shape[0], 3):
        raise ValueError("pbc must have shape [B, 3]")
    if batch_idx.ndim != 1 or batch_idx.shape[0] != positions.shape[0]:
        raise ValueError("batch_idx must have shape [N]")
    if batch_idx.device != positions.device or cell.device != positions.device:
        raise ValueError("positions, cell, and batch_idx must share a device")
    if pbc.device != positions.device:
        raise ValueError("pbc must share the positions device")
    if positions.dtype not in (torch.float32, torch.float64):
        raise TypeError("positions must use float32 or float64")

    idx = batch_idx.to(torch.int64)
    if idx.numel() and bool(torch.any(idx < 0)):
        raise ValueError("batch_idx contains a negative system index")
    if idx.numel() and bool(torch.any(idx >= cell.shape[0])):
        raise ValueError("batch_idx contains a system index outside cell")

    if positions.shape[0] == 0:
        return None
    selected_cell = cell.index_select(0, idx)
    # For row-vector coordinates, solve cell.T @ frac.T = pos.T.
    fractional = torch.linalg.solve(
        selected_cell.transpose(-2, -1), positions.unsqueeze(-1)
    ).squeeze(-1)
    wrapped_fractional = torch.remainder(fractional, 1.0)
    wrapped = torch.bmm(wrapped_fractional.unsqueeze(1), selected_cell).squeeze(1)
    # The public helper defines non-periodic *Cartesian* dimensions as a
    # no-op.  Restore them after converting back from fractional coordinates;
    # retaining a fractional component would change a skewed cell's Cartesian
    # coordinate through the off-diagonal lattice vectors.
    per_atom_pbc = pbc.index_select(0, idx)
    wrapped = torch.where(per_atom_pbc, wrapped, positions)
    with torch.no_grad():
        positions.copy_(wrapped)
    return None


@wrap_positions_into_cell.register_fake
def _wrap_positions_into_cell_fake(
    positions: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    batch_idx: torch.Tensor,
) -> None:
    return None
