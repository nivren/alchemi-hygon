# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend-selected stress conversion, independent of barostat imports."""

from __future__ import annotations

import torch
from nvalchemiops.backend import BackendSelection
from nvalchemiops.executor import execute_selected

from nvalchemi._backend import resolve_compute_backend

__all__ = ["stress_to_cell_force"]


@torch.library.custom_op("nvalchemi::stress_to_cell_force", mutates_args=())
def _stress_to_cell_force_warp(
    stress: torch.Tensor,
    cell: torch.Tensor,
    volume: torch.Tensor,
    keep_aligned: bool = True,
) -> torch.Tensor:
    """Preserve the old torch.ops entrypoint with lazy legacy Warp execution."""
    import warp as wp
    from nvalchemiops.dynamics.utils.cell_filter import stress_to_cell_force as convert

    from nvalchemi.dynamics._ops._bridge import _mat_type, _scalar_type

    mat_t = _mat_type(stress.dtype)
    scl_t = _scalar_type(stress.dtype)
    result = torch.empty_like(stress, memory_format=torch.contiguous_format)
    convert(
        wp.from_torch(stress, dtype=mat_t),
        wp.from_torch(cell, dtype=mat_t),
        wp.from_torch(volume, dtype=scl_t),
        wp.from_torch(result, dtype=mat_t),
        keep_aligned=keep_aligned,
    )
    return result


@_stress_to_cell_force_warp.register_fake
def _stress_to_cell_force_warp_fake(stress, cell, volume, keep_aligned=True):
    """Preserve the legacy operation's fake implementation."""
    return stress.new_empty(stress.shape[0], 3, 3)


def stress_to_cell_force(
    stress: torch.Tensor,
    cell: torch.Tensor,
    volume: torch.Tensor,
    keep_aligned: bool = True,
    *,
    backend: str | None = None,
    selection: BackendSelection | None = None,
) -> torch.Tensor:
    r"""Convert tensile-positive Cauchy stress to raw row-cell force.

    Parameters
    ----------
    stress : torch.Tensor
        Per-system Cartesian stress ``[B, 3, 3]``, in eV/Å³.
    cell : torch.Tensor
        Row lattice vectors ``[B, 3, 3]``, in Å; same dtype/device as stress.
    volume : torch.Tensor
        Physical positive volume ``[B]``, in Å³; same dtype/device.
    keep_aligned : bool, optional
        Zero [0,1], [0,2], [1,2] to preserve a lower-triangular layout.
    backend : str or None, optional
        None/warp preserves legacy Warp selection. Explicit torch_reference
        uses a differentiable device-local implementation without Warp.
    selection : BackendSelection or None, optional
        Pre-resolved stress_to_cell_force selection; never resolved again.

    Returns
    -------
    torch.Tensor
        Independent contiguous ``[B, 3, 3]`` cell force in eV/Å.

    Notes
    -----
    CELL-STRESS-ROW / UBF-009: ``F_H = -V * inverse(H).T @ sigma`` is the
    fixed-fractional-coordinate energy negative gradient for row lattices.
    Reference has first/second derivatives; legacy autograd is not added.
    No optimizer-specific scaling, cell update or alignment is performed.
    """
    selected = resolve_compute_backend(
        backend,
        operation="stress_to_cell_force",
        device=stress.device,
        dtype=stress.dtype,
        gradient_order=2,
        features={"row_cell"},
        selection=selection,
    )
    return execute_selected(
        selected,
        "stress_to_cell_force",
        _stress_to_cell_force_warp,
        stress,
        cell,
        volume,
        keep_aligned,
    )
