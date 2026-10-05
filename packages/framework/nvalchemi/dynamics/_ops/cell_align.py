# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PyTorch binding for cell alignment to lower-triangular form.

Retains the legacy ``torch.library.custom_op`` and its fake registration.
Explicit Torch reference uses a device-local, forward-only eager executor;
its autograd/compile behavior is not supported by this port.

Functions
---------
align_cell
    Align periodic cells to lower-triangular form and rotate positions
    to preserve fractional coordinates.
"""

from __future__ import annotations

import torch
import torch.library
from nvalchemiops.backend import BackendSelection
from nvalchemiops.executor import execute_selected

from nvalchemi._backend import resolve_compute_backend

__all__ = ["align_cell"]


# ---------------------------------------------------------------------------
# Internal custom op
# ---------------------------------------------------------------------------


@torch.library.custom_op("nvalchemi::align_cell", mutates_args={"positions", "cell"})
def _align_cell_op(
    positions: torch.Tensor,
    cell: torch.Tensor,
    batch_idx: torch.Tensor,
) -> None:
    """Align cells to lower-triangular form and transform positions in-place.

    Parameters
    ----------
    positions : torch.Tensor
        Atomic positions ``[N, 3]``, float32 or float64.
    cell : torch.Tensor
        Per-system cell matrices ``[M, 3, 3]``, same dtype.  Overwritten
        with the aligned (lower-triangular) cells.
    batch_idx : torch.Tensor
        Per-atom system index ``[N]``, int32, non-decreasing.
    """
    import warp as wp
    from nvalchemiops.dynamics.utils import align_cell as _align_cell

    from nvalchemi.dynamics._ops._bridge import _mat_type, _vec_type

    dtype = positions.dtype
    vec_t = _vec_type(dtype)
    mat_t = _mat_type(dtype)

    cell_c = cell.contiguous()
    positions_c = positions.contiguous()
    transform = torch.empty_like(cell_c)

    wp_device = wp.device_from_torch(positions.device)
    _align_cell(
        wp.from_torch(positions_c, dtype=vec_t),
        wp.from_torch(cell_c, dtype=mat_t),
        wp.from_torch(transform, dtype=mat_t),
        batch_idx=wp.from_torch(
            batch_idx.to(dtype=torch.int32).contiguous(), dtype=wp.int32
        ),
        device=wp_device,
    )
    # UBF-010: contiguous() can allocate; rotated strided positions must be
    # copied back as well as the cell, otherwise fractional geometry changes.
    positions.copy_(positions_c)
    cell.copy_(cell_c)


@_align_cell_op.register_fake
def _align_cell_op_fake(
    positions: torch.Tensor,
    cell: torch.Tensor,
    batch_idx: torch.Tensor,
) -> None:
    pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def align_cell(
    positions: torch.Tensor,
    cell: torch.Tensor,
    batch_idx: torch.Tensor | None = None,
    *,
    backend: str | None = None,
    selection: BackendSelection | None = None,
) -> None:
    """Align periodic cells to lower-triangular form and rotate positions.

    This is a one-time preprocessing step before variable-cell optimization.
    The cell is transformed to the standard lower-triangular form, and
    positions are rotated to maintain their fractional coordinates.

    Parameters
    ----------
    positions : torch.Tensor
        Atomic positions ``[N, 3]``, float32 or float64.  Modified in-place.
    cell : torch.Tensor
        Per-system cell matrices ``[M, 3, 3]``, same dtype.  Overwritten
        with aligned (lower-triangular) cells.
    batch_idx : torch.Tensor, optional
        Per-atom system index ``[N]``, int32.  If ``None``, all atoms are
        assumed to belong to a single system.
    backend : str or None, optional
        None/warp retains legacy execution; explicit torch_reference is
        forward-only and requires independent, non-overlapping buffers.
    selection : BackendSelection, optional
        Pre-resolved cell_tools selection, reused without another resolution.
    """
    selected = resolve_compute_backend(
        backend,
        operation="cell_tools",
        device=positions.device,
        dtype=positions.dtype,
        gradient_order=0,
        features={"row_cell"},
        selection=selection,
    )
    if batch_idx is None:
        batch_idx = torch.zeros(
            positions.shape[0], dtype=torch.int32, device=positions.device
        )

    def legacy_align(positions, cell, transform, batch_idx):
        _align_cell_op(positions, cell, batch_idx)

    execute_selected(
        selected,
        "align_cell",
        legacy_align,
        positions,
        cell,
        torch.empty_like(cell),
        batch_idx,
    )
