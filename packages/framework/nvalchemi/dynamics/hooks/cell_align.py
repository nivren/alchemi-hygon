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
Cell alignment hook for variable-cell optimization.

Provides :class:`AlignCellHook`, which aligns periodic simulation cells to
lower-triangular (right-handed) form before the first optimizer step, and
:func:`_align_atomic_data_cell`, a standalone utility for aligning a single
:class:`~nvalchemi.data.AtomicData` instance.
"""

from __future__ import annotations

from enum import Enum

import torch

from nvalchemi._backend import resolve_compute_backend
from nvalchemi.dynamics._ops.cell_align import align_cell
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.hooks._context import DynamicsContext

__all__ = ["AlignCellHook"]


class AlignCellHook:
    r"""Align periodic cells before the first variable-cell FIRE2 step.

    Transforms each periodic system's cell matrix to the standard
    lower-triangular (right-handed) form:

    .. math::

       H = \begin{bmatrix}
       a & 0 & 0 \\
       b\cos\gamma & b\sin\gamma & 0 \\
       c_1 & c_2 & c_3
       \end{bmatrix}

    and rotates positions to preserve fractional coordinates.  This
    representation reduces rotational ambiguity (improving optimizer
    stability) and has 6 independent parameters instead of 9.

    The hook fires at :attr:`~DynamicsStage.BEFORE_STEP` and skips
    non-periodic systems.

    Parameters
    ----------
    frequency : int, optional
        Run every ``frequency`` steps.  Default ``1``.

    Attributes
    ----------
    frequency : int
        Hook execution frequency.
    stage : DynamicsStage
        Always :attr:`DynamicsStage.BEFORE_STEP`.

    Examples
    --------
    >>> from nvalchemi.dynamics.hooks import AlignCellHook
    >>> hook = AlignCellHook()
    >>> optimizer = FIRE2VariableCell(model=model, hooks=[hook])
    """

    def __init__(self, frequency: int = 1, *, backend: str | None = None) -> None:
        self.stage = DynamicsStage.BEFORE_STEP
        self.frequency = frequency
        self.backend = backend
        self._selection = None

    def __call__(self, ctx: DynamicsContext, stage: Enum) -> None:
        """Align the current batch when any periodic cell is not triangular."""
        del stage
        batch = ctx.batch

        if not hasattr(batch, "cell") or batch.cell is None:
            return
        if not hasattr(batch, "pbc") or batch.pbc is None:
            return

        # Determine which systems are periodic
        pbc = batch.pbc
        if pbc.dim() == 1:
            if pbc.shape[0] == batch.num_graphs:
                periodic_mask = pbc.to(dtype=torch.bool)
            else:
                periodic_mask = pbc.unsqueeze(0).any(dim=-1)
        else:
            periodic_mask = pbc.any(dim=-1)
        # The fork's newer context carries active_graph_mask. Keep the locked
        # context/schema: derive its equivalent from existing status/exit_status.
        active = getattr(ctx, "active_graph_mask", None)
        status = getattr(batch, "status", None)
        exit_status = getattr(ctx.workflow, "exit_status", None)
        if active is None and status is not None and exit_status is not None:
            active = status.reshape(-1) < exit_status
        if active is not None:
            periodic_mask = periodic_mask & active
        if not periodic_mask.any():
            return
        # UBF-014: fork b12b54c protects the periodic/active subset. Also
        # rotate cached vectors/tensors, which the fork still leaves unchanged.
        # Alignment is a preprocessing operation, not a repeated optimizer step.
        nonlower = torch.triu(batch.cell, 1).ne(0).any(dim=(-1, -2))
        nonpositive = (batch.cell.diagonal(dim1=-2, dim2=-1) <= 0).any(-1)
        selected_systems = periodic_mask & (nonlower | nonpositive)
        if not selected_systems.any():
            return
        atom_mask = selected_systems[batch.batch_idx.long()]
        positions = batch.positions[atom_mask].clone()
        cell = batch.cell[selected_systems].clone()
        if positions.dtype not in (torch.float32, torch.float64):
            raise TypeError(
                "Cell alignment only supports float32/float64 positions, got "
                f"{positions.dtype}."
            )
        workflow = ctx.workflow
        if self._selection is None:
            inherited = getattr(workflow, "_cell_tools_selection", None)
            request = (
                self.backend
                if self.backend is not None
                else getattr(workflow, "backend", None)
            )
            self._selection = resolve_compute_backend(
                request,
                operation="cell_tools",
                device=positions.device,
                dtype=positions.dtype,
                gradient_order=0,
                features={"row_cell"},
                selection=inherited if self.backend is None else None,
            )
        if cell.dtype != positions.dtype:
            if self._selection.family != "warp":
                raise TypeError("cell must share positions dtype")
            cell = cell.to(dtype=positions.dtype)  # Retain the legacy hook cast.
        state = getattr(workflow, "_state", None)
        for name in ("cell_velocities", "cell_velocity"):
            motion = getattr(state, name, None)
            if motion is not None and bool(motion[selected_systems].ne(0).any()):
                raise ValueError("align cells before initializing cell motion")
        systems = torch.full(
            (batch.num_graphs,), -1, device=positions.device, dtype=torch.int64
        )
        systems[selected_systems] = torch.arange(
            int(selected_systems.sum()), device=positions.device
        )
        batch_idx = systems[batch.batch_idx.long()[atom_mask]].int()
        old_cell = cell.clone()
        align_cell(positions, cell, batch_idx, selection=self._selection)
        transform = torch.linalg.solve(cell, old_cell)
        with torch.no_grad():
            batch.positions[atom_mask] = positions
            batch.cell[selected_systems] = cell
            for name in ("velocities", "forces"):
                field = getattr(batch, name, None)
                if field is not None:
                    field[atom_mask] = torch.bmm(
                        transform[batch_idx.long()], field[atom_mask, :, None]
                    ).squeeze(-1)
            stress = getattr(batch, "stress", None)
            if stress is not None:
                stress[selected_systems] = (
                    transform @ stress[selected_systems] @ transform.mT
                )
