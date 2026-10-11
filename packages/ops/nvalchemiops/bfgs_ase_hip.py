# SPDX-FileCopyrightText: Copyright (c) 2026 alchemi-hygon contributors
# SPDX-License-Identifier: Apache-2.0
"""Explicit HIP eigensolver executor; all BFGS math is shared with reference."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from nvalchemiops._bfgs_eigh_hip import _check_runtime, _eigh, _load_extension
from nvalchemiops.bfgs_ase import ASEBFGSState, ASEBFGSStep
from nvalchemiops.bfgs_ase_reference import _step_with_eigh
from nvalchemiops.bfgs_ase_reference import bfgs_ase_prepare_state as _prepare

__all__ = ["bfgs_ase_prepare_state", "bfgs_ase_step", "bfgs_ase_batch_step"]


def bfgs_ase_batch_step(
    positions: Sequence[torch.Tensor],
    forces: Sequence[torch.Tensor],
    states: Sequence[ASEBFGSState],
) -> tuple[ASEBFGSStep, ...]:
    """Use the same grouped BFGS formula with the native batched solver.

    Parameters
    ----------
    positions, forces : Sequence[torch.Tensor]
        Independent contiguous FP64 generalized coordinates and forces on HCU.
    states : Sequence[ASEBFGSState]
        Per-system state validated by the public dispatcher.

    Returns
    -------
    tuple[ASEBFGSStep, ...]
        Per-system diagnostics after successful validation of every group.
    """
    from nvalchemiops.bfgs_ase_batch_reference import _step_batch_with_eigh

    return _step_batch_with_eigh(positions, forces, states, _eigh)


def bfgs_ase_prepare_state(
    num_scalar_dofs: int,
    *,
    alpha: float,
    maxstep: float,
    dtype: torch.dtype,
    device: torch.device,
    system_id: int | None,
    coordinate_mode: str = "cartesian",
) -> ASEBFGSState:
    """Check explicit native availability and allocate device-local state."""
    _check_runtime()
    _load_extension()
    return _prepare(
        num_scalar_dofs,
        alpha=alpha,
        maxstep=maxstep,
        dtype=dtype,
        device=device,
        system_id=system_id,
        coordinate_mode=coordinate_mode,
    )


def bfgs_ase_step(
    positions: torch.Tensor, forces: torch.Tensor, state: ASEBFGSState
) -> ASEBFGSStep:
    """Use checked native eigh with unchanged Hessian/direction/commit math."""
    return _step_with_eigh(positions, forces, state, _eigh)
