# SPDX-FileCopyrightText: Copyright (c) 2026 alchemi-hygon contributors
# SPDX-License-Identifier: Apache-2.0
"""Explicit dynamic HIP Jacobi executor; shared ASE math and bounded group workers."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from nvalchemiops._bfgs_jacobi_hip import _eigh, _grouped_eigh, _load_extension
from nvalchemiops.backend import BackendUnavailableError
from nvalchemiops.bfgs_ase import ASEBFGSState, ASEBFGSStep
from nvalchemiops.bfgs_ase_batch_reference import _step_batch_with_eigh
from nvalchemiops.bfgs_ase_reference import _step_with_eigh
from nvalchemiops.bfgs_ase_reference import bfgs_ase_prepare_state as _prepare

__all__ = ["bfgs_ase_prepare_state", "bfgs_ase_step", "bfgs_ase_batch_step"]


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
    """Validate the device and JIT runtime, then allocate unchanged state tensors."""
    if torch.device(device).type != "cuda":
        raise BackendUnavailableError("HIP BFGS Jacobi requires a HIP device")
    _load_extension(torch.device(device))
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
    """Apply shared single-system ASE math with checked device-local Jacobi."""
    return _step_with_eigh(positions, forces, state, _eigh)


def bfgs_ase_batch_step(
    positions: Sequence[torch.Tensor],
    forces: Sequence[torch.Tensor],
    states: Sequence[ASEBFGSState],
) -> tuple[ASEBFGSStep, ...]:
    """Prepare all Hessians, jointly solve width groups, then atomically commit."""
    return _step_batch_with_eigh(
        positions, forces, states, _eigh, grouped_eigh=_grouped_eigh
    )
