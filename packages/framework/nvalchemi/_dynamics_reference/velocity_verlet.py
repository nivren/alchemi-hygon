# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Torch reference velocity-Verlet state updates.

The functions in this module deliberately have no dependency on
``nvalchemi.dynamics`` or on Warp.  They implement the same in-place state
transition as the upstream kernels and are suitable as a correctness oracle
on CPU and on the Hygon Torch device.
"""

from __future__ import annotations

import torch

__all__ = ["vv_position_update", "vv_velocity_finalize"]


def _validate_state(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    forces: torch.Tensor,
    masses: torch.Tensor,
    dt: torch.Tensor,
    batch_idx: torch.Tensor,
) -> None:
    if positions.ndim != 2 or positions.shape[-1] != 3:
        raise ValueError(f"positions must have shape [N, 3], got {tuple(positions.shape)}")
    if velocities.shape != positions.shape or forces.shape != positions.shape:
        raise ValueError("velocities and forces must have the same shape as positions")
    if masses.ndim == 2 and masses.shape[-1] == 1:
        mass_count = masses.shape[0]
    elif masses.ndim == 1:
        mass_count = masses.shape[0]
    else:
        raise ValueError(f"masses must have shape [N] or [N, 1], got {tuple(masses.shape)}")
    if mass_count != positions.shape[0]:
        raise ValueError("masses and positions must contain the same number of atoms")
    if dt.ndim != 1:
        raise ValueError(f"dt must have shape [M], got {tuple(dt.shape)}")
    if batch_idx.ndim != 1 or batch_idx.shape[0] != positions.shape[0]:
        raise ValueError("batch_idx must have shape [N]")
    if not batch_idx.dtype in (torch.int32, torch.int64):
        raise TypeError("batch_idx must use int32 or int64")
    tensors = (velocities, forces, masses, dt, batch_idx)
    if any(t.device != positions.device for t in tensors):
        raise ValueError("all state tensors must be on the same device")
    if velocities.dtype != positions.dtype or forces.dtype != positions.dtype:
        raise TypeError("positions, velocities, and forces must have the same dtype")
    if masses.dtype != positions.dtype or dt.dtype != positions.dtype:
        raise TypeError("masses and dt must have the same dtype as positions")
    if positions.dtype not in (torch.float32, torch.float64):
        raise TypeError("velocity-Verlet reference supports float32 and float64")
    if batch_idx.numel() and dt.numel():
        if bool(torch.any(batch_idx < 0)) or bool(torch.any(batch_idx >= dt.shape[0])):
            raise ValueError("batch_idx contains a system index outside dt")


def _dt_per_atom(
    dt: torch.Tensor, batch_idx: torch.Tensor, *, dtype: torch.dtype
) -> torch.Tensor:
    return dt.index_select(0, batch_idx.to(torch.int64)).to(dtype).unsqueeze(-1)


@torch.library.custom_op(
    "nvalchemi::reference_vv_position_update",
    mutates_args={"positions", "velocities"},
)
def vv_position_update(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    forces: torch.Tensor,
    masses: torch.Tensor,
    dt: torch.Tensor,
    batch_idx: torch.Tensor,
) -> None:
    r"""Apply the position and first half velocity Verlet updates in-place."""
    _validate_state(positions, velocities, forces, masses, dt, batch_idx)
    mass = masses.reshape(-1, 1)
    dt_atom = _dt_per_atom(dt, batch_idx, dtype=positions.dtype)
    with torch.no_grad():
        acceleration = forces / mass
        positions.add_(velocities * dt_atom + 0.5 * acceleration * dt_atom.square())
        velocities.add_(0.5 * acceleration * dt_atom)


@vv_position_update.register_fake
def _vv_position_update_fake(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    forces: torch.Tensor,
    masses: torch.Tensor,
    dt: torch.Tensor,
    batch_idx: torch.Tensor,
) -> None:
    # The custom op mutates existing tensors and has no tensor return value.
    return None


@torch.library.custom_op(
    "nvalchemi::reference_vv_velocity_finalize", mutates_args={"velocities"}
)
def vv_velocity_finalize(
    velocities: torch.Tensor,
    forces_new: torch.Tensor,
    masses: torch.Tensor,
    dt: torch.Tensor,
    batch_idx: torch.Tensor,
) -> None:
    r"""Apply the final velocity Verlet half kick in-place."""
    _validate_state(velocities, velocities, forces_new, masses, dt, batch_idx)
    mass = masses.reshape(-1, 1)
    dt_atom = _dt_per_atom(dt, batch_idx, dtype=velocities.dtype)
    with torch.no_grad():
        velocities.add_(0.5 * (forces_new / mass) * dt_atom)


@vv_velocity_finalize.register_fake
def _vv_velocity_finalize_fake(
    velocities: torch.Tensor,
    forces_new: torch.Tensor,
    masses: torch.Tensor,
    dt: torch.Tensor,
    batch_idx: torch.Tensor,
) -> None:
    return None
