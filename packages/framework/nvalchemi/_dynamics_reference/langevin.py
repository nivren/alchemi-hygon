# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Warp-free Torch reference for fixed-cell BAOAB Langevin dynamics."""

from __future__ import annotations

import torch

__all__ = ["langevin_half_step", "langevin_finalize"]


def _validate_common(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    forces: torch.Tensor,
    masses: torch.Tensor,
    dt: torch.Tensor,
    temperature: torch.Tensor,
    friction: torch.Tensor,
    batch_idx: torch.Tensor,
) -> None:
    if positions.ndim != 2 or positions.shape[-1] != 3:
        raise ValueError(
            f"positions must have shape [N, 3], got {tuple(positions.shape)}"
        )
    if velocities.shape != positions.shape or forces.shape != positions.shape:
        raise ValueError("velocities and forces must have the same shape as positions")
    if masses.ndim == 2 and masses.shape[-1] == 1:
        mass_count = masses.shape[0]
    elif masses.ndim == 1:
        mass_count = masses.shape[0]
    else:
        raise ValueError(
            f"masses must have shape [N] or [N, 1], got {tuple(masses.shape)}"
        )
    if mass_count != positions.shape[0]:
        raise ValueError("masses and positions must contain the same number of atoms")
    if batch_idx.ndim != 1 or batch_idx.shape[0] != positions.shape[0]:
        raise ValueError("batch_idx must have shape [N]")
    if batch_idx.dtype not in (torch.int32, torch.int64):
        raise TypeError("batch_idx must use int32 or int64")

    parameters = (dt, temperature, friction)
    if any(parameter.ndim != 1 for parameter in parameters):
        raise ValueError("dt, temperature, and friction must have shape [M]")
    if not (dt.shape == temperature.shape == friction.shape):
        raise ValueError("dt, temperature, and friction must have the same shape")

    tensors = (velocities, forces, masses, dt, temperature, friction, batch_idx)
    if any(tensor.device != positions.device for tensor in tensors):
        raise ValueError("all Langevin state tensors must be on the same device")
    if any(
        tensor.dtype != positions.dtype
        for tensor in (velocities, forces, masses, dt, temperature, friction)
    ):
        raise TypeError("floating-point Langevin tensors must have the positions dtype")
    if positions.dtype not in (torch.float32, torch.float64):
        raise TypeError("Langevin reference supports float32 and float64")

    if batch_idx.numel() and not dt.numel():
        raise ValueError("dt, temperature, and friction cannot be empty for non-empty positions")
    if batch_idx.numel() and (
        bool(torch.any(batch_idx < 0))
        or bool(torch.any(batch_idx >= dt.shape[0]))
    ):
        raise ValueError("batch_idx contains a system index outside parameter arrays")
    for name, tensor, predicate in (
        ("masses", masses, lambda value: value > 0),
        ("dt", dt, lambda value: value > 0),
        ("temperature", temperature, lambda value: value >= 0),
        ("friction", friction, lambda value: value >= 0),
    ):
        if not bool(torch.all(torch.isfinite(tensor))):
            raise ValueError(f"{name} must contain only finite values")
        if not bool(torch.all(predicate(tensor))):
            raise ValueError(f"{name} contains an invalid value")


def _per_atom(
    values: torch.Tensor, batch_idx: torch.Tensor, *, dtype: torch.dtype
) -> torch.Tensor:
    return values.index_select(0, batch_idx.to(torch.int64)).to(dtype).unsqueeze(-1)


def _noise_like(values: torch.Tensor, random_seed: int) -> torch.Tensor:
    generator = torch.Generator(device=values.device)
    generator.manual_seed(int(random_seed))
    return torch.randn(
        values.shape,
        dtype=values.dtype,
        device=values.device,
        generator=generator,
    )


@torch.library.custom_op(
    "nvalchemi::reference_langevin_half_step",
    mutates_args={"positions", "velocities"},
)
def langevin_half_step(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    forces: torch.Tensor,
    masses: torch.Tensor,
    dt: torch.Tensor,
    temperature: torch.Tensor,
    friction: torch.Tensor,
    random_seed: int,
    batch_idx: torch.Tensor,
) -> None:
    """Apply the BAOAB ``B-A-O-A`` half-step in-place.

    ``temperature`` is ``kT`` in the framework's internal energy unit, not
    Kelvin. The operation is a forward-only state transition; it does not
    provide a differentiable path through the stochastic update.
    """
    _validate_common(
        positions,
        velocities,
        forces,
        masses,
        dt,
        temperature,
        friction,
        batch_idx,
    )
    mass = masses.reshape(-1, 1)
    dt_atom = _per_atom(dt, batch_idx, dtype=positions.dtype)
    kT_atom = _per_atom(temperature, batch_idx, dtype=positions.dtype)
    friction_atom = _per_atom(friction, batch_idx, dtype=positions.dtype)

    with torch.no_grad():
        half_dt = 0.5 * dt_atom
        velocity = velocities + half_dt * forces / mass
        position = positions + half_dt * velocity
        c1 = torch.exp(-friction_atom * dt_atom)
        c2 = torch.sqrt(kT_atom * (1.0 - c1.square()) / mass)
        velocity = c1 * velocity + c2 * _noise_like(velocity, random_seed)
        position = position + half_dt * velocity
        positions.copy_(position)
        velocities.copy_(velocity)


@langevin_half_step.register_fake
def _langevin_half_step_fake(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    forces: torch.Tensor,
    masses: torch.Tensor,
    dt: torch.Tensor,
    temperature: torch.Tensor,
    friction: torch.Tensor,
    random_seed: int,
    batch_idx: torch.Tensor,
) -> None:
    del (
        positions,
        velocities,
        forces,
        masses,
        dt,
        temperature,
        friction,
        random_seed,
        batch_idx,
    )


@torch.library.custom_op(
    "nvalchemi::reference_langevin_finalize", mutates_args={"velocities"}
)
def langevin_finalize(
    velocities: torch.Tensor,
    forces_new: torch.Tensor,
    masses: torch.Tensor,
    dt: torch.Tensor,
    batch_idx: torch.Tensor,
) -> None:
    """Apply the final BAOAB ``B`` half-kick in-place."""
    _validate_common(
        velocities,
        velocities,
        forces_new,
        masses,
        dt,
        torch.zeros_like(dt),
        torch.zeros_like(dt),
        batch_idx,
    )
    mass = masses.reshape(-1, 1)
    dt_atom = _per_atom(dt, batch_idx, dtype=velocities.dtype)
    with torch.no_grad():
        velocities.add_(0.5 * forces_new / mass * dt_atom)


@langevin_finalize.register_fake
def _langevin_finalize_fake(
    velocities: torch.Tensor,
    forces_new: torch.Tensor,
    masses: torch.Tensor,
    dt: torch.Tensor,
    batch_idx: torch.Tensor,
) -> None:
    del velocities, forces_new, masses, dt, batch_idx
