# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Forward-only mutating NHC ABI backed by functional ops Torch reference."""

from __future__ import annotations

import torch
from nvalchemiops.torch_nhc import (
    nhc_chain_update_functional,
    nhc_position_update_functional,
    nhc_velocity_half_step_functional,
)
from nvalchemiops.torch_nhc import (
    nhc_compute_masses as nhc_compute_masses,
)


def _check_mutation(outputs, inputs=()):
    for i, out in enumerate(outputs):
        for other in (*outputs[i + 1 :], *inputs):
            if out.numel() and other.numel() and torch._C._overlaps(out, other):
                raise ValueError("NHC mutated tensors must not alias other inputs")


@torch.library.custom_op(
    "nvalchemi::reference_nhc_chain_update",
    mutates_args={
        "velocities",
        "eta",
        "eta_dot",
        "ke2",
        "total_scale",
        "step_scale",
        "dt_chain",
    },
)
def nhc_chain_update(
    velocities: torch.Tensor,
    masses: torch.Tensor,
    eta: torch.Tensor,
    eta_dot: torch.Tensor,
    Q: torch.Tensor,
    temperature: torch.Tensor,
    dt: torch.Tensor,
    ndof: torch.Tensor,
    ke2: torch.Tensor,
    total_scale: torch.Tensor,
    step_scale: torch.Tensor,
    dt_chain: torch.Tensor,
    batch_idx: torch.Tensor,
    compute_ke: bool = True,
    nloops: int = 1,
) -> None:
    outputs = (velocities, eta, eta_dot, ke2, total_scale, step_scale, dt_chain)
    _check_mutation(outputs, (masses, Q, temperature, dt, ndof))
    for scratch in (ke2, total_scale, step_scale, dt_chain):
        if (
            scratch.shape != temperature.shape
            or scratch.dtype != velocities.dtype
            or scratch.device != velocities.device
        ):
            raise ValueError(
                "NHC scratch tensors must match per-system shape/dtype/device"
            )
    values = nhc_chain_update_functional(
        velocities,
        masses,
        eta,
        eta_dot,
        Q,
        temperature,
        dt,
        ndof,
        batch_idx,
        nloops=nloops,
        ke2=None if compute_ke else ke2,
    )
    for out, value in zip(outputs, values):
        out.copy_(value)


@nhc_chain_update.register_fake
def _chain_fake(
    velocities,
    masses,
    eta,
    eta_dot,
    Q,
    temperature,
    dt,
    ndof,
    ke2,
    total_scale,
    step_scale,
    dt_chain,
    batch_idx,
    compute_ke=True,
    nloops=1,
):
    pass


@torch.library.custom_op(
    "nvalchemi::reference_nhc_velocity_half_step", mutates_args={"velocities"}
)
def nhc_velocity_half_step(
    velocities: torch.Tensor,
    forces: torch.Tensor,
    masses: torch.Tensor,
    dt: torch.Tensor,
    batch_idx: torch.Tensor,
) -> None:
    _check_mutation((velocities,), (forces, masses, dt))
    velocities.copy_(
        nhc_velocity_half_step_functional(velocities, forces, masses, dt, batch_idx)
    )


@nhc_velocity_half_step.register_fake
def _kick_fake(velocities, forces, masses, dt, batch_idx):
    pass


@torch.library.custom_op(
    "nvalchemi::reference_nhc_position_update", mutates_args={"positions"}
)
def nhc_position_update(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    dt: torch.Tensor,
    batch_idx: torch.Tensor,
) -> None:
    _check_mutation((positions,), (velocities, dt))
    positions.copy_(
        nhc_position_update_functional(positions, velocities, dt, batch_idx)
    )


@nhc_position_update.register_fake
def _drift_fake(positions, velocities, dt, batch_idx):
    pass
