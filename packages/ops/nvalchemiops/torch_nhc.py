# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Device-resident Torch NHC reference, derived from locked Warp chain sweeps.

Temperatures are kT in eV, times use framework internal units. Functional
operations keep autograd; framework mutating custom ops are forward-only.
"""

from __future__ import annotations

import torch


def _floats(anchor: torch.Tensor, **tensors: torch.Tensor) -> None:
    if anchor.dtype not in (torch.float32, torch.float64):
        raise TypeError("NHC supports float32 and float64")
    for name, value in tensors.items():
        if value.device != anchor.device:
            raise ValueError(f"{name} must be on the same device")
        if value.dtype != anchor.dtype:
            raise TypeError(f"{name} must have the same dtype")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite")


def _positive(**tensors: torch.Tensor) -> None:
    for name, value in tensors.items():
        if not bool((value > 0).all()):
            raise ValueError(f"{name} must be positive")


def _index(
    velocities: torch.Tensor,
    masses: torch.Tensor,
    batch_idx: torch.Tensor,
    systems: int,
) -> torch.Tensor:
    if velocities.ndim != 2 or velocities.shape[1] != 3:
        raise ValueError("velocities must have shape [N,3]")
    if masses.shape not in ((len(velocities),), (len(velocities), 1)):
        raise ValueError("masses must have shape [N] or [N,1]")
    if batch_idx.shape != (len(velocities),):
        raise ValueError("batch_idx must have shape [N]")
    if batch_idx.dtype not in (torch.int32, torch.int64):
        raise TypeError("batch_idx must be int32 or int64")
    if batch_idx.device != velocities.device:
        raise ValueError("batch_idx must be on the same device")
    idx = batch_idx.long()
    if bool(((idx < 0) | (idx >= systems)).any()):
        raise ValueError("batch_idx contains an out-of-range system")
    _floats(velocities, velocities=velocities, masses=masses)
    _positive(masses=masses)
    return idx


def nhc_compute_masses(temperature, thermostat_time, masses, batch_idx, chain_length):
    if (
        not isinstance(chain_length, int)
        or isinstance(chain_length, bool)
        or not 1 <= chain_length <= 8
    ):
        raise ValueError("chain_length must be an integer in [1,8]")
    if temperature.ndim != 1 or thermostat_time.shape != temperature.shape:
        raise ValueError("temperature and thermostat_time must have shape [M]")
    _floats(
        temperature,
        temperature=temperature,
        thermostat_time=thermostat_time,
        masses=masses,
    )
    _positive(temperature=temperature, thermostat_time=thermostat_time)
    idx = _index(
        torch.zeros((masses.shape[0], 3), device=masses.device, dtype=masses.dtype),
        masses,
        batch_idx,
        len(temperature),
    )
    counts = torch.bincount(idx, minlength=len(temperature)).to(temperature.dtype)
    _positive(atom_counts=counts)
    base = temperature * thermostat_time.square()
    return torch.cat(
        ((3 * counts * base)[:, None], base[:, None].expand(-1, chain_length - 1)),
        dim=1,
    )


def nhc_compute_2ke(velocities, masses, batch_idx, num_systems):
    idx = _index(velocities, masses, batch_idx, num_systems)
    values = masses.reshape(-1) * velocities.square().sum(dim=1)
    return values.new_zeros(num_systems).index_add(0, idx, values)


def nhc_chain_update_functional(
    velocities,
    masses,
    eta,
    eta_dot,
    Q,
    temperature,
    dt,
    ndof,
    batch_idx,
    *,
    nloops=1,
    ke2=None,
):
    """Return updated v, eta, eta_dot, ke2, total/last scale and weighted dt.

    The bounded loops traverse chain links/substeps, never atoms or systems.
    A supplied ke2 implements upstream compute_ke=False without a collective.
    """
    if (
        isinstance(nloops, bool)
        or not isinstance(nloops, int)
        or nloops not in (1, 3, 5)
    ):
        raise ValueError("nloops must be 1, 3 or 5")
    if eta.ndim != 2 or not 1 <= eta.shape[1] <= 8:
        raise ValueError("eta must have shape [M,C], C in [1,8]")
    M, C = eta.shape
    if eta_dot.shape != eta.shape or Q.shape != eta.shape:
        raise ValueError("eta_dot and Q must have the eta shape")
    if any(x.shape != (M,) for x in (temperature, dt, ndof)):
        raise ValueError("temperature, dt and ndof must have shape [M]")
    idx = _index(velocities, masses, batch_idx, M)
    _floats(
        velocities,
        eta=eta,
        eta_dot=eta_dot,
        Q=Q,
        temperature=temperature,
        dt=dt,
        ndof=ndof,
    )
    _positive(Q=Q, temperature=temperature, ndof=ndof)
    # Negative dt is allowed in this functional primitive for reversibility.
    if bool((dt == 0).any()):
        raise ValueError("dt must be nonzero")
    if ke2 is None:
        kinetic = nhc_compute_2ke(velocities, masses, batch_idx, M)
    else:
        if ke2.shape != (M,):
            raise ValueError("ke2 must have shape [M]")
        _floats(velocities, ke2=ke2)
        if bool((ke2 < 0).any()):
            raise ValueError("ke2 must be non-negative")
        kinetic = ke2
    if nloops == 1:
        weights = (1.0,)
    elif nloops == 3:
        w = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
        weights = (w, 1.0 - 2.0 * w, w)
    else:
        w = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))
        weights = (w, w, 1.0 - 4.0 * w, w, w)
    e = list(eta.unbind(dim=1))
    u = list(eta_dot.unbind(dim=1))
    q = list(Q.unbind(dim=1))
    total = torch.ones_like(dt)
    for weight in weights:
        weighted_dt = dt * weight
        half, quarter, eighth = (
            weighted_dt * 0.5,
            weighted_dt * 0.25,
            weighted_dt * 0.125,
        )
        # Upstream fix NHC-ETA: the thermostat operator advances dt/2;
        # each symmetric eta drift must be dt/4, not dt/2. See the
        # small-step derivative and extended-energy regression tests.
        e = [a + quarter * b for a, b in zip(e, u)]
        if C > 1:
            u[-1] = u[-1] + quarter * (q[-2] * u[-2].square() - temperature) / q[-1]
        for k in range(C - 2, 0, -1):
            scale = torch.exp(-eighth * u[k + 1])
            u[k] = (
                u[k] * scale
                + quarter * (q[k - 1] * u[k - 1].square() - temperature) / q[k]
            ) * scale
        scale = torch.exp(-eighth * u[1]) if C > 1 else torch.ones_like(dt)
        u[0] = (u[0] * scale + quarter * (kinetic - ndof * temperature) / q[0]) * scale
        step = torch.exp(-half * u[0])
        kinetic = kinetic * step.square()
        u[0] = (u[0] * scale + quarter * (kinetic - ndof * temperature) / q[0]) * scale
        for k in range(1, C - 1):
            scale = torch.exp(-eighth * u[k + 1])
            u[k] = (
                u[k] * scale
                + quarter * (q[k - 1] * u[k - 1].square() - temperature) / q[k]
            ) * scale
        if C > 1:
            u[-1] = u[-1] + quarter * (q[-2] * u[-2].square() - temperature) / q[-1]
        e = [a + quarter * b for a, b in zip(e, u)]
        total = total * step
    return (
        velocities * total[idx, None],
        torch.stack(e, dim=1),
        torch.stack(u, dim=1),
        kinetic,
        total,
        step,
        weighted_dt,
    )


def nhc_velocity_half_step_functional(velocities, forces, masses, dt, batch_idx):
    idx = _index(velocities, masses, batch_idx, len(dt))
    if forces.shape != velocities.shape or dt.ndim != 1:
        raise ValueError("forces must have shape [N,3], dt [M]")
    _floats(velocities, forces=forces, dt=dt)
    return velocities + 0.5 * dt[idx, None] * forces / masses.reshape(-1, 1)


def nhc_position_update_functional(positions, velocities, dt, batch_idx):
    _index(velocities, torch.ones_like(velocities[:, 0]), batch_idx, len(dt))
    if positions.shape != velocities.shape or dt.ndim != 1:
        raise ValueError("positions must have shape [N,3], dt [M]")
    _floats(velocities, positions=positions, dt=dt)
    return positions + dt[batch_idx.long(), None] * velocities


def nhc_compute_chain_energy(eta, eta_dot, Q, temperature, ndof):
    kinetic = (0.5 * Q * eta_dot.square()).sum(dim=1)
    potential = temperature * (ndof * eta[:, 0] + eta[:, 1:].sum(dim=1))
    return kinetic, potential
