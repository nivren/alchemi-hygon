# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Warp-free fixed-cell FIRE and FIRE2 reference updates.

The implementation follows the locked upstream ``batch_idx`` path.  It keeps
the optimizer state per system, updates tensors in place, and uses Torch
segment reductions so heterogeneous batches remain independent.  The
variable-cell FIRE2 path is intentionally not present: it requires a verified
virial/stress contract and remains an explicit unsupported operation.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

__all__ = ["fire_step", "fire_update", "fire2_step_coord", "fire2_step_coord_cell"]


def _batch_index(
    batch_idx: torch.Tensor | None, n_atoms: int, num_systems: int, device: torch.device
) -> torch.Tensor:
    if batch_idx is None:
        result = torch.zeros(n_atoms, dtype=torch.int64, device=device)
    else:
        if batch_idx.ndim != 1 or batch_idx.shape[0] != n_atoms:
            raise ValueError("batch_idx must have shape [N]")
        if batch_idx.dtype not in (torch.int32, torch.int64):
            raise TypeError("batch_idx must use int32 or int64")
        result = batch_idx.to(torch.int64)
    if result.numel():
        if bool(torch.any(result < 0)) or bool(torch.any(result >= num_systems)):
            raise ValueError("batch_idx contains a system index outside parameters")
        if result.numel() > 1 and bool(torch.any(result[1:] < result[:-1])):
            raise ValueError("batch_idx must be sorted in non-decreasing order")
    return result


def _check_vectors(
    tensors: Iterable[torch.Tensor],
    size: int,
    dtype: torch.dtype,
    device: torch.device,
    names: Iterable[str],
) -> None:
    for name, tensor in zip(names, tensors, strict=True):
        if tensor.ndim != 1 or tensor.shape[0] != size:
            raise ValueError(f"{name} must have shape [{size}]")
        if tensor.dtype != dtype:
            raise TypeError(f"{name} must have dtype {dtype}")
        if tensor.device != device:
            raise ValueError(f"{name} must share the state device")


def _check_atom_state(
    velocities: torch.Tensor,
    forces: torch.Tensor,
    masses: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if velocities.ndim != 2 or velocities.shape[-1] != 3:
        raise ValueError("velocities must have shape [N, 3]")
    if forces.shape != velocities.shape:
        raise ValueError("forces must have the same shape as velocities")
    if not velocities.is_floating_point() or forces.dtype != velocities.dtype:
        raise TypeError("velocities and forces must share a floating-point dtype")
    if forces.device != velocities.device:
        raise ValueError("velocities and forces must share a device")
    if positions is not None and (
        positions.shape != velocities.shape
        or positions.dtype != velocities.dtype
        or positions.device != velocities.device
    ):
        raise ValueError("positions must match velocities in shape, dtype, and device")
    if masses is None:
        return None
    if masses.ndim == 2 and masses.shape[-1] == 1:
        masses = masses.squeeze(-1)
    elif masses.ndim != 1:
        raise ValueError("masses must have shape [N] or [N, 1]")
    if masses.shape[0] != velocities.shape[0]:
        raise ValueError("masses and velocities must contain the same number of atoms")
    if masses.dtype != velocities.dtype or masses.device != velocities.device:
        raise TypeError("masses must share the velocity dtype and device")
    return masses


def _scratch(
    value: torch.Tensor | None,
    size: int,
    dtype: torch.dtype,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    if value is None:
        return torch.zeros(size, dtype=dtype, device=device)
    if value.ndim != 1 or value.shape[0] != size:
        raise ValueError(f"{name} must have shape [{size}]")
    if value.dtype != dtype or value.device != device:
        raise TypeError(f"{name} must share the state dtype and device")
    return value


def _reduce_fire(
    velocities: torch.Tensor,
    forces: torch.Tensor,
    batch_idx: torch.Tensor,
    vf: torch.Tensor,
    vv: torch.Tensor,
    ff: torch.Tensor,
    compute_reductions: bool,
) -> None:
    if not compute_reductions:
        return
    vf.zero_()
    vv.zero_()
    ff.zero_()
    vf.index_add_(0, batch_idx, (velocities * forces).sum(dim=-1))
    vv.index_add_(0, batch_idx, (velocities * velocities).sum(dim=-1))
    ff.index_add_(0, batch_idx, (forces * forces).sum(dim=-1))


def _fire_mix_and_update(
    velocities: torch.Tensor,
    forces: torch.Tensor,
    batch_idx: torch.Tensor,
    alpha: torch.Tensor,
    dt: torch.Tensor,
    n_steps_positive: torch.Tensor,
    alpha_start: torch.Tensor,
    f_alpha: torch.Tensor,
    dt_min: torch.Tensor,
    dt_max: torch.Tensor,
    n_min: torch.Tensor,
    f_dec: torch.Tensor,
    f_inc: torch.Tensor,
    vf: torch.Tensor,
    vv: torch.Tensor,
    ff: torch.Tensor,
    compute_reductions: bool,
) -> torch.Tensor:
    """Apply the shared FIRE reduction, parameter update, and velocity mix."""
    _reduce_fire(velocities, forces, batch_idx, vf, vv, ff, compute_reductions)
    local_dt = dt.clone()
    positive = vf > 0
    nsi = torch.where(positive, n_steps_positive + 1, torch.zeros_like(n_steps_positive))
    reached = nsi >= n_min
    new_dt = torch.where(
        positive,
        torch.where(reached, torch.minimum(local_dt * f_inc, dt_max), local_dt),
        torch.maximum(local_dt * f_dec, dt_min),
    )
    new_alpha = torch.where(
        positive,
        torch.where(reached, alpha * f_alpha, alpha),
        alpha_start,
    )
    dt.copy_(new_dt)
    alpha.copy_(new_alpha)
    n_steps_positive.copy_(nsi)

    ratio = torch.where(
        ff > 0,
        torch.sqrt(torch.where(ff > 0, vv / ff, vv * 0)),
        vv * 0,
    )
    a = new_alpha.index_select(0, batch_idx)
    mixed = (1.0 - a).unsqueeze(-1) * velocities + (
        a * ratio.index_select(0, batch_idx)
    ).unsqueeze(-1) * forces
    velocities.copy_(
        torch.where(
            positive.index_select(0, batch_idx).unsqueeze(-1), mixed, velocities * 0
        )
    )
    return local_dt


def fire_update(
    velocities: torch.Tensor,
    forces: torch.Tensor,
    alpha: torch.Tensor,
    dt: torch.Tensor,
    n_steps_positive: torch.Tensor,
    alpha_start: torch.Tensor,
    f_alpha: torch.Tensor,
    dt_min: torch.Tensor,
    dt_max: torch.Tensor,
    n_min: torch.Tensor,
    f_dec: torch.Tensor,
    f_inc: torch.Tensor,
    *,
    vf: torch.Tensor | None = None,
    vv: torch.Tensor | None = None,
    ff: torch.Tensor | None = None,
    batch_idx: torch.Tensor | None = None,
    compute_reductions: bool = True,
) -> None:
    """FIRE velocity mixing and adaptive state update without a position step."""
    masses = _check_atom_state(velocities, forces)
    del masses
    num_systems = alpha.shape[0]
    if alpha.ndim != 1:
        raise ValueError("alpha must have shape [M]")
    _check_vectors(
        (dt, alpha_start, f_alpha, dt_min, dt_max, f_dec, f_inc),
        num_systems,
        velocities.dtype,
        velocities.device,
        ("dt", "alpha_start", "f_alpha", "dt_min", "dt_max", "f_dec", "f_inc"),
    )
    _check_vectors(
        (n_steps_positive, n_min),
        num_systems,
        n_steps_positive.dtype,
        velocities.device,
        ("n_steps_positive", "n_min"),
    )
    if n_steps_positive.dtype not in (torch.int32, torch.int64):
        raise TypeError("n_steps_positive and n_min must be integer tensors")
    if alpha.dtype != velocities.dtype or alpha.device != velocities.device:
        raise TypeError("alpha must share the state dtype and device")
    indices = _batch_index(batch_idx, velocities.shape[0], num_systems, velocities.device)
    if not compute_reductions and (vf is None or vv is None or ff is None):
        raise ValueError("vf, vv, and ff are required when compute_reductions=False")
    vf = _scratch(vf, num_systems, velocities.dtype, velocities.device, "vf")
    vv = _scratch(vv, num_systems, velocities.dtype, velocities.device, "vv")
    ff = _scratch(ff, num_systems, velocities.dtype, velocities.device, "ff")
    with torch.no_grad():
        _fire_mix_and_update(
            velocities,
            forces,
            indices,
            alpha,
            dt,
            n_steps_positive,
            alpha_start,
            f_alpha,
            dt_min,
            dt_max,
            n_min,
            f_dec,
            f_inc,
            vf,
            vv,
            ff,
            compute_reductions,
        )


def fire_step(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    forces: torch.Tensor,
    masses: torch.Tensor,
    alpha: torch.Tensor,
    dt: torch.Tensor,
    n_steps_positive: torch.Tensor,
    alpha_start: torch.Tensor,
    f_alpha: torch.Tensor,
    dt_min: torch.Tensor,
    dt_max: torch.Tensor,
    maxstep: torch.Tensor,
    n_min: torch.Tensor,
    f_dec: torch.Tensor,
    f_inc: torch.Tensor,
    uphill_flag: torch.Tensor,
    *,
    vf: torch.Tensor | None = None,
    vv: torch.Tensor | None = None,
    ff: torch.Tensor | None = None,
    batch_idx: torch.Tensor | None = None,
    compute_reductions: bool = True,
) -> None:
    """FIRE integration, mixing, and adaptive update for fixed-cell batches."""
    masses = _check_atom_state(velocities, forces, masses, positions)
    assert masses is not None  # noqa: S101
    if uphill_flag.ndim != 1 or uphill_flag.shape[0] != alpha.shape[0]:
        raise ValueError("uphill_flag must have shape [M]")
    if bool(torch.any(uphill_flag != 0)):
        raise NotImplementedError("FIRE uphill energy-check branch is not in reference")
    num_systems = alpha.shape[0]
    if maxstep.ndim != 1 or maxstep.shape[0] != num_systems:
        raise ValueError("maxstep must have shape [M]")
    if maxstep.dtype != positions.dtype or maxstep.device != positions.device:
        raise TypeError("maxstep must share the state dtype and device")
    # Keep parameter validation and update semantics identical to fire_update.
    indices = _batch_index(batch_idx, positions.shape[0], num_systems, positions.device)
    if not compute_reductions and (vf is None or vv is None or ff is None):
        raise ValueError("vf, vv, and ff are required when compute_reductions=False")
    vf = _scratch(vf, num_systems, positions.dtype, positions.device, "vf")
    vv = _scratch(vv, num_systems, positions.dtype, positions.device, "vv")
    ff = _scratch(ff, num_systems, positions.dtype, positions.device, "ff")
    if alpha.ndim != 1 or alpha.dtype != positions.dtype or alpha.device != positions.device:
        raise TypeError("alpha must be a state-dtype vector on the state device")
    _check_vectors(
        (dt, alpha_start, f_alpha, dt_min, dt_max, f_dec, f_inc),
        num_systems,
        positions.dtype,
        positions.device,
        ("dt", "alpha_start", "f_alpha", "dt_min", "dt_max", "f_dec", "f_inc"),
    )
    if n_steps_positive.dtype not in (torch.int32, torch.int64):
        raise TypeError("n_steps_positive and n_min must be integer tensors")
    _check_vectors(
        (n_steps_positive, n_min),
        num_systems,
        n_steps_positive.dtype,
        positions.device,
        ("n_steps_positive", "n_min"),
    )
    with torch.no_grad():
        local_dt = _fire_mix_and_update(
            velocities,
            forces,
            indices,
            alpha,
            dt,
            n_steps_positive,
            alpha_start,
            f_alpha,
            dt_min,
            dt_max,
            n_min,
            f_dec,
            f_inc,
            vf,
            vv,
            ff,
            compute_reductions,
        )
        dt_atom = local_dt.index_select(0, indices).unsqueeze(-1)
        inv_mass = torch.where(masses > 0, masses.reciprocal(), masses * 0).unsqueeze(-1)
        velocities.add_(dt_atom * forces * inv_mass)
        displacement = dt_atom * velocities
        norm = displacement.norm(dim=-1)
        scale = torch.where(
            norm > 0,
            torch.minimum(torch.ones_like(norm), maxstep.index_select(0, indices) / norm),
            torch.ones_like(norm),
        )
        positions.add_(displacement * scale.unsqueeze(-1))


def _reduce_fire2(
    velocities: torch.Tensor,
    forces: torch.Tensor,
    batch_idx: torch.Tensor,
    dt: torch.Tensor,
    vf: torch.Tensor,
    v_sumsq: torch.Tensor,
    f_sumsq: torch.Tensor,
    compute_reductions: bool,
) -> None:
    if not compute_reductions:
        return
    dt_atom = dt.index_select(0, batch_idx).unsqueeze(-1)
    v_upd = velocities + forces * dt_atom
    vf.zero_()
    v_sumsq.zero_()
    f_sumsq.zero_()
    vf.index_add_(0, batch_idx, (v_upd * forces).sum(dim=-1))
    v_sumsq.index_add_(0, batch_idx, (v_upd * v_upd).sum(dim=-1))
    f_sumsq.index_add_(0, batch_idx, (forces * forces).sum(dim=-1))


def fire2_step_coord(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    forces: torch.Tensor,
    batch_idx: torch.Tensor,
    alpha: torch.Tensor,
    dt: torch.Tensor,
    nsteps_inc: torch.Tensor,
    *,
    vf: torch.Tensor | None = None,
    v_sumsq: torch.Tensor | None = None,
    f_sumsq: torch.Tensor | None = None,
    max_norm: torch.Tensor | None = None,
    delaystep: int = 60,
    dtgrow: float = 1.05,
    dtshrink: float = 0.75,
    alphashrink: float = 0.985,
    alpha0: float = 0.09,
    tmax: float = 0.08,
    tmin: float = 0.005,
    maxstep: float = 0.1,
    compute_reductions: bool = True,
) -> None:
    """FIRE2 coordinate-only step, matching the locked upstream phases."""
    masses = _check_atom_state(velocities, forces, positions=positions)
    del masses
    if alpha.ndim != 1 or alpha.dtype != positions.dtype or alpha.device != positions.device:
        raise TypeError("alpha must share the state dtype and device")
    systems = alpha.shape[0]
    if dt.ndim != 1 or dt.shape[0] != systems or dt.dtype != positions.dtype:
        raise TypeError("dt must be a state-dtype vector with shape [M]")
    if nsteps_inc.ndim != 1 or nsteps_inc.shape[0] != systems:
        raise ValueError("nsteps_inc must have shape [M]")
    if nsteps_inc.dtype not in (torch.int32, torch.int64):
        raise TypeError("nsteps_inc must be an integer tensor")
    indices = _batch_index(batch_idx, positions.shape[0], systems, positions.device)
    if not compute_reductions and (vf is None or v_sumsq is None or f_sumsq is None):
        raise ValueError(
            "vf, v_sumsq, and f_sumsq are required when compute_reductions=False"
        )
    vf = _scratch(vf, systems, positions.dtype, positions.device, "vf")
    v_sumsq = _scratch(v_sumsq, systems, positions.dtype, positions.device, "v_sumsq")
    f_sumsq = _scratch(f_sumsq, systems, positions.dtype, positions.device, "f_sumsq")
    max_norm = _scratch(max_norm, systems, positions.dtype, positions.device, "max_norm")
    maxstep_t = torch.as_tensor(maxstep, dtype=positions.dtype, device=positions.device)
    if maxstep_t.ndim == 0:
        maxstep_t = maxstep_t.expand(systems)
    if maxstep_t.shape != (systems,):
        raise ValueError("maxstep must be a scalar or shape [M]")
    if torch.any(maxstep_t <= 0):
        raise ValueError("maxstep must be positive")
    if positions.shape[0] == 0:
        vf.zero_()
        v_sumsq.zero_()
        f_sumsq.zero_()
        max_norm.zero_()
        return
    with torch.no_grad():
        _reduce_fire2(
            velocities, forces, indices, dt, vf, v_sumsq, f_sumsq, compute_reductions
        )
        zero = torch.zeros((), dtype=positions.dtype, device=positions.device)
        one = torch.ones((), dtype=positions.dtype, device=positions.device)
        downhill = vf > zero
        nsi_new = torch.where(downhill, nsteps_inc + 1, torch.zeros_like(nsteps_inc))
        grow = downhill & (nsi_new > delaystep)
        dtgrow_t = torch.as_tensor(dtgrow, dtype=dt.dtype, device=dt.device)
        dtshrink_t = torch.as_tensor(dtshrink, dtype=dt.dtype, device=dt.device)
        tmax_t = torch.as_tensor(tmax, dtype=dt.dtype, device=dt.device)
        tmin_t = torch.as_tensor(tmin, dtype=dt.dtype, device=dt.device)
        dt_new = torch.where(
            grow,
            torch.minimum(dtgrow_t * dt, tmax_t),
            torch.where(~downhill, torch.maximum(dtshrink_t * dt, tmin_t), dt),
        )
        alphashrink_t = torch.as_tensor(
            alphashrink, dtype=alpha.dtype, device=alpha.device
        )
        alpha0_t = torch.as_tensor(alpha0, dtype=alpha.dtype, device=alpha.device)
        alpha_new = torch.where(
            grow, alphashrink_t * alpha, torch.where(~downhill, alpha0_t, alpha)
        )
        ratio = torch.where(
            f_sumsq > zero,
            torch.sqrt(torch.where(f_sumsq > zero, v_sumsq / f_sumsq, zero)),
            zero,
        )
        a = alpha_new.index_select(0, indices)
        dt_old = dt.index_select(0, indices)
        ratio_atom = ratio.index_select(0, indices)
        coeff = (1.0 - a) * dt_old + a * ratio_atom
        velocities.copy_((1.0 - a).unsqueeze(-1) * velocities + coeff.unsqueeze(-1) * forces)
        dt_atom_new = dt_new.index_select(0, indices).unsqueeze(-1)
        uphill = (~downhill).index_select(0, indices).unsqueeze(-1)
        step = torch.where(
            uphill,
            -0.5 * dt_atom_new * velocities,
            dt_atom_new * velocities,
        )
        max_norm.zero_()
        max_norm.scatter_reduce_(
            0, indices, step.norm(dim=-1), reduce="amax", include_self=True
        )
        inv = torch.clamp(maxstep_t / torch.where(max_norm > zero, max_norm, one), max=one)
        positions.add_(step * inv.index_select(0, indices).unsqueeze(-1))
        velocities.masked_fill_(uphill, 0.0)
        dt.copy_(dt_new * inv)
        nsteps_inc.copy_(nsi_new)
        alpha.copy_(alpha_new)


def fire2_step_coord_cell(*args: object, **kwargs: object) -> None:
    """Variable-cell FIRE2 remains blocked on the virial/stress contract."""
    del args, kwargs
    raise NotImplementedError(
        "reference FIRE2 variable-cell path requires a verified virial/stress contract"
    )
