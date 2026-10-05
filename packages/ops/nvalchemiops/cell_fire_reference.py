# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native variable-cell FIRE2 and fork-repaired full-cell FIRE, eager/forward-only.

Sources: locked cell_filter/torch.fire2 and fork framework b12b54c, ops eadc3ee.
No ASE filter parameterization. See TORCH_VARIABLE_CELL_FIRE_CONTRACT.md.
"""

from __future__ import annotations

import math
import warnings

import torch

from nvalchemiops import cell_reference as cell_tools


def _check(atoms, cell_state, idx, dt, *, writes=()):
    cell_tools._floats(*atoms, *cell_state, dt)
    n, b = atoms[0].shape[0], dt.numel()
    for value in atoms:
        cell_tools._shape(value, (n, 3))
    for value in cell_state:
        cell_tools._shape(value, (b, 3, 3))
    cell_tools._shape(dt, (b,))
    cell_tools._finite(*atoms, *cell_state, dt)
    cell_tools._idx(idx, n, b, atoms[0])
    if bool((idx[1:] < idx[:-1]).any()):
        raise ValueError("batch_idx must be nondecreasing")
    if bool((dt <= 0).any()):
        raise ValueError("dt must be positive")
    all_inputs = (*atoms, *cell_state, dt)
    for out in writes:
        if sum(t is out for t in all_inputs) != 1:
            raise ValueError("state outputs must be independent")
        cell_tools._independent(out, *(t for t in all_inputs if t is not out))
    return n, b, idx.long()


def _cell(h):
    if bool((torch.linalg.det(h) == 0).any()):
        raise ValueError("cell must be invertible")


def _scratch(value, shape, anchor):
    if value is None:
        return torch.empty(shape, dtype=anchor.dtype, device=anchor.device)
    cell_tools._floats(anchor, value)
    cell_tools._shape(value, shape)
    return value


def _controls(alpha, dt, count):
    cell_tools._floats(dt, alpha)
    cell_tools._shape(alpha, dt.shape)
    cell_tools._finite(alpha)
    cell_tools._integer(count, dt)
    cell_tools._shape(count, dt.shape)
    if bool(((alpha < 0) | (alpha > 1)).any()) or bool((count < 0).any()):
        raise ValueError("alpha must be in [0,1] and counters nonnegative")
    cell_tools._independent(alpha, dt)


def _control_storage(alpha, count, *buffers):
    """Controls are mutable state too; reject views into atom/cell buffers."""
    cell_tools._independent(alpha, count, *buffers)
    cell_tools._independent(count, *buffers)


def _positive_scalar(name, value):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _reduce(v, f, idx, b):
    result = v.new_zeros(3, b)
    result[0].index_add_(0, idx, (v * f).sum(-1))
    result[1].index_add_(0, idx, v.square().sum(-1))
    result[2].index_add_(0, idx, f.square().sum(-1))
    return result.unbind(0)


def _ratio(vv, ff):
    # Preserve source zero-safety without clipping positive subnormal norms.
    denominator = torch.where(ff > 0, ff, 1)
    return torch.sqrt(torch.where(ff > 0, vv / denominator, 0))


def _affine_step(r, h, dh, translation, idx):
    # UBF-011 / fork c4c3bb5: row cells require r @ H^-1 @ dH;
    # the locked Warp code used (H+dH) @ H^-1 on column positions.
    transform = torch.linalg.solve(h, dh)
    return torch.bmm(r.unsqueeze(1), transform[idx]).squeeze(1) + translation


def fire2_step_coord_cell(
    positions,
    velocities,
    forces,
    cell,
    cell_velocities,
    cell_force,
    batch_idx,
    alpha,
    dt,
    nsteps_inc,
    *,
    atom_ptr=None,
    ext_atom_ptr=None,
    ext_positions=None,
    ext_velocities=None,
    ext_forces=None,
    ext_batch_idx=None,
    vf=None,
    v_sumsq=None,
    f_sumsq=None,
    max_norm=None,
    delaystep=60,
    dtgrow=1.05,
    dtshrink=0.75,
    alphashrink=0.985,
    alpha0=0.09,
    tmax=0.08,
    tmin=0.005,
    maxstep=0.1,
    cell_force_scale=1.0,
    compute_reductions=True,
):
    """Original six-DOF FIRE2 ABI, with the fork row-cell motion correction."""
    n, b, idx = _check(
        (positions, velocities, forces),
        (cell, cell_velocities, cell_force),
        batch_idx,
        dt,
        writes=(positions, velocities, cell, cell_velocities, dt),
    )
    _cell(cell)
    _controls(alpha, dt, nsteps_inc)
    _control_storage(
        alpha,
        nsteps_inc,
        positions,
        velocities,
        forces,
        cell,
        cell_velocities,
        cell_force,
        dt,
    )
    if bool((torch.triu(cell, 1) != 0).any()):
        raise ValueError(
            "FIRE2 cell must be lower triangular; align it before optimization"
        )
    for name, value in (
        ("maxstep", maxstep),
        ("cell_force_scale", cell_force_scale),
        ("dtgrow", dtgrow),
        ("dtshrink", dtshrink),
        ("tmin", tmin),
        ("tmax", tmax),
    ):
        _positive_scalar(name, value)
    if (
        delaystep < 0
        or not isinstance(delaystep, int)
        or tmax < tmin
        or not 0 <= alpha0 <= 1
        or not 0 <= alphashrink <= 1
    ):
        raise ValueError("invalid FIRE2 controls")
    if ext_positions is not None:
        warnings.warn(
            "fire2_step_coord_cell(..., ext_positions=...) is deprecated and ignored.",
            DeprecationWarning,
            stacklevel=2,
        )
    scratches = [_scratch(x, (b,), positions) for x in (vf, v_sumsq, f_sumsq, max_norm)]
    for out in scratches:
        cell_tools._independent(
            out,
            positions,
            velocities,
            forces,
            cell,
            cell_velocities,
            cell_force,
            alpha,
            dt,
            nsteps_inc,
        )
    for i, out in enumerate(scratches):
        cell_tools._independent(out, *scratches[:i])
    if not compute_reductions and any(x is None for x in (vf, v_sumsq, f_sumsq)):
        raise ValueError("vf/v_sumsq/f_sumsq required when compute_reductions=False")
    if n == 0:
        for out in scratches:
            out.zero_()
        return
    counts = torch.bincount(idx, minlength=b)
    if bool((counts <= 0).any()):
        raise ValueError("fire2_step_coord_cell requires at least one atom per system")
    expected_ptr = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    if atom_ptr is None:
        atom_ptr = expected_ptr
    else:
        cell_tools._integer(atom_ptr, positions)
        if not torch.equal(atom_ptr.long(), expected_ptr):
            raise ValueError("invalid atom_ptr")
    if ext_atom_ptr is None:
        ext_atom_ptr = torch.empty_like(atom_ptr)
        cell_tools.extend_atom_ptr(atom_ptr, ext_atom_ptr)
    atom_rows, cell_rows = cell_tools._layout(
        n, b, positions, atom_ptr, ext_atom_ptr, idx
    )
    ext_idx = torch.repeat_interleave(
        torch.arange(b, device=positions.device), counts + 2
    )
    if ext_batch_idx is not None:
        cell_tools._idx(ext_batch_idx, n + 2 * b, b, positions)
        if not torch.equal(ext_idx, ext_batch_idx.long()):
            raise ValueError("invalid interleaved ext_batch_idx")
    ev = _scratch(ext_velocities, (n + 2 * b, 3), positions)
    ef = _scratch(ext_forces, (n + 2 * b, 3), positions)
    for out in (ev, ef):
        cell_tools._independent(
            out,
            positions,
            velocities,
            forces,
            cell,
            cell_velocities,
            cell_force,
            alpha,
            dt,
            nsteps_inc,
            *scratches,
        )
    cell_tools._independent(ev, ef)
    work_force = cell_force / (
        counts.to(positions.dtype)[:, None, None] * cell_force_scale
    )
    cell_tools.pack_velocities_with_cell(
        velocities, cell_velocities, ev, atom_ptr, ext_atom_ptr, batch_idx=idx
    )
    cell_tools.pack_forces_with_cell(
        forces, work_force, ef, atom_ptr, ext_atom_ptr, batch_idx=idx
    )
    dt_old = dt.clone()
    if compute_reductions:
        reductions = _reduce(ev + dt_old[ext_idx, None] * ef, ef, ext_idx, b)
    else:
        cell_tools._finite(*scratches[:3])
        if bool((scratches[1] < 0).any()) or bool((scratches[2] < 0).any()):
            raise ValueError("squared norms must be nonnegative")
        reductions = tuple(x.clone() for x in scratches[:3])
    power, vv, ff = reductions
    down = power > 0
    next_n = torch.where(down, nsteps_inc + 1, 0)
    grow = down & (next_n > delaystep)
    next_dt = torch.where(
        grow,
        (dt_old * dtgrow).clamp_max(tmax),
        torch.where(down, dt_old, (dt_old * dtshrink).clamp_min(tmin)),
    )
    next_a = torch.where(grow, alpha * alphashrink, torch.where(down, alpha, alpha0))
    a = next_a[ext_idx, None]
    mixed = (1 - a) * ev + (
        (1 - a) * dt_old[ext_idx, None] + a * _ratio(vv, ff)[ext_idx, None]
    ) * ef
    new_v = mixed[atom_rows]
    new_hv = torch.zeros_like(cell)
    new_hv[:, :, 0] = mixed[cell_rows]
    new_hv[:, [1, 2, 2], [1, 1, 2]] = mixed[cell_rows + 1]
    factor = torch.where(down, 1.0, -0.5).to(dt.dtype) * next_dt
    dh = factor[:, None, None] * new_hv
    dr = _affine_step(positions, cell, dh, factor[idx, None] * new_v, idx)
    norm = positions.new_zeros(b).scatter_reduce_(
        0, idx, dr.norm(dim=-1), reduce="amax", include_self=True
    )
    scale = maxstep / norm.clamp_min(maxstep)
    proposed_h = cell + scale[:, None, None] * dh
    proposed_r = positions + scale[idx, None] * dr
    cell_tools._finite(proposed_h, proposed_r, mixed, next_dt, next_a)
    _cell(proposed_h)
    positions.copy_(proposed_r)
    cell.copy_(proposed_h)
    velocities.copy_(torch.where(down[idx, None], new_v, 0))
    cell_velocities.copy_(torch.where(down[:, None, None], new_hv, 0))
    ev.copy_(mixed)  # Upstream scratch contains mixed, before uphill reset.
    dt.copy_(next_dt * scale)
    alpha.copy_(next_a)
    nsteps_inc.copy_(next_n)
    for out, value in zip(scratches, (*reductions, norm), strict=True):
        out.copy_(value)


def _mass(mass, v):
    cell_tools._floats(v, mass)
    if mass.shape not in ((v.shape[0],), (v.shape[0], 1)):
        raise ValueError("mass requires [N] or [N,1]")
    cell_tools._finite(mass)
    if bool((mass <= 0).any()):
        raise ValueError("mass must be positive")
    return mass.reshape(-1, 1)


def _full_drift(positions, velocities, cell, cell_velocities, dt, idx, maxstep):
    """Fork b12b54c drift: include Cartesian atoms and cell rows in the cap."""
    dh = dt[:, None, None] * cell_velocities
    dr = _affine_step(positions, cell, dh, dt[idx, None] * velocities, idx)
    norm = cell.new_zeros(dt.numel()).scatter_reduce_(
        0, idx, dr.norm(dim=-1), reduce="amax", include_self=True
    )
    norm = torch.maximum(norm, dh.norm(dim=-1).amax(-1))
    scale = maxstep / norm.clamp_min(maxstep)
    new_h = cell + scale[:, None, None] * dh
    new_r = positions + scale[idx, None] * dr
    cell_tools._finite(new_h, new_r)
    _cell(new_h)
    positions.copy_(new_r)
    cell.copy_(new_h)


def fire_cell_pre(
    positions,
    velocities,
    forces,
    masses,
    cell,
    cell_velocity,
    cell_force,
    dt,
    batch_idx,
    *,
    maxstep=0.2,
):
    """UBF-012 / fork b12b54c: kick both atom and full-cell state, capped drift."""
    _, _, idx = _check(
        (positions, velocities, forces),
        (cell, cell_velocity, cell_force),
        batch_idx,
        dt,
        writes=(positions, velocities, cell, cell_velocity),
    )
    _cell(cell)
    mass = _mass(masses, velocities)
    cell_tools._independent(masses, positions, velocities, cell, cell_velocity)
    _positive_scalar("maxstep", maxstep)
    next_v = velocities + 0.5 * dt[idx, None] * forces / mass
    next_hv = cell_velocity + 0.5 * dt[:, None, None] * cell_force
    cell_tools._finite(next_v, next_hv)
    _full_drift(positions, next_v, cell, next_hv, dt, idx, maxstep)
    velocities.copy_(next_v)
    cell_velocity.copy_(next_hv)


def _full_pack(v, hv, f, hf, idx, b):
    n = v.shape[0]
    atom_rows = torch.arange(n, device=v.device) + 3 * idx
    counts = torch.bincount(idx, minlength=b)
    cell_rows = (counts.cumsum(0) + 3 * torch.arange(b, device=v.device))[
        :, None
    ] + torch.arange(3, device=v.device)
    ev = v.new_empty(n + 3 * b, 3)
    ef = f.new_empty(n + 3 * b, 3)
    ev[atom_rows] = v
    ef[atom_rows] = f
    ev[cell_rows.reshape(-1)] = hv.reshape(-1, 3)
    ef[cell_rows.reshape(-1)] = hf.reshape(-1, 3)
    ext_idx = torch.repeat_interleave(torch.arange(b, device=v.device), counts + 3)
    return ev, ef, ext_idx, atom_rows, cell_rows


def fire_cell_post(
    velocities,
    forces,
    masses,
    cell_velocity,
    cell_force,
    batch_idx,
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
    *,
    vf=None,
    vv=None,
    ff=None,
):
    """Complete kicks, update joint controls once, then mix all 3N+9 DOFs."""
    _, b, idx = _check(
        (velocities, forces),
        (cell_velocity, cell_force),
        batch_idx,
        dt,
        writes=(velocities, cell_velocity, dt),
    )
    mass = _mass(masses, velocities)
    _controls(alpha, dt, n_steps_positive)
    _control_storage(
        alpha,
        n_steps_positive,
        velocities,
        forces,
        cell_velocity,
        cell_force,
        dt,
        masses,
    )
    cell_tools._independent(masses, velocities, cell_velocity, dt)
    for value in (alpha_start, f_alpha, dt_min, dt_max, f_dec, f_inc):
        cell_tools._floats(dt, value)
        cell_tools._shape(value, (b,))
        cell_tools._finite(value)
    cell_tools._integer(n_min, dt)
    cell_tools._shape(n_min, (b,))
    for out in (velocities, cell_velocity, dt, alpha, n_steps_positive):
        cell_tools._independent(
            out, alpha_start, f_alpha, dt_min, dt_max, n_min, f_dec, f_inc
        )
    if bool(
        (
            # LOCAL-004: upstream max(dt*f_dec, dt_min) permits a zero floor;
            # only the actual dt must stay positive (checked by _controls).
            (dt_min < 0)
            | (dt_max < dt_min)
            | (f_dec <= 0)
            | (f_inc <= 0)
            | (n_min < 0)
            | (alpha_start < 0)
            | (alpha_start > 1)
            | (f_alpha < 0)
            | (f_alpha > 1)
        ).any()
    ):
        raise ValueError("invalid FIRE controls")
    scratches = [_scratch(x, (b,), dt) for x in (vf, vv, ff)]
    for i, out in enumerate(scratches):
        cell_tools._independent(
            out,
            velocities,
            forces,
            cell_velocity,
            cell_force,
            dt,
            alpha,
            n_steps_positive,
            masses,
            alpha_start,
            f_alpha,
            dt_min,
            dt_max,
            n_min,
            f_dec,
            f_inc,
            *scratches[:i],
        )
    ev, ef, ext_idx, atom_rows, cell_rows = _full_pack(
        velocities + 0.5 * dt[idx, None] * forces / mass,
        cell_velocity + 0.5 * dt[:, None, None] * cell_force,
        forces,
        cell_force,
        idx,
        b,
    )
    power, norm_v, norm_f = _reduce(ev, ef, ext_idx, b)
    positive = power > 0
    next_n = torch.where(positive, n_steps_positive + 1, 0)
    reached = positive & (next_n >= n_min)
    next_dt = torch.where(
        reached,
        torch.minimum(dt * f_inc, dt_max),
        torch.where(positive, dt, torch.maximum(dt * f_dec, dt_min)),
    )
    next_a = torch.where(
        reached, alpha * f_alpha, torch.where(positive, alpha, alpha_start)
    )
    # UBF-013 / fork eadc3ee: controls are updated once per system before
    # mixing every atomic/cell DOF, avoiding cross-thread state races.
    a = next_a[ext_idx, None]
    mixed = torch.where(
        positive[ext_idx, None],
        (1 - a) * ev + a * _ratio(norm_v, norm_f)[ext_idx, None] * ef,
        0,
    )
    cell_tools._finite(mixed, next_dt, next_a)
    velocities.copy_(mixed[atom_rows])
    cell_velocity.copy_(mixed[cell_rows.reshape(-1)].reshape(b, 3, 3))
    dt.copy_(next_dt)
    alpha.copy_(next_a)
    n_steps_positive.copy_(next_n)
    for out, value in zip(scratches, (power, norm_v, norm_f), strict=True):
        out.copy_(value)
