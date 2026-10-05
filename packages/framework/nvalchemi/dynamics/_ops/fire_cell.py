# SPDX-License-Identifier: Apache-2.0
"""Full-cell FIRE adapters: fork semantics, explicit selected executor or Warp."""

import torch
from nvalchemiops.executor import execute_selected

from nvalchemi._backend import resolve_compute_backend


def _legacy_pre(
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
    from nvalchemiops.cell_fire_reference import _full_drift

    from nvalchemi.dynamics._ops.velocity_verlet import vv_velocity_finalize

    # UBF-012 / fork b12b54c: Warp atomic kick, direct unit-mass cell kick,
    # then the same Torch affine/capped drift used by the fork framework.
    with torch.no_grad():
        vv_velocity_finalize(velocities, forces, masses, dt, batch_idx)
        cell_velocity.add_(0.5 * dt[:, None, None] * cell_force)
        _full_drift(
            positions, velocities, cell, cell_velocity, dt, batch_idx.long(), maxstep
        )


def _legacy_post(
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
    from nvalchemiops.cell_fire_reference import _full_pack

    from nvalchemi.dynamics._ops.fire import fire_update
    from nvalchemi.dynamics._ops.velocity_verlet import vv_velocity_finalize

    with torch.no_grad():
        vv_velocity_finalize(velocities, forces, masses, dt, batch_idx)
        cell_velocity.add_(0.5 * dt[:, None, None] * cell_force)
        ev, ef, ext_idx, atoms, cells = _full_pack(
            velocities,
            cell_velocity,
            forces,
            cell_force,
            batch_idx.long(),
            dt.numel(),
        )
        fire_update(
            ev,
            ef,
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
            vf=vf,
            vv=vv,
            ff=ff,
            batch_idx=ext_idx.int(),
        )
        velocities.copy_(ev[atoms])
        cell_velocity.copy_(ev[cells.reshape(-1)].reshape_as(cell_velocity))


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
    backend=None,
    selection=None,
):
    selected = resolve_compute_backend(
        backend,
        operation="fire_cell",
        device=positions.device,
        dtype=positions.dtype,
        gradient_order=0,
        features={"variable_cell", "full_cell"},
        selection=selection,
    )
    return execute_selected(
        selected,
        "fire_cell_pre",
        _legacy_pre,
        positions,
        velocities,
        forces,
        masses,
        cell,
        cell_velocity,
        cell_force,
        dt,
        batch_idx,
        maxstep=maxstep,
    )


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
    backend=None,
    selection=None,
):
    selected = resolve_compute_backend(
        backend,
        operation="fire_cell",
        device=velocities.device,
        dtype=velocities.dtype,
        gradient_order=0,
        features={"variable_cell", "full_cell"},
        selection=selection,
    )
    return execute_selected(
        selected,
        "fire_cell_post",
        _legacy_post,
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
        vf=vf,
        vv=vv,
        ff=ff,
    )
