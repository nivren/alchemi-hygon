"""Reference FIRE/FIRE2 contracts without importing Warp."""

from __future__ import annotations

import math
import sys

import pytest
import torch

from nvalchemi._dynamics_reference.fire import (
    fire2_step_coord,
    fire2_step_coord_cell,
    fire_step,
    fire_update,
)


def _fire_params(dtype: torch.dtype = torch.float64) -> dict[str, torch.Tensor]:
    return {
        "alpha": torch.tensor([0.05], dtype=dtype),
        "dt": torch.tensor([0.01], dtype=dtype),
        "n_steps_positive": torch.tensor([10], dtype=torch.int32),
        "alpha_start": torch.tensor([0.1], dtype=dtype),
        "f_alpha": torch.tensor([0.99], dtype=dtype),
        "dt_min": torch.tensor([0.001], dtype=dtype),
        "dt_max": torch.tensor([0.1], dtype=dtype),
        "maxstep": torch.tensor([0.1], dtype=dtype),
        "n_min": torch.tensor([5], dtype=torch.int32),
        "f_dec": torch.tensor([0.5], dtype=dtype),
        "f_inc": torch.tensor([1.1], dtype=dtype),
        "uphill_flag": torch.tensor([0], dtype=torch.int32),
    }


def test_fire_downhill_and_uphill_match_locked_batch_idx_semantics() -> None:
    pos = torch.zeros((3, 3), dtype=torch.float64)
    vel = torch.ones_like(pos)
    force = torch.ones_like(pos)
    masses = torch.ones(3, dtype=torch.float64)
    params = _fire_params()
    fire_step(pos, vel, force, masses, batch_idx=torch.zeros(3, dtype=torch.int32), **params)
    assert torch.allclose(params["dt"], torch.tensor([0.011], dtype=torch.float64))
    assert torch.allclose(params["alpha"], torch.tensor([0.0495], dtype=torch.float64))
    assert torch.allclose(vel, torch.full_like(vel, 1.01))
    assert torch.allclose(pos, torch.full_like(pos, 0.0101))

    pos.zero_()
    vel.fill_(1.0)
    force.fill_(-1.0)
    params = _fire_params()
    fire_step(pos, vel, force, masses, batch_idx=torch.zeros(3, dtype=torch.int32), **params)
    assert torch.allclose(params["dt"], torch.tensor([0.005], dtype=torch.float64))
    assert torch.allclose(params["alpha"], params["alpha_start"])
    assert torch.allclose(vel, torch.full_like(vel, -0.01))
    assert torch.allclose(pos, torch.full_like(pos, -0.0001))
    assert "warp" not in sys.modules


def test_fire_update_keeps_positions_out_of_the_no_md_path() -> None:
    velocities = torch.ones((2, 3), dtype=torch.float64)
    forces = torch.ones_like(velocities)
    params = _fire_params()
    params["n_steps_positive"].fill_(0)
    fire_update(
        velocities,
        forces,
        batch_idx=torch.zeros(2, dtype=torch.int32),
        **{key: value for key, value in params.items() if key != "maxstep" and key != "uphill_flag"},
    )
    # n_steps_positive=0 does not yet reach n_min, so alpha/dt stay unchanged;
    # the ratio is one and the velocity remains one in the no-MD path.
    assert torch.allclose(velocities, torch.ones_like(velocities))
    assert torch.allclose(params["dt"], torch.tensor([0.01], dtype=torch.float64))


def _fire2_scalar_reference(state: tuple[torch.Tensor, ...], **kwargs: float) -> None:
    positions, velocities, forces, batch_idx, alpha, dt, nsteps_inc = state
    systems = alpha.shape[0]
    vf = torch.zeros(systems, dtype=positions.dtype)
    vv = torch.zeros_like(vf)
    ff = torch.zeros_like(vf)
    for atom in range(positions.shape[0]):
        system = int(batch_idx[atom])
        v_upd = velocities[atom] + dt[system] * forces[atom]
        vf[system] += torch.dot(v_upd, forces[atom])
        vv[system] += torch.dot(v_upd, v_upd)
        ff[system] += torch.dot(forces[atom], forces[atom])
    dt_old = dt.clone()
    downhill = vf > 0
    nsi = torch.where(downhill, nsteps_inc + 1, torch.zeros_like(nsteps_inc))
    grow = downhill & (nsi > kwargs["delaystep"])
    dt_new = torch.where(
        grow,
        torch.minimum(kwargs["dtgrow"] * dt, torch.full_like(dt, kwargs["tmax"])),
        torch.where(
            ~downhill,
            torch.maximum(kwargs["dtshrink"] * dt, torch.full_like(dt, kwargs["tmin"])),
            dt,
        ),
    )
    alpha_new = torch.where(
        grow,
        kwargs["alphashrink"] * alpha,
        torch.where(~downhill, torch.full_like(alpha, kwargs["alpha0"]), alpha),
    )
    for atom in range(positions.shape[0]):
        system = int(batch_idx[atom])
        ratio = math.sqrt(float(vv[system] / ff[system])) if ff[system] > 0 else 0.0
        coeff = (1.0 - float(alpha_new[system])) * float(dt_old[system]) + float(
            alpha_new[system]
        ) * ratio
        velocities[atom] = (1.0 - alpha_new[system]) * velocities[atom] + coeff * forces[atom]
    step = torch.empty_like(positions)
    for atom in range(positions.shape[0]):
        system = int(batch_idx[atom])
        step[atom] = (
            -0.5 * dt_new[system] * velocities[atom]
            if not downhill[system]
            else dt_new[system] * velocities[atom]
        )
    max_norm = torch.zeros(systems, dtype=positions.dtype)
    for atom in range(positions.shape[0]):
        system = int(batch_idx[atom])
        max_norm[system] = torch.maximum(max_norm[system], step[atom].norm())
    inv = torch.where(max_norm > 0, torch.minimum(torch.ones_like(max_norm), kwargs["maxstep"] / max_norm), torch.ones_like(max_norm))
    positions += step * inv[batch_idx].unsqueeze(-1)
    for atom in range(positions.shape[0]):
        if not downhill[batch_idx[atom]]:
            velocities[atom].zero_()
    dt.copy_(dt_new * inv)
    alpha.copy_(alpha_new)
    nsteps_inc.copy_(nsi)


def test_fire2_matches_independent_scalar_reference_for_mixed_batch() -> None:
    kwargs = dict(
        delaystep=2,
        dtgrow=1.05,
        dtshrink=0.75,
        alphashrink=0.985,
        alpha0=0.09,
        tmax=0.08,
        tmin=0.005,
        maxstep=0.1,
    )
    generator = torch.Generator().manual_seed(12)
    batch_idx = torch.tensor([0, 0, 1, 1, 1], dtype=torch.int64)
    state = (
        torch.randn((5, 3), generator=generator, dtype=torch.float64),
        torch.randn((5, 3), generator=generator, dtype=torch.float64),
        torch.randn((5, 3), generator=generator, dtype=torch.float64),
        batch_idx,
        torch.tensor([0.05, 0.04], dtype=torch.float64),
        torch.tensor([0.04, 0.03], dtype=torch.float64),
        torch.tensor([1, 3], dtype=torch.int32),
    )
    got = tuple(t.clone() for t in state)
    want = tuple(t.clone() for t in state)
    fire2_step_coord(*got, **kwargs)
    _fire2_scalar_reference(want, **kwargs)
    for actual, expected in zip(got, want, strict=True):
        assert torch.allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_fire2_variable_cell_is_explicitly_blocked() -> None:
    with pytest.raises(NotImplementedError, match="virial/stress"):
        fire2_step_coord_cell()
