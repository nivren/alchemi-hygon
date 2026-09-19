"""Warp-free fixed-cell BAOAB Langevin contract tests."""

from __future__ import annotations

import importlib
import sys

import pytest
import torch


def _state(dtype: torch.dtype = torch.float64):
    positions = torch.tensor(
        [[0.2, -0.1, 0.3], [1.0, 0.5, -0.4], [-0.2, 0.7, 0.1]], dtype=dtype
    )
    velocities = torch.tensor(
        [[0.4, 0.1, -0.2], [-0.3, 0.2, 0.5], [0.6, -0.4, 0.1]], dtype=dtype
    )
    forces = torch.tensor(
        [[1.0, -2.0, 0.5], [0.2, 0.3, -0.4], [-1.5, 0.7, 0.9]], dtype=dtype
    )
    masses = torch.tensor([2.0, 3.0, 4.0], dtype=dtype)
    batch_idx = torch.tensor([0, 0, 1], dtype=torch.int32)
    return positions, velocities, forces, masses, batch_idx


def _variable_batch_state(dtype: torch.dtype = torch.float64):
    positions = torch.arange(18, dtype=dtype).reshape(6, 3) / 10
    velocities = torch.linspace(-0.3, 0.4, 18, dtype=dtype).reshape(6, 3)
    forces = torch.linspace(0.2, 1.0, 18, dtype=dtype).reshape(6, 3)
    masses = torch.tensor(
        [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]], dtype=dtype
    )
    batch_idx = torch.tensor([0, 0, 1, 2, 2, 2], dtype=torch.int32)
    return positions, velocities, forces, masses, batch_idx


def test_reference_langevin_module_does_not_import_warp() -> None:
    sys.modules.pop("warp", None)
    module = importlib.import_module("nvalchemi._dynamics_reference.langevin")
    assert "warp" not in sys.modules
    assert module.langevin_half_step is not None


def test_friction_zero_matches_velocity_verlet_position_half_step() -> None:
    from nvalchemi._dynamics_reference.langevin import langevin_half_step
    from nvalchemi._dynamics_reference.velocity_verlet import vv_position_update

    positions, velocities, forces, masses, batch_idx = _state()
    vv_positions = positions.clone()
    vv_velocities = velocities.clone()
    langevin_positions = positions.clone()
    langevin_velocities = velocities.clone()
    dt = torch.tensor([0.1, 0.2], dtype=positions.dtype)

    vv_position_update(
        vv_positions, vv_velocities, forces, masses, dt, batch_idx
    )
    langevin_half_step(
        langevin_positions,
        langevin_velocities,
        forces,
        masses,
        dt,
        torch.zeros_like(dt),
        torch.zeros_like(dt),
        17,
        batch_idx,
    )

    assert torch.allclose(langevin_positions, vv_positions)
    assert torch.allclose(langevin_velocities, vv_velocities)


def test_heterogeneous_batch_uses_per_system_parameters() -> None:
    from nvalchemi._dynamics_reference.langevin import langevin_half_step

    positions, velocities, forces, masses, batch_idx = _state()
    dt = torch.tensor([0.1, 0.2], dtype=positions.dtype)
    friction = torch.tensor([0.5, 1.0], dtype=positions.dtype)
    temperature = torch.zeros_like(dt)
    dt_atom = dt[batch_idx].unsqueeze(-1)
    friction_atom = friction[batch_idx].unsqueeze(-1)
    expected_velocity = velocities + 0.5 * dt_atom * forces / masses[:, None]
    expected_velocity = torch.exp(-friction_atom * dt_atom) * expected_velocity
    expected_positions = positions + 0.5 * dt_atom * (
        velocities + 0.5 * dt_atom * forces / masses[:, None]
    )
    expected_positions = expected_positions + 0.5 * dt_atom * expected_velocity

    langevin_half_step(
        positions,
        velocities,
        forces,
        masses,
        dt,
        temperature,
        friction,
        0,
        batch_idx,
    )

    assert torch.allclose(positions, expected_positions)
    assert torch.allclose(velocities, expected_velocity)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_variable_batch_sizes_and_column_masses_follow_reference_contract(
    dtype: torch.dtype,
) -> None:
    from nvalchemi._dynamics_reference.langevin import langevin_half_step

    positions, velocities, forces, masses, batch_idx = _variable_batch_state(dtype)
    dt = torch.tensor([0.1, 0.2, 0.3], dtype=dtype)
    temperature = torch.zeros_like(dt)
    friction = torch.tensor([0.0, 0.4, 0.8], dtype=dtype)
    dt_atom = dt[batch_idx].unsqueeze(-1)
    friction_atom = friction[batch_idx].unsqueeze(-1)
    expected_velocity = velocities + 0.5 * dt_atom * forces / masses
    expected_velocity = torch.exp(-friction_atom * dt_atom) * expected_velocity
    expected_positions = positions + 0.5 * dt_atom * (
        velocities + 0.5 * dt_atom * forces / masses
    )
    expected_positions = expected_positions + 0.5 * dt_atom * expected_velocity

    langevin_half_step(
        positions,
        velocities,
        forces,
        masses,
        dt,
        temperature,
        friction,
        0,
        batch_idx,
    )

    assert torch.allclose(positions, expected_positions)
    assert torch.allclose(velocities, expected_velocity)


def test_half_step_mutates_only_positions_and_velocities() -> None:
    from nvalchemi.dynamics._ops.langevin import langevin_half_step

    positions, velocities, forces, masses, batch_idx = _state()
    forces_before = forces.clone()
    masses_before = masses.clone()
    dt = torch.tensor([0.1, 0.2], dtype=positions.dtype)
    temperature = torch.tensor([0.5, 0.7], dtype=positions.dtype)
    friction = torch.tensor([0.5, 1.0], dtype=positions.dtype)

    langevin_half_step(
        positions,
        velocities,
        forces,
        masses,
        dt,
        temperature,
        friction,
        23,
        batch_idx,
        backend="torch_reference",
    )

    assert torch.equal(forces, forces_before)
    assert torch.equal(masses, masses_before)
    assert positions.shape == (3, 3)
    assert velocities.shape == (3, 3)


def test_finalize_handles_variable_batch_sizes_and_preserves_inputs() -> None:
    from nvalchemi.dynamics._ops.langevin import langevin_finalize

    _, velocities, forces, masses, batch_idx = _variable_batch_state()
    velocities_before = velocities.clone()
    forces_before = forces.clone()
    masses_before = masses.clone()
    dt = torch.tensor([0.1, 0.2, 0.3], dtype=velocities.dtype)
    expected = velocities_before + 0.5 * dt[batch_idx].unsqueeze(-1) * forces / masses

    langevin_finalize(
        velocities,
        forces,
        masses,
        dt,
        batch_idx,
        backend="torch_reference",
    )

    assert torch.allclose(velocities, expected)
    assert torch.equal(forces, forces_before)
    assert torch.equal(masses, masses_before)


def test_invalid_batch_indices_fail_explicitly() -> None:
    from nvalchemi._dynamics_reference.langevin import langevin_half_step

    positions, velocities, forces, masses, _ = _state()
    dt = torch.ones(2, dtype=positions.dtype)
    with pytest.raises(ValueError, match="outside parameter arrays"):
        langevin_half_step(
            positions,
            velocities,
            forces,
            masses,
            dt,
            torch.zeros_like(dt),
            torch.zeros_like(dt),
            0,
            torch.tensor([0, 1, 2], dtype=torch.int32),
        )


def test_seed_reproducibility_and_finalize() -> None:
    from nvalchemi._dynamics_reference.langevin import (
        langevin_finalize,
        langevin_half_step,
    )

    state = _state()
    positions_a, velocities_a, forces, masses, batch_idx = state
    positions_b, velocities_b = positions_a.clone(), velocities_a.clone()
    dt = torch.tensor([0.1, 0.2], dtype=positions_a.dtype)
    temperature = torch.tensor([0.5, 0.7], dtype=positions_a.dtype)
    friction = torch.tensor([0.5, 1.0], dtype=positions_a.dtype)
    langevin_half_step(
        positions_a, velocities_a, forces, masses, dt, temperature, friction, 41, batch_idx
    )
    langevin_half_step(
        positions_b, velocities_b, forces, masses, dt, temperature, friction, 41, batch_idx
    )
    assert torch.equal(positions_a, positions_b)
    assert torch.equal(velocities_a, velocities_b)

    forces_new = -forces
    before_finalize = velocities_a.clone()
    langevin_finalize(velocities_a, forces_new, masses, dt, batch_idx)
    dt_atom = dt[batch_idx].unsqueeze(-1)
    expected = before_finalize + 0.5 * forces_new / masses[:, None] * dt_atom
    assert torch.allclose(velocities_a, expected)


def test_empty_input_and_invalid_parameters_are_explicit() -> None:
    from nvalchemi._dynamics_reference.langevin import langevin_half_step

    empty = torch.empty((0, 3), dtype=torch.float64)
    empty_params = torch.ones(1, dtype=torch.float64)
    langevin_half_step(
        empty,
        empty.clone(),
        empty.clone(),
        torch.empty(0, dtype=torch.float64),
        empty_params,
        empty_params,
        empty_params,
        0,
        torch.empty(0, dtype=torch.int32),
    )
    with pytest.raises(ValueError, match="friction"):
        langevin_half_step(
            torch.zeros((1, 3), dtype=torch.float64),
            torch.zeros((1, 3), dtype=torch.float64),
            torch.zeros((1, 3), dtype=torch.float64),
            torch.ones(1, dtype=torch.float64),
            torch.ones(1, dtype=torch.float64),
            torch.ones(1, dtype=torch.float64),
            -torch.ones(1, dtype=torch.float64),
            0,
            torch.zeros(1, dtype=torch.int32),
        )


def test_dispatcher_selects_reference_and_rejects_unknown_backend() -> None:
    from nvalchemi.dynamics._ops.langevin import langevin_half_step
    from nvalchemiops.backend import BackendUnavailableError, resolve_backend

    positions, velocities, forces, masses, batch_idx = _state()
    dt = torch.ones(2, dtype=positions.dtype) * 0.1
    zeros = torch.zeros_like(dt)
    langevin_half_step(
        positions,
        velocities,
        forces,
        masses,
        dt,
        zeros,
        zeros,
        0,
        batch_idx,
        backend="torch_reference",
    )
    selection = resolve_backend(
        "torch_reference",
        operation="langevin",
        device="cpu",
        dtype=torch.float64,
        features={"fixed_cell"},
    )
    assert selection.implementation_id == "torch_reference.langevin-v1"
    with pytest.raises(BackendUnavailableError, match="unknown backend"):
        langevin_half_step(
            positions,
            velocities,
            forces,
            masses,
            dt,
            zeros,
            zeros,
            0,
            batch_idx,
            backend="does_not_exist",
        )


def test_public_nvt_langevin_selects_reference_backend() -> None:
    from nvalchemi.data import AtomicData, Batch
    from nvalchemi.dynamics import NVTLangevin
    from nvalchemi.models.demo import DemoModel, DemoModelWrapper

    data = AtomicData(
        positions=torch.zeros((3, 3)),
        atomic_numbers=torch.ones(3, dtype=torch.long),
        atomic_masses=torch.ones(3),
        forces=torch.zeros((3, 3)),
        energy=torch.zeros((1, 1)),
    )
    data.add_node_property("velocities", torch.zeros((3, 3)))
    batch = Batch.from_data_list([data])
    dynamics = NVTLangevin(
        model=DemoModelWrapper(DemoModel()),
        dt=0.1,
        temperature=300.0,
        friction=0.1,
        backend="torch_reference",
    )

    dynamics.step(batch)

    assert dynamics._backend_selection is not None
    assert (
        dynamics._backend_selection.implementation_id
        == "torch_reference.langevin-v1"
    )
    assert torch.isfinite(batch.positions).all()
    assert torch.isfinite(batch.velocities).all()
