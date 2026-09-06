"""Warp-free reference velocity-Verlet contract tests."""

from __future__ import annotations

import importlib
import sys

import pytest
import torch


def test_reference_module_does_not_import_warp() -> None:
    sys.modules.pop("warp", None)
    module = importlib.import_module("nvalchemi._dynamics_reference.velocity_verlet")
    assert "warp" not in sys.modules
    assert module.vv_position_update is not None


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_position_update_matches_analytic_first_half_step(dtype: torch.dtype) -> None:
    from nvalchemi._dynamics_reference.velocity_verlet import vv_position_update

    positions = torch.zeros((2, 3), dtype=dtype)
    velocities = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=dtype)
    forces = torch.tensor([[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=dtype)
    masses = torch.tensor([[2.0], [4.0]], dtype=dtype)
    dt = torch.tensor([0.5], dtype=dtype)
    batch_idx = torch.zeros(2, dtype=torch.int32)

    vv_position_update(positions, velocities, forces, masses, dt, batch_idx)

    # Both atoms have acceleration 1; r <- r + v*dt + 1/2*a*dt**2.
    expected_positions = torch.tensor(
        [[0.625, 0.0, 0.0], [1.125, 0.0, 0.0]], dtype=dtype
    )
    expected_velocities = torch.tensor(
        [[1.25, 0.0, 0.0], [2.25, 0.0, 0.0]], dtype=dtype
    )
    assert torch.allclose(positions, expected_positions)
    assert torch.allclose(velocities, expected_velocities)


def test_heterogeneous_batch_and_finalize_mutate_inputs() -> None:
    from nvalchemi._dynamics_reference.velocity_verlet import (
        vv_position_update,
        vv_velocity_finalize,
    )

    positions = torch.zeros((3, 3), dtype=torch.float64)
    velocities = torch.zeros_like(positions)
    forces = torch.tensor(
        [[2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [6.0, 0.0, 0.0]], dtype=torch.float64
    )
    masses = torch.tensor([2.0, 2.0, 3.0], dtype=torch.float64)
    dt = torch.tensor([0.1, 0.2], dtype=torch.float64)
    batch_idx = torch.tensor([0, 0, 1], dtype=torch.int64)

    vv_position_update(positions, velocities, forces, masses, dt, batch_idx)
    assert torch.allclose(
        positions[:, 0], torch.tensor([0.005, 0.005, 0.04], dtype=torch.float64)
    )
    assert torch.allclose(
        velocities[:, 0], torch.tensor([0.05, 0.05, 0.2], dtype=torch.float64)
    )

    forces_new = -forces
    vv_velocity_finalize(velocities, forces_new, masses, dt, batch_idx)
    assert torch.allclose(velocities[:, 0], torch.zeros(3, dtype=torch.float64))


def test_invalid_batch_index_is_explicit() -> None:
    from nvalchemi._dynamics_reference.velocity_verlet import vv_position_update

    state = torch.zeros((1, 3), dtype=torch.float64)
    with pytest.raises(ValueError, match="outside dt"):
        vv_position_update(
            state.clone(),
            state.clone(),
            state.clone(),
            torch.ones(1, dtype=torch.float64),
            torch.ones(1, dtype=torch.float64),
            torch.tensor([1], dtype=torch.int32),
        )


def test_batched_kinetic_energy_and_temperature_are_warp_free() -> None:
    from nvalchemi._dynamics_reference import (
        KB_EV,
        kinetic_energy_per_graph,
        temperature_per_graph,
    )

    velocities = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64
    )
    masses = torch.tensor([[2.0], [3.0], [4.0]], dtype=torch.float64)
    batch_idx = torch.tensor([0, 0, 1], dtype=torch.int32)
    kinetic = kinetic_energy_per_graph(velocities, masses, batch_idx, 2)
    assert torch.allclose(kinetic[:, 0], torch.tensor([7.0, 6.0], dtype=torch.float64))

    temperature = temperature_per_graph(
        velocities,
        masses,
        batch_idx,
        2,
        atoms_per_graph=torch.tensor([2, 1]),
    )
    expected = torch.tensor(
        [2.0 * 7.0 / (3.0 * 2.0 * KB_EV), 2.0 * 6.0 / (3.0 * KB_EV)],
        dtype=torch.float64,
    )
    assert torch.allclose(temperature, expected)
    assert "warp" not in sys.modules


def test_kinetic_empty_batch_and_invalid_temperature_count() -> None:
    from nvalchemi._dynamics_reference import (
        kinetic_energy_per_graph,
        temperature_per_graph,
    )

    empty = torch.empty((0, 3), dtype=torch.float32)
    masses = torch.empty(0, dtype=torch.float32)
    batch_idx = torch.empty(0, dtype=torch.int64)
    result = kinetic_energy_per_graph(empty, masses, batch_idx, 2)
    assert result.shape == (2, 1)
    with pytest.raises(ValueError, match=r"shape \[num_graphs\]"):
        temperature_per_graph(empty, masses, batch_idx, 2, torch.ones(1))
