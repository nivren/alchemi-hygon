"""Minimal continuation-state tests for the fixed-cell Langevin reference."""

from __future__ import annotations

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import NVTLangevin
from nvalchemi.models.demo import DemoModel, DemoModelWrapper


def _batch(device: torch.device) -> Batch:
    data = AtomicData(
        positions=torch.tensor(
            [[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]],
            dtype=torch.float64,
        ),
        atomic_numbers=torch.ones(2, dtype=torch.long),
        atomic_masses=torch.ones(2, dtype=torch.float64),
        forces=torch.zeros(2, 3, dtype=torch.float64),
        energy=torch.zeros(1, 1, dtype=torch.float64),
    )
    data.add_node_property(
        "velocities",
        torch.tensor(
            [[0.2, 0.0, -0.1], [-0.3, 0.4, 0.1]],
            dtype=torch.float64,
        ),
    )
    return Batch.from_data_list([data]).to(device)


def _dynamics(model: DemoModelWrapper) -> NVTLangevin:
    return NVTLangevin(
        model=model,
        dt=0.1,
        temperature=300.0,
        friction=0.1,
        random_seed=41,
        backend="torch_reference",
    )


def test_langevin_restart_continues_same_seed_stream() -> None:
    """A split run matches an uninterrupted run after restoring integrator state."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DemoModelWrapper(DemoModel()).to(device)
    continuous = _batch(device)

    uninterrupted = _dynamics(model)
    uninterrupted.run(continuous, n_steps=3)
    split = continuous.clone()
    restart_state = uninterrupted.state_dict()
    uninterrupted.run(continuous, n_steps=2)

    resumed = _dynamics(model)
    resumed.load_state_dict(restart_state)
    resumed.run(split, n_steps=2)

    for key in ("positions", "velocities", "forces", "energy"):
        torch.testing.assert_close(getattr(split, key), getattr(continuous, key))
    assert resumed.step_count == uninterrupted.step_count == 5


def test_langevin_restart_rejects_invalid_state() -> None:
    dynamics = _dynamics(DemoModelWrapper(DemoModel()))
    with pytest.raises(ValueError, match="version"):
        dynamics.load_state_dict(
            {"version": 99, "step_count": 0, "random_seed": 41}
        )

    with pytest.raises(ValueError, match="non-negative"):
        dynamics.load_state_dict(
            {"version": 1, "step_count": -1, "random_seed": 41}
        )
