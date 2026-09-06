"""Regression tests for mixed-status dynamics output preservation."""

from __future__ import annotations

import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.demo import DemoDynamics
from nvalchemi.models.demo import DemoModel, DemoModelWrapper


def test_step_preserves_outputs_for_graduated_graph() -> None:
    """Forces and energy remain aligned with a frozen graph's coordinates."""
    data = [
        AtomicData(
            atomic_numbers=torch.tensor([6, 8], dtype=torch.long),
            positions=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        ),
        AtomicData(
            atomic_numbers=torch.tensor([1, 1, 8], dtype=torch.long),
            positions=torch.tensor([[0.2, 0.1, 0.0], [1.1, 0.0, 0.0], [0.4, 0.8, 0.0]]),
        ),
    ]
    batch = Batch.from_data_list(data)
    batch.forces = torch.full((batch.num_nodes, 3), 7.0)
    batch.energy = torch.full((batch.num_graphs, 1), 11.0)
    batch.velocities = torch.zeros_like(batch.positions)
    batch.status = torch.tensor([[0], [1]], dtype=torch.long)

    frozen_positions = batch.positions[2:].clone()
    frozen_forces = batch.forces[2:].clone()
    frozen_energy = batch.energy[1:].clone()

    dynamics = DemoDynamics(
        model=DemoModelWrapper(DemoModel()), dt=0.1, n_steps=1, exit_status=1
    )
    dynamics.step(batch)

    assert torch.equal(batch.positions[2:], frozen_positions)
    assert torch.equal(batch.forces[2:], frozen_forces)
    assert torch.equal(batch.energy[1:], frozen_energy)
