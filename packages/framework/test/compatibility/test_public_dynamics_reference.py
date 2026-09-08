"""Public dynamics classes running the Warp-free Torch reference backend."""

from __future__ import annotations

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import FIRE, FIRE2, NVE
from nvalchemi.dynamics.hooks._utils import scatter_reduce_per_graph
from nvalchemi.models.demo import DemoModel, DemoModelWrapper


def _batch() -> Batch:
    systems = []
    for n_atoms, seed in ((3, 11), (5, 17)):
        generator = torch.Generator().manual_seed(seed)
        data = AtomicData(
            positions=torch.randn(n_atoms, 3, generator=generator),
            atomic_numbers=torch.ones(n_atoms, dtype=torch.long),
            atomic_masses=torch.ones(n_atoms),
            forces=torch.zeros(n_atoms, 3),
            energy=torch.zeros(1, 1),
        )
        data.add_node_property("velocities", torch.zeros(n_atoms, 3))
        systems.append(data)
    return Batch.from_data_list(systems)


@pytest.mark.parametrize("dynamics_type", (NVE, FIRE, FIRE2))
def test_public_reference_dynamics_runs_heterogeneous_batch(dynamics_type) -> None:
    model = DemoModelWrapper(DemoModel())
    dynamics = dynamics_type(
        model=model,
        dt=0.01,
        n_steps=2,
        backend="torch_reference",
    )
    batch = _batch()
    dynamics.run(batch)

    assert batch.batch_ptr.tolist() == [0, 3, 8]
    assert batch.energy.shape == (2, 1)
    assert batch.forces.shape == (8, 3)
    assert torch.isfinite(batch.positions).all()
    assert torch.isfinite(batch.forces).all()


@pytest.mark.parametrize(
    ("reduce", "expected"),
    [
        ("sum", (4.0, 1.0)),
        ("amax", (3.0, 2.0)),
        ("amin", (1.0, -1.0)),
        ("mean", (2.0, 0.5)),
    ],
)
def test_reference_observer_segment_reduce(reduce, expected) -> None:
    """Observer reductions stay Warp-free for every supported operation."""
    values = torch.tensor([3.0, 1.0, 2.0, -1.0])
    batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
    result = scatter_reduce_per_graph(
        values, batch_idx, num_graphs=2, reduce=reduce, backend="torch_reference"
    )
    assert torch.allclose(result, torch.tensor(expected))


def test_reference_observer_segment_reduce_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        scatter_reduce_per_graph(
            torch.ones(1), torch.zeros(1, dtype=torch.int32), 1, backend="bad"
        )
