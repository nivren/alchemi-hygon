"""Functional contract for FIRE2 with the Torch reference skin cache."""

from __future__ import annotations

import torch

from nvalchemi._dynamics_stage import DynamicsStage
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import FIRE2
from nvalchemi.hooks import NeighborListHook
from nvalchemi.models.base import NeighborConfig, NeighborListFormat
from nvalchemi.models.demo import DemoModel, DemoModelWrapper


def _batch() -> Batch:
    systems = []
    for n_atoms, offset in ((3, 0.0), (5, 10.0)):
        positions = torch.zeros(n_atoms, 3)
        positions[:, 0] = offset + torch.arange(n_atoms, dtype=torch.float32)
        systems.append(
            AtomicData(
                positions=positions,
                atomic_numbers=torch.ones(n_atoms, dtype=torch.long),
                forces=torch.zeros(n_atoms, 3),
                energy=torch.zeros(1, 1),
            )
        )
    return Batch.from_data_list(systems)


def _assert_no_cross_system_edges(batch: Batch) -> None:
    edges = batch.neighbor_list
    if edges.numel() == 0:
        return
    assert torch.equal(batch.batch_idx[edges[:, 0]], batch.batch_idx[edges[:, 1]])


def test_fire2_reference_reuses_and_selectively_rebuilds_skin_cache() -> None:
    batch = _batch()
    hook = NeighborListHook(
        NeighborConfig(cutoff=1.1, format=NeighborListFormat.COO),
        skin=0.5,
        stage=DynamicsStage.BEFORE_COMPUTE,
        backend="torch_reference",
    )
    dynamics = FIRE2(
        model=DemoModelWrapper(DemoModel()),
        dt=1.0e-5,
        n_steps=1,
        hooks=[hook],
        backend="torch_reference",
    )

    # The first FIRE2 step builds the neighbor list before model evaluation.
    dynamics.step(batch)
    assert batch.batch_ptr.tolist() == [0, 3, 8]
    assert batch.batch_idx.tolist() == [0, 0, 0, 1, 1, 1, 1, 1]
    _assert_no_cross_system_edges(batch)
    first_reference = hook._ref_positions.clone()

    # A sub-skin displacement is intentionally below the rebuild threshold.
    batch.positions[0, 0] += 0.1
    dynamics.step(batch)
    assert torch.equal(hook._ref_positions, first_reference)
    _assert_no_cross_system_edges(batch)

    # Move only system 0 beyond skin/2.  FIRE2 still runs as part of this
    # step; the hook must refresh only the stale system's reference positions.
    before_selective = hook._ref_positions.clone()
    batch.positions[0, 0] += 0.3
    dynamics.step(batch)
    assert torch.equal(hook._ref_positions[3:], before_selective[3:])
    assert torch.equal(hook._ref_positions[:3], batch.positions[:3])
    assert torch.isfinite(batch.positions).all()
    assert torch.isfinite(batch.forces).all()
    _assert_no_cross_system_edges(batch)
