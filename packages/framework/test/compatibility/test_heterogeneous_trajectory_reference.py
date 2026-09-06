# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reference trajectory contract for heterogeneous batches.

This test supplements the upstream sink tests with the Batch shape used by the
DCU migration: different atom counts, stable system identities, and explicit
snapshot frame labels.  It does not claim checkpoint/restart support.
"""

from __future__ import annotations

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.hooks import SnapshotHook
from nvalchemi.dynamics.sinks import HostMemory
from nvalchemi.hooks import DynamicsContext


def _make_batch(device: str) -> Batch:
    data = []
    for n_atoms, offset in ((3, 0.0), (5, 100.0)):
        positions = (
            torch.arange(n_atoms * 3, dtype=torch.float32).reshape(n_atoms, 3)
            + offset
        )
        data.append(
            AtomicData(
                atomic_numbers=torch.full((n_atoms,), 6, dtype=torch.long),
                positions=positions,
            )
        )
    batch = Batch.from_data_list(data).to(device)
    batch.velocities = torch.zeros_like(batch.positions)
    batch.forces = torch.ones_like(batch.positions)
    batch.energy = torch.tensor([[3.0], [5.0]], device=batch.device)
    batch.status = torch.tensor([[0], [1]], dtype=torch.long, device=batch.device)
    batch.system_id = torch.tensor([[101], [202]], dtype=torch.long, device=batch.device)
    batch.trajectory_step = torch.ones(
        (batch.num_graphs, 1), dtype=torch.long, device=batch.device
    )
    return batch


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_snapshot_hook_preserves_heterogeneous_frames(device: str) -> None:
    if device == "cuda" and (not torch.version.hip or not torch.cuda.is_available()):
        pytest.skip("Hygon Torch device is unavailable")

    batch = _make_batch(device)
    sink = HostMemory(capacity=4)
    hook = SnapshotHook(sink=sink, frequency=1)
    first_positions = batch.positions.detach().clone()
    first_forces = batch.forces.detach().clone()
    first_velocities = batch.velocities.detach().clone()

    hook(
        DynamicsContext(batch=batch, step_count=1),
        DynamicsStage.AFTER_STEP,
    )

    batch.positions.add_(10.0)
    batch.forces.add_(2.0)
    batch.velocities.add_(3.0)
    batch.energy.add_(1.0)
    batch.trajectory_step.fill_(2)
    hook(
        DynamicsContext(batch=batch, step_count=2),
        DynamicsStage.AFTER_STEP,
    )

    if batch.device.type == "cuda":
        torch.cuda.synchronize()
    stored = sink.read()

    assert len(sink) == 4
    assert stored.num_graphs == 4
    assert stored.batch_ptr.tolist() == [0, 3, 8, 11, 16]
    assert torch.all(stored.batch_idx[1:] >= stored.batch_idx[:-1])
    assert stored.system_id.squeeze(-1).tolist() == [101, 202, 101, 202]
    assert stored.status.squeeze(-1).tolist() == [0, 1, 0, 1]
    assert stored.trajectory_step.squeeze(-1).tolist() == [1, 1, 2, 2]
    assert torch.allclose(
        stored.positions,
        torch.cat((first_positions.cpu(), first_positions.cpu() + 10.0)),
    )
    assert torch.allclose(
        stored.forces,
        torch.cat((first_forces.cpu(), first_forces.cpu() + 2.0)),
    )
    assert torch.allclose(
        stored.velocities,
        torch.cat((first_velocities.cpu(), first_velocities.cpu() + 3.0)),
    )
    assert stored.energy.squeeze(-1).tolist() == [3.0, 5.0, 4.0, 6.0]
