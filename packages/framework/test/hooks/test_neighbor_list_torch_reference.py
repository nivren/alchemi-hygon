# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contract tests for the explicit Warp-independent NeighborListHook path."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.hooks import NeighborListHook
from nvalchemi.hooks._context import HookContext
from nvalchemi.models.base import NeighborConfig, NeighborListFormat


def _batch() -> Batch:
    return Batch.from_data_list(
        [
            AtomicData(
                positions=torch.tensor([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]]),
                atomic_numbers=torch.tensor([1, 1]),
            ),
            AtomicData(
                positions=torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]),
                atomic_numbers=torch.tensor([1, 1]),
            ),
        ]
    )


def _call_eager(hook: NeighborListHook, batch: Batch) -> None:
    NeighborListHook.__call__.__wrapped__(
        hook, HookContext(batch=batch), None
    )


def test_reference_hook_import_and_matrix_writeback_without_warp():
    assert "warp" not in sys.modules
    hook = NeighborListHook(
        NeighborConfig(cutoff=2.0, format=NeighborListFormat.MATRIX),
        backend="torch_reference",
    )
    batch = _batch()
    _call_eager(hook, batch)

    assert "warp" not in sys.modules
    assert batch.neighbor_matrix[:, :1].tolist() == [[1], [0], [3], [2]]
    assert batch.num_neighbors.tolist() == [1, 1, 1, 1]


def test_reference_hook_compiled_entrypoint():
    hook = NeighborListHook(NeighborConfig(cutoff=2.0), backend="torch_reference")
    batch = _batch()
    hook(HookContext(batch=batch), None)
    assert batch.neighbor_list.tolist() == [
        [0, 1],
        [1, 0],
        [2, 3],
        [3, 2],
    ]


def test_reference_hook_supports_coo_and_auto():
    hook = NeighborListHook(
        NeighborConfig(cutoff=2.0, format=NeighborListFormat.COO),
        backend="auto",
    )
    batch = _batch()
    _call_eager(hook, batch)
    assert batch.neighbor_list.tolist() == [[0, 1], [1, 0], [2, 3], [3, 2]]


def test_reference_hook_supports_full_pbc_and_writes_shifts():
    hook = NeighborListHook(
        NeighborConfig(cutoff=0.5, format=NeighborListFormat.MATRIX),
        backend="torch_reference",
    )
    batch = Batch.from_data_list(
        [
            AtomicData(
                positions=torch.tensor(
                    [[0.1, 0.0, 0.0], [1.9, 0.0, 0.0]], dtype=torch.float64
                ),
                atomic_numbers=torch.tensor([1, 1]),
                cell=torch.diag(
                    torch.tensor([2.0, 10.0, 10.0], dtype=torch.float64)
                ).unsqueeze(0),
                pbc=torch.tensor([[True, False, False]]),
            )
        ]
    )
    _call_eager(hook, batch)
    assert batch.neighbor_matrix[:, :1].tolist() == [[1], [0]]
    assert batch.neighbor_matrix_shifts[:, :1].tolist() == [[[-1, 0, 0]], [[1, 0, 0]]]


def test_reference_hook_selects_periodic_cell_list_strategy():
    hook = NeighborListHook(
        NeighborConfig(cutoff=0.5, format=NeighborListFormat.MATRIX),
        method="cell_list",
        backend="torch_reference",
        skin=0.2,
    )
    batch = Batch.from_data_list(
        [
            AtomicData(
                positions=torch.tensor(
                    [[0.1, 0.0, 0.0], [1.9, 0.0, 0.0]], dtype=torch.float64
                ),
                atomic_numbers=torch.tensor([1, 1]),
                cell=torch.diag(
                    torch.tensor([2.0, 10.0, 10.0], dtype=torch.float64)
                ).unsqueeze(0),
                pbc=torch.tensor([[True, False, False]]),
            )
        ]
    )
    _call_eager(hook, batch)
    first = batch.neighbor_matrix.clone()
    batch.positions[0, 1] += 0.05
    _call_eager(hook, batch)
    assert torch.equal(batch.neighbor_matrix, first)
    assert batch.neighbor_matrix_shifts[:, :1].tolist() == [[[-1, 0, 0]], [[1, 0, 0]]]


def test_shared_dynamics_stage_keeps_stage_timing_domain_without_warp():
    framework = Path(__file__).parents[2]
    environment = os.environ | {"PYTHONPATH": f"{framework}:{framework.parent / 'ops'}"}
    script = """
import sys
from nvalchemi._dynamics_stage import DynamicsStage
from nvalchemi.hooks.stage_timing import _stage_domain
assert _stage_domain(DynamicsStage.BEFORE_COMPUTE) == 'dynamics'
assert 'nvalchemi.dynamics' not in sys.modules
assert 'warp' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=framework.parent.parent,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_reference_hook_selects_cell_list_strategy():
    hook = NeighborListHook(
        NeighborConfig(cutoff=2.0),
        method="cell_list",
        backend="torch_reference",
    )
    batch = _batch()
    _call_eager(hook, batch)
    assert batch.neighbor_list.tolist() == [[0, 1], [1, 0], [2, 3], [3, 2]]

    unsupported = NeighborListHook(
        NeighborConfig(cutoff=2.0),
        method="naive",
        backend="torch_reference",
    )
    with pytest.raises(RuntimeError, match="no verified capability"):
        _call_eager(unsupported, _batch())


def test_reference_hook_rejects_negative_skin():
    with pytest.raises(ValueError, match="skin must be non-negative"):
        NeighborListHook(NeighborConfig(cutoff=2.0), backend="torch_reference", skin=-1.0)


def test_reference_hook_reuses_skin_cache_until_displacement_threshold():
    hook = NeighborListHook(
        NeighborConfig(cutoff=2.0),
        backend="torch_reference",
        skin=0.5,
    )
    batch = _batch()
    _call_eager(hook, batch)
    first_reference = hook._ref_positions.clone()

    batch.positions[0, 0] += 0.1
    _call_eager(hook, batch)
    assert torch.equal(hook._ref_positions, first_reference)

    batch.positions[0, 0] += 0.2
    _call_eager(hook, batch)
    assert torch.equal(hook._ref_positions, batch.positions)


def test_reference_skin_rebuild_updates_only_changed_system():
    hook = NeighborListHook(
        NeighborConfig(cutoff=2.0, format=NeighborListFormat.MATRIX),
        backend="torch_reference",
        skin=0.5,
    )
    batch = _batch()
    _call_eager(hook, batch)
    first_reference = hook._ref_positions.clone()
    first_matrix = batch.neighbor_matrix.clone()

    batch.positions[0, 0] += 0.3
    _call_eager(hook, batch)

    assert torch.equal(hook._ref_positions[2:], first_reference[2:])
    assert torch.equal(hook._ref_positions[:2], batch.positions[:2])
    assert torch.equal(batch.neighbor_matrix, first_matrix)


def test_reference_skin_rebuild_preserves_coo_batch_offsets():
    hook = NeighborListHook(
        NeighborConfig(cutoff=2.0, format=NeighborListFormat.COO),
        backend="torch_reference",
        skin=0.5,
    )
    batch = _batch()
    _call_eager(hook, batch)
    first_edges = batch.neighbor_list.clone()

    batch.positions[0, 0] += 0.3
    _call_eager(hook, batch)

    assert torch.equal(batch.neighbor_list, first_edges)
    assert batch.neighbor_list.tolist() == [[0, 1], [1, 0], [2, 3], [3, 2]]


def test_reference_hook_rebuilds_when_periodic_cell_changes():
    hook = NeighborListHook(
        NeighborConfig(cutoff=0.5, format=NeighborListFormat.MATRIX),
        backend="torch_reference",
        skin=0.5,
    )
    batch = Batch.from_data_list(
        [
            AtomicData(
                positions=torch.tensor(
                    [[0.1, 0.0, 0.0], [1.9, 0.0, 0.0]], dtype=torch.float64
                ),
                atomic_numbers=torch.tensor([1, 1]),
                cell=torch.diag(
                    torch.tensor([2.0, 10.0, 10.0], dtype=torch.float64)
                ).unsqueeze(0),
                pbc=torch.tensor([[True, False, False]]),
            )
        ]
    )
    _call_eager(hook, batch)
    first_reference = hook._ref_cell.clone()
    batch.cell[0, 0, 0] = 2.2
    _call_eager(hook, batch)
    assert not torch.equal(hook._ref_cell, first_reference)


def _capacity_batch(num_atoms: int = 4) -> Batch:
    positions = torch.zeros(num_atoms, 3)
    positions[:, 0] = torch.arange(num_atoms, dtype=torch.float32)
    return Batch.from_data_list(
        [
            AtomicData(
                positions=positions,
                atomic_numbers=torch.ones(num_atoms, dtype=torch.long),
            )
        ]
    )


def _make_sparse_after_rebuild(batch: Batch) -> None:
    with torch.no_grad():
        batch.positions[0, 0] += 0.3
        batch.positions[2:, 0] = torch.arange(2, batch.num_nodes, dtype=batch.positions.dtype) * 100


def test_reference_capacity_grows_and_retries_after_overflow():
    hook = NeighborListHook(
        NeighborConfig(cutoff=5.0, format=NeighborListFormat.MATRIX),
        backend="torch_reference",
        max_neighbors=1,
    )
    batch = _capacity_batch()
    _call_eager(hook, batch)

    assert hook._max_neighbors == 16
    assert batch.neighbor_matrix.shape == (4, 16)
    assert batch.num_neighbors.tolist() == [3, 3, 3, 3]


def test_reference_capacity_grows_for_heterogeneous_batch():
    batch = Batch.from_data_list(
        [
            AtomicData(
                positions=torch.tensor(
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
                ),
                atomic_numbers=torch.ones(4, dtype=torch.long),
            ),
            AtomicData(
                positions=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                atomic_numbers=torch.ones(2, dtype=torch.long),
            ),
        ]
    )
    hook = NeighborListHook(
        NeighborConfig(cutoff=5.0, format=NeighborListFormat.MATRIX),
        backend="torch_reference",
        max_neighbors=1,
    )
    _call_eager(hook, batch)

    assert hook._max_neighbors == 16
    assert batch.batch_ptr.tolist() == [0, 4, 6]
    assert batch.neighbor_matrix.shape == (6, 16)
    assert batch.num_neighbors.tolist() == [3, 3, 3, 3, 1, 1]
    assert batch.neighbor_matrix[4:, :1].tolist() == [[5], [4]]


def test_reference_capacity_shrinks_after_skin_rebuild():
    hook = NeighborListHook(
        NeighborConfig(cutoff=20.0, format=NeighborListFormat.MATRIX),
        backend="torch_reference",
        skin=0.5,
    )
    batch = _capacity_batch(num_atoms=17)
    _call_eager(hook, batch)
    assert hook._max_neighbors == 32

    _make_sparse_after_rebuild(batch)
    _call_eager(hook, batch)

    assert hook._max_neighbors == 16
    assert batch.neighbor_matrix.shape == (17, 16)
    assert int(batch.num_neighbors.max()) == 1


def test_reference_capacity_shrink_respects_override_floor():
    hook = NeighborListHook(
        NeighborConfig(cutoff=20.0, format=NeighborListFormat.MATRIX),
        backend="torch_reference",
        skin=0.5,
        max_neighbors=24,
    )
    batch = _capacity_batch(num_atoms=17)
    _call_eager(hook, batch)

    _make_sparse_after_rebuild(batch)
    _call_eager(hook, batch)

    assert hook._max_neighbors == 24
    assert batch.neighbor_matrix.shape == (17, 24)
    assert int(batch.num_neighbors.max()) == 1


def test_reference_capacity_survives_repeated_skin_rebuilds():
    hook = NeighborListHook(
        NeighborConfig(cutoff=20.0, format=NeighborListFormat.MATRIX),
        backend="torch_reference",
        skin=0.5,
    )
    batch = _capacity_batch(num_atoms=17)
    capacities: list[int] = []
    for step in range(6):
        with torch.no_grad():
            if step % 2 == 0:
                batch.positions[:, 0] = torch.arange(
                    batch.num_nodes, dtype=batch.positions.dtype
                )
            else:
                batch.positions[:, 0] = torch.arange(
                    batch.num_nodes, dtype=batch.positions.dtype
                ) * 100
                batch.positions[1, 0] = 1.0
            batch.positions[0, 0] += 0.3
        _call_eager(hook, batch)
        capacities.append(hook._max_neighbors)

    assert capacities == [32, 16, 32, 16, 32, 16]
    assert int(batch.num_neighbors.max()) == 1
