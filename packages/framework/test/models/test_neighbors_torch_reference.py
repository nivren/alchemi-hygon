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

"""Framework neighbor entry point tests for the Warp-independent backend."""

from __future__ import annotations

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.lj import LennardJonesModelWrapper
from nvalchemi.models.base import NeighborListFormat
from nvalchemi.neighbors import compute_neighbors


def _make_batch() -> Batch:
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


def test_compute_neighbors_torch_reference_preserves_batch_boundaries():
    """The explicit reference backend writes global indices without cross-pairs."""
    batch = _make_batch()
    compute_neighbors(
        batch,
        cutoff=2.0,
        format=NeighborListFormat.MATRIX,
        backend="torch_reference",
    )
    assert batch.neighbor_matrix.tolist() == [[1], [0], [3], [2]]
    assert batch.num_neighbors.tolist() == [1, 1, 1, 1]

    coo_batch = _make_batch()
    compute_neighbors(
        coo_batch,
        cutoff=2.0,
        format=NeighborListFormat.COO,
        backend="auto",
    )
    assert coo_batch.neighbor_list.tolist() == [[0, 1], [1, 0], [2, 3], [3, 2]]


def test_compute_neighbors_torch_reference_cell_list_preserves_batch_boundaries():
    """The opt-in cell-list strategy keeps the framework batch contract."""
    batch = _make_batch()
    compute_neighbors(
        batch,
        cutoff=2.0,
        format=NeighborListFormat.MATRIX,
        backend="torch_reference",
        method="cell_list",
    )
    assert batch.neighbor_matrix.tolist() == [[1], [0], [3], [2]]
    assert batch.num_neighbors.tolist() == [1, 1, 1, 1]


def test_compute_neighbors_rejects_unpublished_cell_list_backend_name():
    """The pre-M1 global strategy spelling is not part of the public API."""
    with pytest.raises(RuntimeError, match="unknown backend request"):
        compute_neighbors(
            _make_batch(),
            cutoff=2.0,
            backend="torch_reference_cell_list",
        )


def test_compute_neighbors_rejects_unregistered_optimized_backend():
    """A framework caller cannot silently fall back from an unknown backend."""
    with pytest.raises(RuntimeError, match="no verified capability"):
        compute_neighbors(_make_batch(), cutoff=2.0, backend="triton")


def test_compute_neighbors_torch_reference_pbc_writes_shifts():
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
    compute_neighbors(
        batch,
        cutoff=0.5,
        format=NeighborListFormat.MATRIX,
        backend="torch_reference",
    )
    assert batch.neighbor_matrix.tolist() == [[1], [0]]
    assert batch.num_neighbors.tolist() == [1, 1]
    assert batch.neighbor_matrix_shifts.tolist() == [[[-1, 0, 0]], [[1, 0, 0]]]

    coo_batch = Batch.from_data_list(
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
    compute_neighbors(
        coo_batch,
        cutoff=0.5,
        format=NeighborListFormat.COO,
        backend="torch_reference",
    )
    assert coo_batch.neighbor_list.tolist() == [[0, 1], [1, 0]]
    assert coo_batch.neighbor_list_shifts.tolist() == [[-1, 0, 0], [1, 0, 0]]


def test_compute_neighbors_periodic_cell_list_supports_full_and_half():
    def make_batch() -> Batch:
        return Batch.from_data_list(
            [
                AtomicData(
                    positions=torch.tensor(
                        [[0.1, 0.0, 0.0], [1.9, 0.0, 0.0]],
                        dtype=torch.float64,
                    ),
                    atomic_numbers=torch.tensor([1, 1]),
                    cell=torch.diag(
                        torch.tensor([2.0, 10.0, 10.0], dtype=torch.float64)
                    ).unsqueeze(0),
                    pbc=torch.tensor([[True, False, False]]),
                )
            ]
        )

    full = make_batch()
    compute_neighbors(
        full,
        cutoff=0.5,
        format=NeighborListFormat.MATRIX,
        backend="torch_reference",
        method="cell_list",
    )
    assert full.neighbor_matrix[:, :1].tolist() == [[1], [0]]
    assert full.neighbor_matrix_shifts[:, :1].tolist() == [[[-1, 0, 0]], [[1, 0, 0]]]

    half = make_batch()
    compute_neighbors(
        half,
        cutoff=0.5,
        format=NeighborListFormat.COO,
        half_list=True,
        backend="torch_reference",
        method="cell_list",
    )
    assert half.neighbor_list.shape == (1, 2)
    assert half.neighbor_list_shifts.shape == (1, 3)


def test_make_neighbor_hooks_accepts_backend_and_compatibility_alias():
    """The public spelling is backend; the old alias remains unambiguous."""
    model = LennardJonesModelWrapper(
        epsilon=1.0,
        sigma=1.0,
        cutoff=2.0,
        backend="torch_reference",
    )
    assert model.make_neighbor_hooks(backend="torch_reference")[0].backend == "torch_reference"
    assert (
        model.make_neighbor_hooks(neighbor_backend="torch_reference")[0].backend
        == "torch_reference"
    )
    with pytest.raises(ValueError, match="must agree"):
        model.make_neighbor_hooks(backend="torch_reference", neighbor_backend="warp")
