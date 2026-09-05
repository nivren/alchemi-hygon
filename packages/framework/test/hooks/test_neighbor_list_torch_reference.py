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

import sys

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
    assert batch.neighbor_matrix.tolist() == [[1], [0], [3], [2]]
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


def test_reference_hook_rejects_method_selection():
    hook = NeighborListHook(
        NeighborConfig(cutoff=2.0),
        method="naive",
        backend="torch_reference",
    )
    with pytest.raises(NotImplementedError, match="does not support method selection"):
        _call_eager(hook, _batch())


@pytest.mark.parametrize(
    "config_kwargs, batch_kwargs, message",
    [
        ({"skin": 0.5}, {}, "requires skin=0"),
        ({}, {"pbc": True}, "does not support PBC"),
    ],
)
def test_reference_hook_rejects_unsupported_state(
    config_kwargs: dict[str, object],
    batch_kwargs: dict[str, object],
    message: str,
):
    hook = NeighborListHook(
        NeighborConfig(cutoff=2.0, **config_kwargs),
        backend="torch_reference",
        skin=float(config_kwargs.get("skin", 0.0)),
    )
    batch = _batch()
    if batch_kwargs.get("pbc"):
        batch = Batch.from_data_list(
            [
                AtomicData(
                    positions=torch.tensor([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]]),
                    atomic_numbers=torch.tensor([1, 1]),
                    cell=torch.eye(3).unsqueeze(0) * 10.0,
                    pbc=torch.ones(1, 3, dtype=torch.bool),
                )
            ]
        )
    with pytest.raises(NotImplementedError, match=message):
        _call_eager(hook, batch)
