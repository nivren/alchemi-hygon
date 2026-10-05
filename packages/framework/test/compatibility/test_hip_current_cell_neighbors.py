# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-HIP current-cell MATRIX calls through compute_neighbors and Hook."""

from collections import Counter

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.hooks import NeighborListHook
from nvalchemi.hooks._context import HookContext
from nvalchemi.models.base import NeighborConfig, NeighborListFormat
from nvalchemi.neighbors import compute_neighbors


pytestmark = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(),
    reason="requires an explicitly allocated visible HIP device",
)


def _batch(dtype):
    data = []
    for n in (2, 3):
        fractions = torch.tensor(
            [[0.06, 0.07, 0.1], [0.21, 0.13, 0.12], [0.93, 0.08, 0.09]],
            dtype=dtype, device="cuda",
        )[:n]
        cell = torch.eye(3, dtype=dtype, device="cuda") * 9
        data.append(AtomicData(
            positions=fractions @ cell,
            atomic_numbers=torch.ones(n, dtype=torch.int64, device="cuda"),
            cell=cell[None], pbc=torch.ones((1, 3), dtype=torch.bool, device="cuda"),
        ))
    return Batch.from_data_list(data, device="cuda")


def _edges(batch):
    matrix = batch.neighbor_matrix.cpu()
    shifts = batch.neighbor_matrix_shifts.cpu()
    return Counter(
        (i, int(matrix[i, k]), *shifts[i, k].tolist())
        for i, n in enumerate(batch.num_neighbors.cpu().tolist()) for k in range(n)
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("entrypoint", ["compute_neighbors", "hook"])
def test_public_hip_rebuilds_after_cell_changes_and_capacity_growth(dtype, entrypoint):
    batch = _batch(dtype)
    expected = _batch(dtype)
    fractions = batch.positions @ torch.eye(3, dtype=dtype, device="cuda") / 9
    config = NeighborConfig(cutoff=2.4, format=NeighborListFormat.MATRIX, skin=0)
    hook = NeighborListHook(config, backend="hip", method="cell_list")
    initial_capacity = None
    for name in ("initial", "contract", "shear", "wrap", "restore"):
        cells = torch.eye(3, dtype=dtype, device="cuda").expand(2, 3, 3).clone() * 9
        current_fractions = fractions.clone()
        if name in {"contract", "shear", "wrap"}:
            cells *= 0.15
            cells[1] *= 1.2  # heterogeneous cells
        if name in {"shear", "wrap"}:
            cells[:, 1, 0] = 0.3
            cells[:, 2, 0] = 0.2
            cells[:, 0, 2] = 0.1  # full triclinic, not just lower triangular
        if name == "wrap":
            current_fractions[0] += fractions.new_tensor([1, -1, 2])
        positions = torch.bmm(
            current_fractions[:, None, :], cells[batch.batch_idx.long()]
        ).squeeze(1)
        for target in (batch, expected):
            target.cell.copy_(cells)
            target.positions.copy_(positions)
        saved_cell, saved_positions = batch.cell.clone(), batch.positions.clone()
        if entrypoint == "hook":
            hook(HookContext(batch=batch), None)
        else:
            compute_neighbors(batch, config=config, backend="hip", method="cell_list")
        compute_neighbors(expected, config=config, backend="torch_reference", method="cell_list")
        assert _edges(batch) == _edges(expected), name
        torch.testing.assert_close(batch.num_neighbors, expected.num_neighbors)
        torch.testing.assert_close(batch.cell, saved_cell, rtol=0, atol=0)
        torch.testing.assert_close(batch.positions, saved_positions, rtol=0, atol=0)
        if name == "initial":
            initial_capacity = batch.neighbor_matrix.shape[1]
        if name == "contract":
            assert int(batch.num_neighbors.max()) > initial_capacity
