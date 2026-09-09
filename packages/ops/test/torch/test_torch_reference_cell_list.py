# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests for the opt-in no-PBC Torch cell-list backend."""

from __future__ import annotations

import pytest
import torch

from nvalchemiops.torch_reference import neighbor_list as dense_neighbor_list
from nvalchemiops.torch_reference_cell_list import neighbor_list


def test_cell_list_matches_dense_reference_for_batch_full_and_half() -> None:
    """Cell-list output matches dense output including geometry and row order."""
    positions = torch.tensor(
        [
            [-3.0, 2.0, 1.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.8, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    batch_ptr = torch.tensor([0, 4, 6], dtype=torch.int64)
    for half_fill in (False, True):
        expected = dense_neighbor_list(
            positions,
            1.5,
            batch_ptr=batch_ptr,
            max_neighbors=positions.shape[0],
            half_fill=half_fill,
            return_neighbor_list=True,
            return_distances=True,
            return_vectors=True,
        )
        actual = neighbor_list(
            positions,
            1.5,
            batch_ptr=batch_ptr,
            max_neighbors=positions.shape[0],
            half_fill=half_fill,
            return_neighbor_list=True,
            return_distances=True,
            return_vectors=True,
        )
        for expected_value, actual_value in zip(expected, actual):
            torch.testing.assert_close(
                expected_value, actual_value, atol=1e-12, rtol=1e-12
            )


def test_cell_list_auto_capacity_and_empty_input() -> None:
    """The opt-in backend discovers capacity and handles an empty batch."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    matrix, counts = neighbor_list(positions, 1.5)
    assert matrix.tolist() == [[1, 3], [0, 2], [1, 3]]
    assert counts.tolist() == [1, 2, 1]

    empty_matrix, empty_counts = neighbor_list(
        torch.empty((0, 3), dtype=torch.float64), 1.5
    )
    assert empty_matrix.shape == (0, 0)
    assert empty_counts.shape == (0,)


def test_cell_list_rejects_periodic_and_overflow_inputs_explicitly() -> None:
    """Unsupported periodic inputs and insufficient capacity never fall back."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64
    )
    with pytest.raises(NotImplementedError, match="no-PBC inputs only"):
        neighbor_list(
            positions,
            1.5,
            cell=torch.eye(3, dtype=torch.float64),
            pbc=torch.ones(3, dtype=torch.bool),
        )
    with pytest.raises(RuntimeError, match="capacity"):
        neighbor_list(positions, 1.5, max_neighbors=0)
