# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the isolated Torch materialization bridge."""

from __future__ import annotations

import pytest
import torch

from nvalchemiops._torch_batch_query_materialization import (
    materialize_batch_query_candidate_into,
    materialize_batch_query_topology_geometry_into,
)
from nvalchemiops.torch_reference_cell_list import batch_cell_list


def _mixed_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positions = torch.tensor(
        [
            [0.1, 0.1, 0.1],
            [1.8, 0.2, 0.1],
            [0.4, 1.5, 0.2],
            [0.2, 0.2, 0.2],
            [1.7, 0.2, 0.2],
            [0.8, 1.4, 0.2],
        ],
        dtype=torch.float64,
    )
    cells = torch.tensor(
        [
            [[2.0, 0.0, 0.0], [0.5, 1.8, 0.0], [0.2, 0.3, 2.1]],
            [[2.0, 0.0, 0.0], [0.3, 2.0, 0.0], [0.0, 0.0, 2.0]],
        ],
        dtype=torch.float64,
    )
    batch_idx = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int32)
    return positions, cells, batch_idx


@pytest.mark.parametrize("half_fill", (False, True))
def test_materialization_matches_reference_for_mixed_pbc_and_permuted_rows(
    half_fill: bool,
) -> None:
    positions, cells, batch_idx = _mixed_batch()
    expected = batch_cell_list(
        positions,
        0.65,
        cells,
        torch.tensor([[True, True, True], [True, False, False]]),
        batch_idx,
        max_neighbors=32,
        half_fill=half_fill,
        return_distances=True,
        return_vectors=True,
    )
    expected_matrix, expected_counts, expected_shifts, expected_distances, expected_vectors = expected

    candidate_matrix = expected_matrix.clone()
    candidate_shifts = expected_shifts.clone()
    candidate_counts = expected_counts.clone()
    for row, count in enumerate(candidate_counts.tolist()):
        if count > 1:
            candidate_matrix[row, :count] = torch.roll(
                candidate_matrix[row, :count], shifts=1, dims=0
            )
            candidate_shifts[row, :count] = torch.roll(
                candidate_shifts[row, :count], shifts=1, dims=0
            )

    actual_matrix = torch.empty_like(candidate_matrix)
    actual_shifts = torch.empty_like(candidate_shifts)
    actual_counts = torch.empty_like(candidate_counts)
    actual_distances = torch.empty_like(expected_distances)
    actual_vectors = torch.empty_like(expected_vectors)
    materialize_batch_query_candidate_into(
        positions,
        cells,
        batch_idx,
        candidate_matrix,
        candidate_shifts,
        candidate_counts,
        actual_matrix,
        actual_shifts,
        actual_counts,
        distances=actual_distances,
        vectors=actual_vectors,
    )

    for actual, reference in zip(
        (actual_matrix, actual_counts, actual_shifts, actual_distances, actual_vectors),
        expected,
        strict=True,
    ):
        torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-12)


def test_materialization_preserves_first_and_second_geometry_gradients() -> None:
    positions = torch.tensor(
        [[0.1, 0.0, 0.0], [1.9, 0.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    cells = (torch.eye(3, dtype=torch.float64) * 2.0).unsqueeze(0).requires_grad_()
    batch_idx = torch.zeros(2, dtype=torch.int32)
    candidate_matrix = torch.tensor([[1, 2], [0, 2]], dtype=torch.int32)
    candidate_shifts = torch.zeros((2, 2, 3), dtype=torch.int32)
    candidate_shifts[0, 0, 0] = -1
    candidate_shifts[1, 0, 0] = 1
    candidate_counts = torch.tensor([1, 1], dtype=torch.int32)
    public_matrix = torch.empty_like(candidate_matrix)
    public_shifts = torch.empty_like(candidate_shifts)
    public_counts = torch.empty_like(candidate_counts)
    distances = torch.empty((2, 2), dtype=torch.float64)
    vectors = torch.empty((2, 2, 3), dtype=torch.float64)

    materialize_batch_query_candidate_into(
        positions,
        cells,
        batch_idx,
        candidate_matrix,
        candidate_shifts,
        candidate_counts,
        public_matrix,
        public_shifts,
        public_counts,
        distances=distances,
        vectors=vectors,
    )
    active = torch.arange(2)[None, :] < public_counts.to(torch.long)[:, None]
    value = distances[active].square().sum() + 0.25 * vectors[active].square().sum()
    grad_positions, grad_cells = torch.autograd.grad(
        value, (positions, cells), create_graph=True
    )
    second = torch.autograd.grad(
        grad_positions.square().sum() + grad_cells.square().sum(),
        (positions, cells),
    )
    assert all(torch.isfinite(value).all() for value in second)


def test_topology_geometry_bridge_matches_reference_and_preserves_gradients() -> None:
    positions, cells, batch_idx = _mixed_batch()
    expected = batch_cell_list(
        positions,
        0.65,
        cells,
        torch.tensor([[True, True, True], [True, False, False]]),
        batch_idx,
        max_neighbors=32,
        half_fill=True,
        return_distances=True,
        return_vectors=True,
    )
    expected_matrix, expected_counts, expected_shifts, expected_distances, expected_vectors = expected

    actual_distances = torch.empty_like(expected_distances)
    actual_vectors = torch.empty_like(expected_vectors)
    materialize_batch_query_topology_geometry_into(
        positions,
        cells,
        batch_idx,
        expected_matrix,
        expected_shifts,
        expected_counts,
        distances=actual_distances,
        vectors=actual_vectors,
    )
    torch.testing.assert_close(actual_distances, expected_distances, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(actual_vectors, expected_vectors, atol=1e-12, rtol=1e-12)

    gradient_positions = positions.clone().requires_grad_()
    gradient_cells = cells.clone().requires_grad_()
    gradient_distances = torch.empty_like(expected_distances)
    gradient_vectors = torch.empty_like(expected_vectors)
    materialize_batch_query_topology_geometry_into(
        gradient_positions,
        gradient_cells,
        batch_idx,
        expected_matrix,
        expected_shifts,
        expected_counts,
        distances=gradient_distances,
        vectors=gradient_vectors,
    )
    active = torch.arange(expected_matrix.shape[1])[None, :] < expected_counts.to(torch.long)[:, None]
    value = gradient_distances[active].square().sum() + 0.25 * gradient_vectors[active].square().sum()
    first = torch.autograd.grad(value, (gradient_positions, gradient_cells), create_graph=True)
    second = torch.autograd.grad(
        first[0].square().sum() + first[1].square().sum(),
        (gradient_positions, gradient_cells),
    )
    assert all(torch.isfinite(value).all() for value in second)


def test_materialization_rejects_candidate_overflow_without_truncation() -> None:
    positions = torch.zeros((1, 3), dtype=torch.float64)
    cells = torch.eye(3, dtype=torch.float64).unsqueeze(0)
    batch_idx = torch.zeros(1, dtype=torch.int32)
    topology = torch.zeros((1, 2), dtype=torch.int32)
    shifts = torch.zeros((1, 2, 3), dtype=torch.int32)
    counts = torch.tensor([3], dtype=torch.int32)
    with pytest.raises(ValueError, match="within the candidate capacity"):
        materialize_batch_query_candidate_into(
            positions,
            cells,
            batch_idx,
            topology,
            shifts,
            counts,
            torch.empty_like(topology),
            torch.empty_like(shifts),
            torch.empty_like(counts),
        )
