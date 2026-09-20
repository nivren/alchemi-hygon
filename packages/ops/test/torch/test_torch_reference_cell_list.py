# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests for the opt-in Torch reference cell-list strategy."""

from __future__ import annotations

import pytest
import torch

from nvalchemiops.torch_reference import neighbor_list as dense_neighbor_list
from nvalchemiops.torch_reference_cell_list import (
    allocate_cell_list,
    batch_cell_list,
    build_cell_list,
    estimate_cell_list_sizes,
    neighbor_list,
    query_cell_list,
)


def _active_periodic_tuples(result, *, sort: bool = True):
    matrix, counts, shifts = result[:3]
    values = []
    for row, count in enumerate(counts.tolist()):
        for column in range(count):
            values.append(
                (row, int(matrix[row, column]), *map(int, shifts[row, column]))
            )
    return sorted(values) if sort else values


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


def test_large_mixed_pbc_batch_uses_consistent_cell_capacity() -> None:
    """The high-level reference path must match its 8192-bin allocation."""
    cells = torch.tensor(
        [
            [[16.0, 0.0, 0.0], [0.5, 16.0, 0.0], [0.3, 0.4, 16.0]],
            [[12.0, 0.0, 0.0], [0.4, 12.0, 0.0], [0.2, 0.3, 12.0]],
        ],
        dtype=torch.float64,
    )
    pbc = torch.tensor([[True, False, True], [False, True, False]])
    positions = torch.cat(
        [
            torch.rand((46, 3), dtype=torch.float64, generator=torch.Generator().manual_seed(1))
            @ cells[0],
            torch.rand((46, 3), dtype=torch.float64, generator=torch.Generator().manual_seed(2))
            @ cells[1],
        ]
    )
    batch_idx = torch.arange(2, dtype=torch.int32).repeat_interleave(46)

    matrix, counts, shifts = neighbor_list(
        positions,
        0.6,
        cell=cells,
        pbc=pbc,
        batch_idx=batch_idx,
        max_neighbors=256,
    )

    assert matrix.shape == (92, 256)
    assert shifts.shape == (92, 256, 3)
    assert torch.all(counts <= 256)


def test_periodic_cell_list_matches_dense_reference_and_overflow_is_explicit() -> None:
    """Periodic pairs/shifts match dense FP64 and capacity never truncates."""
    positions = torch.tensor(
        [[0.1, 0.2, 0.3], [1.9, 0.2, 0.3], [1.0, 1.0, 1.0]],
        dtype=torch.float64,
    )
    cell = torch.eye(3, dtype=torch.float64) * 2.0
    pbc = torch.ones(3, dtype=torch.bool)
    expected = dense_neighbor_list(positions, 0.5, cell=cell, pbc=pbc)
    actual = neighbor_list(positions, 0.5, cell=cell, pbc=pbc)
    assert _active_periodic_tuples(actual) == _active_periodic_tuples(expected)
    actual_order = _active_periodic_tuples(actual, sort=False)
    assert actual_order == sorted(actual_order)
    with pytest.raises(RuntimeError, match="capacity"):
        neighbor_list(positions, 0.5, cell=cell, pbc=pbc, max_neighbors=0)


def test_periodic_half_list_is_exactly_one_of_each_reverse_pair() -> None:
    positions = torch.tensor(
        [[0.1, 0.1, 0.1], [1.9, 0.1, 0.1], [1.0, 1.0, 1.0]],
        dtype=torch.float64,
    )
    cell = torch.eye(3, dtype=torch.float64) * 2.0
    pbc = torch.ones(3, dtype=torch.bool)
    full = _active_periodic_tuples(
        neighbor_list(positions, 1.25, cell=cell, pbc=pbc)
    )
    half = _active_periodic_tuples(
        neighbor_list(positions, 1.25, cell=cell, pbc=pbc, half_fill=True)
    )
    assert len(full) == 2 * len(half)
    half_set = set(half)
    for row, col, sx, sy, sz in full:
        reverse = (col, row, -sx, -sy, -sz)
        assert ((row, col, sx, sy, sz) in half_set) ^ (reverse in half_set)


def test_triclinic_mixed_pbc_batch_matches_dense_pair_set() -> None:
    positions = torch.tensor(
        [
            [0.1, 0.1, 0.1],
            [1.8, 0.2, 0.1],
            [0.4, 1.5, 0.2],
            [0.2, 0.2, 0.2],
            [1.7, 0.2, 0.2],
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
    pbc = torch.tensor([[True, True, True], [True, False, False]])
    batch_ptr = torch.tensor([0, 3, 5], dtype=torch.int64)
    expected = dense_neighbor_list(
        positions, 0.65, cell=cells, pbc=pbc, batch_ptr=batch_ptr
    )
    actual = neighbor_list(
        positions, 0.65, cell=cells, pbc=pbc, batch_ptr=batch_ptr
    )
    assert _active_periodic_tuples(actual) == _active_periodic_tuples(expected)


def test_layered_build_query_matches_high_level_and_preserves_false_rebuild() -> None:
    positions = torch.tensor(
        [[0.1, 0.0, 0.0], [1.9, 0.0, 0.0]], dtype=torch.float64
    )
    cell = torch.eye(3, dtype=torch.float64) * 2.0
    pbc = torch.ones(3, dtype=torch.bool)
    max_cells, radius = estimate_cell_list_sizes(
        cell, pbc, 0.5, min_cells_per_dimension=1
    )
    scratch = allocate_cell_list(2, max_cells, radius, positions.device)
    build_cell_list(positions, 0.5, cell, pbc, *scratch, min_cells_per_dimension=1)
    matrix = torch.full((2, 4), 2, dtype=torch.int32)
    shifts = torch.zeros((2, 4, 3), dtype=torch.int32)
    counts = torch.zeros(2, dtype=torch.int32)
    distances = torch.zeros((2, 4), dtype=torch.float64)
    vectors = torch.zeros((2, 4, 3), dtype=torch.float64)
    query_cell_list(
        positions,
        0.5,
        cell,
        pbc,
        *scratch,
        matrix,
        shifts,
        counts,
        return_distances=True,
        return_vectors=True,
        neighbor_distances=distances,
        neighbor_vectors=vectors,
    )
    expected = neighbor_list(
        positions,
        0.5,
        cell=cell,
        pbc=pbc,
        max_neighbors=4,
        return_distances=True,
        return_vectors=True,
    )
    for got, want in zip((matrix, counts, shifts, distances, vectors), expected):
        torch.testing.assert_close(got, want)

    saved = matrix.clone()
    query_cell_list(
        positions + 0.25,
        0.5,
        cell,
        pbc,
        *scratch,
        matrix,
        shifts,
        counts,
        rebuild_flags=torch.tensor(False),
    )
    assert torch.equal(matrix, saved)


@pytest.mark.parametrize("periodic", (False, True))
@pytest.mark.parametrize("half_fill", (False, True))
def test_query_canonicalizes_permuted_atom_order_within_each_cell(
    periodic: bool, half_fill: bool
) -> None:
    """Public query output is independent of an internal cell-list permutation."""
    positions = torch.tensor(
        [
            [0.11, 0.12, 0.13],
            [0.19, 0.18, 0.17],
            [0.28, 0.24, 0.20],
            [0.35, 0.31, 0.27],
            [0.42, 0.38, 0.34],
        ],
        dtype=torch.float64,
    )
    cell = torch.eye(3, dtype=torch.float64) * 2.0
    pbc = torch.full((3,), periodic, dtype=torch.bool)
    max_cells, radius = estimate_cell_list_sizes(
        cell, pbc, 0.45, min_cells_per_dimension=1
    )
    scratch = allocate_cell_list(positions.shape[0], max_cells, radius, positions.device)
    build_cell_list(positions, 0.45, cell, pbc, *scratch, min_cells_per_dimension=1)
    shifts, mapping, counts, starts, atom_list = scratch[2:]
    del mapping
    permuted = atom_list.clone()
    for count, start in zip(counts.tolist(), starts.tolist(), strict=True):
        if count > 1:
            permuted[start : start + count] = torch.flip(
                permuted[start : start + count], dims=(0,)
            )

    def query(atom_list_input: torch.Tensor) -> tuple[torch.Tensor, ...]:
        matrix = torch.full((positions.shape[0], 32), positions.shape[0], dtype=torch.int32)
        matrix_shifts = torch.zeros((*matrix.shape, 3), dtype=torch.int32)
        neighbor_counts = torch.zeros(positions.shape[0], dtype=torch.int32)
        distances = torch.zeros(matrix.shape, dtype=positions.dtype)
        vectors = torch.zeros((*matrix.shape, 3), dtype=positions.dtype)
        query_cell_list(
            positions,
            0.45,
            cell,
            pbc,
            scratch[0],
            scratch[1],
            shifts,
            scratch[3],
            counts,
            starts,
            atom_list_input,
            matrix,
            matrix_shifts,
            neighbor_counts,
            half_fill=half_fill,
            return_distances=True,
            return_vectors=True,
            neighbor_distances=distances,
            neighbor_vectors=vectors,
        )
        return matrix, neighbor_counts, matrix_shifts, distances, vectors

    expected = query(atom_list)
    actual = query(permuted)
    for actual_value, expected_value in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_value, expected_value, atol=1e-12, rtol=1e-12)


def test_batch_selective_rebuild_preserves_unselected_system() -> None:
    positions = torch.tensor(
        [[0.1, 0.0, 0.0], [1.9, 0.0, 0.0], [0.1, 0.0, 0.0], [1.9, 0.0, 0.0]],
        dtype=torch.float64,
    )
    cells = torch.eye(3, dtype=torch.float64).repeat(2, 1, 1) * 2.0
    pbc = torch.ones((2, 3), dtype=torch.bool)
    batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
    matrix, counts, shifts = batch_cell_list(
        positions, 0.5, cells, pbc, batch_idx, max_neighbors=4
    )
    system_one = matrix[2:].clone(), counts[2:].clone(), shifts[2:].clone()
    moved = positions.clone()
    moved[0, 0] = 0.8
    batch_cell_list(
        moved,
        0.5,
        cells,
        pbc,
        batch_idx,
        neighbor_matrix=matrix,
        neighbor_matrix_shifts=shifts,
        num_neighbors=counts,
        rebuild_flags=torch.tensor([True, False]),
    )
    assert torch.equal(matrix[2:], system_one[0])
    assert torch.equal(counts[2:], system_one[1])
    assert torch.equal(shifts[2:], system_one[2])


def test_periodic_vectors_and_distances_retain_first_and_second_gradients() -> None:
    positions = torch.tensor(
        [[0.1, 0.0, 0.0], [1.9, 0.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    cell = (torch.eye(3, dtype=torch.float64) * 2.0).requires_grad_()
    matrix, counts, _shifts, distances, vectors = neighbor_list(
        positions,
        0.5,
        cell=cell,
        pbc=torch.ones(3, dtype=torch.bool),
        return_distances=True,
        return_vectors=True,
    )
    active = torch.arange(matrix.shape[1])[None, :] < counts.to(torch.long)[:, None]
    value = distances[active].square().sum() + 0.25 * vectors[active].square().sum()
    grad_positions, grad_cell = torch.autograd.grad(
        value, (positions, cell), create_graph=True
    )
    second = torch.autograd.grad(
        grad_positions.square().sum() + grad_cell.square().sum(),
        (positions, cell),
    )
    assert all(torch.isfinite(value).all() for value in second)


def test_zero_volume_cell_and_deferred_features_fail_explicitly() -> None:
    positions = torch.zeros((1, 3), dtype=torch.float64)
    with pytest.raises(RuntimeError, match="volume == 0"):
        neighbor_list(
            positions,
            1.0,
            cell=torch.zeros((3, 3), dtype=torch.float64),
            pbc=torch.ones(3, dtype=torch.bool),
        )
    with pytest.raises(NotImplementedError, match="target_indices"):
        neighbor_list(positions, 1.0, target_indices=torch.tensor([0]))
