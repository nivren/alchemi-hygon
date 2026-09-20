# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for the shared stage-two cell-key build ABI."""

from __future__ import annotations

import pytest
import torch

from nvalchemiops._cell_list_abi import (
    build_batch_cell_key_counts_reference_into,
    build_cell_atom_list_reference_into,
    build_cell_counts_reference_into,
    build_cell_csr_reference,
    build_cell_csr_reference_into,
    build_cell_keys_reference,
    build_cell_keys_reference_into,
    build_cell_starts_reference_into,
)


def test_batch_cell_key_count_reference_matches_concatenated_system_oracles() -> None:
    cells = torch.tensor(
        [
            [[4.0, 0.0, 0.0], [0.7, 3.5, 0.0], [0.4, 0.6, 4.2]],
            [[3.0, 0.0, 0.0], [0.2, 2.5, 0.0], [0.1, 0.3, 3.1]],
        ],
        dtype=torch.float64,
    )
    positions = torch.tensor(
        [[4.7, 4.1, -1.0], [0.2, 0.3, 1.7], [3.2, -0.4, 0.2]],
        dtype=torch.float64,
    )
    dimensions = torch.tensor([[4, 5, 6], [3, 4, 2]], dtype=torch.int32)
    pbc = torch.tensor([[True, False, True], [False, True, False]])
    batch_idx = torch.tensor([1, 0, 1], dtype=torch.int32)
    offsets = torch.tensor([0, 120], dtype=torch.int32)
    shifts = torch.full((3, 3), -9, dtype=torch.int32)
    mapping = torch.full((3, 3), -9, dtype=torch.int32)
    keys = torch.full((3,), -9, dtype=torch.int32)
    counts = torch.full((144,), -9, dtype=torch.int32)

    build_batch_cell_key_counts_reference_into(
        positions,
        torch.linalg.inv(cells),
        dimensions,
        pbc,
        batch_idx,
        offsets,
        shifts,
        mapping,
        keys,
        counts,
    )

    expected_shifts = torch.empty_like(shifts)
    expected_mapping = torch.empty_like(mapping)
    expected_keys = torch.empty_like(keys)
    expected_counts = torch.zeros_like(counts)
    for system, offset in enumerate(offsets.tolist()):
        atom_indices = torch.nonzero(batch_idx == system, as_tuple=False).flatten()
        result = build_cell_keys_reference(
            positions[atom_indices],
            torch.linalg.inv(cells[system]),
            dimensions[system],
            pbc[system],
        )
        expected_shifts[atom_indices] = result.atom_periodic_shifts
        expected_mapping[atom_indices] = result.atom_to_cell_mapping
        expected_keys[atom_indices] = result.cell_keys + offset
        build_cell_counts_reference_into(
            result.cell_keys,
            expected_counts[offset : offset + int(dimensions[system].prod())],
        )
    assert torch.equal(shifts, expected_shifts)
    assert torch.equal(mapping, expected_mapping)
    assert torch.equal(keys, expected_keys)
    assert torch.equal(counts, expected_counts)


def test_batch_cell_key_count_reference_rejects_noncontiguous_cell_offsets() -> None:
    tensors = dict(
        positions=torch.empty((0, 3), dtype=torch.float32),
        inverse_cells=torch.eye(3, dtype=torch.float32).unsqueeze(0),
        cells_per_dimension=torch.tensor([[2, 2, 2]], dtype=torch.int32),
        pbc=torch.ones((1, 3), dtype=torch.bool),
        batch_idx=torch.empty(0, dtype=torch.int32),
        cell_offsets=torch.tensor([1], dtype=torch.int32),
        atom_periodic_shifts=torch.empty((0, 3), dtype=torch.int32),
        atom_to_cell_mapping=torch.empty((0, 3), dtype=torch.int32),
        cell_keys=torch.empty(0, dtype=torch.int32),
        cell_counts=torch.empty(8, dtype=torch.int32),
    )
    with pytest.raises(ValueError, match="contiguous global"):
        build_batch_cell_key_counts_reference_into(**tensors)


def test_cell_key_build_matches_triclinic_mixed_pbc_contract() -> None:
    cell = torch.tensor(
        [[4.0, 0.0, 0.0], [0.7, 3.5, 0.0], [0.4, 0.6, 4.2]],
        dtype=torch.float64,
    )
    fractional = torch.tensor(
        [[-0.1, 1.25, -0.25], [1.1, -0.2, 0.6]], dtype=torch.float64
    )
    result = build_cell_keys_reference(
        fractional @ cell,
        torch.linalg.inv(cell),
        torch.tensor([4, 5, 6], dtype=torch.int32),
        torch.tensor([True, False, True]),
    )
    assert result.atom_periodic_shifts.tolist() == [[-1, 0, -1], [1, 0, 0]]
    assert result.atom_to_cell_mapping.tolist() == [[3, 4, 4], [0, 0, 3]]
    assert result.cell_keys.tolist() == [99, 60]


def test_cell_key_build_into_mutates_only_declared_outputs() -> None:
    positions = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
    inverse = torch.eye(3, dtype=torch.float32)
    dimensions = torch.tensor([4, 4, 4], dtype=torch.int32)
    pbc = torch.tensor([True, True, True])
    shifts = torch.full((1, 3), -9, dtype=torch.int32)
    mapping = torch.full((1, 3), -9, dtype=torch.int32)
    keys = torch.full((1,), -9, dtype=torch.int32)
    build_cell_keys_reference_into(
        positions, inverse, dimensions, pbc, shifts, mapping, keys
    )
    assert shifts.tolist() == [[0, 0, 0]]
    assert mapping.tolist() == [[0, 0, 1]]
    assert keys.tolist() == [16]


def test_cell_key_csr_count_and_exclusive_starts_contract() -> None:
    result = build_cell_csr_reference(
        torch.tensor([3, 0, 3, 1], dtype=torch.int32),
        5,
        global_atom_offset=10,
    )
    assert result.cell_counts.tolist() == [1, 1, 0, 2, 0]
    assert result.cell_starts.tolist() == [10, 11, 12, 12, 14]


def test_cell_key_count_overwrites_the_complete_active_buffer() -> None:
    counts = torch.full((5,), -1, dtype=torch.int32)
    build_cell_counts_reference_into(
        torch.tensor([3, 0, 3, 1], dtype=torch.int32), counts
    )
    assert counts.tolist() == [1, 1, 0, 2, 0]


def test_cell_starts_reference_is_batch_slice_agnostic_and_preserves_counts() -> None:
    counts = torch.tensor([1, 1, 0, 2, 0, 3], dtype=torch.int32)
    starts = torch.full_like(counts, -1)
    build_cell_starts_reference_into(counts, starts, global_atom_offset=10)
    assert counts.tolist() == [1, 1, 0, 2, 0, 3]
    assert starts.tolist() == [10, 11, 12, 12, 14, 14]


def test_cell_starts_reference_rejects_negative_counts_and_index_overflow() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        build_cell_starts_reference_into(
            torch.tensor([1, -1], dtype=torch.int32),
            torch.empty(2, dtype=torch.int32),
        )
    with pytest.raises(ValueError, match="range exceeds int32"):
        build_cell_starts_reference_into(
            torch.tensor([1], dtype=torch.int32),
            torch.empty(1, dtype=torch.int32),
            global_atom_offset=torch.iinfo(torch.int32).max,
        )


def test_cell_atom_list_fill_preserves_stable_order_and_public_csr_buffers() -> None:
    keys = torch.tensor([2, 0, 2, 1], dtype=torch.int32)
    counts = torch.tensor([1, 1, 2], dtype=torch.int32)
    starts = torch.tensor([10, 11, 12], dtype=torch.int32)
    counts_before = counts.clone()
    starts_before = starts.clone()
    atom_list = torch.full((6,), -9, dtype=torch.int32)
    cursor = torch.full_like(counts, -7)

    build_cell_atom_list_reference_into(
        keys,
        counts,
        starts,
        atom_list,
        cursor,
        global_atom_offset=10,
    )

    assert atom_list.tolist() == [11, 13, 10, 12, -9, -9]
    assert cursor.tolist() == [1, 1, 2]
    assert torch.equal(counts, counts_before)
    assert torch.equal(starts, starts_before)


def test_cell_atom_list_fill_empty_input_and_invalid_capacity() -> None:
    counts = torch.zeros(2, dtype=torch.int32)
    starts = torch.full_like(counts, 7)
    atom_list = torch.full((1,), -9, dtype=torch.int32)
    cursor = torch.full_like(counts, -7)
    build_cell_atom_list_reference_into(
        torch.empty(0, dtype=torch.int32),
        counts,
        starts,
        atom_list,
        cursor,
        global_atom_offset=7,
    )
    assert atom_list.tolist() == [-9]
    assert cursor.tolist() == [0, 0]
    with pytest.raises(ValueError, match="capacity"):
        build_cell_atom_list_reference_into(
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([1, 1], dtype=torch.int32),
            torch.tensor([0, 1], dtype=torch.int32),
            torch.empty(1, dtype=torch.int32),
            torch.empty(2, dtype=torch.int32),
        )


def test_cell_key_csr_empty_input_writes_zero_counts_and_offset_starts() -> None:
    counts = torch.full((3,), -1, dtype=torch.int32)
    starts = torch.full((3,), -1, dtype=torch.int32)
    build_cell_csr_reference_into(
        torch.empty(0, dtype=torch.int32),
        counts,
        starts,
        global_atom_offset=7,
    )
    assert counts.tolist() == [0, 0, 0]
    assert starts.tolist() == [7, 7, 7]


@pytest.mark.parametrize(
    ("dimensions", "pbc", "message"),
    [
        (torch.tensor([4, 4, 4], dtype=torch.int64), torch.ones(3, dtype=torch.bool), "int32"),
        (torch.tensor([4, 0, 4], dtype=torch.int32), torch.ones(3, dtype=torch.bool), "positive"),
        (torch.tensor([4, 4, 4], dtype=torch.int32), torch.ones(3, dtype=torch.int32), "bool"),
        (torch.tensor([46341, 46341, 46341], dtype=torch.int32), torch.ones(3, dtype=torch.bool), "int32"),
    ],
)
def test_cell_key_build_rejects_invalid_abi_inputs(
    dimensions: torch.Tensor, pbc: torch.Tensor, message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        build_cell_keys_reference(
            torch.zeros((1, 3), dtype=torch.float32),
            torch.eye(3, dtype=torch.float32),
            dimensions,
            pbc,
        )


@pytest.mark.parametrize(
    ("keys", "total_cells", "message"),
    [
        (torch.tensor([-1], dtype=torch.int32), 2, "cell_keys"),
        (torch.tensor([2], dtype=torch.int32), 2, "cell_keys"),
        (torch.tensor([0], dtype=torch.int64), 2, "int32"),
    ],
)
def test_cell_key_csr_rejects_invalid_keys(
    keys: torch.Tensor, total_cells: int, message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        build_cell_csr_reference(keys, total_cells)
