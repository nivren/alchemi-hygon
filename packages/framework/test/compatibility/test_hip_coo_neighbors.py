# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic COO adapter oracle and actual HIP public/Hook integration."""

from collections import Counter

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.hooks import NeighborListHook
from nvalchemi.hooks._context import HookContext
from nvalchemi.models.base import NeighborConfig, NeighborListFormat
from nvalchemi.neighbors import _neighbor_matrix_to_coo, _write_neighbor_data_to_batch, compute_neighbors


def _batch(dtype=torch.float64, device="cpu"):
    systems = []
    for n in (3, 1):
        h = torch.eye(3, dtype=dtype, device=device) * 9
        s = torch.tensor([[0.06, 0.07, 0.1], [0.21, 0.13, 0.12], [0.93, 0.08, 0.09]], dtype=dtype, device=device)[:n]
        systems.append(AtomicData(
            positions=s @ h, atomic_numbers=torch.ones(n, dtype=torch.int64, device=device),
            cell=h[None], pbc=torch.ones((1, 3), dtype=torch.bool, device=device),
        ))
    return Batch.from_data_list(systems, device=device)


@pytest.mark.parametrize("empty", [False, True])
def test_torch_coo_adapter_preserves_images_padding_and_batch_segments(empty):
    batch = _batch()
    matrix = torch.tensor([[1, 1, 4], [0, 4, 4], [4, 4, 4], [3, 3, 4]], dtype=torch.int32)
    counts = torch.tensor([2, 1, 0, 2], dtype=torch.int32)
    shifts = torch.zeros((4, 3, 3), dtype=torch.int32)
    shifts[0, 1, 0] = 1
    shifts[3, 0, 0], shifts[3, 1, 0] = -2, 2
    if empty:
        counts.zero_()
    saved = tuple(v.clone() for v in (matrix, counts, shifts))
    _write_neighbor_data_to_batch(batch, matrix, counts, shifts, NeighborListFormat.COO, 2.4, backend="hip")
    expected = [] if empty else [(0, 1, 0, 0, 0), (0, 1, 1, 0, 0), (1, 0, 0, 0, 0), (3, 3, -2, 0, 0), (3, 3, 2, 0, 0)]
    actual = torch.cat((batch.neighbor_list, batch.neighbor_list_shifts), dim=1)
    assert actual.tolist() == [list(row) for row in expected]
    assert batch._storage.groups["edges"].segment_lengths.tolist() == ([0, 0] if empty else [3, 2])
    assert batch.neighbor_list.dtype == batch.neighbor_list_shifts.dtype == torch.int32
    for value, old in zip((matrix, counts, shifts), saved):
        torch.testing.assert_close(value, old, rtol=0, atol=0)
    # A rebuilt edge group must leave references to earlier outputs intact.
    old_edges = batch.neighbor_list.clone()
    previous = batch.neighbor_list
    _write_neighbor_data_to_batch(batch, matrix, torch.zeros_like(counts), shifts, NeighborListFormat.COO, 2.4, backend="hip")
    torch.testing.assert_close(previous, old_edges)
    assert batch.neighbor_list.shape == (0, 2)


def test_torch_coo_adapter_zero_atoms():
    result, shifts = _neighbor_matrix_to_coo(
        torch.empty((0, 0), dtype=torch.int32), torch.empty(0, dtype=torch.int32),
        torch.empty((0, 0, 3), dtype=torch.int32),
    )
    assert result.shape == (2, 0) and shifts.shape == (0, 3)


@pytest.mark.parametrize("periodic", [False, True])
def test_hook_can_resize_overallocated_matrix_without_losing_active_neighbors(periodic):
    from nvalchemiops.torch_reference import NeighborOverflowError

    matrix = torch.full((2, 80), 2, dtype=torch.int32)
    counts = torch.tensor([54, 51], dtype=torch.int32)
    matrix[0, :54], matrix[1, :51] = 1, 0
    shifts = torch.arange(2 * 80 * 3, dtype=torch.int32).reshape(2, 80, 3)
    result = (matrix, counts, shifts) if periodic else (matrix, counts)
    pbc = torch.ones((1, 3), dtype=torch.bool) if periodic else None
    resized = NeighborListHook._pad_reference_result(result, pbc=pbc, capacity=64, fill_value=2)
    assert resized[0].shape == (2, 64)
    torch.testing.assert_close(resized[0], matrix[:, :64])
    torch.testing.assert_close(resized[1], counts)
    if periodic:
        torch.testing.assert_close(resized[2], shifts[:, :64])
    with pytest.raises(NeighborOverflowError, match="capacity"):
        NeighborListHook._pad_reference_result(result, pbc=pbc, capacity=32, fill_value=2)


@pytest.mark.skipif(not torch.version.hip or not torch.cuda.is_available(), reason="requires allocated HIP device")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("entrypoint", ["compute", "hook"])
@pytest.mark.parametrize("nondefault_stream", [False, True])
def test_hip_public_coo_cell_changes_capacity_and_stream(dtype, entrypoint, nondefault_stream):
    batch, expected = _batch(dtype, "cuda"), _batch(dtype, "cuda")
    fractions = batch.positions / 9
    config = NeighborConfig(cutoff=2.4, format=NeighborListFormat.COO, skin=0)
    hook = NeighborListHook(config, backend="hip", method="cell_list")
    stream = torch.cuda.Stream() if nondefault_stream else torch.cuda.current_stream()
    stream.wait_stream(torch.cuda.current_stream())
    initial_capacity = None
    with torch.cuda.stream(stream):
        for name in ("initial", "contract", "shear", "wrap", "no_edges", "restore"):
            cells = torch.eye(3, dtype=dtype, device="cuda").expand(2, 3, 3).clone() * 9
            s = fractions.clone()
            if name in {"contract", "shear", "wrap"}:
                cells *= 0.15
                cells[1] *= 1.2
            if name in {"shear", "wrap"}:
                cells[:, 1, 0], cells[:, 2, 0], cells[:, 0, 2] = 0.3, 0.2, 0.1
            if name == "wrap":
                s[0] += s.new_tensor([1, -1, 2])
            if name == "no_edges":
                cells *= 4
            positions = torch.bmm(s[:, None], cells[batch.batch_idx.long()]).squeeze(1)
            for target in (batch, expected):
                target.cell.copy_(cells)
                target.positions.copy_(positions)
            before = batch.positions.clone(), batch.cell.clone()
            if entrypoint == "hook":
                hook(HookContext(batch=batch), None)
            else:
                compute_neighbors(batch, config=config, backend="hip", method="cell_list")
            compute_neighbors(expected, config=config, backend="torch_reference", method="cell_list")
            def signature(data):
                return Counter(tuple(row) for row in torch.cat((data.neighbor_list, data.neighbor_list_shifts), 1).cpu().tolist())
            assert signature(batch) == signature(expected), name
            e = batch.neighbor_list.long()
            assert torch.equal(batch.batch_idx[e[:, 0]], batch.batch_idx[e[:, 1]])
            torch.testing.assert_close(batch._storage.groups["edges"].segment_lengths, expected._storage.groups["edges"].segment_lengths)
            torch.testing.assert_close(batch.positions, before[0], rtol=0, atol=0)
            torch.testing.assert_close(batch.cell, before[1], rtol=0, atol=0)
            counts = torch.bincount(e[:, 0], minlength=batch.num_nodes)
            if name == "initial":
                initial_capacity = hook._max_neighbors if entrypoint == "hook" else int(counts.max())
            if name == "contract":
                assert int(counts.max()) > initial_capacity
            if name == "no_edges":
                assert e.shape == (0, 2)
    stream.synchronize()
