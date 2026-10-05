# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Current-cell snapshot oracle and real-HIP regression (no mocked kernels)."""

from __future__ import annotations

import itertools
from collections import Counter

import pytest
import torch

from nvalchemiops import _hip_neighbor_executor as hip
from nvalchemiops.torch_reference import NeighborOverflowError
from nvalchemiops.torch_reference_cell_list import neighbor_list as reference


SNAPSHOTS = ("initial", "shrink", "expand", "shear", "wrap", "multi_image", "restore")
requires_hip = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(),
    reason="requires an explicitly allocated visible HIP device",
)


def _snapshot(name: str, dtype: torch.dtype, device: str = "cpu"):
    cells = torch.tensor(
        [[[10.5, 0, 0], [0.3, 9.5, 0], [0.1, 0.4, 8.5]],
         [[8.5, 0.2, 0], [0.4, 7.5, 0.1], [0.2, 0.1, 6.5]]],
        dtype=dtype, device=device,
    )
    fractions = torch.tensor(
        [[0.06, 0.07, 0.1], [0.21, 0.13, 0.12], [0.93, 0.08, 0.09],
         [0.16, 0.12, 0.17], [0.81, 0.19, 0.21]], dtype=dtype, device=device,
    )
    if name == "shrink":
        cells *= 0.37
    elif name == "expand":
        cells *= 1.8
    elif name in {"shear", "wrap"}:
        cells = torch.tensor(
            [[[7.8, 0.8, 0.2], [1.1, 6.4, 0.3], [0.5, 0.7, 5.9]],
             [[4.2, 1.2, 0.4], [0.8, 5.1, 0.3], [0.2, 0.7, 3.9]]],
            dtype=dtype, device=device,
        )
        if name == "wrap":
            fractions[0] += fractions.new_tensor([1, -1, 0])
            fractions[3] += fractions.new_tensor([-1, 0, 2])
    elif name == "multi_image":
        cells = torch.tensor(
            [[[1.1, 0, 0], [0.15, 1.6, 0], [0.1, 0.2, 1.8]],
             [[1.5, 0.15, 0.1], [0.2, 1.2, 0.1], [0.1, 0.2, 1.7]]],
            dtype=dtype, device=device,
        )
    elif name not in {"initial", "restore"}:
        raise ValueError(name)
    idx = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int32, device=device)
    ptr = torch.tensor([0, 3, 5], dtype=torch.int32, device=device)
    positions = torch.bmm(fractions.unsqueeze(1), cells[idx.long()]).squeeze(1)
    pbc = torch.ones((2, 3), dtype=torch.bool, device=device)
    return positions, cells, pbc, idx, ptr


def _edges(result):
    matrix, counts, shifts = (value.detach().cpu() for value in result)
    entries = []
    for i, count in enumerate(counts.tolist()):
        for k in range(count):
            entries.append((i, int(matrix[i, k]), *shifts[i, k].tolist()))
    return Counter(entries)


def _enumerated_edges(positions, cells, ptr, cutoff=2.4):
    """Independent finite mirror enumeration; no reference grid/search helpers."""
    images = torch.tensor(list(itertools.product(range(-6, 7), repeat=3)), dtype=cells.dtype)
    entries = []
    for b, (lo, hi) in enumerate(zip(ptr[:-1].tolist(), ptr[1:].tolist())):
        r = positions[lo:hi]
        frac = r @ torch.linalg.inv(cells[b])
        # Prove the fixture's finite image box covers every possible hit.
        span = frac.amax(0) - frac.amin(0)
        bound = span + cutoff * torch.linalg.vector_norm(torch.linalg.inv(cells[b]), dim=0)
        assert bool(torch.all(bound < 6))
        dr = r[None, :, None, :] - r[:, None, None, :] + images @ cells[b]
        d2 = dr.square().sum(-1)
        active = (d2 > 0) & (d2 < cutoff * cutoff)
        # Fixture avoids cutoff ties, including float32-sensitive comparisons.
        assert float((d2 - cutoff * cutoff).abs().min()) > 1e-4
        for i, j, image in active.nonzero().tolist():
            entries.append((lo + i, lo + j, *images[image].to(torch.int32).tolist()))
    return Counter(entries)


def _run(executor, inputs, capacity=None):
    positions, cells, pbc, idx, ptr = inputs
    return executor(
        positions, 2.4, cell=cells, pbc=pbc, batch_idx=idx, batch_ptr=ptr,
        max_neighbors=capacity,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("name", SNAPSHOTS)
def test_reference_current_cell_snapshots_against_independent_images(name, dtype):
    inputs = _snapshot(name, dtype)
    actual = _edges(_run(reference, inputs))
    assert actual == _enumerated_edges(inputs[0], inputs[1], inputs[4])
    if name == "multi_image":
        assert any(i == j and max(abs(s) for s in shift) >= 2 for i, j, *shift in actual)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_current_cell_metadata_refreshes_after_in_place_cell_change(dtype):
    cells = _snapshot("initial", dtype)[1]
    first = hip._current_cell_metadata(cells, 2.4)
    cells.copy_(_snapshot("multi_image", dtype)[1])
    second = hip._current_cell_metadata(cells, 2.4)
    assert not torch.equal(first[0], second[0])  # grid dimensions
    assert not torch.equal(first[1], second[1])  # search radius
    assert not torch.equal(first[3], second[3])  # inverse cell
    torch.testing.assert_close(second[3] @ cells, torch.eye(3, dtype=dtype).expand(2, 3, 3))
    assert second[2][0] == 0
    assert second[2][1] == second[0][0].prod()


@requires_hip
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("nondefault_stream", [False, True])
def test_hip_sequence_uses_current_geometry_and_preserves_inputs(dtype, nondefault_stream):
    inputs = _snapshot("initial", dtype, "cuda")
    stream = torch.cuda.Stream() if nondefault_stream else torch.cuda.current_stream()
    # The inputs were created on the default stream.
    stream.wait_stream(torch.cuda.current_stream())
    previous = None
    initial_capacity = None
    with torch.cuda.stream(stream):
        for name in SNAPSHOTS:
            current = _snapshot(name, dtype, "cuda")
            for target, source in zip(inputs, current):
                target.copy_(source)  # same tensor identity across cell changes
            before = tuple(value.clone() for value in inputs)
            result = _run(hip.neighbor_list, inputs)
            expected = _run(reference, inputs)
            assert _edges(result) == _edges(expected), name
            torch.testing.assert_close(result[1], expected[1])
            for value, saved in zip(inputs, before):
                torch.testing.assert_close(value, saved, rtol=0, atol=0)
            if previous is not None:
                old_result, old_saved = previous
                for value, saved in zip(old_result, old_saved):
                    torch.testing.assert_close(value, saved, rtol=0, atol=0)
            columns = torch.arange(result[0].shape[1], device="cuda")
            padding = columns[None, :] >= result[1][:, None]
            assert bool(torch.all(result[0][padding] == inputs[0].shape[0]))
            assert bool(torch.all(result[2][padding] == 0))
            if name == "initial":
                initial_capacity = result[0].shape[1]
            if name == "multi_image":
                assert int(result[1].max()) > initial_capacity
            previous = result, tuple(value.clone() for value in result)
    stream.synchronize()


@requires_hip
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_hip_cell_contraction_explicit_overflow_auto_growth_and_recovery(dtype):
    initial = _snapshot("initial", dtype, "cuda")
    contracted = _snapshot("multi_image", dtype, "cuda")
    initial_result = _run(hip.neighbor_list, initial)
    capacity = initial_result[0].shape[1]
    before = tuple(value.clone() for value in contracted)
    with pytest.raises(NeighborOverflowError, match="capacity overflow"):
        _run(hip.neighbor_list, contracted, capacity)
    for value, saved in zip(contracted, before):
        torch.testing.assert_close(value, saved, rtol=0, atol=0)
    result = _run(hip.neighbor_list, contracted)
    assert int(result[1].max()) > capacity
    assert _edges(result) == _edges(_run(reference, contracted))
    recovered = _run(hip.neighbor_list, initial)
    assert _edges(recovered) == _edges(initial_result)


@requires_hip
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("nondefault_stream", [False, True])
def test_hip_dispatch_coo_matches_reference_snapshots_and_empty(dtype, nondefault_stream):
    from nvalchemiops.dispatch import dispatch_neighbor_list

    stream = torch.cuda.Stream() if nondefault_stream else torch.cuda.current_stream()
    with torch.cuda.stream(stream):
        for name in SNAPSHOTS + ("empty",):
            inputs = _snapshot("initial" if name == "empty" else name, dtype, "cuda")
            r, h, pbc, idx, ptr = inputs
            if name == "empty":
                r, idx, ptr = r[:0], idx[:0], torch.zeros_like(ptr)
            before = tuple(v.clone() for v in (r, h, pbc, idx, ptr))
            kwargs = dict(cell=h, pbc=pbc, batch_idx=idx, batch_ptr=ptr, return_neighbor_list=True)
            result, selection = dispatch_neighbor_list(
                r, 2.4, **kwargs, backend="hip", method="cell_list", return_backend=True,
            )
            expected = reference(r, 2.4, **kwargs)
            edges, row_ptr, shifts = result
            def signature(values):
                return Counter(tuple(row) for row in torch.cat((values[0].T, values[2]), 1).cpu().tolist())
            assert signature(result) == signature(expected), name
            torch.testing.assert_close(row_ptr, expected[1])
            assert row_ptr.dtype == torch.int64
            assert edges.shape == (2, int(row_ptr[-1]))
            assert shifts.shape == (edges.shape[1], 3)
            assert edges.dtype == shifts.dtype == torch.int32
            assert "coo" in selection.features and selection.family == "hip"
            assert all(v.device == r.device for v in result)
            for value, saved in zip((r, h, pbc, idx, ptr), before):
                torch.testing.assert_close(value, saved, rtol=0, atol=0)
            if name == "multi_image":
                with pytest.raises(NeighborOverflowError, match="capacity overflow"):
                    dispatch_neighbor_list(r, 2.4, **kwargs, max_neighbors=2, backend="hip")
    stream.synchronize()
