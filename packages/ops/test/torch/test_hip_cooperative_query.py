# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cooperative query row ownership, canonical ordering and overflow gates."""
from __future__ import annotations

from collections import Counter
from itertools import product

import pytest
import torch

from nvalchemiops._hip_batch_cell_query import batch_query_cell_list_hip_into
from nvalchemiops._hip_neighbor_executor import neighbor_list
from nvalchemiops.torch_reference import NeighborOverflowError


requires_hip = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(),
    reason="requires an explicitly allocated visible HIP device",
)


def _fixture(n, dtype):
    generator = torch.Generator().manual_seed(20261006)
    cell = torch.tensor([[8.3, 0, 0], [0.3, 7.7, 0], [0.1, 0.2, 8.1]], dtype=dtype)
    positions = torch.rand(n, 3, generator=generator, dtype=dtype) @ cell
    return positions, cell[None]


def _oracle(positions, cells):
    images = torch.tensor(list(product(range(-1, 2), repeat=3)), dtype=positions.dtype)
    # Fractional span<1 and cutoff * reciprocal norm<0.32: +/-1 covers this fixture.
    vectors = positions[:, None, None] - positions[None, :, None] - images @ cells[0]
    d2 = vectors.square().sum(-1)
    assert float((d2 - 2.4**2).abs().min()) > 1e-5
    valid = (d2 > 0) & (d2 < 2.4**2)
    return Counter((i, j, *images[s].int().tolist()) for i, j, s in valid.nonzero().tolist())


@pytest.mark.parametrize("n", [67, 129])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cooperative_fixture_oracle(n, dtype):
    """The independent finite-image fixture has no cutoff ties or duplicates."""
    edges = _oracle(*_fixture(n, dtype))
    assert edges and all(count == 1 for count in edges.values())


@requires_hip
@pytest.mark.parametrize("n", [67, 129])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cooperative_public_rows_are_exact_repeatable_and_overflow(n, dtype):
    """Non-round atom counts preserve rows despite parallel candidate insertion."""
    cpu = _fixture(n, dtype)
    expected = _oracle(*cpu)
    positions, cells = (value.cuda() for value in cpu)
    pbc = torch.ones((1, 3), dtype=torch.bool, device="cuda")
    previous = None
    for _ in range(4):
        result = neighbor_list(positions, 2.4, cell=cells, pbc=pbc, max_neighbors=128)
        matrix, counts, shifts = (value.cpu() for value in result)
        edges = Counter((i, int(matrix[i, k]), *shifts[i, k].tolist())
                        for i, count in enumerate(counts.tolist()) for k in range(count))
        assert edges == expected
        if previous is not None:
            for actual, saved in zip(result, previous):
                assert torch.equal(actual, saved)
        previous = tuple(value.clone() for value in result)
    with pytest.raises(NeighborOverflowError, match="capacity overflow"):
        neighbor_list(positions, 2.4, cell=cells, pbc=pbc, max_neighbors=1)


@requires_hip
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cooperative_offset_product_overflow_is_explicit(dtype):
    """Huge valid int32 radii must fail before flattened int64 iteration wraps."""
    device = "cuda"
    zeros3 = torch.zeros((1, 3), dtype=torch.int32, device=device)
    zeros1 = torch.zeros(1, dtype=torch.int32, device=device)
    ones3 = torch.ones_like(zeros3)
    with pytest.raises(RuntimeError, match="search-offset count exceeds int64"):
        batch_query_cell_list_hip_into(
            torch.zeros((1, 3), dtype=dtype, device=device),
            torch.eye(3, dtype=dtype, device=device)[None],
            torch.ones((1, 3), dtype=torch.bool, device=device),
            0.5, zeros1, ones3, torch.full_like(ones3, 1 << 30), zeros1,
            zeros3, zeros3, torch.ones_like(zeros1), zeros1, zeros1,
            torch.empty((1, 4), dtype=torch.int32, device=device),
            torch.empty((1, 4, 3), dtype=torch.int32, device=device),
            torch.empty_like(zeros1),
        )
