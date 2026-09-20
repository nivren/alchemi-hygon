# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU boundary tests for the isolated native HIP geometry candidate."""

from __future__ import annotations

import sys

import pytest
import torch

from nvalchemiops._hip_batch_query_geometry import (
    materialize_batch_query_geometry_hip_into,
)


def _empty_case() -> tuple[torch.Tensor, ...]:
    positions = torch.empty((0, 3), dtype=torch.float32)
    cells = torch.eye(3, dtype=torch.float32).unsqueeze(0)
    batch_idx = torch.empty(0, dtype=torch.int32)
    public = torch.empty((0, 4), dtype=torch.int32)
    shifts = torch.empty((0, 4, 3), dtype=torch.int32)
    counts = torch.empty(0, dtype=torch.int32)
    distances = torch.empty((0, 4), dtype=torch.float32)
    vectors = torch.empty((0, 4, 3), dtype=torch.float32)
    return positions, cells, batch_idx, public, shifts, counts, distances, vectors


def test_native_hip_batch_query_geometry_boundary_does_not_compile_on_import() -> None:
    assert "nvalchemiops._hip_batch_query_geometry" in sys.modules


def test_native_hip_batch_query_geometry_rejects_cpu_without_fallback() -> None:
    with pytest.raises(RuntimeError, match="visible HIP Torch device"):
        materialize_batch_query_geometry_hip_into(*_empty_case())


def test_native_hip_batch_query_geometry_rejects_autograd_inputs() -> None:
    values = list(_empty_case())
    values[0] = values[0].requires_grad_()
    with pytest.raises(RuntimeError, match="forward-only.*autograd"):
        materialize_batch_query_geometry_hip_into(*values)
