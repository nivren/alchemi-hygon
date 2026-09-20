# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU boundary tests for the isolated single-composite-key candidate."""

from __future__ import annotations

import pytest
import torch

from nvalchemiops._hip_batch_query_materialize import (
    _composite_key_layout,
    allocate_batch_query_topology_composite_workspace,
    materialize_batch_query_topology_composite_hip_into,
)


def _empty_case() -> tuple[torch.Tensor, ...]:
    candidate = torch.empty((0, 4), dtype=torch.int32)
    candidate_shifts = torch.empty((0, 4, 3), dtype=torch.int32)
    counts = torch.empty(0, dtype=torch.int32)
    public = torch.empty((0, 5), dtype=torch.int32)
    public_shifts = torch.empty((0, 5, 3), dtype=torch.int32)
    public_counts = torch.empty(0, dtype=torch.int32)
    return candidate, candidate_shifts, counts, public, public_shifts, public_counts


def test_composite_key_layout_reports_signed_int64_overflow() -> None:
    with pytest.raises(ValueError, match="exceeds signed int64"):
        _composite_key_layout(5, 20, 1)


def test_composite_key_layout_reports_shift_bias_range() -> None:
    with pytest.raises(ValueError, match="shift_bias"):
        _composite_key_layout(5, 4, 16)


def test_native_composite_candidate_rejects_cpu_without_fallback() -> None:
    with pytest.raises(RuntimeError, match="visible HIP Torch device"):
        materialize_batch_query_topology_composite_hip_into(*_empty_case())


def test_composite_workspace_rejects_cpu_without_fallback() -> None:
    candidate, _, counts, *_ = _empty_case()
    with pytest.raises(RuntimeError, match="visible HIP Torch device"):
        allocate_batch_query_topology_composite_workspace(candidate, counts)
