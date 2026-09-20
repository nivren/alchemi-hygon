# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU boundary tests for the isolated native HIP topology materializer."""

from __future__ import annotations

import sys

import pytest
import torch

from nvalchemiops._hip_batch_query_materialize import (
    materialize_batch_query_topology_hip_into,
)


def _empty_case() -> tuple[torch.Tensor, ...]:
    candidate = torch.empty((0, 4), dtype=torch.int32)
    candidate_shifts = torch.empty((0, 4, 3), dtype=torch.int32)
    counts = torch.empty(0, dtype=torch.int32)
    public = torch.empty((0, 5), dtype=torch.int32)
    public_shifts = torch.empty((0, 5, 3), dtype=torch.int32)
    public_counts = torch.empty(0, dtype=torch.int32)
    return candidate, candidate_shifts, counts, public, public_shifts, public_counts


def test_native_hip_batch_query_materialize_boundary_does_not_compile_on_import() -> None:
    assert "nvalchemiops._hip_batch_query_materialize" in sys.modules


def test_native_hip_batch_query_materialize_rejects_cpu_without_fallback() -> None:
    with pytest.raises(RuntimeError, match="visible HIP Torch device"):
        materialize_batch_query_topology_hip_into(*_empty_case())
