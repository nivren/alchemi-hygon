# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-side tests for the explicit native HIP CSR scan boundary."""

from __future__ import annotations

import sys

import pytest
import torch

from nvalchemiops._hip_cell_scan import (
    build_cell_starts_hip_into,
    cell_starts_hip_workspace_size,
)


def test_native_hip_scan_boundary_does_not_load_warp_or_compile_on_import() -> None:
    assert "nvalchemiops._hip_cell_scan" in sys.modules


def test_native_hip_scan_boundary_rejects_cpu_without_fallback() -> None:
    counts = torch.tensor([1, 2], dtype=torch.int32)
    starts = torch.empty_like(counts)
    with pytest.raises(RuntimeError, match="visible HIP Torch device"):
        cell_starts_hip_workspace_size(counts, starts)
    with pytest.raises(RuntimeError, match="visible HIP Torch device"):
        build_cell_starts_hip_into(
            counts,
            starts,
            torch.empty(0, dtype=torch.uint8),
        )
