# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-side tests for the explicit native HIP cell-key count boundary."""

from __future__ import annotations

import sys

import pytest
import torch

from nvalchemiops._hip_cell_count import build_cell_counts_hip_into


def test_native_hip_count_boundary_does_not_load_warp_or_compile_on_import() -> None:
    assert "nvalchemiops._hip_cell_count" in sys.modules


def test_native_hip_count_boundary_rejects_cpu_without_fallback() -> None:
    with pytest.raises(RuntimeError, match="visible HIP Torch device"):
        build_cell_counts_hip_into(
            torch.tensor([0, 1], dtype=torch.int32),
            torch.empty(2, dtype=torch.int32),
        )
