# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-side tests for the explicit native HIP cell-key boundary."""

from __future__ import annotations

import sys

import pytest
import torch

from nvalchemiops._hip_cell_key import build_cell_keys_hip_into


def test_native_hip_boundary_does_not_load_warp_or_compile_on_import() -> None:
    assert "nvalchemiops._hip_cell_key" in sys.modules


def test_native_hip_boundary_rejects_cpu_without_fallback() -> None:
    outputs = [
        torch.empty((1, 3), dtype=torch.int32),
        torch.empty((1, 3), dtype=torch.int32),
        torch.empty(1, dtype=torch.int32),
    ]
    with pytest.raises(RuntimeError, match="visible HIP Torch device"):
        build_cell_keys_hip_into(
            torch.zeros((1, 3), dtype=torch.float32),
            torch.eye(3, dtype=torch.float32),
            torch.tensor([4, 4, 4], dtype=torch.int32),
            torch.ones(3, dtype=torch.bool),
            *outputs,
        )
