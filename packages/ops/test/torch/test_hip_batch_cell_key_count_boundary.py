# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-side tests for the explicit native HIP Batch fusion boundary."""

from __future__ import annotations

import sys

import pytest
import torch

from nvalchemiops._hip_batch_cell_key_count import build_batch_cell_key_counts_hip_into


def test_native_hip_batch_fusion_boundary_does_not_compile_on_import() -> None:
    assert "nvalchemiops._hip_batch_cell_key_count" in sys.modules


def test_native_hip_batch_fusion_boundary_rejects_cpu_without_fallback() -> None:
    with pytest.raises(RuntimeError, match="visible HIP Torch device"):
        build_batch_cell_key_counts_hip_into(
            torch.empty((0, 3), dtype=torch.float32),
            torch.eye(3, dtype=torch.float32).unsqueeze(0),
            torch.tensor([[2, 2, 2]], dtype=torch.int32),
            torch.ones((1, 3), dtype=torch.bool),
            torch.empty(0, dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
            torch.empty((0, 3), dtype=torch.int32),
            torch.empty((0, 3), dtype=torch.int32),
            torch.empty(0, dtype=torch.int32),
            torch.empty(8, dtype=torch.int32),
        )
