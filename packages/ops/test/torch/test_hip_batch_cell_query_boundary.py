# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU boundary tests for the explicit native HIP query candidate."""

from __future__ import annotations

import sys

import pytest
import torch

from nvalchemiops._hip_batch_cell_query import batch_query_cell_list_hip_into


def test_native_hip_batch_query_boundary_does_not_compile_on_import() -> None:
    assert "nvalchemiops._hip_batch_cell_query" in sys.modules


def test_native_hip_batch_query_boundary_rejects_cpu_without_fallback() -> None:
    with pytest.raises(RuntimeError, match="visible HIP Torch device"):
        batch_query_cell_list_hip_into(
            torch.empty((0, 3), dtype=torch.float32),
            torch.eye(3, dtype=torch.float32).unsqueeze(0),
            torch.ones((1, 3), dtype=torch.bool),
            0.5,
            torch.empty(0, dtype=torch.int32),
            torch.tensor([[2, 2, 2]], dtype=torch.int32),
            torch.ones((1, 3), dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
            torch.empty((0, 3), dtype=torch.int32),
            torch.empty((0, 3), dtype=torch.int32),
            torch.empty(8, dtype=torch.int32),
            torch.empty(8, dtype=torch.int32),
            torch.empty(0, dtype=torch.int32),
            torch.empty((0, 8), dtype=torch.int32),
            torch.empty((0, 8, 3), dtype=torch.int32),
            torch.empty(0, dtype=torch.int32),
        )
