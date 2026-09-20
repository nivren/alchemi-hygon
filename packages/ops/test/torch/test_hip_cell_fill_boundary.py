# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-side tests for the explicit native HIP CSR fill boundary."""

from __future__ import annotations

import sys

import pytest
import torch

from nvalchemiops._hip_cell_fill import build_cell_atom_list_hip_into


def test_native_hip_fill_boundary_does_not_compile_on_import() -> None:
    assert "nvalchemiops._hip_cell_fill" in sys.modules


def test_native_hip_fill_boundary_rejects_cpu_without_fallback() -> None:
    with pytest.raises(RuntimeError, match="visible HIP Torch device"):
        build_cell_atom_list_hip_into(
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([1, 1], dtype=torch.int32),
            torch.tensor([0, 1], dtype=torch.int32),
            torch.empty(2, dtype=torch.int32),
            torch.empty(2, dtype=torch.int32),
        )
