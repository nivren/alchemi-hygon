# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional native HIP atomic count for the shared cell-key CSR ABI.

Importing this module registers only a mutation-only custom op.  The HIP
extension is compiled lazily on the first explicit call, into a caller-selected
cache directory.  This is a candidate implementation of the count phase only:
the Torch reference remains responsible for prefix scan, atom-list fill, and
neighbor querying.
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import ModuleType

import torch

from nvalchemiops._cell_list_abi import _validate_cell_count_inputs
from nvalchemiops._hip_extension_loader import load_hip_extension


_EXTENSION: ModuleType | None = None
_EXTENSION_LOCK = threading.Lock()


def _validate_hip_call(
    cell_keys: torch.Tensor,
    cell_counts: torch.Tensor,
) -> None:
    _validate_cell_count_inputs(cell_keys, cell_counts)
    if cell_keys.device.type != "cuda" or not torch.version.hip:
        raise RuntimeError("native HIP cell-key count requires a visible HIP Torch device")
    if not cell_counts.is_contiguous():
        raise ValueError("native HIP cell-count output buffer must be contiguous")


def _load_jit_extension() -> ModuleType:
    """Compile/load the native HIP count module into an explicit cache."""

    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    with _EXTENSION_LOCK:
        if _EXTENSION is not None:
            return _EXTENSION
        _EXTENSION = load_hip_extension(
            source_root=Path(__file__).with_name("_native"),
            extension_name="nvalchemi_cell_key_count_hip",
            source_names=("cell_key_count.cpp", "cell_key_count.cu"),
            build_env_var="NVALCHEMI_HIP_CELL_COUNT_BUILD_DIR",
            verbose_env_var="NVALCHEMI_HIP_CELL_COUNT_VERBOSE",
        )
    return _EXTENSION


@torch.library.custom_op(
    "nvalchemiops::_cell_key_count_hip",
    mutates_args=("cell_counts",),
)
def _cell_key_count_hip(
    cell_keys: torch.Tensor,
    cell_counts: torch.Tensor,
) -> None:
    """Write per-cell counts with a HIP zero-and-atomic-add implementation."""

    _validate_hip_call(cell_keys, cell_counts)
    _load_jit_extension().cell_key_count_into(cell_keys, cell_counts)


@_cell_key_count_hip.register_fake
def _cell_key_count_hip_fake(*args: object, **kwargs: object) -> None:
    """Declare mutation-only behavior for fake/compile tracing."""

    del args, kwargs
    return None


def build_cell_counts_hip_into(
    cell_keys: torch.Tensor,
    cell_counts: torch.Tensor,
) -> None:
    """Execute the explicit native HIP count phase into a caller buffer.

    Cell topology is discrete, so this candidate is forward-only.  CPU tensors
    and non-HIP PyTorch builds fail rather than falling back to Torch.
    """

    _validate_hip_call(cell_keys, cell_counts)
    _cell_key_count_hip(cell_keys, cell_counts)


__all__ = ["build_cell_counts_hip_into"]
