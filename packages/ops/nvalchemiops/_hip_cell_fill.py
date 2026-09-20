# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional native HIP atomic CSR atom-list fill candidate.

This module owns no runtime selection.  It exposes only an explicit forward
operation using caller-owned list and cursor buffers.  Atomic insertion order
is intentionally unspecified; callers must not expose it as the stable public
neighbor order without canonicalization.
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import ModuleType

import torch

from nvalchemiops._cell_list_abi import _validate_cell_atom_list_fill_inputs
from nvalchemiops._hip_extension_loader import load_hip_extension


_EXTENSION: ModuleType | None = None
_EXTENSION_LOCK = threading.Lock()


def _validate_hip_call(
    cell_keys: torch.Tensor,
    cell_counts: torch.Tensor,
    cell_starts: torch.Tensor,
    cell_atom_list: torch.Tensor,
    cell_cursor: torch.Tensor,
    global_atom_offset: int,
) -> None:
    _validate_cell_atom_list_fill_inputs(
        cell_keys,
        cell_counts,
        cell_starts,
        cell_atom_list,
        cell_cursor,
        global_atom_offset=global_atom_offset,
    )
    if cell_keys.device.type != "cuda" or not torch.version.hip:
        raise RuntimeError("native HIP CSR fill requires a visible HIP Torch device")


def _load_jit_extension() -> ModuleType:
    """Compile/load the native HIP fill extension into an explicit cache."""

    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    with _EXTENSION_LOCK:
        if _EXTENSION is not None:
            return _EXTENSION
        _EXTENSION = load_hip_extension(
            source_root=Path(__file__).with_name("_native"),
            extension_name="nvalchemi_cell_fill_hip",
            source_names=("cell_atom_list_fill.cpp", "cell_atom_list_fill.cu"),
            build_env_var="NVALCHEMI_HIP_CELL_FILL_BUILD_DIR",
            verbose_env_var="NVALCHEMI_HIP_CELL_FILL_VERBOSE",
        )
    return _EXTENSION


@torch.library.custom_op(
    "nvalchemiops::_cell_atom_list_fill_hip",
    mutates_args=("cell_atom_list", "cell_cursor"),
)
def _cell_atom_list_fill_hip(
    cell_keys: torch.Tensor,
    cell_counts: torch.Tensor,
    cell_starts: torch.Tensor,
    cell_atom_list: torch.Tensor,
    cell_cursor: torch.Tensor,
    global_atom_offset: int,
) -> None:
    """Atomically fill one active CSR atom-list slice on the current stream."""

    _validate_hip_call(
        cell_keys,
        cell_counts,
        cell_starts,
        cell_atom_list,
        cell_cursor,
        global_atom_offset,
    )
    _load_jit_extension().cell_atom_list_fill_into(
        cell_keys,
        cell_counts,
        cell_starts,
        cell_atom_list,
        cell_cursor,
        global_atom_offset,
    )


@_cell_atom_list_fill_hip.register_fake
def _cell_atom_list_fill_hip_fake(*args: object, **kwargs: object) -> None:
    """Declare mutation-only behavior for fake/compile tracing."""

    del args, kwargs
    return None


def build_cell_atom_list_hip_into(
    cell_keys: torch.Tensor,
    cell_counts: torch.Tensor,
    cell_starts: torch.Tensor,
    cell_atom_list: torch.Tensor,
    cell_cursor: torch.Tensor,
    *,
    global_atom_offset: int = 0,
) -> None:
    """Run explicit forward-only native HIP fill with unordered cell slots."""

    _validate_hip_call(
        cell_keys,
        cell_counts,
        cell_starts,
        cell_atom_list,
        cell_cursor,
        global_atom_offset,
    )
    _cell_atom_list_fill_hip(
        cell_keys,
        cell_counts,
        cell_starts,
        cell_atom_list,
        cell_cursor,
        global_atom_offset,
    )


__all__ = ["build_cell_atom_list_hip_into"]
