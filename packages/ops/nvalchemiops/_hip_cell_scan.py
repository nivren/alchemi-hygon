# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional native HIP rocPRIM scan for shared cell-list CSR starts.

The module is lazy and forward-only.  It operates on one flattened active cell
slice, which represents both a single system and concatenated Batch cell
slices after their per-system cell offsets have been applied.  It does not
connect to the cell-list dispatcher or choose a runtime backend.
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import ModuleType

import torch

from nvalchemiops._cell_list_abi import _validate_cell_scan_inputs
from nvalchemiops._hip_extension_loader import load_hip_extension


_EXTENSION: ModuleType | None = None
_EXTENSION_LOCK = threading.Lock()


def _validate_hip_call(
    cell_counts: torch.Tensor,
    cell_starts: torch.Tensor,
    workspace: torch.Tensor | None,
    global_atom_offset: int,
) -> None:
    _validate_cell_scan_inputs(cell_counts, cell_starts, global_atom_offset)
    if cell_counts.device.type != "cuda" or not torch.version.hip:
        raise RuntimeError("native HIP cell-start scan requires a visible HIP Torch device")
    if not cell_starts.is_contiguous():
        raise ValueError("native HIP cell-start output buffer must be contiguous")
    if workspace is None:
        return
    if workspace.ndim != 1 or workspace.dtype != torch.uint8:
        raise ValueError("cell-start workspace must have shape (W,) and dtype uint8")
    if workspace.device != cell_counts.device:
        raise ValueError("cell-start workspace must be on the same device as counts")
    if not workspace.is_contiguous():
        raise ValueError("cell-start workspace must be contiguous")
    if workspace.numel() and workspace.data_ptr() in {
        cell_counts.data_ptr(),
        cell_starts.data_ptr(),
    }:
        raise ValueError("cell-start workspace must not alias counts or starts")


def _load_jit_extension() -> ModuleType:
    """Compile/load the native HIP rocPRIM module into an explicit cache."""

    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    with _EXTENSION_LOCK:
        if _EXTENSION is not None:
            return _EXTENSION
        _EXTENSION = load_hip_extension(
            source_root=Path(__file__).with_name("_native"),
            extension_name="nvalchemi_cell_start_scan_hip",
            source_names=("cell_start_scan.cpp", "cell_start_scan.cu"),
            build_env_var="NVALCHEMI_HIP_CELL_SCAN_BUILD_DIR",
            verbose_env_var="NVALCHEMI_HIP_CELL_SCAN_VERBOSE",
        )
    return _EXTENSION


def cell_starts_hip_workspace_size(
    cell_counts: torch.Tensor,
    cell_starts: torch.Tensor,
    *,
    global_atom_offset: int = 0,
) -> int:
    """Return rocPRIM temporary-storage bytes for this active CSR slice."""

    _validate_hip_call(cell_counts, cell_starts, None, global_atom_offset)
    return int(
        _load_jit_extension().cell_start_scan_workspace_size(
            cell_counts,
            cell_starts,
        )
    )


@torch.library.custom_op(
    "nvalchemiops::_cell_start_scan_hip",
    mutates_args=("cell_starts", "workspace"),
)
def _cell_start_scan_hip(
    cell_counts: torch.Tensor,
    cell_starts: torch.Tensor,
    workspace: torch.Tensor,
    global_atom_offset: int,
) -> None:
    """Write CSR starts with rocPRIM into caller-owned output and scratch."""

    _validate_hip_call(cell_counts, cell_starts, workspace, global_atom_offset)
    _load_jit_extension().cell_start_scan_into(
        cell_counts,
        cell_starts,
        workspace,
        global_atom_offset,
    )


@_cell_start_scan_hip.register_fake
def _cell_start_scan_hip_fake(*args: object, **kwargs: object) -> None:
    """Declare mutation-only behavior for fake/compile tracing."""

    del args, kwargs
    return None


def build_cell_starts_hip_into(
    cell_counts: torch.Tensor,
    cell_starts: torch.Tensor,
    workspace: torch.Tensor,
    *,
    global_atom_offset: int = 0,
) -> None:
    """Execute an explicit native HIP scan without modifying ``cell_counts``."""

    _validate_hip_call(cell_counts, cell_starts, workspace, global_atom_offset)
    required = cell_starts_hip_workspace_size(
        cell_counts,
        cell_starts,
        global_atom_offset=global_atom_offset,
    )
    if workspace.numel() < required:
        raise ValueError(
            f"cell-start workspace has {workspace.numel()} bytes, need {required}"
        )
    _cell_start_scan_hip(
        cell_counts,
        cell_starts,
        workspace,
        global_atom_offset,
    )


__all__ = ["build_cell_starts_hip_into", "cell_starts_hip_workspace_size"]
