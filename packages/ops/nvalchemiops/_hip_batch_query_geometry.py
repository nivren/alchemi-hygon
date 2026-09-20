# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isolated native HIP forward geometry candidate for Batch query outputs.

The input topology is already in the public canonical layout.  This candidate
only computes distance/vector outputs from positions, cells, batch indices and
integer image shifts.  It is deliberately forward-only: the Torch geometry
bridge remains the correctness and differentiable reference path.
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import ModuleType

import torch
from nvalchemiops._hip_extension_loader import load_hip_extension


_EXTENSION: ModuleType | None = None
_EXTENSION_LOCK = threading.Lock()


def _validate_inputs(
    positions: torch.Tensor,
    cells: torch.Tensor,
    batch_idx: torch.Tensor,
    public_matrix: torch.Tensor,
    public_shifts: torch.Tensor,
    public_counts: torch.Tensor,
    distances: torch.Tensor,
    vectors: torch.Tensor,
) -> None:
    if positions.ndim != 2 or positions.shape[1:] != (3,):
        raise ValueError("positions must have shape (N, 3)")
    if positions.dtype not in (torch.float32, torch.float64):
        raise TypeError("positions must have dtype float32 or float64")
    if positions.requires_grad or cells.requires_grad:
        raise RuntimeError(
            "native HIP Batch query geometry is forward-only and does not support autograd"
        )
    if cells.ndim != 3 or cells.shape[1:] != (3, 3) or cells.dtype != positions.dtype:
        raise ValueError("cells must have shape (B, 3, 3) and match positions dtype")
    if batch_idx.shape != (positions.shape[0],) or batch_idx.dtype != torch.int32:
        raise ValueError("batch_idx must have shape (N,) and dtype int32")
    if public_matrix.ndim != 2 or public_matrix.shape[0] != positions.shape[0]:
        raise ValueError("public_matrix must have shape (N, K)")
    if public_matrix.shape[1] <= 0:
        raise ValueError("public_matrix must provide positive capacity K")
    if public_shifts.shape != (*public_matrix.shape, 3):
        raise ValueError("public_shifts must have shape (N, K, 3)")
    if public_counts.shape != (positions.shape[0],):
        raise ValueError("public_counts must have shape (N,)")
    if distances.shape != public_matrix.shape or distances.dtype != positions.dtype:
        raise ValueError("distances must have shape (N, K) and match positions dtype")
    if vectors.shape != public_shifts.shape or vectors.dtype != positions.dtype:
        raise ValueError("vectors must have shape (N, K, 3) and match positions dtype")
    values = (
        cells,
        batch_idx,
        public_matrix,
        public_shifts,
        public_counts,
        distances,
        vectors,
    )
    if any(value.device != positions.device for value in values):
        raise ValueError("HIP geometry tensors must share one device")
    if any(not value.is_contiguous() for value in values):
        raise ValueError("HIP geometry tensors must be contiguous")
    if positions.device.type != "cuda" or not torch.version.hip:
        raise RuntimeError("native HIP Batch query geometry requires a visible HIP Torch device")
    if cells.shape[0] <= 0:
        raise ValueError("HIP geometry requires at least one system")
    if bool(torch.any(batch_idx < 0)) or bool(torch.any(batch_idx >= cells.shape[0])):
        raise ValueError("batch_idx contains an invalid system index")
    if any(
        value.dtype != torch.int32
        for value in (public_matrix, public_shifts, public_counts)
    ):
        raise TypeError("public topology buffers must have dtype int32")
    if bool(torch.any(public_counts < 0)) or bool(
        torch.any(public_counts > public_matrix.shape[1])
    ):
        raise ValueError("public_counts must be within the public capacity")
    active = torch.arange(
        public_matrix.shape[1], device=public_matrix.device
    )[None, :] < public_counts[:, None]
    active_columns = public_matrix[active]
    if bool(torch.any(active_columns < 0)) or bool(
        torch.any(active_columns >= positions.shape[0])
    ):
        raise ValueError("public_matrix contains an invalid active neighbor index")
    if distances.numel() and vectors.numel() and distances.data_ptr() == vectors.data_ptr():
        raise ValueError("distance and vector outputs must not alias")


def _load_jit_extension() -> ModuleType:
    """Compile/load the isolated native HIP geometry candidate."""

    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    with _EXTENSION_LOCK:
        if _EXTENSION is not None:
            return _EXTENSION
        _EXTENSION = load_hip_extension(
            source_root=Path(__file__).with_name("_native"),
            extension_name="nvalchemi_batch_query_geometry_hip",
            source_names=("batch_query_geometry.cpp", "batch_query_geometry.cu"),
            build_env_var="NVALCHEMI_HIP_BATCH_QUERY_GEOMETRY_BUILD_DIR",
            verbose_env_var="NVALCHEMI_HIP_BATCH_QUERY_GEOMETRY_VERBOSE",
        )
    return _EXTENSION


@torch.library.custom_op(
    "nvalchemiops::_batch_query_geometry_hip",
    mutates_args=("distances", "vectors"),
)
def _batch_query_geometry_hip(
    positions: torch.Tensor,
    cells: torch.Tensor,
    batch_idx: torch.Tensor,
    public_matrix: torch.Tensor,
    public_shifts: torch.Tensor,
    public_counts: torch.Tensor,
    distances: torch.Tensor,
    vectors: torch.Tensor,
) -> None:
    _validate_inputs(
        positions,
        cells,
        batch_idx,
        public_matrix,
        public_shifts,
        public_counts,
        distances,
        vectors,
    )
    _load_jit_extension().batch_query_geometry_into(
        positions,
        cells,
        batch_idx,
        public_matrix,
        public_shifts,
        public_counts,
        distances,
        vectors,
    )


@_batch_query_geometry_hip.register_fake
def _batch_query_geometry_hip_fake(*args: object, **kwargs: object) -> None:
    """Declare mutation-only fake/compile behavior for the candidate."""

    del args, kwargs
    return None


def materialize_batch_query_geometry_hip_into(
    positions: torch.Tensor,
    cells: torch.Tensor,
    batch_idx: torch.Tensor,
    public_matrix: torch.Tensor,
    public_shifts: torch.Tensor,
    public_counts: torch.Tensor,
    distances: torch.Tensor,
    vectors: torch.Tensor,
) -> None:
    """Compute forward distance/vector outputs with a native HIP kernel."""

    _batch_query_geometry_hip(
        positions,
        cells,
        batch_idx,
        public_matrix,
        public_shifts,
        public_counts,
        distances,
        vectors,
    )


__all__ = ["materialize_batch_query_geometry_hip_into"]
