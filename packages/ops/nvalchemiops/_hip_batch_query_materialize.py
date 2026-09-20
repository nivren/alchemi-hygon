# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isolated native HIP topology materialization candidate.

This module owns only the discrete part of Batch query materialization:
stable public ordering, per-row rank computation, and scatter into caller
owned full/half-list buffers.  It consumes an unordered native query
candidate and intentionally does not compute distances or vectors.

The implementation is a correctness candidate for the point20 hotspot.  It
is not registered with the runtime dispatcher and does not participate in
``auto`` or explicit backend selection.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import torch
from nvalchemiops._hip_extension_loader import load_hip_extension


_EXTENSION: ModuleType | None = None
_EXTENSION_LOCK = threading.Lock()


@dataclass(frozen=True)
class CompositeTopologyWorkspace:
    """Caller-owned scratch buffers for repeated composite topology calls."""

    keys_a: torch.Tensor
    keys_b: torch.Tensor
    order_a: torch.Tensor
    order_b: torch.Tensor
    row_starts: torch.Tensor
    sort_workspace: torch.Tensor
    scan_workspace: torch.Tensor
    atoms: int
    candidate_capacity: int
    row_bits: int
    total_bits: int
    shift_bits: int
    shift_bias: int


def _composite_key_layout(
    atoms: int, shift_bits: int, shift_bias: int
) -> tuple[int, int]:
    """Validate the packed-key layout without touching a device."""

    if isinstance(shift_bits, bool) or not isinstance(shift_bits, int):
        raise TypeError("shift_bits must be an int")
    if isinstance(shift_bias, bool) or not isinstance(shift_bias, int):
        raise TypeError("shift_bias must be an int")
    if shift_bits <= 0 or shift_bits >= 64:
        raise ValueError("shift_bits must be in [1, 63]")
    row_bits = max(1, int(atoms).bit_length())
    total_bits = 2 * row_bits + 3 * shift_bits
    if total_bits > 63:
        raise ValueError(
            "composite topology key exceeds signed int64 width: "
            f"{total_bits} bits"
        )
    max_shift_code = (1 << shift_bits) - 1
    if shift_bias < 0 or shift_bias > max_shift_code:
        raise ValueError("shift_bias must fit the selected shift_bits")
    return row_bits, total_bits


def _validate_inputs(
    candidate_matrix: torch.Tensor,
    candidate_shifts: torch.Tensor,
    candidate_counts: torch.Tensor,
    public_matrix: torch.Tensor,
    public_shifts: torch.Tensor,
    public_counts: torch.Tensor,
    fill_value: int,
) -> None:
    if candidate_matrix.ndim != 2 or candidate_matrix.shape[1] <= 0:
        raise ValueError("candidate_matrix must have shape (N, Kc) with positive Kc")
    atoms, candidate_capacity = candidate_matrix.shape
    if candidate_shifts.shape != (atoms, candidate_capacity, 3):
        raise ValueError("candidate_shifts must have shape (N, Kc, 3)")
    if candidate_counts.shape != (atoms,):
        raise ValueError("candidate_counts must have shape (N,)")
    if public_matrix.ndim != 2 or public_matrix.shape[0] != atoms:
        raise ValueError("public_matrix must have shape (N, Kp)")
    if public_matrix.shape[1] <= 0:
        raise ValueError("public_matrix must provide positive capacity Kp")
    public_capacity = public_matrix.shape[1]
    if public_shifts.shape != (atoms, public_capacity, 3):
        raise ValueError("public_shifts must have shape (N, Kp, 3)")
    if public_counts.shape != (atoms,):
        raise ValueError("public_counts must have shape (N,)")
    topology = (
        candidate_matrix,
        candidate_shifts,
        candidate_counts,
        public_matrix,
        public_shifts,
        public_counts,
    )
    if any(value.dtype != torch.int32 for value in topology):
        raise TypeError("HIP topology materialization buffers must have dtype int32")
    if any(value.device != candidate_matrix.device for value in topology):
        raise ValueError("HIP topology materialization buffers must share one device")
    if any(not value.is_contiguous() for value in topology):
        raise ValueError("HIP topology materialization buffers must be contiguous")
    if candidate_matrix.device.type != "cuda" or not torch.version.hip:
        raise RuntimeError(
            "native HIP Batch query topology materialization requires a visible HIP Torch device"
        )
    if isinstance(fill_value, bool) or not isinstance(fill_value, int):
        raise TypeError("fill_value must be an int")
    if fill_value < atoms or fill_value > torch.iinfo(torch.int32).max:
        raise ValueError("fill_value must be >= num atoms and fit int32")
    if bool(torch.any(candidate_counts < 0)) or bool(
        torch.any(candidate_counts > candidate_capacity)
    ):
        raise ValueError("candidate_counts must be within candidate capacity")
    if bool(torch.any(candidate_counts > public_capacity)):
        raise ValueError("candidate_counts exceed public capacity")
    if atoms:
        active = torch.arange(candidate_capacity, device=candidate_matrix.device)[None, :] < (
            candidate_counts[:, None]
        )
        active_columns = candidate_matrix[active]
        if bool(torch.any(active_columns < 0)) or bool(torch.any(active_columns >= atoms)):
            raise ValueError("candidate_matrix contains an invalid active neighbor index")
    input_ptrs = {value.data_ptr() for value in topology[:3] if value.numel()}
    output_ptrs = {value.data_ptr() for value in topology[3:] if value.numel()}
    if input_ptrs & output_ptrs:
        raise ValueError("candidate and public topology buffers must not alias")
    if len(output_ptrs) != len([value for value in topology[3:] if value.numel()]):
        raise ValueError("public topology output buffers must not alias")


def _load_jit_extension() -> ModuleType:
    """Compile/load the isolated native HIP topology candidate."""

    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    with _EXTENSION_LOCK:
        if _EXTENSION is not None:
            return _EXTENSION
        _EXTENSION = load_hip_extension(
            source_root=Path(__file__).with_name("_native"),
            extension_name="nvalchemi_batch_query_materialize_hip",
            source_names=("batch_query_materialize.cpp", "batch_query_materialize.cu"),
            build_env_var="NVALCHEMI_HIP_BATCH_QUERY_MATERIALIZE_BUILD_DIR",
            verbose_env_var="NVALCHEMI_HIP_BATCH_QUERY_MATERIALIZE_VERBOSE",
        )
    return _EXTENSION


def _validate_composite_inputs(
    candidate_matrix: torch.Tensor,
    candidate_shifts: torch.Tensor,
    candidate_counts: torch.Tensor,
    public_matrix: torch.Tensor,
    public_shifts: torch.Tensor,
    public_counts: torch.Tensor,
    fill_value: int,
    shift_bits: int,
    shift_bias: int,
) -> tuple[int, int]:
    row_bits, total_bits = _composite_key_layout(
        candidate_matrix.shape[0], shift_bits, shift_bias
    )
    _validate_inputs(
        candidate_matrix,
        candidate_shifts,
        candidate_counts,
        public_matrix,
        public_shifts,
        public_counts,
        fill_value,
    )
    min_shift = -shift_bias
    max_shift = (1 << shift_bits) - 1 - shift_bias
    if candidate_shifts.numel() and bool(torch.any(candidate_shifts < min_shift)):
        raise ValueError(
            "candidate_shifts contains a value below the composite key range"
        )
    if candidate_shifts.numel() and bool(torch.any(candidate_shifts > max_shift)):
        raise ValueError(
            "candidate_shifts contains a value above the composite key range"
        )
    return row_bits, total_bits


def _validate_workspace_inputs(
    candidate_matrix: torch.Tensor, candidate_counts: torch.Tensor
) -> None:
    if candidate_matrix.ndim != 2 or candidate_matrix.shape[1] <= 0:
        raise ValueError("candidate_matrix must have shape (N, Kc) with positive Kc")
    if candidate_counts.shape != (candidate_matrix.shape[0],):
        raise ValueError("candidate_counts must have shape (N,)")
    if candidate_matrix.dtype != torch.int32 or candidate_counts.dtype != torch.int32:
        raise TypeError("composite workspace inputs must have dtype int32")
    if candidate_matrix.device.type != "cuda" or not torch.version.hip:
        raise RuntimeError(
            "native HIP Batch query topology workspace requires a visible HIP Torch device"
        )
    if candidate_counts.device != candidate_matrix.device:
        raise ValueError("workspace inputs must share one device")
    if not candidate_matrix.is_contiguous() or not candidate_counts.is_contiguous():
        raise ValueError("workspace inputs must be contiguous")
    candidate_capacity = candidate_matrix.shape[1]
    if bool(torch.any(candidate_counts < 0)) or bool(
        torch.any(candidate_counts > candidate_capacity)
    ):
        raise ValueError("candidate_counts must be within candidate capacity")


def allocate_batch_query_topology_composite_workspace(
    candidate_matrix: torch.Tensor,
    candidate_counts: torch.Tensor,
    *,
    shift_bits: int = 8,
    shift_bias: int = 128,
) -> CompositeTopologyWorkspace:
    """Allocate reusable scratch for repeated composite topology materialization."""

    _validate_workspace_inputs(candidate_matrix, candidate_counts)
    row_bits, total_bits = _composite_key_layout(
        candidate_matrix.shape[0], int(shift_bits), int(shift_bias)
    )
    sort_bytes, scan_bytes = _load_jit_extension().batch_query_materialize_topology_composite_workspace_size(
        candidate_matrix, candidate_counts, int(total_bits)
    )
    total = candidate_matrix.numel()
    atoms, candidate_capacity = candidate_matrix.shape
    return CompositeTopologyWorkspace(
        keys_a=torch.empty((total,), dtype=torch.int64, device=candidate_matrix.device),
        keys_b=torch.empty((total,), dtype=torch.int64, device=candidate_matrix.device),
        order_a=torch.empty((total,), dtype=torch.int32, device=candidate_matrix.device),
        order_b=torch.empty((total,), dtype=torch.int32, device=candidate_matrix.device),
        row_starts=torch.empty(
            (atoms,), dtype=torch.int32, device=candidate_matrix.device
        ),
        sort_workspace=torch.empty(
            (int(sort_bytes),), dtype=torch.uint8, device=candidate_matrix.device
        ),
        scan_workspace=torch.empty(
            (int(scan_bytes),), dtype=torch.uint8, device=candidate_matrix.device
        ),
        atoms=int(atoms),
        candidate_capacity=int(candidate_capacity),
        row_bits=int(row_bits),
        total_bits=int(total_bits),
        shift_bits=int(shift_bits),
        shift_bias=int(shift_bias),
    )


@torch.library.custom_op(
    "nvalchemiops::_batch_query_materialize_topology_hip",
    mutates_args=("public_matrix", "public_shifts", "public_counts"),
)
def _batch_query_materialize_topology_hip(
    candidate_matrix: torch.Tensor,
    candidate_shifts: torch.Tensor,
    candidate_counts: torch.Tensor,
    public_matrix: torch.Tensor,
    public_shifts: torch.Tensor,
    public_counts: torch.Tensor,
    fill_value: int,
) -> None:
    _validate_inputs(
        candidate_matrix,
        candidate_shifts,
        candidate_counts,
        public_matrix,
        public_shifts,
        public_counts,
        fill_value,
    )
    _load_jit_extension().batch_query_materialize_topology_into(
        candidate_matrix,
        candidate_shifts,
        candidate_counts,
        public_matrix,
        public_shifts,
        public_counts,
        int(fill_value),
    )


@_batch_query_materialize_topology_hip.register_fake
def _batch_query_materialize_topology_hip_fake(
    candidate_matrix: torch.Tensor,
    candidate_shifts: torch.Tensor,
    candidate_counts: torch.Tensor,
    public_matrix: torch.Tensor,
    public_shifts: torch.Tensor,
    public_counts: torch.Tensor,
    fill_value: int,
) -> None:
    del candidate_matrix, candidate_shifts, candidate_counts, fill_value
    del public_matrix, public_shifts, public_counts
    return None


@torch.library.custom_op(
    "nvalchemiops::_batch_query_materialize_topology_composite_hip",
    mutates_args=("public_matrix", "public_shifts", "public_counts"),
)
def _batch_query_materialize_topology_composite_hip(
    candidate_matrix: torch.Tensor,
    candidate_shifts: torch.Tensor,
    candidate_counts: torch.Tensor,
    public_matrix: torch.Tensor,
    public_shifts: torch.Tensor,
    public_counts: torch.Tensor,
    fill_value: int,
    shift_bits: int,
    shift_bias: int,
) -> None:
    row_bits, total_bits = _validate_composite_inputs(
        candidate_matrix,
        candidate_shifts,
        candidate_counts,
        public_matrix,
        public_shifts,
        public_counts,
        fill_value,
        shift_bits,
        shift_bias,
    )
    _load_jit_extension().batch_query_materialize_topology_composite_into(
        candidate_matrix,
        candidate_shifts,
        candidate_counts,
        public_matrix,
        public_shifts,
        public_counts,
        int(fill_value),
        int(row_bits),
        int(total_bits),
        int(shift_bits),
        int(shift_bias),
    )


@_batch_query_materialize_topology_composite_hip.register_fake
def _batch_query_materialize_topology_composite_hip_fake(
    candidate_matrix: torch.Tensor,
    candidate_shifts: torch.Tensor,
    candidate_counts: torch.Tensor,
    public_matrix: torch.Tensor,
    public_shifts: torch.Tensor,
    public_counts: torch.Tensor,
    fill_value: int,
    shift_bits: int,
    shift_bias: int,
) -> None:
    del candidate_matrix, candidate_shifts, candidate_counts, fill_value
    del public_matrix, public_shifts, public_counts, shift_bits, shift_bias
    return None


def materialize_batch_query_topology_hip_into(
    candidate_matrix: torch.Tensor,
    candidate_shifts: torch.Tensor,
    candidate_counts: torch.Tensor,
    public_matrix: torch.Tensor,
    public_shifts: torch.Tensor,
    public_counts: torch.Tensor,
    *,
    fill_value: int | None = None,
) -> None:
    """Materialize an unordered native candidate into stable public topology."""

    resolved_fill = candidate_matrix.shape[0] if fill_value is None else int(fill_value)
    _batch_query_materialize_topology_hip(
        candidate_matrix,
        candidate_shifts,
        candidate_counts,
        public_matrix,
        public_shifts,
        public_counts,
        resolved_fill,
    )


def materialize_batch_query_topology_composite_hip_into(
    candidate_matrix: torch.Tensor,
    candidate_shifts: torch.Tensor,
    candidate_counts: torch.Tensor,
    public_matrix: torch.Tensor,
    public_shifts: torch.Tensor,
    public_counts: torch.Tensor,
    *,
    fill_value: int | None = None,
    shift_bits: int = 8,
    shift_bias: int = 128,
) -> None:
    """Test a one-sort packed-key topology materialization candidate."""

    resolved_fill = candidate_matrix.shape[0] if fill_value is None else int(fill_value)
    _batch_query_materialize_topology_composite_hip(
        candidate_matrix,
        candidate_shifts,
        candidate_counts,
        public_matrix,
        public_shifts,
        public_counts,
        resolved_fill,
        int(shift_bits),
        int(shift_bias),
    )


def materialize_batch_query_topology_composite_hip_into_with_workspace(
    candidate_matrix: torch.Tensor,
    candidate_shifts: torch.Tensor,
    candidate_counts: torch.Tensor,
    public_matrix: torch.Tensor,
    public_shifts: torch.Tensor,
    public_counts: torch.Tensor,
    workspace: CompositeTopologyWorkspace,
    *,
    fill_value: int | None = None,
    shift_bits: int = 8,
    shift_bias: int = 128,
) -> None:
    """Materialize topology using caller-owned reusable composite scratch."""

    resolved_fill = candidate_matrix.shape[0] if fill_value is None else int(fill_value)
    row_bits, total_bits = _validate_composite_inputs(
        candidate_matrix,
        candidate_shifts,
        candidate_counts,
        public_matrix,
        public_shifts,
        public_counts,
        resolved_fill,
        int(shift_bits),
        int(shift_bias),
    )
    if (
        workspace.atoms != candidate_matrix.shape[0]
        or workspace.candidate_capacity != candidate_matrix.shape[1]
        or workspace.row_bits != row_bits
        or workspace.total_bits != total_bits
        or workspace.shift_bits != int(shift_bits)
        or workspace.shift_bias != int(shift_bias)
    ):
        raise ValueError("composite workspace shape or key layout does not match inputs")
    _load_jit_extension().batch_query_materialize_topology_composite_workspace_into(
        candidate_matrix,
        candidate_shifts,
        candidate_counts,
        public_matrix,
        public_shifts,
        public_counts,
        workspace.keys_a,
        workspace.keys_b,
        workspace.order_a,
        workspace.order_b,
        workspace.row_starts,
        workspace.sort_workspace,
        workspace.scan_workspace,
        int(resolved_fill),
        int(row_bits),
        int(total_bits),
        int(shift_bits),
        int(shift_bias),
    )


__all__ = [
    "CompositeTopologyWorkspace",
    "allocate_batch_query_topology_composite_workspace",
    "materialize_batch_query_topology_composite_hip_into",
    "materialize_batch_query_topology_composite_hip_into_with_workspace",
    "materialize_batch_query_topology_hip_into",
]
