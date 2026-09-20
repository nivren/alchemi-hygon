# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional native HIP fused Batch geometry/PBC/key/count candidate.

The explicit operation owns no cell-list policy: it validates and writes the
shared build outputs for a flattened Batch, but does not perform scan, fill,
query, dispatcher registration, or runtime backend selection.
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import ModuleType

import torch

from nvalchemiops._cell_list_abi import _validate_batch_cell_key_count_inputs
from nvalchemiops._hip_extension_loader import load_hip_extension


_EXTENSION: ModuleType | None = None
_EXTENSION_LOCK = threading.Lock()


def _validate_hip_call(
    positions: torch.Tensor,
    inverse_cells: torch.Tensor,
    cells_per_dimension: torch.Tensor,
    pbc: torch.Tensor,
    batch_idx: torch.Tensor,
    cell_offsets: torch.Tensor,
    atom_periodic_shifts: torch.Tensor,
    atom_to_cell_mapping: torch.Tensor,
    cell_keys: torch.Tensor,
    cell_counts: torch.Tensor,
) -> None:
    _validate_batch_cell_key_count_inputs(
        positions,
        inverse_cells,
        cells_per_dimension,
        pbc,
        batch_idx,
        cell_offsets,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        cell_keys,
        cell_counts,
    )
    if positions.device.type != "cuda" or not torch.version.hip:
        raise RuntimeError(
            "native HIP batch cell-key/count requires a visible HIP Torch device"
        )
    if not all(
        value.is_contiguous()
        for value in (
            atom_periodic_shifts,
            atom_to_cell_mapping,
            cell_keys,
            cell_counts,
        )
    ):
        raise ValueError("native HIP batch cell-key/count output buffers must be contiguous")


def _load_jit_extension() -> ModuleType:
    """Compile/load the fused native HIP module into an explicit cache."""

    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    with _EXTENSION_LOCK:
        if _EXTENSION is not None:
            return _EXTENSION
        _EXTENSION = load_hip_extension(
            source_root=Path(__file__).with_name("_native"),
            extension_name="nvalchemi_batch_cell_key_count_hip",
            source_names=("batch_cell_key_count.cpp", "batch_cell_key_count.cu"),
            build_env_var="NVALCHEMI_HIP_BATCH_CELL_KEY_COUNT_BUILD_DIR",
            verbose_env_var="NVALCHEMI_HIP_BATCH_CELL_KEY_COUNT_VERBOSE",
        )
    return _EXTENSION


@torch.library.custom_op(
    "nvalchemiops::_batch_cell_key_count_hip",
    mutates_args=(
        "atom_periodic_shifts",
        "atom_to_cell_mapping",
        "cell_keys",
        "cell_counts",
    ),
)
def _batch_cell_key_count_hip(
    positions: torch.Tensor,
    inverse_cells: torch.Tensor,
    cells_per_dimension: torch.Tensor,
    pbc: torch.Tensor,
    batch_idx: torch.Tensor,
    cell_offsets: torch.Tensor,
    atom_periodic_shifts: torch.Tensor,
    atom_to_cell_mapping: torch.Tensor,
    cell_keys: torch.Tensor,
    cell_counts: torch.Tensor,
) -> None:
    """Write Batch key-build outputs and global cell counts on the current stream."""

    _validate_hip_call(
        positions,
        inverse_cells,
        cells_per_dimension,
        pbc,
        batch_idx,
        cell_offsets,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        cell_keys,
        cell_counts,
    )
    _load_jit_extension().batch_cell_key_count_into(
        positions,
        inverse_cells,
        cells_per_dimension,
        pbc,
        batch_idx,
        cell_offsets,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        cell_keys,
        cell_counts,
    )


@_batch_cell_key_count_hip.register_fake
def _batch_cell_key_count_hip_fake(*args: object, **kwargs: object) -> None:
    """Declare the mutation-only fake/compile behavior."""

    del args, kwargs
    return None


def build_batch_cell_key_counts_hip_into(
    positions: torch.Tensor,
    inverse_cells: torch.Tensor,
    cells_per_dimension: torch.Tensor,
    pbc: torch.Tensor,
    batch_idx: torch.Tensor,
    cell_offsets: torch.Tensor,
    atom_periodic_shifts: torch.Tensor,
    atom_to_cell_mapping: torch.Tensor,
    cell_keys: torch.Tensor,
    cell_counts: torch.Tensor,
) -> None:
    """Run the explicit forward-only fused native HIP candidate."""

    _validate_hip_call(
        positions,
        inverse_cells,
        cells_per_dimension,
        pbc,
        batch_idx,
        cell_offsets,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        cell_keys,
        cell_counts,
    )
    _batch_cell_key_count_hip(
        positions,
        inverse_cells,
        cells_per_dimension,
        pbc,
        batch_idx,
        cell_offsets,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        cell_keys,
        cell_counts,
    )


__all__ = ["build_batch_cell_key_counts_hip_into"]
