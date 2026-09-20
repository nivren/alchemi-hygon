# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional native HIP custom-op boundary for the shared cell-key ABI.

The module is intentionally lazy: importing it registers only the Torch
operator and does not compile or load HIP code.  The extension is built into a
caller-selected cache on first explicit HIP use.  There is no CPU fallback.
The source files are also suitable for a future package-level AOT build, but
this small step does not change the Hatchling wheel build.
"""

from __future__ import annotations

import importlib.util
import sys
import threading
from pathlib import Path
from types import ModuleType

import torch

from nvalchemiops._cell_list_abi import (
    CellKeyBuildResult,
    _validate_inputs,
    _validate_outputs,
)
from nvalchemiops._hip_extension_loader import load_hip_extension


_EXTENSION: ModuleType | None = None
_EXTENSION_LOCK = threading.Lock()


def _validate_hip_call(
    positions: torch.Tensor,
    inverse_cell: torch.Tensor,
    cells_per_dimension: torch.Tensor,
    pbc: torch.Tensor,
    atom_periodic_shifts: torch.Tensor,
    atom_to_cell_mapping: torch.Tensor,
    cell_keys: torch.Tensor,
) -> None:
    _validate_inputs(positions, inverse_cell, cells_per_dimension, pbc)
    _validate_outputs(
        positions, atom_periodic_shifts, atom_to_cell_mapping, cell_keys
    )
    if positions.device.type != "cuda" or not torch.version.hip:
        raise RuntimeError(
            "native HIP cell-key build requires a visible HIP Torch device"
        )
    if not all(
        value.is_contiguous()
        for value in (atom_periodic_shifts, atom_to_cell_mapping, cell_keys)
    ):
        raise ValueError("native HIP cell-key output buffers must be contiguous")


def _load_packaged_extension() -> ModuleType | None:
    """Load an AOT artifact embedded in a platform-specific wheel, if present."""

    artifact = Path(__file__).with_name("_native") / "nvalchemi_cell_key_hip.so"
    if not artifact.is_file():
        return None
    module_name = "nvalchemi_cell_key_hip"
    loaded = sys.modules.get(module_name)
    if loaded is not None:
        return loaded
    specification = importlib.util.spec_from_file_location(module_name, artifact)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load packaged native HIP extension: {artifact}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    try:
        specification.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _load_jit_extension() -> ModuleType:
    """Compile/load the native HIP module into an explicit cache directory."""

    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    with _EXTENSION_LOCK:
        if _EXTENSION is not None:
            return _EXTENSION
        _EXTENSION = load_hip_extension(
            source_root=Path(__file__).with_name("_native"),
            extension_name="nvalchemi_cell_key_hip",
            source_names=("cell_key_build.cpp", "cell_key_build.cu"),
            build_env_var="NVALCHEMI_HIP_CELL_KEY_BUILD_DIR",
            verbose_env_var="NVALCHEMI_HIP_CELL_KEY_VERBOSE",
        )
    return _EXTENSION


def _load_native_extension() -> ModuleType:
    """Load packaged AOT code when present, otherwise use the explicit JIT path."""

    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    with _EXTENSION_LOCK:
        if _EXTENSION is not None:
            return _EXTENSION
        packaged = _load_packaged_extension()
        if packaged is not None:
            _EXTENSION = packaged
            return _EXTENSION
    return _load_jit_extension()


@torch.library.custom_op(
    "nvalchemiops::_cell_key_build_hip",
    mutates_args=(
        "atom_periodic_shifts",
        "atom_to_cell_mapping",
        "cell_keys",
    ),
)
def _cell_key_build_hip(
    positions: torch.Tensor,
    inverse_cell: torch.Tensor,
    cells_per_dimension: torch.Tensor,
    pbc: torch.Tensor,
    atom_periodic_shifts: torch.Tensor,
    atom_to_cell_mapping: torch.Tensor,
    cell_keys: torch.Tensor,
) -> None:
    """Run native HIP cell-key build into caller-owned output buffers."""

    _validate_hip_call(
        positions,
        inverse_cell,
        cells_per_dimension,
        pbc,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        cell_keys,
    )
    _load_native_extension().cell_key_build_into(
        positions,
        inverse_cell,
        cells_per_dimension,
        pbc,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        cell_keys,
    )


@_cell_key_build_hip.register_fake
def _cell_key_build_hip_fake(*args: object, **kwargs: object) -> None:
    """Declare the mutation-only shape behavior for fake/compile tracing."""

    del args, kwargs
    return None


def build_cell_keys_hip_into(
    positions: torch.Tensor,
    inverse_cell: torch.Tensor,
    cells_per_dimension: torch.Tensor,
    pbc: torch.Tensor,
    atom_periodic_shifts: torch.Tensor,
    atom_to_cell_mapping: torch.Tensor,
    cell_keys: torch.Tensor,
) -> None:
    """Validate the shared ABI and execute its explicit native HIP path.

    This function is forward-only: cell topology and integer outputs are not
    differentiable.  A CPU tensor or a non-HIP Torch build fails explicitly.
    """

    _validate_hip_call(
        positions,
        inverse_cell,
        cells_per_dimension,
        pbc,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        cell_keys,
    )
    _cell_key_build_hip(
        positions,
        inverse_cell,
        cells_per_dimension,
        pbc,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        cell_keys,
    )


def build_cell_keys_hip(
    positions: torch.Tensor,
    inverse_cell: torch.Tensor,
    cells_per_dimension: torch.Tensor,
    pbc: torch.Tensor,
) -> CellKeyBuildResult:
    """Allocate shared ABI outputs and invoke :func:`build_cell_keys_hip_into`."""

    outputs = CellKeyBuildResult(
        atom_periodic_shifts=torch.empty(
            (positions.shape[0], 3), dtype=torch.int32, device=positions.device
        ),
        atom_to_cell_mapping=torch.empty(
            (positions.shape[0], 3), dtype=torch.int32, device=positions.device
        ),
        cell_keys=torch.empty(
            positions.shape[0], dtype=torch.int32, device=positions.device
        ),
    )
    build_cell_keys_hip_into(
        positions,
        inverse_cell,
        cells_per_dimension,
        pbc,
        outputs.atom_periodic_shifts,
        outputs.atom_to_cell_mapping,
        outputs.cell_keys,
    )
    return outputs


__all__ = ["build_cell_keys_hip", "build_cell_keys_hip_into"]
