# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Warp-independent Torch operation dispatcher for the first HCU slice."""

from __future__ import annotations

from typing import Any

import torch

from nvalchemiops.backend import (
    BackendName,
    BackendSelection,
    BackendUnavailableError,
    resolve_backend,
)
from nvalchemiops.torch_reference import lj_energy_forces as _lj_energy_forces
from nvalchemiops.torch_reference import neighbor_list as _neighbor_list
from nvalchemiops.torch_reference_cell_list import (
    neighbor_list as _neighbor_list_cell_list,
)


def dispatch_neighbor_list(
    positions: torch.Tensor,
    cutoff: float,
    cell: torch.Tensor | None = None,
    pbc: torch.Tensor | None = None,
    batch_idx: torch.Tensor | None = None,
    batch_ptr: torch.Tensor | None = None,
    max_neighbors: int | None = None,
    half_fill: bool = False,
    fill_value: int | None = None,
    return_neighbor_list: bool = False,
    *,
    return_distances: bool = False,
    return_vectors: bool = False,
    target_indices: torch.Tensor | None = None,
    backend: BackendName = "torch_reference",
    selection: BackendSelection | None = None,
    return_backend: bool = False,
    **kwargs: object,
) -> Any:
    """Dispatch neighbor construction and optionally return its audit record.

    ``selection`` is an optional pre-resolved registry decision supplied by a
    framework facade.  When present, it avoids resolving the same request a
    second time; direct ops callers can continue to pass ``backend``.
    """
    features = {
        "periodic" if cell is not None or pbc is not None else "no_pbc",
        "half" if half_fill else "full",
        "coo" if return_neighbor_list else "matrix",
    }
    if return_distances:
        features.add("distances")
    if return_vectors:
        features.add("vectors")
    if selection is None:
        selection = resolve_backend(
            backend,
            operation="neighbor_list",
            device=positions.device,
            dtype=positions.dtype,
            features=features,
        )
    elif selection.operation != "neighbor_list":
        raise ValueError(
            "pre-resolved backend selection must target operation "
            f"'neighbor_list', got {selection.operation!r}"
        )
    if selection.selected not in {"torch_reference", "torch_reference_cell_list"}:
        raise BackendUnavailableError(
            "the Torch dispatcher does not execute legacy Warp; use the framework "
            "legacy entry point or request a registered Torch backend"
        )
    neighbor_impl = (
        _neighbor_list
        if selection.selected == "torch_reference"
        else _neighbor_list_cell_list
    )
    result = neighbor_impl(
        positions,
        cutoff,
        cell=cell,
        pbc=pbc,
        batch_idx=batch_idx,
        batch_ptr=batch_ptr,
        max_neighbors=max_neighbors,
        half_fill=half_fill,
        fill_value=fill_value,
        return_neighbor_list=return_neighbor_list,
        return_distances=return_distances,
        return_vectors=return_vectors,
        target_indices=target_indices,
        **kwargs,
    )
    return (result, selection) if return_backend else result


def dispatch_lj_energy_forces(
    positions: torch.Tensor,
    neighbor_matrix: torch.Tensor,
    num_neighbors: torch.Tensor,
    *,
    epsilon: float,
    sigma: float,
    cutoff: float,
    half_list: bool = False,
    fill_value: int | None = None,
    switch_width: float = 0.0,
    cell: torch.Tensor | None = None,
    batch_idx: torch.Tensor | None = None,
    neighbor_matrix_shifts: torch.Tensor | None = None,
    backend: BackendName = "torch_reference",
    return_backend: bool = False,
) -> Any:
    """Dispatch LJ energy/force evaluation and optionally return its audit record."""
    features = {
        "periodic" if neighbor_matrix_shifts is not None else "no_pbc",
        "half" if half_list else "full",
        "forces",
    }
    selection = resolve_backend(
        backend,
        operation="lj_energy_forces",
        device=positions.device,
        dtype=positions.dtype,
        gradient_order=2,
        features=features,
    )
    if selection.selected != "torch_reference":
        raise BackendUnavailableError(
            "the Torch dispatcher does not execute legacy Warp; use the framework "
            "legacy entry point or request a registered Torch backend"
        )
    result = _lj_energy_forces(
        positions,
        neighbor_matrix,
        num_neighbors,
        epsilon=epsilon,
        sigma=sigma,
        cutoff=cutoff,
        half_list=half_list,
        fill_value=fill_value,
        switch_width=switch_width,
        cell=cell,
        batch_idx=batch_idx,
        neighbor_matrix_shifts=neighbor_matrix_shifts,
    )
    return (result, selection) if return_backend else result


__all__ = ["dispatch_lj_energy_forces", "dispatch_neighbor_list"]
