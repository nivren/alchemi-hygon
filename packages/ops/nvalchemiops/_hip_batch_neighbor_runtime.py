# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Admission contract for the staged native-HIP Batch neighbor pipeline.

This module deliberately describes the runtime boundary without registering an
implementation in :mod:`nvalchemiops.backend`.  The isolated native stages
already have executable boundaries, but the framework must not select them
until their combined ABI and unsupported-request behavior are explicit.

The contract is intentionally small:

``checked build init -> trusted reusable CSR workspace -> native candidate
query -> canonical full topology -> Torch geometry``.

The topology stages are forward-only and discrete.  Continuous distance/vector
geometry remains the Torch reference stage, so first- and second-order
gradients are part of this candidate contract.  This is a semantic admission
gate, not a claim that the native pipeline is registered or production-ready.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from nvalchemiops.backend import BackendUnavailableError


@dataclass(frozen=True, slots=True)
class NativeHipNeighborPipelineABI:
    """Minimum cross-stage ABI for the fixed-cell full-list candidate."""

    build_stage: str = "checked_init_then_trusted_reusable_csr"
    query_stage: str = "native_unordered_candidate_query"
    topology_stage: str = "native_composite_key_to_canonical_full_matrix"
    geometry_stage: str = "torch_reference_distance_vector"
    workspace_lifetime: str = (
        "caller_owned; reuse requires unchanged shape/dtype/device/alias/"
        "cell metadata/capacity; structural changes reinitialize"
    )
    query_order: str = "unspecified_until_topology_materialization"
    list_mode: str = "full"
    cell_mode: str = "fixed"
    geometry_gradient_order: int = 2
    supports_skin: bool = False
    supports_rebuild: bool = False
    supports_variable_cell: bool = False
    supports_target_indices: bool = False
    supports_pair_outputs: bool = False
    supports_distributed: bool = False
    native_geometry_registered: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Return a stable, report-friendly ABI record."""
        return asdict(self)


NATIVE_HIP_BATCH_NEIGHBOR_ABI = NativeHipNeighborPipelineABI()


@dataclass(frozen=True, slots=True)
class NativeHipNeighborPipelineRequest:
    """Normalized semantic request checked by the native-HIP admission gate.

    ``device_type`` and ``dtype`` are labels rather than Torch objects so the
    gate can be unit-tested without importing or initializing a HIP runtime.
    Framework integration can derive them from the actual tensors at the
    explicit backend boundary.
    """

    device_type: str
    dtype: str
    batch: bool = True
    fixed_cell: bool = True
    full_list: bool = True
    geometry_backend: str = "torch_reference"
    gradient_order: int = 2
    skin: float = 0.0
    rebuild: bool = False
    target_indices: bool = False
    pair_outputs: bool = False
    distributed: bool = False
    native_geometry: bool = False


def _reject(request: NativeHipNeighborPipelineRequest, reason: str) -> None:
    raise BackendUnavailableError(
        "native HIP Batch neighbor pipeline is not admitted for this request: "
        f"{reason}; request={request!r}"
    )


def validate_native_hip_neighbor_pipeline(
    request: NativeHipNeighborPipelineRequest,
) -> NativeHipNeighborPipelineABI:
    """Validate the minimum ABI without compiling or selecting native HIP.

    The function only checks semantic eligibility.  Device availability and
    implementation registration remain separate responsibilities of the
    central backend registry.  In particular, a CPU request is rejected rather
    than silently falling back to Torch.
    """

    if request.device_type not in {"cuda", "hip"}:
        _reject(request, "requires a HIP-capable Torch device (cuda/hip type)")
    if request.dtype not in {"float32", "float64"}:
        _reject(request, "only float32 and float64 are covered by the native ABI")
    if not request.batch:
        _reject(request, "Batch input is required by the current native ABI")
    if not request.fixed_cell:
        _reject(request, "variable-cell requests are not admitted")
    if not request.full_list:
        _reject(request, "half-list requests are not admitted")
    if request.geometry_backend != "torch_reference":
        _reject(
            request,
            "the admitted continuous geometry stage is Torch reference geometry",
        )
    if request.native_geometry:
        _reject(
            request,
            "native HIP geometry is an isolated forward-only candidate, not the runtime ABI",
        )
    if request.gradient_order not in {0, 1, 2}:
        _reject(request, "supported continuous geometry gradient orders are 0, 1, 2")
    if request.skin != 0.0 or request.rebuild:
        _reject(request, "skin/rebuild lifecycle is not admitted")
    if request.target_indices:
        _reject(request, "target-index neighbor queries are not admitted")
    if request.pair_outputs:
        _reject(request, "pair-function outputs are not admitted")
    if request.distributed:
        _reject(request, "distributed/domain-parallel execution is not admitted")
    return NATIVE_HIP_BATCH_NEIGHBOR_ABI


def run_native_hip_batch_neighbor_into(
    build_plan: Any,
    cells: torch.Tensor,
    cutoff: float,
    neighbor_search_radius: torch.Tensor,
    candidate_matrix: torch.Tensor,
    candidate_shifts: torch.Tensor,
    candidate_counts: torch.Tensor,
    public_matrix: torch.Tensor,
    public_shifts: torch.Tensor,
    public_counts: torch.Tensor,
    distances: torch.Tensor | None,
    vectors: torch.Tensor | None,
    topology_workspace: Any,
    *,
    fill_value: int | None = None,
    gradient_order: int = 2,
) -> None:
    """Run the explicitly admitted native-HIP hybrid neighbor pipeline.

    ``build_plan`` must be an already initialized
    :class:`_TrustedBatchCellBuildPlan`.  Initialization is intentionally
    separate: it performs the checked public build once, while this wrapper
    reuses the exact caller-owned build buffers and workspace on every call.
    The wrapper mutates only the candidate/public/geometry output buffers and
    the buffers owned by ``build_plan``.  ``distances`` and ``vectors`` may
    both be ``None`` for the framework topology-only executor; when either is
    supplied, the Torch geometry bridge preserves the differentiable path.

    This is an explicit ops-side call boundary.  It is not a registry executor,
    does not implement ``auto`` selection, and never falls back to Torch
    topology when the native request is not admitted.  Torch is used only for
    the final differentiable distance/vector geometry stage.
    """

    try:
        positions = build_plan.positions
        pbc = build_plan.pbc
        batch_idx = build_plan.batch_idx
        cells_per_dimension = build_plan.cells_per_dimension
        cell_offsets = build_plan.cell_offsets
        atom_periodic_shifts = build_plan.atom_periodic_shifts
        atom_to_cell_mapping = build_plan.atom_to_cell_mapping
        cell_counts = build_plan.cell_counts
        cell_starts = build_plan.cell_starts
        cell_atom_list = build_plan.cell_atom_list
    except AttributeError as exc:
        raise TypeError(
            "build_plan must be an initialized trusted Batch cell-build plan"
        ) from exc
    if not callable(getattr(build_plan, "run", None)):
        raise TypeError(
            "build_plan must be an initialized trusted Batch cell-build plan"
        )

    validate_native_hip_neighbor_pipeline(
        NativeHipNeighborPipelineRequest(
            device_type=str(getattr(positions.device, "type", positions.device)),
            dtype=str(positions.dtype).removeprefix("torch."),
            gradient_order=gradient_order,
        )
    )

    from nvalchemiops._hip_batch_cell_query import (  # noqa: PLC0415
        batch_query_cell_list_hip_into,
    )
    from nvalchemiops._hip_batch_query_materialize import (  # noqa: PLC0415
        materialize_batch_query_topology_composite_hip_into_with_workspace,
    )
    from nvalchemiops._torch_batch_query_materialization import (  # noqa: PLC0415
        materialize_batch_query_topology_geometry_into,
    )

    resolved_fill = positions.shape[0] if fill_value is None else int(fill_value)

    # The plan owns the checked/trusted build stages and all CSR buffers.
    build_plan.run()
    batch_query_cell_list_hip_into(
        positions,
        cells,
        pbc,
        cutoff,
        batch_idx,
        cells_per_dimension,
        neighbor_search_radius,
        cell_offsets,
        atom_periodic_shifts,
        atom_to_cell_mapping,
        cell_counts,
        cell_starts,
        cell_atom_list,
        candidate_matrix,
        candidate_shifts,
        candidate_counts,
        half_fill=False,
        fill_value=resolved_fill,
    )
    materialize_batch_query_topology_composite_hip_into_with_workspace(
        candidate_matrix,
        candidate_shifts,
        candidate_counts,
        public_matrix,
        public_shifts,
        public_counts,
        topology_workspace,
        fill_value=resolved_fill,
    )
    if distances is not None or vectors is not None:
        materialize_batch_query_topology_geometry_into(
            positions,
            cells,
            batch_idx,
            public_matrix,
            public_shifts,
            public_counts,
            distances=distances,
            vectors=vectors,
        )


__all__ = [
    "NATIVE_HIP_BATCH_NEIGHBOR_ABI",
    "NativeHipNeighborPipelineABI",
    "NativeHipNeighborPipelineRequest",
    "run_native_hip_batch_neighbor_into",
    "validate_native_hip_neighbor_pipeline",
]
