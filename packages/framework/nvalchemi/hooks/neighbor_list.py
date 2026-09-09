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
"""Neighbor list hook for on-the-fly neighbor list construction.

This module provides :class:`NeighborListHook`, which runs at the
``BEFORE_COMPUTE`` stage to compute or refresh the neighbor list stored in
the batch before the model forward pass.  It supports an optional Verlet
skin buffer to avoid recomputing neighbors every step.

Both ``MATRIX`` and ``COO`` neighbor formats are supported for dynamic
updates (i.e. updates each dynamics step).  For ``COO`` format the hook
creates or replaces the edges group on the batch each step so that
``batch.neighbor_list`` (shape ``(E, 2)``) and ``batch.neighbor_list_shifts``
(shape ``(E, 3)``, PBC only) are always up to date.  The companion
``Batch.edge_ptr`` property derives the per-atom CSR pointer on demand.

Pre-allocation
--------------
The hook maintains *staging buffers* — persistent GPU tensors that are
refreshed each step via ``Tensor.copy_()`` — to avoid per-step dynamic
allocation inside the ``neighbor_list`` dispatcher.

For PBC systems, ``NeighborListHook`` runs the ``nvalchemiops`` selector
once per batch shape and caches the resulting explicit strategy name (for
example ``batch_cell_list_pair_centric`` or ``batch_cluster_tile``). For
non-PBC systems it keeps the legacy size threshold between naive and cell-list
methods. The selected paths normally allocate auxiliary tensors on demand with
CPU-GPU syncs (e.g. ``.item()`` calls), so
:meth:`NeighborListHook._alloc_nl_kwargs` computes these **once** when the batch
shape is first seen (or changes) and caches them in
``NeighborListHook._buf_nl_kwargs``:

* *Naive, no PBC*: no extra kwargs needed.
* *Naive, PBC*: ``shift_range_per_dimension``, ``num_shifts_per_system``,
  ``max_shifts_per_system``, and ``max_atoms_per_system``.
* *Cell list*: seven cell-list scratch tensors via ``allocate_cell_list``.
* *Cluster tile*: batch cluster-tile sort/group/tile scratch tensors.

**NPT note**: geometry-dependent kwargs (shift ranges, cell-list sizes) are
fixed when the staging buffers are first allocated for a given ``(N, B)``
shape.  For NPT (variable-cell) simulations the pre-computed values may
become stale as the cell changes; accuracy is maintained by keeping the
cutoff + skin well below the shortest cell dimension throughout the run.
"""

from __future__ import annotations

from enum import Enum

import torch
from nvalchemiops.backend import (
    BackendSelection,
    BackendUnavailableError,
    validate_backend_name,
)
from nvalchemiops.torch_backend import dispatch_neighbor_list
from nvalchemiops.torch_reference import NeighborOverflowError

from nvalchemi._backend import resolve_compute_backend
from nvalchemi.data import Batch
from nvalchemi.hooks._context import HookContext
from nvalchemi.models.base import NeighborConfig, NeighborListFormat
from nvalchemi.neighbors import _write_neighbor_data_to_batch

# Warp-facing dependencies are loaded only when the legacy backend is selected.
# This keeps an explicit Torch reference hook usable in environments where Warp
# is intentionally absent, while preserving the original implementation once
# the default path is entered.
_WARP_DEPENDENCIES_READY = False
neighbor_list_strategy_run_args = None
estimate_max_neighbors = None
neighbor_list = None
suggest_neighbor_list_method = None
batch_neighbor_list_needs_rebuild = None
estimate_batch_cell_list_sizes = None
allocate_batch_cluster_tile_list = None
estimate_batch_max_tiles_per_group = None
allocate_cell_list = None
compute_naive_num_shifts = None


def _initialize_warp_dependencies() -> None:
    """Import the legacy Warp neighbor implementation on demand."""
    global _WARP_DEPENDENCIES_READY
    global neighbor_list_strategy_run_args, estimate_max_neighbors, neighbor_list
    global suggest_neighbor_list_method, batch_neighbor_list_needs_rebuild
    global estimate_batch_cell_list_sizes
    global allocate_batch_cluster_tile_list, estimate_batch_max_tiles_per_group
    global allocate_cell_list, compute_naive_num_shifts

    if _WARP_DEPENDENCIES_READY:
        return

    from nvalchemiops.neighbors.base_dispatch import (
        neighbor_list_strategy_run_args as _neighbor_list_strategy_run_args,
    )
    from nvalchemiops.neighbors.neighbor_utils import (
        estimate_max_neighbors as _estimate_max_neighbors,
    )
    from nvalchemiops.torch.neighbors import (
        neighbor_list as _neighbor_list,
    )
    from nvalchemiops.torch.neighbors import (
        suggest_neighbor_list_method as _suggest_neighbor_list_method,
    )
    from nvalchemiops.torch.neighbors.rebuild_detection import (
        batch_neighbor_list_needs_rebuild as _batch_neighbor_list_needs_rebuild,
    )

    try:
        from nvalchemiops.torch.neighbors.batch_cell_list import (
            estimate_batch_cell_list_sizes as _estimate_batch_cell_list_sizes,
        )
    except ImportError:
        _estimate_batch_cell_list_sizes = None

    try:
        from nvalchemiops.torch.neighbors.batch_cluster_tile import (
            allocate_batch_cluster_tile_list as _allocate_batch_cluster_tile_list,
        )
        from nvalchemiops.torch.neighbors.batch_cluster_tile import (
            estimate_batch_max_tiles_per_group as _estimate_batch_max_tiles_per_group,
        )
    except ImportError:
        _allocate_batch_cluster_tile_list = None
        _estimate_batch_max_tiles_per_group = None

    try:
        from nvalchemiops.torch.neighbors.neighbor_utils import (
            allocate_cell_list as _allocate_cell_list,
        )
        from nvalchemiops.torch.neighbors.neighbor_utils import (
            compute_naive_num_shifts as _compute_naive_num_shifts,
        )
    except ImportError:
        _allocate_cell_list = None
        _compute_naive_num_shifts = None

    neighbor_list_strategy_run_args = _neighbor_list_strategy_run_args
    estimate_max_neighbors = _estimate_max_neighbors
    neighbor_list = _neighbor_list
    suggest_neighbor_list_method = _suggest_neighbor_list_method
    batch_neighbor_list_needs_rebuild = _batch_neighbor_list_needs_rebuild
    estimate_batch_cell_list_sizes = _estimate_batch_cell_list_sizes
    allocate_batch_cluster_tile_list = _allocate_batch_cluster_tile_list
    estimate_batch_max_tiles_per_group = _estimate_batch_max_tiles_per_group
    allocate_cell_list = _allocate_cell_list
    compute_naive_num_shifts = _compute_naive_num_shifts
    _WARP_DEPENDENCIES_READY = True


class NeighborListHook:
    """Compute and cache neighbor lists before each model evaluation.

    This hook runs at :attr:`~DynamicsStage.BEFORE_COMPUTE` and writes
    neighbor data into the batch so that the model's ``adapt_input`` can
    read it.  An optional Verlet skin buffer avoids rebuilding the list
    every step: the list is only recomputed when the maximum raw Cartesian
    displacement since the last build exceeds ``config.skin / 2``, or when
    the set of active systems changes (detected via ``system_id``).

    For ``MATRIX`` format the following tensors are written to the atoms
    group of the batch (and thus accessible as ``batch.neighbor_matrix``
    etc.):

    * ``neighbor_matrix`` — shape ``(N, max_neighbors)``, int32
    * ``num_neighbors``   — shape ``(N,)``, int32
    * ``neighbor_matrix_shifts`` — shape ``(N, max_neighbors, 3)``, int32
      (only written when PBC is active)

    For ``COO`` format the edges group of the batch is created or replaced
    on every rebuild, making the following accessible:

    * ``batch.neighbor_list`` — shape ``(E, 2)``, int32 (nvalchemi convention)
    * ``batch.neighbor_list_shifts`` — shape ``(E, 3)``, int32 (only when PBC active)
    * ``batch.edge_ptr`` — shape ``(N+1,)``, int32, derived on demand via
      the :attr:`~nvalchemi.data.Batch.edge_ptr` property

    Parameters
    ----------
    config : NeighborConfig
        Neighbor list configuration read from the model config.
    skin : float, optional
        Verlet skin distance in the same length units as positions.
        The neighbor list is searched out to ``cutoff + skin`` so that
        atoms crossing the skin boundary but not the bare cutoff are
        already included.  The list is only rebuilt when any atom's raw
        Cartesian position has moved more than ``skin / 2`` since the
        previous build (requires ``nvalchemiops >= 0.4``); set to ``0.0``
        (default) to rebuild every step.
    max_neighbors : int | None, optional
        Maximum number of neighbors per atom for MATRIX format.  When
        ``None`` (default), auto-estimated from the cutoff via
        ``estimate_max_neighbors(cutoff)``.  Ignored for COO format.
    stage : Enum | None, optional
        Explicit workflow stage at which this hook runs.  Defaults to
        ``None``; when ``None`` the hook fires at
        ``DynamicsStage.BEFORE_COMPUTE`` (see the class description above).
    method : str | None, optional
        Neighbor operation strategy for an explicit backend family. For the
        current reference path, ``backend="torch_reference",
        method="cell_list"`` selects the no-PBC uniform cell-list
        implementation. The legacy Warp path retains its upstream method
        handling when ``backend`` is ``None`` or ``"warp"``.
    backend : str | None, optional
        Compute backend request. It is resolved for this operation and its
        requested features by :func:`nvalchemiops.backend.resolve_backend`;
        see :func:`nvalchemiops.backend.backend_capabilities` for the
        authoritative capability table. ``None`` preserves the upstream Warp
        path. The current reference hook supports full periodic lists and a
        small-input cached skin path; unsupported combinations fail
        explicitly. ``auto`` does not choose an operation strategy in M1.
    """

    def __init__(
        self,
        config: NeighborConfig,
        skin: float = 0.0,
        max_neighbors: int | None = None,
        stage: Enum | None = None,
        method: str | None = None,
        backend: str | None = None,
    ) -> None:
        validate_backend_name(backend)
        if backend in (None, "warp"):
            _initialize_warp_dependencies()
        if skin < 0.0:
            raise ValueError("skin must be non-negative")

        self.config = config
        self.skin = skin
        self.stage = stage
        self.method = method
        self.backend = backend
        self._max_neighbors_override = max_neighbors
        self.frequency = 1
        self._neighbor_list_flag = config.format == NeighborListFormat.COO

        # Skin-buffer state: populated after the first build.
        self._ref_positions: torch.Tensor | None = None
        self._rebuild_flags: torch.Tensor | None = None
        self._ref_cell: torch.Tensor | None = None
        self._ref_pbc: torch.Tensor | None = None
        self._ref_batch_idx: torch.Tensor | None = None
        self._ref_batch_ptr: torch.Tensor | None = None

        # Neighbor Matrix state: populated after the first build.
        self._neighbor_matrix: torch.Tensor | None = None
        self._col_range: torch.Tensor | None = None
        self._num_neighbors: torch.Tensor | None = None
        self._neighbor_matrix_shifts: torch.Tensor | None = None
        self._max_neighbors: int | None = None

        # Shape the staging buffers were allocated for; used to detect when
        # re-allocation is needed (e.g. inflight batching with variable load).
        self._alloc_N: int | None = None
        self._alloc_B: int | None = None

        # Staging buffers — persistent GPU tensors refreshed each step via
        # copy_() to avoid per-step dynamic allocation inside the dispatcher.
        self._buf_positions: torch.Tensor | None = None
        self._buf_batch_idx: torch.Tensor | None = None
        self._buf_batch_ptr: torch.Tensor | None = None
        self._buf_cell: torch.Tensor | None = None  # PBC only
        self._buf_pbc: torch.Tensor | None = None  # PBC only

        # Algorithm-specific pre-allocated kwargs forwarded to neighbor_list.
        self._buf_nl_kwargs: dict[str, torch.Tensor | int] = {}
        self._neighbor_list_method: str | None = None

        # Adaptive K-dimension state.
        self._actual_max_k: torch.Tensor | None = None  # GPU scalar from last build
        self._first_build: bool = True  # Force sync check after first kernel call

    # ------------------------------------------------------------------
    # Main hook entry point
    # ------------------------------------------------------------------
    @torch.compile(fullgraph=False, mode="max-autotune-no-cudagraphs")
    def __call__(self, ctx: HookContext, stage: Enum) -> None:
        """Recompute the neighbor list if needed and write it to the batch.

        When ``skin > 0``,
        :func:`~nvalchemiops.torch.neighbors.rebuild_detection.batch_neighbor_list_needs_rebuild`,
        the list is only rebuilt when at least one atom's raw Cartesian position
        has moved more than ``skin / 2`` since the previous build.  The reference
        positions are updated in-place on the GPU whenever a rebuild occurs.
        """
        pbc = getattr(ctx.batch, "pbc", None)
        if pbc is not None and not bool(pbc.any()):
            pbc = None
        selection = resolve_compute_backend(
            self.backend,
            operation="neighbor_list",
            device=ctx.batch.positions.device,
            dtype=ctx.batch.positions.dtype,
            features={
                "periodic" if pbc is not None else "no_pbc",
                "half" if self.config.half_list else "full",
                "matrix"
                if self.config.format == NeighborListFormat.MATRIX
                else "coo",
            },
            strategy=(
                self.method if self.backend not in (None, "warp") else None
            ),
        )
        if selection.family == "torch_reference":
            self._rebuild_reference(ctx.batch, selection=selection)
            return

        if selection.selected != "warp":
            raise BackendUnavailableError(
                f"NeighborListHook has no executor for selected backend "
                f"{selection.selected!r}"
            )

        self._rebuild(ctx.batch)

        # First build: initialise the skin-buffer reference (one-time clone).
        if self.skin > 0.0 and self._ref_positions is None:
            self._init_ref_positions(ctx.batch.positions)

    @torch.compiler.disable
    def _rebuild_reference(
        self, batch: Batch, *, selection: BackendSelection
    ) -> None:
        """Run the restricted Warp-independent reference neighbor path."""
        pbc = getattr(batch, "pbc", None)
        cell = getattr(batch, "cell", None)
        if pbc is not None and not bool(pbc.any()):
            pbc = None
            cell = None

        rebuild_flags = None
        if self.skin > 0.0:
            rebuild_flags = self._reference_rebuild_flags(
                batch, cell=cell, pbc=pbc
            )
            if rebuild_flags is not None:
                if not bool(torch.any(rebuild_flags)):
                    self._write_reference_cache(batch)
                    return
                self._rebuild_reference_systems(
                    batch,
                    cell=cell,
                    pbc=pbc,
                    rebuild_flags=rebuild_flags,
                    selection=selection,
                )
                self._write_reference_cache(batch)
                return

        shape_changed = (
            self._neighbor_matrix is None
            or self._neighbor_matrix.shape[0] != batch.num_nodes
        )
        if shape_changed:
            self._max_neighbors = self._max_neighbors_override
        result, capacity = self._dispatch_reference(
            positions=batch.positions,
            cutoff=self.config.cutoff + self.skin,
            cell=cell,
            pbc=pbc,
            batch_idx=batch.batch_idx,
            batch_ptr=batch.batch_ptr,
            fill_value=batch.num_nodes,
            capacity=self._max_neighbors,
            selection=selection,
        )
        neighbor_matrix, num_neighbors, neighbor_matrix_shifts = self._unpack_reference_result(
            result, pbc=pbc
        )
        self._max_neighbors = capacity
        self._neighbor_matrix = neighbor_matrix
        self._num_neighbors = num_neighbors
        self._neighbor_matrix_shifts = neighbor_matrix_shifts
        self._shrink_reference_capacity(batch.num_nodes, pbc=pbc)
        if self.skin > 0.0:
            self._ref_positions = batch.positions.detach().clone()
            self._ref_cell = None if cell is None else cell.detach().clone()
            self._ref_pbc = None if pbc is None else pbc.detach().clone()
            self._ref_batch_idx = batch.batch_idx.detach().clone()
            self._ref_batch_ptr = batch.batch_ptr.detach().clone()
        _write_neighbor_data_to_batch(
            batch=batch,
            neighbor_matrix=neighbor_matrix,
            num_neighbors=num_neighbors,
            neighbor_matrix_shifts=neighbor_matrix_shifts,
            format=self.config.format,
            cutoff=self.config.cutoff,
            backend=self.backend,
        )

    @staticmethod
    def _round_reference_k(value: int) -> int:
        """Round a staging K dimension up to the 16-entry alignment."""
        if value <= 0:
            return 0
        return ((value + 15) // 16) * 16

    @staticmethod
    def _unpack_reference_result(
        result: tuple[torch.Tensor, ...], *, pbc: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if pbc is None:
            neighbor_matrix, num_neighbors = result
            return neighbor_matrix, num_neighbors, None
        neighbor_matrix, num_neighbors, neighbor_matrix_shifts = result
        return neighbor_matrix, num_neighbors, neighbor_matrix_shifts

    @staticmethod
    def _pad_reference_result(
        result: tuple[torch.Tensor, ...],
        *,
        pbc: torch.Tensor | None,
        capacity: int,
        fill_value: int,
    ) -> tuple[torch.Tensor, ...]:
        """Pad a reference result to the staging capacity without changing counts."""
        matrix, counts, shifts = NeighborListHook._unpack_reference_result(
            result, pbc=pbc
        )
        if matrix.shape[1] == capacity:
            return result
        padded = torch.full(
            (matrix.shape[0], capacity),
            fill_value,
            dtype=matrix.dtype,
            device=matrix.device,
        )
        if matrix.shape[1] > 0:
            padded[:, : matrix.shape[1]].copy_(matrix)
        if pbc is None:
            return padded, counts
        padded_shifts = torch.zeros(
            matrix.shape[0], capacity, 3, dtype=shifts.dtype, device=shifts.device
        )
        if shifts.shape[1] > 0:
            padded_shifts[:, : shifts.shape[1]].copy_(shifts)
        return padded, counts, padded_shifts

    def _reference_growth_capacity(self, actual: int, current: int) -> int:
        """Return the aligned capacity required by the reference grow contract."""
        new_capacity = self._round_reference_k(int(actual * 1.5))
        if new_capacity <= current:
            new_capacity = self._round_reference_k(current + 1)
        if self._max_neighbors_override is not None:
            new_capacity = max(new_capacity, self._max_neighbors_override)
        return new_capacity

    @torch.compiler.disable
    def _dispatch_reference(
        self,
        *,
        positions: torch.Tensor,
        cutoff: float,
        cell: torch.Tensor | None,
        pbc: torch.Tensor | None,
        batch_idx: torch.Tensor | None,
        batch_ptr: torch.Tensor | None,
        fill_value: int,
        capacity: int | None,
        selection: BackendSelection,
    ) -> tuple[tuple[torch.Tensor, ...], int]:
        """Dispatch reference neighbors with staging grow-and-retry semantics."""

        def run(limit: int | None) -> tuple[torch.Tensor, ...]:
            return dispatch_neighbor_list(
                positions=positions,
                cutoff=cutoff,
                cell=cell,
                pbc=pbc,
                batch_idx=batch_idx,
                batch_ptr=batch_ptr,
                max_neighbors=limit,
                half_fill=self.config.half_list,
                fill_value=fill_value,
                selection=selection,
            )

        if capacity is None:
            result = run(None)
            _, counts, _ = self._unpack_reference_result(result, pbc=pbc)
            actual = int(counts.max().item()) if counts.numel() else 0
            capacity = self._round_reference_k(actual)
            if actual > 0 and actual >= capacity:
                capacity = self._reference_growth_capacity(actual, capacity)
            return (
                self._pad_reference_result(
                    result,
                    pbc=pbc,
                    capacity=capacity,
                    fill_value=fill_value,
                ),
                capacity,
            )

        try:
            result = run(capacity)
        except NeighborOverflowError:
            discovered = run(None)
            _, counts, _ = self._unpack_reference_result(discovered, pbc=pbc)
            actual = int(counts.max().item()) if counts.numel() else 0
            if actual <= capacity:
                raise
            capacity = self._reference_growth_capacity(actual, capacity)
            result = run(capacity)
            return result, capacity

        _, counts, _ = self._unpack_reference_result(result, pbc=pbc)
        actual = int(counts.max().item()) if counts.numel() else 0
        if actual > 0 and actual >= capacity:
            capacity = self._reference_growth_capacity(actual, capacity)
            result = run(capacity)
        return result, capacity

    @torch.compiler.disable
    def _resize_reference_k(
        self, N: int, capacity: int, *, pbc: torch.Tensor | None
    ) -> None:
        """Resize the reference staging matrix while preserving cached systems."""
        if self._neighbor_matrix is None or self._num_neighbors is None:
            self._max_neighbors = capacity
            return
        old_capacity = self._neighbor_matrix.shape[1]
        if old_capacity == capacity:
            self._max_neighbors = capacity
            return
        fill_value = N
        matrix = torch.full(
            (N, capacity),
            fill_value,
            dtype=self._neighbor_matrix.dtype,
            device=self._neighbor_matrix.device,
        )
        copy_width = min(old_capacity, capacity)
        if copy_width > 0:
            matrix[:, :copy_width].copy_(self._neighbor_matrix[:, :copy_width])
        self._neighbor_matrix = matrix
        if pbc is None:
            self._neighbor_matrix_shifts = None
        else:
            shifts = torch.zeros(
                N, capacity, 3, dtype=torch.int32, device=self._neighbor_matrix.device
            )
            if self._neighbor_matrix_shifts is not None and copy_width > 0:
                shifts[:, :copy_width].copy_(
                    self._neighbor_matrix_shifts[:, :copy_width]
                )
            self._neighbor_matrix_shifts = shifts
        self._max_neighbors = capacity
        self._col_range = torch.arange(
            capacity, dtype=torch.int32, device=self._neighbor_matrix.device
        )

    @torch.compiler.disable
    def _shrink_reference_capacity(
        self, N: int, *, pbc: torch.Tensor | None
    ) -> None:
        """Trim an idle reference staging matrix without rerunning neighbors."""
        if self._max_neighbors is None or self._num_neighbors is None:
            return
        actual = int(self._num_neighbors.max().item()) if self._num_neighbors.numel() else 0
        if actual <= 0 or actual >= 0.5 * self._max_neighbors:
            return
        new_capacity = self._round_reference_k(actual * 2)
        if self._max_neighbors_override is not None:
            new_capacity = max(new_capacity, self._max_neighbors_override)
        if new_capacity < self._max_neighbors:
            self._resize_reference_k(N, new_capacity, pbc=pbc)

    def _write_reference_cache(self, batch: Batch) -> None:
        if self._neighbor_matrix is None or self._num_neighbors is None:
            raise RuntimeError("reference neighbor cache is incomplete")
        _write_neighbor_data_to_batch(
            batch=batch,
            neighbor_matrix=self._neighbor_matrix,
            num_neighbors=self._num_neighbors,
            neighbor_matrix_shifts=self._neighbor_matrix_shifts,
            format=self.config.format,
            cutoff=self.config.cutoff,
            backend=self.backend,
        )

    @torch.compiler.disable
    def _rebuild_reference_systems(
        self,
        batch: Batch,
        *,
        cell: torch.Tensor | None,
        pbc: torch.Tensor | None,
        rebuild_flags: torch.Tensor,
        selection: BackendSelection,
    ) -> None:
        """Refresh only systems whose reference skin has become stale.

        This is deliberately an eager Torch reference implementation.  It
        keeps the global matrix/index contract while rebuilding each changed
        system with local indices and then restoring the batch offset.  The
        optimized Warp path retains its own preallocated per-system rebuild
        kernel; this method does not claim cell-list performance or dynamic
        capacity growth.
        """
        if self._neighbor_matrix is None or self._num_neighbors is None:
            raise RuntimeError("reference neighbor cache is incomplete")
        capacity = self._max_neighbors
        if capacity is None:
            capacity = self._neighbor_matrix.shape[1]
        fill_value = batch.num_nodes
        systems = torch.nonzero(rebuild_flags, as_tuple=False).flatten().tolist()
        while True:
            pending: list[tuple[int, tuple[torch.Tensor, ...]]] = []
            grew = False
            for system in systems:
                start = int(batch.batch_ptr[system].item())
                end = int(batch.batch_ptr[system + 1].item())
                local_cell = None if cell is None else cell[system]
                local_pbc = None if pbc is None else pbc[system]
                result, new_capacity = self._dispatch_reference(
                    positions=batch.positions[start:end],
                    cutoff=self.config.cutoff + self.skin,
                    cell=local_cell,
                    pbc=local_pbc,
                    batch_idx=None,
                    batch_ptr=None,
                    fill_value=fill_value,
                    capacity=capacity,
                    selection=selection,
                )
                if new_capacity != capacity:
                    self._resize_reference_k(batch.num_nodes, new_capacity, pbc=pbc)
                    capacity = new_capacity
                    grew = True
                    break
                pending.append((system, result))
            if grew:
                continue

            for system, result in pending:
                start = int(batch.batch_ptr[system].item())
                end = int(batch.batch_ptr[system + 1].item())
                local_pbc = None if pbc is None else pbc[system]
                local_matrix, local_counts, local_shifts = self._unpack_reference_result(
                    result, pbc=local_pbc
                )
                active = torch.arange(capacity, device=batch.device).unsqueeze(0)
                active = active < local_counts.to(torch.long).unsqueeze(1)
                global_matrix = local_matrix.clone()
                global_matrix[active] += start
                self._neighbor_matrix[start:end].copy_(global_matrix)
                self._num_neighbors[start:end].copy_(local_counts)
                if self._neighbor_matrix_shifts is not None:
                    if local_shifts is None:
                        self._neighbor_matrix_shifts[start:end].zero_()
                    else:
                        self._neighbor_matrix_shifts[start:end].copy_(local_shifts)

                with torch.no_grad():
                    self._ref_positions[start:end].copy_(batch.positions[start:end])
                    if self._ref_cell is not None and cell is not None:
                        self._ref_cell[system].copy_(cell[system])
                    if self._ref_pbc is not None and pbc is not None:
                        self._ref_pbc[system].copy_(pbc[system])
            break

        self._shrink_reference_capacity(batch.num_nodes, pbc=pbc)

    @torch.compiler.disable
    def _reference_needs_rebuild(
        self,
        batch: Batch,
        *,
        cell: torch.Tensor | None,
        pbc: torch.Tensor | None,
    ) -> bool:
        """Return whether any cached reference system is stale."""
        flags = self._reference_rebuild_flags(batch, cell=cell, pbc=pbc)
        return flags is None or bool(torch.any(flags))

    def _reference_rebuild_flags(
        self,
        batch: Batch,
        *,
        cell: torch.Tensor | None,
        pbc: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Return per-system stale flags, or ``None`` when a full rebuild is needed."""
        if self._ref_positions is None or self._neighbor_matrix is None:
            return None
        if self._ref_positions.shape != batch.positions.shape:
            return None
        if self._ref_positions.device != batch.positions.device:
            return None
        if self._ref_batch_idx is None or not torch.equal(
            self._ref_batch_idx, batch.batch_idx
        ):
            return None
        if self._ref_batch_ptr is None or not torch.equal(
            self._ref_batch_ptr, batch.batch_ptr
        ):
            return None
        if (self._ref_cell is None) != (cell is None) or (self._ref_pbc is None) != (
            pbc is None
        ):
            return None
        rebuild_flags = torch.zeros(
            batch.num_graphs, dtype=torch.bool, device=batch.positions.device
        )
        if cell is not None and self._ref_cell is not None:
            rebuild_flags |= torch.any(self._ref_cell != cell, dim=(1, 2))
        if pbc is not None and self._ref_pbc is not None:
            rebuild_flags |= torch.any(self._ref_pbc != pbc, dim=1)
        displacement = batch.positions - self._ref_positions
        threshold_sq = (self.skin / 2.0) ** 2
        for system in range(batch.num_graphs):
            start = int(batch.batch_ptr[system].item())
            end = int(batch.batch_ptr[system + 1].item())
            if end > start and bool(
                torch.any(displacement[start:end].square().sum(dim=1) > threshold_sq)
            ):
                rebuild_flags[system] = True
        return rebuild_flags

    @torch.compiler.disable
    def _init_ref_positions(self, positions: torch.Tensor) -> None:
        """One-time clone of positions into the skin-buffer reference.

        Marked ``@torch.compiler.disable`` because the attribute assignment
        is a Python mutation that creates a graph break.  Called only on the
        first step for a given batch shape.
        """
        self._ref_positions = positions.detach().clone()

    # ------------------------------------------------------------------
    # Neighbor list construction
    # ------------------------------------------------------------------

    def _rebuild(self, batch: Batch) -> None:
        """Build the neighbor list and write results into the batch."""
        positions = batch.positions  # (N, 3)
        batch_ptr = batch.batch_ptr  # (B+1,)
        N = batch.num_nodes
        B = batch.num_graphs

        # Detect PBC.  getattr avoids a try/except which is a graph break.
        pbc = getattr(batch, "pbc", None)  # (B, 3) bool or None
        cell = getattr(batch, "cell", None)  # (B, 3, 3) float or None

        # ------------------------------------------------------------------
        # Allocate (or reallocate) the output tensors when shape changes.
        # Reallocation also resets the skin-buffer state so that the first
        # subsequent step forces a full rebuild and re-initialises
        # _ref_positions for the new atom count.
        # ------------------------------------------------------------------
        if self._neighbor_matrix is None or self._neighbor_matrix.shape[0] != N:
            self._alloc_output_tensors(N, batch, pbc)

        # ------------------------------------------------------------------
        # (Re)allocate staging buffers and algorithm kwargs on shape change.
        # ------------------------------------------------------------------
        if self._alloc_N != N or self._alloc_B != B:
            # Composition changed — check K before staging realloc.
            self._check_and_resize_k(N, batch.device, pbc)
            self._alloc_staging_buffers(
                N,
                B,
                positions.dtype,
                batch.device,
                cell,
                pbc,
                batch_ptr,
                positions=positions,
                batch_idx=batch.batch_idx,
            )
            self._alloc_N = N
            self._alloc_B = B

        # Refresh staging buffers from the current batch.
        self._copy_to_staging_buffers(positions, batch_ptr, batch.batch_idx, cell, pbc)

        # ------------------------------------------------------------------
        # Skin check: decide per-system whether the neighbor list needs
        # rebuilding based on raw Cartesian displacement since the last
        # build.  Minimum-image displacement is not valid here: wrapping an
        # atom changes the coordinate representation that cached periodic
        # shifts refer to and must therefore trigger a rebuild.
        # ------------------------------------------------------------------
        if self.skin > 0.0 and self._ref_positions is not None:
            self._rebuild_flags = batch_neighbor_list_needs_rebuild(
                reference_positions=self._ref_positions,
                current_positions=self._buf_positions,
                batch_idx=self._buf_batch_idx,
                skin_distance_threshold=self.skin / 2,
                update_reference_positions=True,
                num_systems=B,
            )

        # ------------------------------------------------------------------
        # Build the neighbor list using pre-allocated buffers.
        # ------------------------------------------------------------------
        neighbor_list(
            positions=self._buf_positions,
            cutoff=self.config.cutoff + self.skin,
            cell=self._buf_cell,
            pbc=self._buf_pbc,
            max_neighbors=self._max_neighbors,
            half_fill=self.config.half_list,
            batch_ptr=self._buf_batch_ptr,
            batch_idx=self._buf_batch_idx,
            neighbor_matrix=self._neighbor_matrix,
            num_neighbors=self._num_neighbors,
            neighbor_matrix_shifts=self._neighbor_matrix_shifts,
            rebuild_flags=self._rebuild_flags,
            method=self._neighbor_list_method,
            **self._buf_nl_kwargs,
        )

        # ------------------------------------------------------------------
        # Adaptive K: first-build check (runs once, then never again).
        # This is the only per-step adaptive K code.  After the first
        # build, all checks are gated on structural events (N/B change)
        # inside _alloc_output_tensors / _alloc_staging_buffers.
        # ------------------------------------------------------------------
        if self._first_build:
            self._first_build = False
            self._actual_max_k = self._num_neighbors.max()
            grew = self._check_and_resize_k(N, batch.device, pbc)
            if grew:
                # K was too small — re-run kernel with larger buffers.
                neighbor_list(
                    positions=self._buf_positions,
                    cutoff=self.config.cutoff + self.skin,
                    cell=self._buf_cell,
                    pbc=self._buf_pbc,
                    max_neighbors=self._max_neighbors,
                    half_fill=self.config.half_list,
                    batch_ptr=self._buf_batch_ptr,
                    batch_idx=self._buf_batch_idx,
                    neighbor_matrix=self._neighbor_matrix,
                    num_neighbors=self._num_neighbors,
                    neighbor_matrix_shifts=self._neighbor_matrix_shifts,
                    rebuild_flags=None,  # Force full rebuild
                    method=self._neighbor_list_method,
                    **self._buf_nl_kwargs,
                )

        # ------------------------------------------------------------------
        # Mark Stale Entries
        # ------------------------------------------------------------------
        stale = self._col_range.unsqueeze(0) >= self._num_neighbors.unsqueeze(1)
        self._neighbor_matrix[stale] = batch.num_nodes
        if self._neighbor_matrix_shifts is not None:
            self._neighbor_matrix_shifts[stale] = 0

        # ------------------------------------------------------------------
        # Post-processing: write results to batch (shared with compute_neighbors)
        # ------------------------------------------------------------------
        _write_neighbor_data_to_batch(
            batch=batch,
            neighbor_matrix=self._neighbor_matrix,
            num_neighbors=self._num_neighbors,
            neighbor_matrix_shifts=self._neighbor_matrix_shifts,
            format=NeighborListFormat.COO
            if self._neighbor_list_flag
            else NeighborListFormat.MATRIX,
            cutoff=self.config.cutoff,
        )

    # ------------------------------------------------------------------
    # Staging buffer management
    # ------------------------------------------------------------------

    @torch.compiler.disable
    def _alloc_output_tensors(
        self,
        N: int,
        batch: "Batch",
        pbc: torch.Tensor | None,
    ) -> None:
        """Allocate neighbor-matrix output tensors for atom count *N*.

        Marked ``@torch.compiler.disable`` because it calls
        ``estimate_max_neighbors`` (CPU work), allocates tensors with
        dynamic shapes, and mutates Python attributes — all graph breaks.
        Called only when the atom count changes.
        """
        device = batch.device
        max_nbrs = self._max_neighbors_override
        if max_nbrs is None:
            max_nbrs = estimate_max_neighbors(
                cutoff=self.config.cutoff + self.skin,
            )
        # Non-PBC hard cap: an atom can see at most (N_system - 1)
        # neighbors without periodic images.  We use max_num_nodes
        # (not max_num_nodes - 1) so that K has one sentinel slot
        # to distinguish "all used" from "overflow" in the adaptive check.
        # Round up to nearest 16 for memory-aligned kernel performance.
        if pbc is None and batch.max_num_nodes > 0:
            cap = ((batch.max_num_nodes + 15) // 16) * 16
            max_nbrs = min(max_nbrs, cap)
        self._max_neighbors = max_nbrs
        self._neighbor_matrix = torch.full(
            (N, max_nbrs), N, dtype=torch.int32, device=device
        )
        self._col_range = torch.arange(max_nbrs, device=device, dtype=torch.int32)
        self._num_neighbors = torch.zeros(N, dtype=torch.int32, device=device)
        if pbc is not None:
            self._neighbor_matrix_shifts = torch.zeros(
                N, max_nbrs, 3, dtype=torch.int32, device=device
            )
        # Reset skin-buffer state so __call__ re-initialises _ref_positions.
        self._ref_positions = None
        self._rebuild_flags = None
        # Reset adaptive K state so first build triggers a sync check.
        self._first_build = True
        self._actual_max_k = None

    @torch.compiler.disable
    def _check_and_resize_k(
        self,
        N: int,
        device: torch.device,
        pbc: torch.Tensor | None,
    ) -> bool:
        """Sync on actual max K and grow/shrink the neighbor matrix if needed.

        Called on structural events (first build, N/B change, cell volume
        change).  The sync cost is acceptable because these events are
        infrequent and the calling code path is already off the compile graph.

        Returns ``True`` if K was grown (caller must re-run the kernel).
        Shrinking trims the existing buffers in-place — no re-run needed.
        """
        if self._actual_max_k is None:
            return False
        actual = int(self._actual_max_k.item())

        if actual >= self._max_neighbors:
            # Overflow — grow with 1.5x headroom and round to nearest 16.  Must re-run kernel.
            self._max_neighbors = ((int(actual * 1.5) + 15) // 16) * 16
            self._realloc_k(N, device, pbc)
            return True
        elif actual < (1 / 2) * self._max_neighbors and actual > 0:
            # 2x+ overestimate — trim existing buffers in-place.
            new_k = ((int(actual * 2) + 15) // 16) * 16
            # Never shrink below the user-provided override — it serves as a
            # hard floor.  We may grow above it on overflow, but not below.
            if self._max_neighbors_override is not None:
                new_k = max(new_k, self._max_neighbors_override)
            if new_k < self._max_neighbors:
                self._max_neighbors = new_k
                self._neighbor_matrix = self._neighbor_matrix[:, :new_k].contiguous()
                self._col_range = self._col_range[:new_k]
                if self._neighbor_matrix_shifts is not None:
                    self._neighbor_matrix_shifts = self._neighbor_matrix_shifts[
                        :, :new_k
                    ].contiguous()
            # num_neighbors unchanged — still valid.
        return False

    @torch.compiler.disable
    def _realloc_k(
        self,
        N: int,
        device: torch.device,
        pbc: torch.Tensor | None,
    ) -> None:
        """Reallocate neighbor-matrix buffers at the current N with a new K.

        Preserves N (no staging-buffer realloc needed) but resets the
        skin state to force a full rebuild on the next step.
        """
        max_nbrs = self._max_neighbors
        self._neighbor_matrix = torch.full(
            (N, max_nbrs), N, dtype=torch.int32, device=device
        )
        self._col_range = torch.arange(max_nbrs, device=device, dtype=torch.int32)
        self._num_neighbors = torch.zeros(N, dtype=torch.int32, device=device)
        if pbc is not None:
            self._neighbor_matrix_shifts = torch.zeros(
                N, max_nbrs, 3, dtype=torch.int32, device=device
            )
        else:
            self._neighbor_matrix_shifts = None
        # Reset skin state to force a full rebuild.
        self._ref_positions = None
        self._rebuild_flags = None

    @torch.compiler.disable
    def _alloc_staging_buffers(
        self,
        N: int,
        B: int,
        dtype: torch.dtype,
        device: torch.device,
        cell: torch.Tensor | None,
        pbc: torch.Tensor | None,
        batch_ptr: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        batch_idx: torch.Tensor | None = None,
    ) -> None:
        """Allocate persistent staging buffers for the current (N, B) shape."""
        self._buf_positions = torch.zeros(N, 3, dtype=dtype, device=device)
        self._buf_batch_idx = torch.zeros(N, dtype=torch.int32, device=device)
        self._buf_batch_ptr = torch.zeros(B + 1, dtype=torch.int32, device=device)
        if cell is not None:
            self._buf_cell = torch.zeros(B, 3, 3, dtype=dtype, device=device)
            self._buf_pbc = torch.zeros(B, 3, dtype=torch.bool, device=device)
        else:
            self._buf_cell = None
            self._buf_pbc = None
        # Initialise rebuild_flags as all-True so the first step, before the skin
        # check has reference positions, builds every system. Subsequent checks
        # replace this tensor with the flags returned by nvalchemiops.
        self._rebuild_flags = torch.ones(B, dtype=torch.bool, device=device)
        # Pre-allocate algorithm-specific kwargs to eliminate on-demand CPU syncs
        # from the neighbor_list dispatcher.  Use the actual batch_ptr (if provided)
        # to compute max_atoms_per_system correctly — the staging buffer is still
        # all-zeros at this point and would give max_atoms = 0.
        ptr = batch_ptr if batch_ptr is not None else self._buf_batch_ptr
        alloc_positions = positions if positions is not None else self._buf_positions
        alloc_batch_idx = batch_idx if batch_idx is not None else self._buf_batch_idx
        self._alloc_nl_kwargs(
            N, B, alloc_positions, alloc_batch_idx, ptr, cell, pbc, device, dtype
        )

    def _copy_to_staging_buffers(
        self,
        positions: torch.Tensor,
        batch_ptr: torch.Tensor,
        batch_idx: torch.Tensor,
        cell: torch.Tensor | None,
        pbc: torch.Tensor | None,
    ) -> None:
        """Refresh staging buffers from the current batch."""
        self._buf_positions.copy_(positions)
        self._buf_batch_ptr.copy_(batch_ptr)
        self._buf_batch_idx.copy_(batch_idx)
        if self._buf_cell is not None and cell is not None:
            self._buf_cell.copy_(cell)
        if self._buf_pbc is not None and pbc is not None:
            self._buf_pbc.copy_(pbc)

    # ------------------------------------------------------------------
    # Algorithm-specific pre-allocation
    # ------------------------------------------------------------------

    def _alloc_nl_kwargs(
        self,
        N: int,
        B: int,
        positions: torch.Tensor,
        batch_idx: torch.Tensor,
        batch_ptr: torch.Tensor,
        cell: torch.Tensor | None,
        pbc: torch.Tensor | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Pre-allocate algorithm-specific kwargs to remove CPU-GPU syncs.

        The ops dispatcher exposes a host-only cost model that can select
        fine-grained strategies such as ``batch_cell_list_pair_centric`` and
        ``batch_cluster_tile``.  Run that selector once when staging buffers are
        allocated, cache the chosen method, and pre-allocate scratch tensors for
        the selected base algorithm.
        """
        self._buf_nl_kwargs = {}
        batch_ptr = batch_ptr.detach().to(dtype=torch.int32).contiguous()
        self._neighbor_list_method = self._select_neighbor_list_method(
            N, B, batch_ptr, cell, pbc, dtype
        )
        base_method = self._base_neighbor_list_method(self._neighbor_list_method)

        if base_method.endswith("cluster_tile"):
            if (
                allocate_batch_cluster_tile_list is None
                or estimate_batch_max_tiles_per_group is None
                or cell is None
            ):
                return
            alloc_cell = cell.to(dtype).contiguous()
            max_tiles_per_group = estimate_batch_max_tiles_per_group(
                batch_ptr, self.config.cutoff + self.skin, alloc_cell
            )
            (
                sorted_atom_index,
                sort_inv,
                sorted_pos_x,
                sorted_pos_y,
                sorted_pos_z,
                batch_idx_sorted,
                batch_ptr_padded,
                group_system,
                group_ptr,
                group_ctr_x,
                group_ctr_y,
                group_ctr_z,
                group_ext_x,
                group_ext_y,
                group_ext_z,
                num_tiles,
                tile_row_group,
                tile_col_group,
                tile_system,
            ) = allocate_batch_cluster_tile_list(
                batch_ptr,
                device,
                dtype=dtype,
                max_tiles_per_group=max_tiles_per_group,
            )
            self._buf_nl_kwargs = {
                "max_tiles_per_group": max_tiles_per_group,
                "sorted_atom_index": sorted_atom_index,
                "sort_inv": sort_inv,
                "sorted_pos_x": sorted_pos_x,
                "sorted_pos_y": sorted_pos_y,
                "sorted_pos_z": sorted_pos_z,
                "batch_idx_sorted": batch_idx_sorted,
                "batch_ptr_padded": batch_ptr_padded,
                "group_system": group_system,
                "group_ptr": group_ptr,
                "group_ctr_x": group_ctr_x,
                "group_ctr_y": group_ctr_y,
                "group_ctr_z": group_ctr_z,
                "group_ext_x": group_ext_x,
                "group_ext_y": group_ext_y,
                "group_ext_z": group_ext_z,
                "num_tiles": num_tiles,
                "tile_row_group": tile_row_group,
                "tile_col_group": tile_col_group,
                "tile_system": tile_system,
            }
            return

        if base_method.endswith("cell_list"):
            if estimate_batch_cell_list_sizes is None or allocate_cell_list is None:
                return
            if cell is not None and pbc is not None:
                alloc_cell = cell.to(dtype).contiguous()
                alloc_pbc = pbc
            else:
                expanded_idx = batch_idx.unsqueeze(1).expand_as(positions)
                pos_min = torch.full((B, 3), float("inf"), dtype=dtype, device=device)
                pos_min.scatter_reduce_(0, expanded_idx, positions, reduce="amin")
                pos_max = torch.full((B, 3), float("-inf"), dtype=dtype, device=device)
                pos_max.scatter_reduce_(0, expanded_idx, positions, reduce="amax")
                cell_lengths = (pos_max - pos_min) * 1.5 + 0.1 * (
                    self.config.cutoff + self.skin
                )
                alloc_cell = torch.diag_embed(cell_lengths)
                alloc_pbc = torch.zeros(B, 3, dtype=torch.bool, device=device)

            max_total_cells, neighbor_search_radius = estimate_batch_cell_list_sizes(
                alloc_cell, alloc_pbc, self.config.cutoff + self.skin
            )
            (
                cells_per_dimension,
                neighbor_search_radius,
                atom_periodic_shifts,
                atom_to_cell_mapping,
                atoms_per_cell_count,
                cell_atom_start_indices,
                cell_atom_list,
            ) = allocate_cell_list(
                N, int(max_total_cells), neighbor_search_radius, device
            )
            self._buf_nl_kwargs = {
                "cells_per_dimension": cells_per_dimension,
                "neighbor_search_radius": neighbor_search_radius,
                "atom_periodic_shifts": atom_periodic_shifts,
                "atom_to_cell_mapping": atom_to_cell_mapping,
                "atoms_per_cell_count": atoms_per_cell_count,
                "cell_atom_start_indices": cell_atom_start_indices,
                "cell_atom_list": cell_atom_list,
            }
            return

        if cell is not None and pbc is not None:
            if compute_naive_num_shifts is None:
                return
            shift_range, num_shifts, max_shifts = compute_naive_num_shifts(
                cell.to(dtype).contiguous(),
                self.config.cutoff + self.skin,
                pbc,
            )
            max_atoms = int((batch_ptr[1:] - batch_ptr[:-1]).max().item())
            self._buf_nl_kwargs = {
                "shift_range_per_dimension": shift_range,
                "num_shifts_per_system": num_shifts,
                "max_shifts_per_system": max_shifts,
                "max_atoms_per_system": max_atoms,
            }

    def _select_neighbor_list_method(
        self,
        N: int,
        B: int,
        batch_ptr: torch.Tensor,
        cell: torch.Tensor | None,
        pbc: torch.Tensor | None,
        dtype: torch.dtype,
    ) -> str:
        """Choose the explicit method to use inside the compiled hot path."""
        if self.method is not None:
            return self.method
        fallback = "cell_list" if N // max(B, 1) >= 2000 else "naive"
        if cell is None or pbc is None:
            return fallback
        try:
            return suggest_neighbor_list_method(
                batch_ptr,
                cell.to(dtype).contiguous(),
                pbc,
                self.config.cutoff + self.skin,
                half_fill=self.config.half_list,
                return_neighbor_list=False,
                positions_dtype=dtype,
            )
        except (RuntimeError, NotImplementedError, ValueError):
            return fallback

    @staticmethod
    def _base_neighbor_list_method(method: str | None) -> str:
        """Return the dispatcher base method for a fine-grained strategy name."""
        if method is None:
            return "naive"
        try:
            return neighbor_list_strategy_run_args(method)[0]
        except ValueError:
            return method
