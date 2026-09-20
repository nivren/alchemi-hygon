# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit native HIP Batch CSR-build composition candidate.

This is deliberately a Python-level composition of the independently validated
fused key/count, rocPRIM scan, and atomic fill operations.  It owns neither a
dispatcher registration nor runtime backend selection.  The resulting atom
list has unspecified within-cell order and must not replace the stable public
cell-list path before query/output canonicalization is validated.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from nvalchemiops._hip_batch_cell_key_count import (
    _load_jit_extension as _load_key_count_extension,
)
from nvalchemiops._hip_batch_cell_key_count import build_batch_cell_key_counts_hip_into
from nvalchemiops._hip_cell_fill import (
    _load_jit_extension as _load_fill_extension,
)
from nvalchemiops._hip_cell_fill import build_cell_atom_list_hip_into
from nvalchemiops._hip_cell_scan import (
    _load_jit_extension as _load_scan_extension,
)
from nvalchemiops._hip_cell_scan import build_cell_starts_hip_into


def build_batch_cell_csr_hip_into(
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
    cell_starts: torch.Tensor,
    cell_atom_list: torch.Tensor,
    cell_cursor: torch.Tensor,
    scan_workspace: torch.Tensor,
    *,
    global_atom_offset: int = 0,
) -> None:
    """Compose native Batch key/count, scan, and unordered CSR fill.

    ``scan_workspace`` remains caller-owned.  Allocate it with the explicit
    scan workspace query before this call; a zeroed count buffer is a valid
    pre-fusion input to that query.  The composition is forward-only and
    preserves the individual phases' mutation and error contracts.
    """

    build_batch_cell_key_counts_hip_into(
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
    build_cell_starts_hip_into(
        cell_counts,
        cell_starts,
        scan_workspace,
        global_atom_offset=global_atom_offset,
    )
    build_cell_atom_list_hip_into(
        cell_keys,
        cell_counts,
        cell_starts,
        cell_atom_list,
        cell_cursor,
        global_atom_offset=global_atom_offset,
    )


def _build_batch_cell_csr_hip_into_trusted(
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
    cell_starts: torch.Tensor,
    cell_atom_list: torch.Tensor,
    cell_cursor: torch.Tensor,
    scan_workspace: torch.Tensor,
    *,
    global_atom_offset: int = 0,
) -> None:
    """Run the CSR build through direct native extensions after validation.

    This is intentionally private.  The caller must have already validated
    the complete shape, dtype, device, alias, range, capacity, and workspace
    contract for this exact buffer set.  The public composition above remains
    the safe checked entry point.  This fast path only removes repeated Python
    custom-op validation; it does not change the three native algorithms or
    their mutation and ordering semantics.
    """

    _load_key_count_extension().batch_cell_key_count_into(
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
    _load_scan_extension().cell_start_scan_into(
        cell_counts,
        cell_starts,
        scan_workspace,
        int(global_atom_offset),
    )
    _load_fill_extension().cell_atom_list_fill_into(
        cell_keys,
        cell_counts,
        cell_starts,
        cell_atom_list,
        cell_cursor,
        int(global_atom_offset),
    )


@dataclass(frozen=True, slots=True)
class _TrustedBatchCellBuildPlan:
    """Validated tensor/workspace lifetime for repeated trusted CSR builds.

    ``initialize`` performs one checked public build and captures the exact
    tensor storage used by the subsequent trusted calls.  The plan may be
    reused while shapes, dtypes, devices, aliases, cell metadata, capacity,
    and workspace remain unchanged; position values may change in place.  A
    structural change requires a new plan through ``initialize``.
    """

    positions: torch.Tensor
    inverse_cells: torch.Tensor
    cells_per_dimension: torch.Tensor
    pbc: torch.Tensor
    batch_idx: torch.Tensor
    cell_offsets: torch.Tensor
    atom_periodic_shifts: torch.Tensor
    atom_to_cell_mapping: torch.Tensor
    cell_keys: torch.Tensor
    cell_counts: torch.Tensor
    cell_starts: torch.Tensor
    cell_atom_list: torch.Tensor
    cell_cursor: torch.Tensor
    scan_workspace: torch.Tensor
    global_atom_offset: int = 0

    @classmethod
    def initialize(
        cls,
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
        cell_starts: torch.Tensor,
        cell_atom_list: torch.Tensor,
        cell_cursor: torch.Tensor,
        scan_workspace: torch.Tensor,
        *,
        global_atom_offset: int = 0,
    ) -> "_TrustedBatchCellBuildPlan":
        """Create a plan after one complete checked public build.

        The initial public build is intentional: it establishes the complete
        Python/reference ABI contract before later calls bypass only the
        repeated high-level validation.  This method mutates the supplied
        CSR outputs exactly as the public build does.
        """

        build_batch_cell_csr_hip_into(
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
            cell_starts,
            cell_atom_list,
            cell_cursor,
            scan_workspace,
            global_atom_offset=global_atom_offset,
        )
        return cls(
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
            cell_starts,
            cell_atom_list,
            cell_cursor,
            scan_workspace,
            global_atom_offset,
        )

    def run(self) -> None:
        """Execute one subsequent trusted build on the captured buffers."""

        _build_batch_cell_csr_hip_into_trusted(
            self.positions,
            self.inverse_cells,
            self.cells_per_dimension,
            self.pbc,
            self.batch_idx,
            self.cell_offsets,
            self.atom_periodic_shifts,
            self.atom_to_cell_mapping,
            self.cell_keys,
            self.cell_counts,
            self.cell_starts,
            self.cell_atom_list,
            self.cell_cursor,
            self.scan_workspace,
            global_atom_offset=self.global_atom_offset,
        )


__all__ = ["build_batch_cell_csr_hip_into"]
