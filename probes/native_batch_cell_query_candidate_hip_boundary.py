#!/usr/bin/env python3
"""Check the isolated native HIP Batch query-enumeration candidate.

The candidate writes unordered per-source-atom pairs and image shifts.  This
probe canonicalizes those pairs with the Torch reference ordering, then checks
the Torch materialized distances/vectors.  No dispatcher or runtime policy is
modified, and the candidate remains forward-only.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch


def _workspace(counts: torch.Tensor, starts: torch.Tensor) -> torch.Tensor:
    from nvalchemiops._hip_cell_scan import cell_starts_hip_workspace_size  # noqa: PLC0415

    counts.zero_()
    return torch.empty(
        cell_starts_hip_workspace_size(counts, starts),
        dtype=torch.uint8,
        device=counts.device,
    )


def _build_outputs(atoms: int, cells: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        torch.full((atoms, 3), -9, dtype=torch.int32, device=device),
        torch.full((atoms, 3), -9, dtype=torch.int32, device=device),
        torch.full((atoms,), -9, dtype=torch.int32, device=device),
        torch.full((cells,), -9, dtype=torch.int32, device=device),
        torch.full((cells,), -9, dtype=torch.int32, device=device),
        torch.full((atoms + 3,), -9, dtype=torch.int32, device=device),
        torch.full((cells,), -9, dtype=torch.int32, device=device),
    )


def _canonical_pairs(
    matrix: torch.Tensor,
    counts: torch.Tensor,
    shifts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows_grid = torch.arange(matrix.shape[0], device=matrix.device)[:, None].expand_as(matrix)
    columns_grid = matrix.to(torch.long)
    active = torch.arange(matrix.shape[1], device=matrix.device)[None, :] < counts[:, None]
    rows = rows_grid[active]
    columns = columns_grid[active]
    pair_shifts = shifts[active]
    order = torch.arange(rows.numel(), device=matrix.device)
    for values in (
        pair_shifts[:, 2],
        pair_shifts[:, 1],
        pair_shifts[:, 0],
        columns,
        rows,
    ):
        order = order[torch.argsort(values[order], stable=True)]
    return rows[order], columns[order], pair_shifts[order]


def _materialize(
    positions: torch.Tensor,
    cells: torch.Tensor,
    batch_idx: torch.Tensor,
    pairs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, columns, shifts = pairs
    geometry = cells[batch_idx[rows].to(torch.long)]
    vectors = positions[rows] - positions[columns] - torch.einsum(
        "pi,pij->pj", shifts.to(positions.dtype), geometry
    )
    return vectors.square().sum(dim=1), vectors


def _query_reference(
    positions: torch.Tensor,
    cells: torch.Tensor,
    pbc: torch.Tensor,
    cutoff: float,
    batch_idx: torch.Tensor,
    dimensions: torch.Tensor,
    radius: torch.Tensor,
    build: tuple[torch.Tensor, ...],
    *,
    half_fill: bool,
    capacity: int,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    from nvalchemiops.torch_reference_cell_list import batch_query_cell_list  # noqa: PLC0415

    atoms = positions.shape[0]
    matrix = torch.full((atoms, capacity), atoms, dtype=torch.int32, device=positions.device)
    shifts = torch.zeros((*matrix.shape, 3), dtype=torch.int32, device=positions.device)
    counts = torch.zeros(atoms, dtype=torch.int32, device=positions.device)
    distances = torch.zeros(matrix.shape, dtype=positions.dtype, device=positions.device)
    vectors = torch.zeros((*matrix.shape, 3), dtype=positions.dtype, device=positions.device)
    atom_shifts, mapping, _, cell_counts, starts, atom_list, _ = build
    batch_query_cell_list(
        positions,
        cells,
        pbc,
        cutoff,
        batch_idx,
        dimensions,
        radius,
        atom_shifts,
        mapping,
        cell_counts,
        starts,
        atom_list,
        matrix,
        shifts,
        counts,
        half_fill=half_fill,
        return_distances=True,
        return_vectors=True,
        neighbor_distances=distances,
        neighbor_vectors=vectors,
    )
    return (matrix, counts, shifts, distances, vectors), build


def _run_case(
    positions: torch.Tensor,
    cells: torch.Tensor,
    pbc: torch.Tensor,
    dimensions: torch.Tensor,
    radius: torch.Tensor,
    batch_idx: torch.Tensor,
    offsets: torch.Tensor,
    *,
    half_fill: bool,
    label: str,
    capacity: int = 256,
) -> None:
    from nvalchemiops._cell_list_abi import (  # noqa: PLC0415
        build_batch_cell_key_counts_reference_into,
        build_cell_atom_list_reference_into,
        build_cell_starts_reference_into,
    )
    from nvalchemiops._hip_batch_cell_build import (  # noqa: PLC0415
        build_batch_cell_csr_hip_into,
    )
    from nvalchemiops._hip_batch_cell_query import (  # noqa: PLC0415
        batch_query_cell_list_hip_into,
    )
    from nvalchemiops._hip_batch_query_materialize import (  # noqa: PLC0415
        allocate_batch_query_topology_composite_workspace,
        materialize_batch_query_topology_composite_hip_into,
        materialize_batch_query_topology_composite_hip_into_with_workspace,
        materialize_batch_query_topology_hip_into,
    )
    from nvalchemiops._hip_batch_query_geometry import (  # noqa: PLC0415
        materialize_batch_query_geometry_hip_into,
    )
    from nvalchemiops._torch_batch_query_materialization import (  # noqa: PLC0415
        materialize_batch_query_candidate_into,
        materialize_batch_query_topology_geometry_into,
    )

    total_cells = int(dimensions.to(torch.int64).prod(dim=1).sum().item())
    actual = _build_outputs(positions.shape[0], total_cells, positions.device)
    expected = _build_outputs(positions.shape[0], total_cells, positions.device)
    workspace = _workspace(actual[3], actual[4])
    build_batch_cell_csr_hip_into(
        positions,
        torch.linalg.inv(cells),
        dimensions,
        pbc,
        batch_idx,
        offsets,
        *actual,
        workspace,
    )
    build_batch_cell_key_counts_reference_into(
        positions,
        torch.linalg.inv(cells),
        dimensions,
        pbc,
        batch_idx,
        offsets,
        *expected[:4],
    )
    build_cell_starts_reference_into(expected[3], expected[4])
    build_cell_atom_list_reference_into(
        expected[2], expected[3], expected[4], expected[5], expected[6]
    )

    candidate_matrix = torch.full(
        (positions.shape[0], capacity), positions.shape[0], dtype=torch.int32, device=positions.device
    )
    candidate_shifts = torch.zeros(
        (*candidate_matrix.shape, 3), dtype=torch.int32, device=positions.device
    )
    candidate_counts = torch.zeros(positions.shape[0], dtype=torch.int32, device=positions.device)
    batch_query_cell_list_hip_into(
        positions,
        cells,
        pbc,
        0.52,
        batch_idx,
        dimensions,
        radius,
        offsets,
        actual[0],
        actual[1],
        actual[3],
        actual[4],
        actual[5],
        candidate_matrix,
        candidate_shifts,
        candidate_counts,
        half_fill=half_fill,
    )
    reference, _ = _query_reference(
        positions,
        cells,
        pbc,
        0.52,
        batch_idx,
        dimensions,
        radius,
        expected,
        half_fill=half_fill,
        capacity=capacity,
    )
    candidate_pairs = _canonical_pairs(candidate_matrix, candidate_counts, candidate_shifts)
    reference_pairs = _canonical_pairs(reference[0], reference[1], reference[2])
    for actual_value, expected_value in zip(candidate_pairs, reference_pairs, strict=True):
        torch.testing.assert_close(actual_value, expected_value, msg=f"{label}: pair mismatch")
    candidate_public = (
        torch.empty_like(candidate_matrix),
        torch.empty_like(candidate_shifts),
        torch.empty_like(candidate_counts),
        torch.empty((positions.shape[0], capacity), dtype=positions.dtype, device=positions.device),
        torch.empty(
            (positions.shape[0], capacity, 3), dtype=positions.dtype, device=positions.device
        ),
    )
    materialize_batch_query_candidate_into(
        positions,
        cells,
        batch_idx,
        candidate_matrix,
        candidate_shifts,
        candidate_counts,
        candidate_public[0],
        candidate_public[1],
        candidate_public[2],
        distances=candidate_public[3],
        vectors=candidate_public[4],
    )
    materialized_public = (
        candidate_public[0],
        candidate_public[2],
        candidate_public[1],
        candidate_public[3],
        candidate_public[4],
    )
    for actual_value, expected_value in zip(materialized_public, reference, strict=True):
        torch.testing.assert_close(
            actual_value,
            expected_value,
            atol=1e-6 if positions.dtype == torch.float32 else 1e-12,
            rtol=1e-6 if positions.dtype == torch.float32 else 1e-12,
        )

    native_topology = (
        torch.empty_like(candidate_matrix),
        torch.empty_like(candidate_shifts),
        torch.empty_like(candidate_counts),
    )
    materialize_batch_query_topology_hip_into(
        candidate_matrix,
        candidate_shifts,
        candidate_counts,
        native_topology[0],
        native_topology[1],
        native_topology[2],
    )
    native_public = (native_topology[0], native_topology[2], native_topology[1])
    for actual_value, expected_value in zip(native_public, reference[:3], strict=True):
        torch.testing.assert_close(actual_value, expected_value)

    composite_topology = (
        torch.empty_like(candidate_matrix),
        torch.empty_like(candidate_shifts),
        torch.empty_like(candidate_counts),
    )
    composite_workspace = allocate_batch_query_topology_composite_workspace(
        candidate_matrix, candidate_counts
    )
    materialize_batch_query_topology_composite_hip_into_with_workspace(
        candidate_matrix,
        candidate_shifts,
        candidate_counts,
        composite_topology[0],
        composite_topology[1],
        composite_topology[2],
        composite_workspace,
    )
    materialize_batch_query_topology_composite_hip_into_with_workspace(
        candidate_matrix,
        candidate_shifts,
        candidate_counts,
        composite_topology[0],
        composite_topology[1],
        composite_topology[2],
        composite_workspace,
    )
    composite_public = (
        composite_topology[0],
        composite_topology[2],
        composite_topology[1],
    )
    for actual_value, expected_value in zip(
        composite_public, reference[:3], strict=True
    ):
        torch.testing.assert_close(actual_value, expected_value)

    native_geometry_distances = torch.empty_like(reference[3])
    native_geometry_vectors = torch.empty_like(reference[4])
    materialize_batch_query_geometry_hip_into(
        positions,
        cells,
        batch_idx,
        composite_topology[0],
        composite_topology[1],
        composite_topology[2],
        native_geometry_distances,
        native_geometry_vectors,
    )
    native_geometry_public = (
        composite_topology[0],
        composite_topology[2],
        composite_topology[1],
        native_geometry_distances,
        native_geometry_vectors,
    )
    for actual_value, expected_value in zip(
        native_geometry_public, reference, strict=True
    ):
        torch.testing.assert_close(
            actual_value,
            expected_value,
            atol=1e-6 if positions.dtype == torch.float32 else 1e-12,
            rtol=1e-6 if positions.dtype == torch.float32 else 1e-12,
        )

    hybrid_distances = torch.empty_like(reference[3])
    hybrid_vectors = torch.empty_like(reference[4])
    materialize_batch_query_topology_geometry_into(
        positions,
        cells,
        batch_idx,
        composite_topology[0],
        composite_topology[1],
        composite_topology[2],
        distances=hybrid_distances,
        vectors=hybrid_vectors,
    )
    hybrid_public = (
        composite_topology[0],
        composite_topology[2],
        composite_topology[1],
        hybrid_distances,
        hybrid_vectors,
    )
    for actual_value, expected_value in zip(hybrid_public, reference, strict=True):
        torch.testing.assert_close(
            actual_value,
            expected_value,
            atol=1e-6 if positions.dtype == torch.float32 else 1e-12,
            rtol=1e-6 if positions.dtype == torch.float32 else 1e-12,
        )

    gradient_positions = positions.detach().clone().requires_grad_()
    gradient_cells = cells.detach().clone().requires_grad_()
    gradient_outputs = (
        torch.empty_like(candidate_matrix),
        torch.empty_like(candidate_shifts),
        torch.empty_like(candidate_counts),
        torch.empty((positions.shape[0], capacity), dtype=positions.dtype, device=positions.device),
        torch.empty(
            (positions.shape[0], capacity, 3), dtype=positions.dtype, device=positions.device
        ),
    )
    materialize_batch_query_candidate_into(
        gradient_positions,
        gradient_cells,
        batch_idx,
        candidate_matrix,
        candidate_shifts,
        candidate_counts,
        gradient_outputs[0],
        gradient_outputs[1],
        gradient_outputs[2],
        distances=gradient_outputs[3],
        vectors=gradient_outputs[4],
    )
    gradient_active = torch.arange(capacity, device=positions.device)[None, :] < (
        gradient_outputs[2].to(torch.long)[:, None]
    )
    gradient_value = (
        gradient_outputs[3][gradient_active].square().sum()
        + 0.25 * gradient_outputs[4][gradient_active].square().sum()
    )
    gradient_positions_first, gradient_cells_first = torch.autograd.grad(
        gradient_value, (gradient_positions, gradient_cells), create_graph=True
    )
    gradient_second = torch.autograd.grad(
        gradient_positions_first.square().sum() + gradient_cells_first.square().sum(),
        (gradient_positions, gradient_cells),
    )
    if not all(torch.isfinite(value).all() for value in gradient_second):
        raise AssertionError(f"{label}: materialized geometry gradients are not finite")
    candidate_geometry = _materialize(positions, cells, batch_idx, candidate_pairs)
    reference_geometry = _materialize(positions, cells, batch_idx, reference_pairs)
    torch.testing.assert_close(candidate_geometry[0], reference_geometry[0], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(candidate_geometry[1], reference_geometry[1], atol=1e-6, rtol=1e-6)
    reference_active = torch.arange(capacity, device=positions.device)[None, :] < reference[1][:, None]
    torch.testing.assert_close(
        reference[3][reference_active],
        reference_geometry[0].sqrt(),
        atol=1e-6 if positions.dtype == torch.float32 else 1e-12,
        rtol=1e-6 if positions.dtype == torch.float32 else 1e-12,
    )
    torch.testing.assert_close(
        reference[4][reference_active],
        reference_geometry[1],
        atol=1e-6 if positions.dtype == torch.float32 else 1e-12,
        rtol=1e-6 if positions.dtype == torch.float32 else 1e-12,
    )


def main() -> None:
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("HIP PyTorch and a visible HCU are required")

    root = Path(__file__).resolve().parent
    for variable, directory in (
        ("NVALCHEMI_HIP_BATCH_CELL_KEY_COUNT_BUILD_DIR", "native-hip-batch-cell-key-count"),
        ("NVALCHEMI_HIP_CELL_SCAN_BUILD_DIR", "native-hip-cell-start-scan"),
        ("NVALCHEMI_HIP_CELL_FILL_BUILD_DIR", "native-hip-cell-fill"),
        ("NVALCHEMI_HIP_BATCH_CELL_QUERY_BUILD_DIR", "native-hip-batch-cell-query"),
        ("NVALCHEMI_HIP_BATCH_QUERY_GEOMETRY_BUILD_DIR", "native-hip-batch-query-geometry"),
    ):
        os.environ.setdefault(variable, str(root.parent / "artifacts" / directory))

    device = torch.device("cuda")
    cells64 = torch.tensor(
        [
            [[3.2, 0.0, 0.0], [0.3, 3.0, 0.0], [0.2, 0.4, 3.1]],
            [[3.0, 0.0, 0.0], [0.2, 2.9, 0.0], [0.1, 0.3, 3.2]],
        ],
        dtype=torch.float64,
        device=device,
    )
    dimensions = torch.tensor([[4, 4, 4], [3, 3, 3]], dtype=torch.int32, device=device)
    offsets = torch.tensor([0, 64], dtype=torch.int32, device=device)
    radius = torch.ones((2, 3), dtype=torch.int32, device=device)
    pbc = torch.tensor([[True, True, True], [False, False, False]], device=device)
    atoms_per_system = (16, 11)
    batch_idx = torch.cat(
        [
            torch.zeros(atoms_per_system[0], dtype=torch.int32, device=device),
            torch.ones(atoms_per_system[1], dtype=torch.int32, device=device),
        ]
    )
    fractional = torch.cat(
        [
            torch.tensor(
                [[0.08 + 0.017 * i, 0.12 + 0.013 * i, 0.16 + 0.011 * i] for i in range(16)],
                dtype=torch.float64,
                device=device,
            ),
            torch.tensor(
                [[0.14 + 0.023 * i, 0.18 + 0.019 * i, 0.22 + 0.017 * i] for i in range(11)],
                dtype=torch.float64,
                device=device,
            ),
        ]
    )
    dtype_parity: dict[str, bool] = {}
    for dtype in (torch.float32, torch.float64):
        cells = cells64.to(dtype)
        positions = torch.bmm(
            fractional.to(dtype).unsqueeze(1), cells[batch_idx.to(torch.long)]
        ).squeeze(1)
        for half_fill in (False, True):
            _run_case(
                positions,
                cells,
                pbc,
                dimensions,
                radius,
                batch_idx,
                offsets,
                half_fill=half_fill,
                label=f"{dtype}, half_fill={half_fill}",
            )
        dtype_parity[str(dtype)] = True

    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        _run_case(
            torch.bmm(
                fractional.to(torch.float32).unsqueeze(1),
                cells64.to(torch.float32)[batch_idx.to(torch.long)],
            ).squeeze(1),
            cells64.to(torch.float32),
            pbc,
            dimensions,
            radius,
            batch_idx,
            offsets,
            half_fill=False,
            label="non-default stream",
        )
    stream.synchronize()
    torch.cuda.synchronize()

    overflow_checked = False
    try:
        # The fixture has more than one active neighbor for at least one row.
        cells = cells64.to(torch.float32)
        positions = torch.bmm(
            fractional.to(torch.float32).unsqueeze(1), cells[batch_idx.to(torch.long)]
        ).squeeze(1)
        _run_case(
            positions,
            cells,
            pbc,
            dimensions,
            radius,
            batch_idx,
            offsets,
            half_fill=False,
            label="overflow",
            capacity=1,
        )
    except RuntimeError as exc:
        if "capacity overflow" not in str(exc):
            raise
        overflow_checked = True

    from nvalchemiops._hip_batch_cell_query import (  # noqa: PLC0415
        batch_query_cell_list_hip_into,
    )
    from nvalchemiops._hip_batch_query_materialize import (  # noqa: PLC0415
        allocate_batch_query_topology_composite_workspace,
        materialize_batch_query_topology_composite_hip_into,
        materialize_batch_query_topology_composite_hip_into_with_workspace,
        materialize_batch_query_topology_hip_into,
    )
    from nvalchemiops._hip_batch_query_geometry import (  # noqa: PLC0415
        materialize_batch_query_geometry_hip_into,
    )

    empty_cells = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0) * 2.0
    empty_matrix = torch.empty((0, 4), dtype=torch.int32, device=device)
    empty_shifts = torch.empty((0, 4, 3), dtype=torch.int32, device=device)
    batch_query_cell_list_hip_into(
        torch.empty((0, 3), dtype=torch.float32, device=device),
        empty_cells,
        torch.ones((1, 3), dtype=torch.bool, device=device),
        0.52,
        torch.empty(0, dtype=torch.int32, device=device),
        torch.tensor([[2, 2, 2]], dtype=torch.int32, device=device),
        torch.ones((1, 3), dtype=torch.int32, device=device),
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.empty((0, 3), dtype=torch.int32, device=device),
        torch.empty((0, 3), dtype=torch.int32, device=device),
        torch.zeros(8, dtype=torch.int32, device=device),
        torch.zeros(8, dtype=torch.int32, device=device),
        torch.empty(0, dtype=torch.int32, device=device),
        empty_matrix,
        empty_shifts,
        torch.empty(0, dtype=torch.int32, device=device),
    )
    empty_candidate = torch.empty((0, 4), dtype=torch.int32, device=device)
    empty_candidate_shifts = torch.empty((0, 4, 3), dtype=torch.int32, device=device)
    empty_candidate_counts = torch.empty(0, dtype=torch.int32, device=device)
    empty_public = torch.empty((0, 5), dtype=torch.int32, device=device)
    empty_public_shifts = torch.empty((0, 5, 3), dtype=torch.int32, device=device)
    empty_public_counts = torch.empty(0, dtype=torch.int32, device=device)
    materialize_batch_query_topology_hip_into(
        empty_candidate,
        empty_candidate_shifts,
        empty_candidate_counts,
        empty_public,
        empty_public_shifts,
        empty_public_counts,
    )
    materialize_batch_query_topology_composite_hip_into(
        empty_candidate,
        empty_candidate_shifts,
        empty_candidate_counts,
        empty_public,
        empty_public_shifts,
        empty_public_counts,
    )
    empty_workspace = allocate_batch_query_topology_composite_workspace(
        empty_candidate, empty_candidate_counts
    )
    materialize_batch_query_topology_composite_hip_into_with_workspace(
        empty_candidate,
        empty_candidate_shifts,
        empty_candidate_counts,
        empty_public,
        empty_public_shifts,
        empty_public_counts,
        empty_workspace,
    )
    empty_distances = torch.empty((0, 5), dtype=torch.float32, device=device)
    empty_vectors = torch.empty((0, 5, 3), dtype=torch.float32, device=device)
    materialize_batch_query_geometry_hip_into(
        torch.empty((0, 3), dtype=torch.float32, device=device),
        empty_cells,
        torch.empty(0, dtype=torch.int32, device=device),
        empty_public,
        empty_public_shifts,
        empty_public_counts,
        empty_distances,
        empty_vectors,
    )

    zero_candidate = torch.zeros((3, 2), dtype=torch.int32, device=device)
    zero_candidate_shifts = torch.zeros(
        (3, 2, 3), dtype=torch.int32, device=device
    )
    zero_candidate_counts = torch.zeros(3, dtype=torch.int32, device=device)
    zero_expected_matrix = torch.full(
        (3, 4), 3, dtype=torch.int32, device=device
    )
    zero_expected_shifts = torch.zeros(
        (3, 4, 3), dtype=torch.int32, device=device
    )
    zero_expected_counts = torch.zeros(3, dtype=torch.int32, device=device)
    for materialize_zero in (
        materialize_batch_query_topology_hip_into,
        materialize_batch_query_topology_composite_hip_into,
    ):
        zero_public = (
            torch.full((3, 4), -7, dtype=torch.int32, device=device),
            torch.full((3, 4, 3), -7, dtype=torch.int32, device=device),
            torch.full((3,), -7, dtype=torch.int32, device=device),
        )
        materialize_zero(
            zero_candidate,
            zero_candidate_shifts,
            zero_candidate_counts,
            *zero_public,
        )
        torch.testing.assert_close(zero_public[0], zero_expected_matrix)
        torch.testing.assert_close(zero_public[1], zero_expected_shifts)
        torch.testing.assert_close(zero_public[2], zero_expected_counts)

    capacity_candidate = torch.tensor(
        [[1, 2], [3, 3], [3, 3]], dtype=torch.int32, device=device
    )
    capacity_shifts = torch.zeros((3, 2, 3), dtype=torch.int32, device=device)
    capacity_counts = torch.tensor([2, 0, 0], dtype=torch.int32, device=device)
    capacity_public = torch.empty((3, 1), dtype=torch.int32, device=device)
    capacity_public_shifts = torch.empty((3, 1, 3), dtype=torch.int32, device=device)
    capacity_public_counts = torch.empty(3, dtype=torch.int32, device=device)
    capacity_rejected = False
    try:
        materialize_batch_query_topology_composite_hip_into(
            capacity_candidate,
            capacity_shifts,
            capacity_counts,
            capacity_public,
            capacity_public_shifts,
            capacity_public_counts,
        )
    except ValueError as exc:
        if "public capacity" not in str(exc):
            raise
        capacity_rejected = True

    print(
        json.dumps(
            {
                "candidate_order": "unordered_within_source_row",
                "atoms_per_system": list(atoms_per_system),
                "batch_systems": 2,
                "capacity_overflow_explicit": overflow_checked,
                "composite_capacity_rejected": capacity_rejected,
                "composite_workspace_reused": True,
                "device": torch.cuda.get_device_name(),
                "dtype_parity": dtype_parity,
                "empty_input": True,
                "hybrid_native_topology_torch_geometry": True,
                "native_hip_geometry_forward": True,
                "half_and_full": True,
                "materialization": "torch_reference_distance_vector",
                "native_topology_materialization": "rocprim_stable_radix_sort_scatter",
                "native_composite_topology_materialization": "rocprim_int64_composite_key_sort_scatter",
                "mixed_pbc": [[True, True, True], [False, False, False]],
                "non_default_stream": True,
                "performance_measured": False,
                "public_order_not_exposed": True,
                "zero_candidate_counts": True,
                "torch_hip": torch.version.hip,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
