#!/usr/bin/env python3
"""Check public Torch-query parity from an unordered native HIP Batch CSR build.

This remains an isolated composition: HIP builds CSR data, while the existing
Torch query produces public neighbor outputs.  No dispatcher or backend policy
is modified.  The test intentionally compares active output order, not only a
set, because the native fill's within-cell atomic order is unspecified.
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


def _query_outputs(
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
) -> tuple[torch.Tensor, ...]:
    from nvalchemiops.torch_reference_cell_list import batch_query_cell_list  # noqa: PLC0415

    atoms = positions.shape[0]
    matrix = torch.full((atoms, 256), atoms, dtype=torch.int32, device=positions.device)
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
    return matrix, counts, shifts, distances, vectors


def _assert_active_equal(
    actual: tuple[torch.Tensor, ...], expected: tuple[torch.Tensor, ...], label: str
) -> None:
    for name, actual_value, expected_value in zip(
        ("neighbor_matrix", "num_neighbors", "pair_shifts", "distances", "vectors"),
        actual,
        expected,
        strict=True,
    ):
        if actual_value.dtype.is_floating_point:
            torch.testing.assert_close(actual_value, expected_value, atol=1e-6, rtol=1e-6)
        elif not torch.equal(actual_value, expected_value):
            raise AssertionError(f"{label}: {name} differs in public order or values")


def _run(
    positions: torch.Tensor,
    inverse_cells: torch.Tensor,
    cells: torch.Tensor,
    pbc: torch.Tensor,
    dimensions: torch.Tensor,
    radius: torch.Tensor,
    batch_idx: torch.Tensor,
    offsets: torch.Tensor,
    *,
    half_fill: bool,
    label: str,
) -> None:
    from nvalchemiops._cell_list_abi import (  # noqa: PLC0415
        build_batch_cell_key_counts_reference_into,
        build_cell_atom_list_reference_into,
        build_cell_starts_reference_into,
    )
    from nvalchemiops._hip_batch_cell_build import (  # noqa: PLC0415
        build_batch_cell_csr_hip_into,
    )

    total_cells = int(dimensions.to(torch.int64).prod(dim=1).sum().item())
    actual = _build_outputs(positions.shape[0], total_cells, positions.device)
    expected = _build_outputs(positions.shape[0], total_cells, positions.device)
    workspace = _workspace(actual[3], actual[4])
    build_batch_cell_csr_hip_into(
        positions,
        inverse_cells,
        dimensions,
        pbc,
        batch_idx,
        offsets,
        *actual,
        workspace,
    )
    build_batch_cell_key_counts_reference_into(
        positions,
        inverse_cells,
        dimensions,
        pbc,
        batch_idx,
        offsets,
        *expected[:4],
    )
    build_cell_starts_reference_into(expected[3], expected[4])
    build_cell_atom_list_reference_into(expected[2], expected[3], expected[4], expected[5], expected[6])
    torch.cuda.synchronize()
    _assert_active_equal(
        _query_outputs(positions, cells, pbc, 0.52, batch_idx, dimensions, radius, actual, half_fill=half_fill),
        _query_outputs(positions, cells, pbc, 0.52, batch_idx, dimensions, radius, expected, half_fill=half_fill),
        label,
    )


def main() -> None:
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("HIP PyTorch and a visible HCU are required")

    root = Path(__file__).resolve().parent
    for variable, directory in (
        ("NVALCHEMI_HIP_BATCH_CELL_KEY_COUNT_BUILD_DIR", "native-hip-batch-cell-key-count"),
        ("NVALCHEMI_HIP_CELL_SCAN_BUILD_DIR", "native-hip-cell-start-scan"),
        ("NVALCHEMI_HIP_CELL_FILL_BUILD_DIR", "native-hip-cell-fill"),
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
            _run(
                positions,
                torch.linalg.inv(cells),
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
        _run(
            torch.bmm(fractional.to(torch.float32).unsqueeze(1), cells64.to(torch.float32)[batch_idx.to(torch.long)]).squeeze(1),
            torch.linalg.inv(cells64.to(torch.float32)),
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
    print(
        json.dumps(
            {
                "atomic_cell_order_canonicalized_by_query": True,
                "atoms_per_system": list(atoms_per_system),
                "batch_systems": 2,
                "device": torch.cuda.get_device_name(),
                "dtype_parity": dtype_parity,
                "half_and_full": True,
                "mixed_pbc": [[True, True, True], [False, False, False]],
                "non_default_stream": True,
                "performance_measured": False,
                "public_pair_shift_order_compared": True,
                "torch_hip": torch.version.hip,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
