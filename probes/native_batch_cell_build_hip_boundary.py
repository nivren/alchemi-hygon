#!/usr/bin/env python3
"""Validate the composed native HIP Batch CSR build against Torch.

The composition is intentionally isolated from the cell-list dispatcher and
query path.  Atomic insertion is compared as a per-cell atom set, not order.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch


def _outputs(atoms: int, cells: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        torch.full((atoms, 3), -9, dtype=torch.int32, device=device),
        torch.full((atoms, 3), -9, dtype=torch.int32, device=device),
        torch.full((atoms,), -9, dtype=torch.int32, device=device),
        torch.full((cells,), -9, dtype=torch.int32, device=device),
        torch.full((cells,), -9, dtype=torch.int32, device=device),
        torch.full((atoms + 3,), -9, dtype=torch.int32, device=device),
        torch.full((cells,), -9, dtype=torch.int32, device=device),
    )


def _assert_atom_sets(
    actual: torch.Tensor,
    expected: torch.Tensor,
    counts: torch.Tensor,
    starts: torch.Tensor,
    *,
    global_atom_offset: int,
    label: str,
) -> None:
    for key, count in enumerate(counts.cpu().tolist()):
        if not count:
            continue
        start = int(starts[key].item()) - global_atom_offset
        if not torch.equal(
            torch.sort(actual[start : start + count]).values,
            torch.sort(expected[start : start + count]).values,
        ):
            raise AssertionError(f"{label}: cell {key} atom set differs")


def _workspace(counts: torch.Tensor, starts: torch.Tensor) -> torch.Tensor:
    from nvalchemiops._hip_cell_scan import cell_starts_hip_workspace_size  # noqa: PLC0415

    # Workspace size is shape-derived.  Zeros satisfy the public validation
    # before the fused key/count phase overwrites this caller-owned buffer.
    counts.zero_()
    return torch.empty(
        cell_starts_hip_workspace_size(counts, starts),
        dtype=torch.uint8,
        device=counts.device,
    )


def _run(
    positions: torch.Tensor,
    inverse_cells: torch.Tensor,
    dimensions: torch.Tensor,
    pbc: torch.Tensor,
    batch_idx: torch.Tensor,
    offsets: torch.Tensor,
    *,
    global_atom_offset: int,
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
    actual = _outputs(positions.shape[0], total_cells, positions.device)
    expected = _outputs(positions.shape[0], total_cells, positions.device)
    actual_shifts, actual_mapping, actual_keys, actual_counts, actual_starts, actual_list, actual_cursor = actual
    expected_shifts, expected_mapping, expected_keys, expected_counts, expected_starts, expected_list, expected_cursor = expected
    workspace = _workspace(actual_counts, actual_starts)

    build_batch_cell_csr_hip_into(
        positions,
        inverse_cells,
        dimensions,
        pbc,
        batch_idx,
        offsets,
        actual_shifts,
        actual_mapping,
        actual_keys,
        actual_counts,
        actual_starts,
        actual_list,
        actual_cursor,
        workspace,
        global_atom_offset=global_atom_offset,
    )
    build_batch_cell_key_counts_reference_into(
        positions,
        inverse_cells,
        dimensions,
        pbc,
        batch_idx,
        offsets,
        expected_shifts,
        expected_mapping,
        expected_keys,
        expected_counts,
    )
    build_cell_starts_reference_into(
        expected_counts,
        expected_starts,
        global_atom_offset=global_atom_offset,
    )
    build_cell_atom_list_reference_into(
        expected_keys,
        expected_counts,
        expected_starts,
        expected_list,
        expected_cursor,
        global_atom_offset=global_atom_offset,
    )
    torch.cuda.synchronize()

    for name, actual_value, expected_value in zip(
        ("shifts", "mapping", "keys", "counts", "starts"),
        actual[:5],
        expected[:5],
        strict=True,
    ):
        if not torch.equal(actual_value, expected_value):
            raise AssertionError(f"{label}: {name} differs from Torch reference")
    _assert_atom_sets(
        actual_list,
        expected_list,
        actual_counts,
        actual_starts,
        global_atom_offset=global_atom_offset,
        label=label,
    )
    if not torch.equal(actual_cursor, actual_counts):
        raise AssertionError(f"{label}: cursor does not equal final counts")
    if not torch.equal(actual_list[positions.shape[0] :], torch.full_like(actual_list[positions.shape[0] :], -9)):
        raise AssertionError(f"{label}: fill wrote beyond active list capacity")


def main() -> None:
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("HIP PyTorch and a visible HCU are required")

    root = Path(__file__).resolve().parent
    os.environ.setdefault(
        "NVALCHEMI_HIP_BATCH_CELL_KEY_COUNT_BUILD_DIR",
        str(root.parent / "artifacts" / "native-hip-batch-cell-key-count"),
    )
    os.environ.setdefault(
        "NVALCHEMI_HIP_CELL_SCAN_BUILD_DIR",
        str(root.parent / "artifacts" / "native-hip-cell-start-scan"),
    )
    os.environ.setdefault(
        "NVALCHEMI_HIP_CELL_FILL_BUILD_DIR",
        str(root.parent / "artifacts" / "native-hip-cell-fill"),
    )
    from nvalchemiops._hip_batch_cell_key_count import _load_jit_extension as load_fusion  # noqa: PLC0415
    from nvalchemiops._hip_cell_fill import _load_jit_extension as load_fill  # noqa: PLC0415
    from nvalchemiops._hip_cell_scan import _load_jit_extension as load_scan  # noqa: PLC0415

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260919)
    dimensions = torch.tensor([[4, 5, 6], [3, 4, 2]], dtype=torch.int32, device=device)
    offsets = torch.tensor([0, 120], dtype=torch.int32, device=device)
    pbc = torch.tensor([[True, False, True], [False, True, False]], device=device)
    cells64 = torch.tensor(
        [
            [[4.0, 0.0, 0.0], [0.7, 3.5, 0.0], [0.4, 0.6, 4.2]],
            [[3.0, 0.0, 0.0], [0.2, 2.5, 0.0], [0.1, 0.3, 3.1]],
        ],
        dtype=torch.float64,
        device=device,
    )
    atoms = 257
    batch_idx = torch.randint(2, (atoms,), dtype=torch.int32, device=device, generator=generator)
    dtype_parity: dict[str, bool] = {}
    for dtype in (torch.float32, torch.float64):
        fractional = torch.rand((atoms, 3), dtype=dtype, device=device, generator=generator)
        positions = torch.bmm(
            (fractional * 2.5 - 0.75).unsqueeze(1), cells64.to(dtype)[batch_idx.to(torch.long)]
        ).squeeze(1)
        _run(
            positions,
            torch.linalg.inv(cells64.to(dtype)),
            dimensions,
            pbc,
            batch_idx,
            offsets,
            global_atom_offset=17,
            label=str(dtype),
        )
        dtype_parity[str(dtype)] = True

    single_dims = torch.tensor([[2, 3, 4]], dtype=torch.int32, device=device)
    _run(
        torch.empty((0, 3), dtype=torch.float32, device=device),
        torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0),
        single_dims,
        torch.ones((1, 3), dtype=torch.bool, device=device),
        torch.empty(0, dtype=torch.int32, device=device),
        torch.zeros(1, dtype=torch.int32, device=device),
        global_atom_offset=7,
        label="empty B=1",
    )

    position_storage = torch.zeros((atoms, 6), dtype=torch.float32, device=device)
    strided_positions = position_storage[:, ::2]
    strided_positions.copy_(torch.rand((atoms, 3), dtype=torch.float32, device=device, generator=generator))
    inverse_storage = torch.zeros((2, 3, 6), dtype=torch.float32, device=device)
    strided_inverse = inverse_storage[:, :, ::2]
    strided_inverse.copy_(torch.linalg.inv(cells64.to(torch.float32)))
    batch_storage = torch.empty(atoms * 2, dtype=torch.int32, device=device)
    strided_batch = batch_storage[::2]
    strided_batch.copy_(batch_idx)
    _run(strided_positions, strided_inverse, dimensions, pbc, strided_batch, offsets, global_atom_offset=0, label="strided read-only inputs")

    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        _run(
            torch.rand((atoms, 3), dtype=torch.float32, device=device, generator=generator),
            torch.linalg.inv(cells64.to(torch.float32)),
            dimensions,
            pbc,
            batch_idx,
            offsets,
            global_atom_offset=0,
            label="non-default stream",
        )
    stream.synchronize()
    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "atomic_order_compared_as_cell_sets": True,
                "atoms": atoms,
                "batch_systems": 2,
                "device": torch.cuda.get_device_name(),
                "dtype_parity": dtype_parity,
                "empty_b1": True,
                "extensions": {
                    "fill": str(Path(load_fill().__file__).resolve()),
                    "fusion": str(Path(load_fusion().__file__).resolve()),
                    "scan": str(Path(load_scan().__file__).resolve()),
                },
                "global_atom_offset": 17,
                "heterogeneous_dimensions": [[4, 5, 6], [3, 4, 2]],
                "mixed_pbc": [[True, False, True], [False, True, False]],
                "non_contiguous_read_only_inputs": True,
                "non_default_stream": True,
                "performance_measured": False,
                "torch_hip": torch.version.hip,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
