#!/usr/bin/env python3
"""Validate native HIP CSR atom-list fill against the stable Torch oracle.

Atomic insertion order is intentionally compared as a per-cell atom set.  This
probe does not register the candidate, canonicalize its output, or measure
performance.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch


def _assert_cell_sets(
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
        actual_values = torch.sort(actual[start : start + count]).values
        expected_values = torch.sort(expected[start : start + count]).values
        if not torch.equal(actual_values, expected_values):
            raise AssertionError(f"{label}: cell {key} atom set differs")


def _outputs(atoms: int, cells: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.full((atoms + 3,), -9, dtype=torch.int32, device=device),
        torch.full((cells,), -7, dtype=torch.int32, device=device),
    )


def _run(
    keys: torch.Tensor,
    counts: torch.Tensor,
    starts: torch.Tensor,
    *,
    global_atom_offset: int,
    label: str,
) -> None:
    from nvalchemiops._cell_list_abi import (  # noqa: PLC0415
        build_cell_atom_list_reference_into,
    )
    from nvalchemiops._hip_cell_fill import (  # noqa: PLC0415
        build_cell_atom_list_hip_into,
    )

    expected_list, expected_cursor = _outputs(keys.numel(), counts.numel(), keys.device)
    actual_list, actual_cursor = _outputs(keys.numel(), counts.numel(), keys.device)
    counts_before = counts.clone()
    starts_before = starts.clone()
    build_cell_atom_list_reference_into(
        keys,
        counts,
        starts,
        expected_list,
        expected_cursor,
        global_atom_offset=global_atom_offset,
    )
    build_cell_atom_list_hip_into(
        keys,
        counts,
        starts,
        actual_list,
        actual_cursor,
        global_atom_offset=global_atom_offset,
    )
    torch.cuda.synchronize()
    _assert_cell_sets(
        actual_list,
        expected_list,
        counts,
        starts,
        global_atom_offset=global_atom_offset,
        label=label,
    )
    if not torch.equal(actual_cursor, counts):
        raise AssertionError(f"{label}: cursor does not equal final counts")
    if not torch.equal(counts, counts_before) or not torch.equal(starts, starts_before):
        raise AssertionError(f"{label}: public CSR buffers were modified")
    if not torch.equal(actual_list[keys.numel() :], torch.full_like(actual_list[keys.numel() :], -9)):
        raise AssertionError(f"{label}: fill wrote beyond active atom-list capacity")


def main() -> None:
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("HIP PyTorch and a visible HCU are required")

    root = Path(__file__).resolve().parent
    os.environ.setdefault(
        "NVALCHEMI_HIP_CELL_FILL_BUILD_DIR",
        str(root.parent / "artifacts" / "native-hip-cell-fill"),
    )
    from nvalchemiops._cell_list_abi import (  # noqa: PLC0415
        build_batch_cell_key_counts_reference_into,
        build_cell_starts_reference_into,
    )
    from nvalchemiops._hip_cell_fill import _load_jit_extension  # noqa: PLC0415

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
    total_cells = 144
    batch_idx = torch.randint(2, (atoms,), dtype=torch.int32, device=device, generator=generator)
    dtype_parity: dict[str, bool] = {}
    saved: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    for dtype in (torch.float32, torch.float64):
        cells = cells64.to(dtype)
        fractional = torch.rand((atoms, 3), dtype=dtype, device=device, generator=generator)
        fractional = fractional * 2.5 - 0.75
        positions = torch.bmm(
            fractional.unsqueeze(1), cells[batch_idx.to(torch.long)]
        ).squeeze(1)
        shifts = torch.empty((atoms, 3), dtype=torch.int32, device=device)
        mapping = torch.empty_like(shifts)
        keys = torch.empty(atoms, dtype=torch.int32, device=device)
        counts = torch.empty(total_cells, dtype=torch.int32, device=device)
        build_batch_cell_key_counts_reference_into(
            positions,
            torch.linalg.inv(cells),
            dimensions,
            pbc,
            batch_idx,
            offsets,
            shifts,
            mapping,
            keys,
            counts,
        )
        starts = torch.empty_like(counts)
        build_cell_starts_reference_into(counts, starts)
        _run(keys, counts, starts, global_atom_offset=0, label=str(dtype))
        dtype_parity[str(dtype)] = True
        if dtype == torch.float32:
            saved = (keys, counts, starts)

    assert saved is not None
    keys, counts, starts = saved
    key_storage = torch.empty(keys.numel() * 2, dtype=torch.int32, device=device)
    strided_keys = key_storage[::2]
    strided_keys.copy_(keys)
    count_storage = torch.empty(counts.numel() * 2, dtype=torch.int32, device=device)
    strided_counts = count_storage[::2]
    strided_counts.copy_(counts)
    start_storage = torch.empty(starts.numel() * 2, dtype=torch.int32, device=device)
    strided_starts = start_storage[::2]
    strided_starts.copy_(starts)
    _run(strided_keys, strided_counts, strided_starts, global_atom_offset=0, label="strided inputs")

    empty_counts = torch.zeros(24, dtype=torch.int32, device=device)
    empty_starts = torch.full_like(empty_counts, 7)
    _run(
        torch.empty(0, dtype=torch.int32, device=device),
        empty_counts,
        empty_starts,
        global_atom_offset=7,
        label="empty B=1",
    )

    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        _run(keys, counts, starts, global_atom_offset=0, label="non-default stream")
    stream.synchronize()
    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "atoms": atoms,
                "atomic_order_compared_as_cell_sets": True,
                "batch_systems": 2,
                "device": torch.cuda.get_device_name(),
                "dtype_parity": dtype_parity,
                "empty_b1": True,
                "extension_path": str(Path(_load_jit_extension().__file__).resolve()),
                "global_counts_and_starts_unchanged": True,
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
