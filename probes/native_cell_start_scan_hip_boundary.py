#!/usr/bin/env python3
"""Validate native HIP rocPRIM CSR start scan output and stream parity."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch


def _workspace(counts: torch.Tensor, starts: torch.Tensor) -> torch.Tensor:
    from nvalchemiops._hip_cell_scan import cell_starts_hip_workspace_size  # noqa: PLC0415

    return torch.empty(
        cell_starts_hip_workspace_size(counts, starts),
        dtype=torch.uint8,
        device=counts.device,
    )


def _assert_parity(
    counts: torch.Tensor,
    starts: torch.Tensor,
    *,
    global_atom_offset: int,
    label: str,
) -> None:
    from nvalchemiops._cell_list_abi import (  # noqa: PLC0415
        build_cell_starts_reference_into,
    )
    from nvalchemiops._hip_cell_scan import (  # noqa: PLC0415
        build_cell_starts_hip_into,
    )

    counts_before = counts.clone()
    expected = torch.empty_like(starts)
    build_cell_starts_hip_into(
        counts,
        starts,
        _workspace(counts, starts),
        global_atom_offset=global_atom_offset,
    )
    build_cell_starts_reference_into(
        counts_before,
        expected,
        global_atom_offset=global_atom_offset,
    )
    torch.cuda.synchronize()
    if not torch.equal(counts, counts_before):
        raise AssertionError(f"{label}: scan modified counts")
    if not torch.equal(starts, expected):
        raise AssertionError(f"{label}: starts differ from Torch reference")


def main() -> None:
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("HIP PyTorch and a visible HCU are required")

    root = Path(__file__).resolve().parent
    os.environ.setdefault(
        "NVALCHEMI_HIP_CELL_SCAN_BUILD_DIR",
        str(root.parent / "artifacts" / "native-hip-cell-start-scan"),
    )
    from nvalchemiops._hip_cell_scan import _load_jit_extension  # noqa: PLC0415

    device = torch.device("cuda")
    counts = torch.tensor([1, 1, 0, 2, 0, 3], dtype=torch.int32, device=device)
    starts = torch.empty_like(counts)
    _assert_parity(counts, starts, global_atom_offset=10, label="regular")

    empty_counts = torch.empty(0, dtype=torch.int32, device=device)
    empty_starts = torch.empty_like(empty_counts)
    _assert_parity(empty_counts, empty_starts, global_atom_offset=7, label="empty")

    strided_storage = torch.zeros(12, dtype=torch.int32, device=device)
    strided_counts = strided_storage[::2]
    strided_counts.copy_(counts)
    strided_starts = torch.empty_like(counts)
    _assert_parity(
        strided_counts,
        strided_starts,
        global_atom_offset=10,
        label="strided read-only counts",
    )

    stream = torch.cuda.Stream(device=device)
    stream_counts = counts.clone()
    stream_starts = torch.empty_like(stream_counts)
    with torch.cuda.stream(stream):
        _assert_parity(
            stream_counts,
            stream_starts,
            global_atom_offset=10,
            label="non-default stream",
        )
    stream.synchronize()

    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "cells": int(counts.numel()),
                "counts_unchanged": True,
                "device": torch.cuda.get_device_name(),
                "empty_input": True,
                "extension_path": str(Path(_load_jit_extension().__file__).resolve()),
                "global_atom_offset": 10,
                "non_contiguous_read_only_counts": True,
                "non_default_stream": True,
                "performance_measured": False,
                "torch_hip": torch.version.hip,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
