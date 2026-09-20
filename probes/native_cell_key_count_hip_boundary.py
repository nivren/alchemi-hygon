#!/usr/bin/env python3
"""Validate native HIP cell-key atomic-count output and stream parity.

This probe is a correctness boundary only.  It neither changes runtime
dispatch nor reports a performance result.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch


def _assert_equal(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    if not torch.equal(actual, expected):
        raise AssertionError(f"{label}: native HIP counts differ from Torch reference")


def main() -> None:
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("HIP PyTorch and a visible HCU are required")

    root = Path(__file__).resolve().parent
    os.environ.setdefault(
        "NVALCHEMI_HIP_CELL_COUNT_BUILD_DIR",
        str(root.parent / "artifacts" / "native-hip-cell-key-count"),
    )
    from nvalchemiops._cell_list_abi import (  # noqa: PLC0415
        build_cell_counts_reference_into,
    )
    from nvalchemiops._hip_cell_count import (  # noqa: PLC0415
        _load_jit_extension,
        build_cell_counts_hip_into,
    )

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260919)
    total_cells = 137
    keys = torch.randint(
        total_cells, (4099,), dtype=torch.int32, device=device, generator=generator
    )
    actual = torch.full((total_cells,), -7, dtype=torch.int32, device=device)
    expected = actual.clone()
    build_cell_counts_hip_into(keys, actual)
    build_cell_counts_reference_into(keys, expected)
    _assert_equal(actual, expected, "regular input")

    empty_actual = torch.full((total_cells,), -7, dtype=torch.int32, device=device)
    empty_expected = empty_actual.clone()
    empty_keys = torch.empty(0, dtype=torch.int32, device=device)
    build_cell_counts_hip_into(empty_keys, empty_actual)
    build_cell_counts_reference_into(empty_keys, empty_expected)
    _assert_equal(empty_actual, empty_expected, "empty input")

    strided_storage = torch.zeros(8198, dtype=torch.int32, device=device)
    strided_keys = strided_storage[::2]
    strided_keys.copy_(keys)
    strided_actual = torch.empty(total_cells, dtype=torch.int32, device=device)
    strided_expected = torch.empty_like(strided_actual)
    build_cell_counts_hip_into(strided_keys, strided_actual)
    build_cell_counts_reference_into(strided_keys, strided_expected)
    _assert_equal(strided_actual, strided_expected, "strided read-only input")

    stream = torch.cuda.Stream(device=device)
    stream_actual = torch.empty(total_cells, dtype=torch.int32, device=device)
    with torch.cuda.stream(stream):
        build_cell_counts_hip_into(keys, stream_actual)
    stream.synchronize()
    stream_expected = torch.empty_like(stream_actual)
    build_cell_counts_reference_into(keys, stream_expected)
    _assert_equal(stream_actual, stream_expected, "non-default stream")

    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "atoms": int(keys.numel()),
                "device": torch.cuda.get_device_name(),
                "empty_input": True,
                "extension_path": str(Path(_load_jit_extension().__file__).resolve()),
                "non_contiguous_read_only_input": True,
                "non_default_stream": True,
                "performance_measured": False,
                "total_cells": total_cells,
                "torch_hip": torch.version.hip,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
