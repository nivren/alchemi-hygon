#!/usr/bin/env python3
"""Validate the package native HIP cell-key custom-op boundary.

This probe checks output parity and stream behavior only.  It does not register
the implementation as a production backend or collect performance numbers.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch


def _assert_equal(actual, expected, label: str) -> None:
    for name in ("atom_periodic_shifts", "atom_to_cell_mapping", "cell_keys"):
        actual_value = getattr(actual, name)
        expected_value = getattr(expected, name)
        if not torch.equal(actual_value, expected_value):
            raise AssertionError(f"{label}: {name} differs from Torch reference")


def main() -> None:
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("HIP PyTorch and a visible HCU are required")

    root = Path(__file__).resolve().parent
    os.environ.setdefault(
        "NVALCHEMI_HIP_CELL_KEY_BUILD_DIR",
        str(root.parent / "artifacts" / "native-hip-cell-key"),
    )
    from nvalchemiops._cell_list_abi import (  # noqa: PLC0415
        CellKeyBuildResult,
        build_cell_keys_reference,
        build_cell_keys_reference_into,
    )
    from nvalchemiops._hip_cell_key import (  # noqa: PLC0415
        _load_native_extension,
        build_cell_keys_hip,
        build_cell_keys_hip_into,
    )

    device = torch.device("cuda")
    extension_path = str(Path(_load_native_extension().__file__).resolve())
    generator = torch.Generator(device=device).manual_seed(20260919)
    cell = torch.tensor(
        [[4.0, 0.0, 0.0], [0.7, 3.5, 0.0], [0.4, 0.6, 4.2]],
        dtype=torch.float64,
        device=device,
    )
    dimensions = torch.tensor([7, 6, 8], dtype=torch.int32, device=device)
    pbc = torch.tensor([True, True, False], dtype=torch.bool, device=device)
    parity: dict[str, bool] = {}
    for dtype in (torch.float32, torch.float64):
        fractional = torch.rand(
            (257, 3), generator=generator, device=device, dtype=dtype
        )
        fractional = fractional * 2.5 - 0.75
        positions = fractional @ cell.to(dtype)
        actual = build_cell_keys_hip(
            positions, torch.linalg.inv(cell.to(dtype)), dimensions, pbc
        )
        expected = build_cell_keys_reference(
            positions, torch.linalg.inv(cell.to(dtype)), dimensions, pbc
        )
        _assert_equal(actual, expected, str(dtype))
        parity[str(dtype)] = True

    empty_positions = torch.empty((0, 3), dtype=torch.float32, device=device)
    empty_outputs = CellKeyBuildResult(
        atom_periodic_shifts=torch.full((0, 3), -7, dtype=torch.int32, device=device),
        atom_to_cell_mapping=torch.full((0, 3), -7, dtype=torch.int32, device=device),
        cell_keys=torch.full((0,), -7, dtype=torch.int32, device=device),
    )
    expected_empty = CellKeyBuildResult(
        atom_periodic_shifts=empty_outputs.atom_periodic_shifts.clone(),
        atom_to_cell_mapping=empty_outputs.atom_to_cell_mapping.clone(),
        cell_keys=empty_outputs.cell_keys.clone(),
    )
    build_cell_keys_hip_into(
        empty_positions,
        torch.eye(3, dtype=torch.float32, device=device),
        dimensions,
        pbc,
        empty_outputs.atom_periodic_shifts,
        empty_outputs.atom_to_cell_mapping,
        empty_outputs.cell_keys,
    )
    build_cell_keys_reference_into(
        empty_positions,
        torch.eye(3, dtype=torch.float32, device=device),
        dimensions,
        pbc,
        expected_empty.atom_periodic_shifts,
        expected_empty.atom_to_cell_mapping,
        expected_empty.cell_keys,
    )
    _assert_equal(empty_outputs, expected_empty, "empty")

    stream = torch.cuda.Stream(device=device)
    fractional = torch.rand(
        (257, 3), generator=generator, device=device, dtype=torch.float32
    ) * 2.5 - 0.75
    stream_positions = fractional @ cell.to(torch.float32)
    stream_inverse = torch.linalg.inv(cell.to(torch.float32))
    with torch.cuda.stream(stream):
        stream_actual = build_cell_keys_hip(
            stream_positions, stream_inverse, dimensions, pbc
        )
    stream.synchronize()
    stream_expected = build_cell_keys_reference(
        stream_positions, stream_inverse, dimensions, pbc
    )
    _assert_equal(stream_actual, stream_expected, "non-default stream")

    non_contiguous = torch.zeros((257, 6), dtype=torch.float32, device=device)[:, ::2]
    non_contiguous_inverse = torch.eye(3, dtype=torch.float32, device=device).t()
    outputs = [
        torch.empty((257, 3), dtype=torch.int32, device=device),
        torch.empty((257, 3), dtype=torch.int32, device=device),
        torch.empty((257,), dtype=torch.int32, device=device),
    ]
    expected_outputs = [output.clone() for output in outputs]
    build_cell_keys_hip_into(
        non_contiguous,
        non_contiguous_inverse,
        dimensions,
        pbc,
        *outputs,
    )
    build_cell_keys_reference_into(
        non_contiguous,
        non_contiguous_inverse,
        dimensions,
        pbc,
        *expected_outputs,
    )
    _assert_equal(
        CellKeyBuildResult(*outputs),
        CellKeyBuildResult(*expected_outputs),
        "non-contiguous read-only inputs",
    )

    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "torch_hip": torch.version.hip,
                "atoms": 257,
                "mixed_pbc": [True, True, False],
                "dtype_parity": parity,
                "empty_input": True,
                "non_default_stream": True,
                "non_contiguous_read_only_inputs": True,
                "performance_measured": False,
                "extension_path": extension_path,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
