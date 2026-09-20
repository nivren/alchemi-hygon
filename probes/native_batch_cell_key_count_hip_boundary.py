#!/usr/bin/env python3
"""Validate the native HIP fused Batch geometry/PBC/key/count boundary.

This is a numerical/stream probe only.  It does not time the candidate or
connect it to a cell-list runtime path.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch


def _outputs(num_atoms: int, total_cells: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        torch.full((num_atoms, 3), -7, dtype=torch.int32, device=device),
        torch.full((num_atoms, 3), -7, dtype=torch.int32, device=device),
        torch.full((num_atoms,), -7, dtype=torch.int32, device=device),
        torch.full((total_cells,), -7, dtype=torch.int32, device=device),
    )


def _assert_equal(
    actual: tuple[torch.Tensor, ...], expected: tuple[torch.Tensor, ...], label: str
) -> None:
    for name, actual_value, expected_value in zip(
        ("shifts", "mapping", "keys", "counts"), actual, expected, strict=True
    ):
        if not torch.equal(actual_value, expected_value):
            raise AssertionError(f"{label}: {name} differs from Torch reference")


def _run(
    positions: torch.Tensor,
    inverse_cells: torch.Tensor,
    dimensions: torch.Tensor,
    pbc: torch.Tensor,
    batch_idx: torch.Tensor,
    offsets: torch.Tensor,
    *,
    label: str,
) -> None:
    from nvalchemiops._cell_list_abi import (  # noqa: PLC0415
        build_batch_cell_key_counts_reference_into,
    )
    from nvalchemiops._hip_batch_cell_key_count import (  # noqa: PLC0415
        build_batch_cell_key_counts_hip_into,
    )

    total_cells = int(dimensions.to(torch.int64).prod(dim=1).sum().item())
    actual = _outputs(positions.shape[0], total_cells, positions.device)
    expected = _outputs(positions.shape[0], total_cells, positions.device)
    build_batch_cell_key_counts_hip_into(
        positions,
        inverse_cells,
        dimensions,
        pbc,
        batch_idx,
        offsets,
        *actual,
    )
    build_batch_cell_key_counts_reference_into(
        positions,
        inverse_cells,
        dimensions,
        pbc,
        batch_idx,
        offsets,
        *expected,
    )
    _assert_equal(actual, expected, label)


def main() -> None:
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("HIP PyTorch and a visible HCU are required")

    root = Path(__file__).resolve().parent
    os.environ.setdefault(
        "NVALCHEMI_HIP_BATCH_CELL_KEY_COUNT_BUILD_DIR",
        str(root.parent / "artifacts" / "native-hip-batch-cell-key-count"),
    )
    from nvalchemiops._hip_batch_cell_key_count import _load_jit_extension  # noqa: PLC0415

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
    batch_idx = torch.randint(2, (257,), dtype=torch.int32, device=device, generator=generator)
    dtype_parity: dict[str, bool] = {}
    for dtype in (torch.float32, torch.float64):
        fractional = torch.rand((257, 3), dtype=dtype, device=device, generator=generator)
        fractional = fractional * 2.5 - 0.75
        cells = cells64.to(dtype)
        positions = torch.bmm(
            fractional.unsqueeze(1), cells[batch_idx.to(torch.long)]
        ).squeeze(1)
        _run(
            positions,
            torch.linalg.inv(cells),
            dimensions,
            pbc,
            batch_idx,
            offsets,
            label=str(dtype),
        )
        dtype_parity[str(dtype)] = True

    single_dims = torch.tensor([[2, 3, 4]], dtype=torch.int32, device=device)
    single_offsets = torch.zeros(1, dtype=torch.int32, device=device)
    single_pbc = torch.ones((1, 3), dtype=torch.bool, device=device)
    _run(
        torch.empty((0, 3), dtype=torch.float32, device=device),
        torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0),
        single_dims,
        single_pbc,
        torch.empty(0, dtype=torch.int32, device=device),
        single_offsets,
        label="empty B=1",
    )

    positions_storage = torch.zeros((257, 6), dtype=torch.float32, device=device)
    strided_positions = positions_storage[:, ::2]
    strided_positions.copy_(torch.rand((257, 3), dtype=torch.float32, device=device, generator=generator))
    inverse_storage = torch.zeros((2, 3, 6), dtype=torch.float32, device=device)
    strided_inverse = inverse_storage[:, :, ::2]
    strided_inverse.copy_(torch.linalg.inv(cells64.to(torch.float32)))
    batch_storage = torch.empty(514, dtype=torch.int32, device=device)
    strided_batch = batch_storage[::2]
    strided_batch.copy_(batch_idx)
    offset_storage = torch.empty(4, dtype=torch.int32, device=device)
    strided_offsets = offset_storage[::2]
    strided_offsets.copy_(offsets)
    _run(
        strided_positions,
        strided_inverse,
        dimensions,
        pbc,
        strided_batch,
        strided_offsets,
        label="strided read-only inputs",
    )

    stream = torch.cuda.Stream(device=device)
    stream_positions = torch.rand((257, 3), dtype=torch.float32, device=device, generator=generator)
    with torch.cuda.stream(stream):
        _run(
            stream_positions,
            torch.linalg.inv(cells64.to(torch.float32)),
            dimensions,
            pbc,
            batch_idx,
            offsets,
            label="non-default stream",
        )
    stream.synchronize()
    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "atoms": 257,
                "batch_systems": 2,
                "device": torch.cuda.get_device_name(),
                "dtype_parity": dtype_parity,
                "empty_b1": True,
                "extension_path": str(Path(_load_jit_extension().__file__).resolve()),
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
