#!/usr/bin/env python3
"""Validate the isolated one-sort composite-key topology candidate.

The candidate is compared with the existing five-stable-sort HIP candidate on
an unordered, mixed-sign PBC-shift fixture.  It is intentionally a topology
only probe: no dispatcher, geometry, autograd, or runtime policy is changed.
"""

from __future__ import annotations

import json

import torch


def _outputs(atoms: int, capacity: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        torch.full((atoms, capacity), atoms, dtype=torch.int32, device=device),
        torch.zeros((atoms, capacity, 3), dtype=torch.int32, device=device),
        torch.zeros(atoms, dtype=torch.int32, device=device),
    )


def _assert_same(actual: tuple[torch.Tensor, ...], expected: tuple[torch.Tensor, ...]) -> None:
    for lhs, rhs in zip(actual, expected, strict=True):
        if not torch.equal(lhs, rhs):
            mismatch = torch.nonzero(lhs != rhs, as_tuple=False)
            first = tuple(int(value) for value in mismatch[0].tolist())
            raise AssertionError(
                f"composite topology differs at {first}: "
                f"actual={lhs[first].item()} expected={rhs[first].item()}"
            )


def main() -> None:
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("HIP PyTorch and a visible HCU are required")

    from nvalchemiops._hip_batch_query_materialize import (  # noqa: PLC0415
        _composite_key_layout,
        materialize_batch_query_topology_composite_hip_into,
        materialize_batch_query_topology_hip_into,
    )

    device = torch.device("cuda")
    atoms = 5
    capacity = 5
    candidate_matrix = torch.tensor(
        [
            [4, 1, 3, 0, 5],
            [4, 0, 3, 5, 5],
            [0, 4, 5, 5, 5],
            [1, 0, 4, 2, 5],
            [3, 5, 5, 5, 5],
        ],
        dtype=torch.int32,
        device=device,
    )
    candidate_shifts = torch.tensor(
        [
            [[0, 0, 0], [-1, 0, 2], [1, -1, 0], [0, 2, -1], [0, 0, 0]],
            [[1, 0, -1], [0, 0, 0], [-2, 1, 0], [0, 0, 0], [0, 0, 0]],
            [[0, -1, 1], [2, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 1, 0], [-1, 0, 1], [2, -2, 0], [0, 0, -1], [0, 0, 0]],
            [[-2, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ],
        dtype=torch.int32,
        device=device,
    )
    candidate_counts = torch.tensor([4, 3, 2, 4, 1], dtype=torch.int32, device=device)
    expected = _outputs(atoms, capacity, device)
    actual = _outputs(atoms, capacity, device)

    materialize_batch_query_topology_hip_into(
        candidate_matrix,
        candidate_shifts,
        candidate_counts,
        *expected,
    )
    materialize_batch_query_topology_composite_hip_into(
        candidate_matrix,
        candidate_shifts,
        candidate_counts,
        *actual,
        shift_bits=4,
        shift_bias=8,
    )
    torch.cuda.synchronize()
    _assert_same(actual, expected)

    stream = torch.cuda.Stream(device=device)
    stream_actual = _outputs(atoms, capacity, device)
    with torch.cuda.stream(stream):
        materialize_batch_query_topology_composite_hip_into(
            candidate_matrix,
            candidate_shifts,
            candidate_counts,
            *stream_actual,
            shift_bits=4,
            shift_bias=8,
        )
    stream.synchronize()
    _assert_same(stream_actual, expected)

    overflow_shifts = candidate_shifts.clone()
    overflow_shifts[0, 0, 0] = 3
    try:
        materialize_batch_query_topology_composite_hip_into(
            candidate_matrix,
            overflow_shifts,
            candidate_counts,
            *_outputs(atoms, capacity, device),
            shift_bits=2,
            shift_bias=2,
        )
    except ValueError as exc:
        if "composite key range" not in str(exc):
            raise AssertionError(f"unexpected shift overflow error: {exc}") from exc
    else:
        raise AssertionError("shift range overflow was not rejected")

    row_bits, total_bits = _composite_key_layout(atoms, 4, 8)
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "torch_hip": torch.version.hip,
                "atoms": atoms,
                "capacity": capacity,
                "row_bits": row_bits,
                "total_key_bits": total_bits,
                "default_parity": True,
                "non_default_stream_parity": True,
                "shift_overflow_rejected": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
