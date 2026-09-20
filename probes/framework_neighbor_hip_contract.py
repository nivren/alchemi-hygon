#!/usr/bin/env python3
"""Validate the Point41 public HIP neighbor compatibility gate on a Hygon DCU.

This probe is correctness/boundary-only.  It covers the registered FP32/FP64
periodic full-list MATRIX path, empty input, automatic capacity growth,
explicit capacity overflow, unsupported public requests, and repeated no-skin
Hook calls.  It does not measure performance or change backend selection
policy.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.hooks import NeighborListHook
from nvalchemi.hooks._context import HookContext
from nvalchemi.models.base import NeighborConfig, NeighborListFormat
from nvalchemi.neighbors import compute_neighbors
from nvalchemiops.backend import BackendUnavailableError


def _make_data(
    atoms_per_system: int,
    batch_systems: int,
    dtype: torch.dtype,
    device: torch.device,
) -> list[AtomicData]:
    cells = torch.tensor(
        [
            [[16.0, 0.0, 0.0], [0.5, 16.0, 0.0], [0.3, 0.4, 16.0]],
            [[12.0, 0.0, 0.0], [0.4, 12.0, 0.0], [0.2, 0.3, 12.0]],
        ],
        dtype=dtype,
        device=device,
    )
    pbc = torch.tensor(
        [[True, False, True], [False, True, False]], device=device
    )
    generator = torch.Generator(device=device).manual_seed(20260920)
    output: list[AtomicData] = []
    for system in range(batch_systems):
        template = system % 2
        cell = cells[template]
        positions = torch.rand(
            (atoms_per_system, 3),
            dtype=dtype,
            device=device,
            generator=generator,
        ) @ cell
        output.append(
            AtomicData(
                positions=positions,
                atomic_numbers=torch.ones(
                    atoms_per_system, dtype=torch.int64, device=device
                ),
                cell=cell.unsqueeze(0),
                pbc=pbc[template].unsqueeze(0),
            )
        )
    return output


def _assert_same(lhs: Batch, rhs: Batch, label: str) -> None:
    torch.testing.assert_close(lhs.neighbor_matrix, rhs.neighbor_matrix, msg=label)
    torch.testing.assert_close(lhs.num_neighbors, rhs.num_neighbors, msg=label)
    torch.testing.assert_close(
        lhs.neighbor_matrix_shifts, rhs.neighbor_matrix_shifts, msg=label
    )


def _compute(batch: Batch, *, dtype: torch.dtype) -> None:
    del dtype
    compute_neighbors(
        batch,
        cutoff=0.6,
        format=NeighborListFormat.MATRIX,
        max_neighbors=256,
        backend="hip",
        method="cell_list",
    )


def _unsupported_requests(data: list[AtomicData]) -> dict[str, str]:
    periodic = Batch.from_data_list(data[:1])
    results: dict[str, str] = {}
    requests = {
        "periodic_half": lambda: compute_neighbors(
            periodic,
            cutoff=0.6,
            format=NeighborListFormat.MATRIX,
            half_list=True,
            backend="hip",
            method="cell_list",
        ),
        "periodic_coo": lambda: compute_neighbors(
            Batch.from_data_list(data[:1]),
            cutoff=0.6,
            format=NeighborListFormat.COO,
            backend="hip",
            method="cell_list",
        ),
        "no_pbc": lambda: compute_neighbors(
            Batch.from_data_list(
                [
                    AtomicData(
                        positions=torch.tensor(
                            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
                            dtype=data[0].positions.dtype,
                            device=data[0].positions.device,
                        ),
                        atomic_numbers=torch.ones(
                            2, dtype=torch.int64, device=data[0].positions.device
                        ),
                    )
                ]
            ),
            cutoff=0.6,
            backend="hip",
            method="cell_list",
        ),
    }
    for name, request in requests.items():
        try:
            request()
        except BackendUnavailableError as exc:
            results[name] = str(exc)
        else:
            raise AssertionError(f"unsupported HIP request unexpectedly passed: {name}")
    return results


def _capacity_overflow(device: torch.device) -> str:
    cell = torch.eye(3, dtype=torch.float32, device=device) * 10.0
    pbc = torch.ones((1, 3), dtype=torch.bool, device=device)
    batch = Batch.from_data_list(
        [
            AtomicData(
                positions=torch.tensor(
                    [[1.0, 1.0, 1.0], [1.1, 1.0, 1.0], [1.2, 1.0, 1.0]],
                    dtype=torch.float32,
                    device=device,
                ),
                atomic_numbers=torch.ones(3, dtype=torch.int64, device=device),
                cell=cell.unsqueeze(0),
                pbc=pbc,
            )
        ]
    )
    try:
        compute_neighbors(
            batch,
            cutoff=0.6,
            format=NeighborListFormat.MATRIX,
            max_neighbors=1,
            backend="hip",
            method="cell_list",
        )
    except RuntimeError as exc:
        if "capacity" not in str(exc):
            raise
        return str(exc)
    raise AssertionError("HIP capacity overflow was silently accepted")


def _periodic_image_data(device: torch.device) -> list[AtomicData]:
    """Create a one-atom periodic cell whose image neighbors force growth."""
    cell = torch.eye(3, dtype=torch.float32, device=device) * 2.0
    return [
        AtomicData(
            positions=torch.tensor(
                [[1.0, 1.0, 1.0]], dtype=torch.float32, device=device
            ),
            atomic_numbers=torch.ones(1, dtype=torch.int64, device=device),
            cell=cell.unsqueeze(0),
            pbc=torch.ones((1, 3), dtype=torch.bool, device=device),
        )
    ]


def _auto_capacity(device: torch.device) -> dict[str, object]:
    """Check None-capacity growth, parity, and the finite retry upper bound."""
    candidate = Batch.from_data_list(_periodic_image_data(device))
    compute_neighbors(
        candidate,
        cutoff=2.1,
        format=NeighborListFormat.MATRIX,
        max_neighbors=None,
        backend="hip",
        method="cell_list",
    )
    capacity = int(candidate.neighbor_matrix.shape[1])
    max_count = int(candidate.num_neighbors.max().item())
    if capacity <= 1:
        raise AssertionError(
            "HIP max_neighbors=None did not grow beyond the one-atom initial capacity"
        )
    if max_count > capacity:
        raise AssertionError("HIP automatic capacity returned truncated neighbors")

    reference = Batch.from_data_list(_periodic_image_data(device))
    compute_neighbors(
        reference,
        cutoff=2.1,
        format=NeighborListFormat.MATRIX,
        max_neighbors=capacity,
        backend="torch_reference",
        method="cell_list",
    )
    _assert_same(candidate, reference, "HIP automatic capacity versus Torch reference")

    # For cell=2I and cutoff=2.1, _grid_spec's conservative radius is 3 in
    # every dimension, so the executor's finite image-cell upper bound is 7^3.
    conservative_upper = 7**3
    if capacity > conservative_upper:
        raise AssertionError(
            f"HIP automatic capacity {capacity} exceeded conservative upper "
            f"bound {conservative_upper}"
        )
    return {
        "status": "passed",
        "initial_capacity": 1,
        "final_capacity": capacity,
        "max_count": max_count,
        "conservative_upper_bound": conservative_upper,
    }


def _empty_input(device: torch.device) -> dict[str, object]:
    dtype = torch.float32
    batch = Batch.from_data_list(
        [
            AtomicData(
                positions=torch.empty((0, 3), dtype=dtype, device=device),
                atomic_numbers=torch.empty((0,), dtype=torch.int64, device=device),
                cell=torch.eye(3, dtype=dtype, device=device).unsqueeze(0),
                pbc=torch.ones((1, 3), dtype=torch.bool, device=device),
            )
        ]
    )
    _compute(batch, dtype=dtype)
    return {
        "matrix_shape": list(batch.neighbor_matrix.shape),
        "counts_shape": list(batch.num_neighbors.shape),
        "shifts_shape": list(batch.neighbor_matrix_shifts.shape),
    }


def _repeat_hook(data: list[AtomicData]) -> dict[str, object]:
    batch = Batch.from_data_list(data)
    hook = NeighborListHook(
        NeighborConfig(cutoff=0.6, format=NeighborListFormat.MATRIX),
        max_neighbors=256,
        method="cell_list",
        backend="hip",
        skin=0.0,
    )
    eager_call = NeighborListHook.__call__.__wrapped__  # type: ignore[attr-defined]
    eager_call(hook, HookContext(batch=batch), None)
    first = (
        batch.neighbor_matrix.clone(),
        batch.num_neighbors.clone(),
        batch.neighbor_matrix_shifts.clone(),
    )
    eager_call(hook, HookContext(batch=batch), None)
    torch.testing.assert_close(batch.neighbor_matrix, first[0])
    torch.testing.assert_close(batch.num_neighbors, first[1])
    torch.testing.assert_close(batch.neighbor_matrix_shifts, first[2])
    return {
        "status": "passed",
        "matrix_shape": list(batch.neighbor_matrix.shape),
        "max_count": int(batch.num_neighbors.max().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atoms-per-system", type=int, default=46)
    parser.add_argument("--batch-systems", type=int, default=32)
    args = parser.parse_args()
    if min(args.atoms_per_system, args.batch_systems) <= 0:
        raise ValueError("atoms-per-system and batch-systems must be positive")
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("HIP PyTorch and a visible HCU are required")

    root = Path(__file__).resolve().parent
    for variable, directory in (
        ("NVALCHEMI_HIP_BATCH_CELL_KEY_COUNT_BUILD_DIR", "point38-cell-key-count"),
        ("NVALCHEMI_HIP_CELL_SCAN_BUILD_DIR", "point38-cell-scan"),
        ("NVALCHEMI_HIP_CELL_FILL_BUILD_DIR", "point38-cell-fill"),
        ("NVALCHEMI_HIP_BATCH_CELL_QUERY_BUILD_DIR", "point38-cell-query"),
        ("NVALCHEMI_HIP_BATCH_QUERY_MATERIALIZE_BUILD_DIR", "point38-topology"),
    ):
        os.environ.setdefault(variable, str(root.parent / "artifacts" / directory))

    device = torch.device("cuda")
    results: dict[str, object] = {}
    for dtype in (torch.float32, torch.float64):
        data = _make_data(args.atoms_per_system, args.batch_systems, dtype, device)
        candidate = Batch.from_data_list(data)
        reference = Batch.from_data_list(_make_data(
            args.atoms_per_system, args.batch_systems, dtype, device
        ))
        _compute(candidate, dtype=dtype)
        compute_neighbors(
            reference,
            cutoff=0.6,
            format=NeighborListFormat.MATRIX,
            max_neighbors=256,
            backend="torch_reference",
            method="cell_list",
        )
        torch.cuda.synchronize()
        _assert_same(candidate, reference, f"HIP versus Torch reference {dtype}")
        results[str(dtype).removeprefix("torch.")] = {
            "status": "passed",
            "matrix_shape": list(candidate.neighbor_matrix.shape),
            "max_count": int(candidate.num_neighbors.max().item()),
        }

    data = _make_data(args.atoms_per_system, args.batch_systems, torch.float32, device)
    results["empty_input"] = _empty_input(device)
    results["auto_capacity"] = _auto_capacity(device)
    results["capacity_overflow"] = {
        "status": "explicitly_rejected",
        "error": _capacity_overflow(device),
    }
    results["unsupported_requests"] = _unsupported_requests(data)
    results["repeated_hook"] = _repeat_hook(data)
    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "status": "passed",
                "device": torch.cuda.get_device_name(),
                "torch_hip": torch.version.hip,
                "atoms_per_system": args.atoms_per_system,
                "batch_systems": args.batch_systems,
                "hip_visible_device": os.environ.get("HIP_VISIBLE_DEVICES"),
                "scope": "Point41 HIP framework compatibility gate; no performance claim",
                "results": results,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
