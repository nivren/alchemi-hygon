#!/usr/bin/env python3
"""Measure the public framework neighbor API on a visible Hygon DCU.

The measured scope is the host API wall-clock of ``compute_neighbors``.  A
sample includes framework selection, ops dispatch, executor allocation,
fixed-cell metadata, native cell-list build/query, topology materialization,
and Batch write-back.  Device synchronization brackets every sample.

This probe intentionally measures one backend per process so that cold/JIT
cost is attributable to that backend.  It does not change backend policy or
claim that the measured backend should become the default.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from statistics import mean, median, pstdev

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
    device: torch.device,
    dtype: torch.dtype,
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
    data: list[AtomicData] = []
    for system in range(batch_systems):
        template = system % 2
        cell = cells[template]
        fractional = torch.rand(
            (atoms_per_system, 3), dtype=dtype, device=device, generator=generator
        )
        positions = fractional @ cell
        data.append(
            AtomicData(
                positions=positions,
                atomic_numbers=torch.ones(
                    atoms_per_system, dtype=torch.int64, device=device
                ),
                cell=cell.unsqueeze(0),
                pbc=pbc[template].unsqueeze(0),
            )
        )
    return data


def _summary(samples: list[float]) -> dict[str, float]:
    med = median(samples)
    return {
        "max_ms": max(samples),
        "mean_ms": mean(samples),
        "median_ms": med,
        "min_ms": min(samples),
        "relative_pstdev_percent": pstdev(samples) / med * 100.0,
    }


def _wall_ms(call: object) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    call()  # type: ignore[operator]
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000.0
    if elapsed <= 0.0:
        raise RuntimeError(f"non-positive API wall-clock sample: {elapsed}")
    return elapsed


def _compute(batch: Batch, backend: str) -> None:
    compute_neighbors(
        batch,
        cutoff=0.6,
        format=NeighborListFormat.MATRIX,
        max_neighbors=256,
        backend=backend,
        method="cell_list",
    )


def _assert_outputs(lhs: Batch, rhs: Batch, label: str) -> None:
    torch.testing.assert_close(lhs.neighbor_matrix, rhs.neighbor_matrix, msg=label)
    torch.testing.assert_close(lhs.num_neighbors, rhs.num_neighbors, msg=label)
    torch.testing.assert_close(
        lhs.neighbor_matrix_shifts, rhs.neighbor_matrix_shifts, msg=label
    )


def _hook_boundary(
    data: list[AtomicData],
    device: torch.device,
    backend: str,
) -> dict[str, object]:
    hook_batch = Batch.from_data_list(data)
    hook = NeighborListHook(
        NeighborConfig(cutoff=0.6, format=NeighborListFormat.MATRIX),
        max_neighbors=256,
        method="cell_list",
        backend=backend,
        skin=0.0,
    )
    NeighborListHook.__call__.__wrapped__(  # type: ignore[attr-defined]
        hook, HookContext(batch=hook_batch), None
    )
    result: dict[str, object] = {
        "skin_zero": {
            "status": "passed",
            "matrix_shape": list(hook_batch.neighbor_matrix.shape),
            "max_count": int(hook_batch.num_neighbors.max().item()),
        }
    }
    if backend == "hip":
        skin_hook = NeighborListHook(
            NeighborConfig(cutoff=0.6, format=NeighborListFormat.MATRIX),
            max_neighbors=256,
            method="cell_list",
            backend="hip",
            skin=0.5,
        )
        try:
            NeighborListHook.__call__.__wrapped__(  # type: ignore[attr-defined]
                skin_hook, HookContext(batch=Batch.from_data_list(data)), None
            )
        except BackendUnavailableError as exc:
            result["skin_nonzero"] = {
                "status": "explicitly_rejected",
                "error": str(exc),
            }
        else:
            raise AssertionError("HIP NeighborListHook unexpectedly accepted skin > 0")
    del device
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("hip", "torch_reference"), required=True)
    parser.add_argument("--atoms-per-system", type=int, default=46)
    parser.add_argument("--batch-systems", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/framework-neighbor-hip-wallclock"),
    )
    args = parser.parse_args()
    if min(
        args.atoms_per_system,
        args.batch_systems,
        args.warmup,
        args.samples,
    ) <= 0:
        raise ValueError("size, warmup, and samples must be positive")
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("HIP PyTorch and a visible HCU are required")

    root = Path(__file__).resolve().parent
    for variable, directory in (
        ("NVALCHEMI_HIP_BATCH_CELL_KEY_COUNT_BUILD_DIR", "point39-cell-key-count"),
        ("NVALCHEMI_HIP_CELL_SCAN_BUILD_DIR", "point39-cell-scan"),
        ("NVALCHEMI_HIP_CELL_FILL_BUILD_DIR", "point39-cell-fill"),
        ("NVALCHEMI_HIP_BATCH_CELL_QUERY_BUILD_DIR", "point39-cell-query"),
        ("NVALCHEMI_HIP_BATCH_QUERY_MATERIALIZE_BUILD_DIR", "point39-topology"),
    ):
        os.environ.setdefault(variable, str(root.parent / "artifacts" / directory))

    device = torch.device("cuda")
    dtype = torch.float32
    data = _make_data(args.atoms_per_system, args.batch_systems, device, dtype)
    batch = Batch.from_data_list(data)

    cold_ms = _wall_ms(lambda: _compute(batch, args.backend))

    correctness: dict[str, object] = {"status": "not_checked"}
    if args.backend == "hip":
        reference = Batch.from_data_list(_make_data(
            args.atoms_per_system, args.batch_systems, device, dtype
        ))
        _compute(reference, "torch_reference")
        torch.cuda.synchronize()
        _assert_outputs(batch, reference, "HIP public API versus Torch reference")
        correctness = {
            "status": "passed",
            "oracle": "framework compute_neighbors backend=torch_reference, method=cell_list",
        }

    for _ in range(args.warmup):
        _compute(batch, args.backend)
    torch.cuda.synchronize()

    samples = [_wall_ms(lambda: _compute(batch, args.backend)) for _ in range(args.samples)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = (
        args.output_dir
        / f"{args.backend}-{args.atoms_per_system}x{args.batch_systems}-warm-ms.txt"
    ).resolve()
    sample_path.write_text("".join(f"{sample:.9g}\n" for sample in samples), encoding="utf-8")

    hook = _hook_boundary(data, device, args.backend)
    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "status": "passed",
                "backend": args.backend,
                "device": torch.cuda.get_device_name(),
                "torch_hip": torch.version.hip,
                "dtype": str(dtype),
                "atoms_per_system": args.atoms_per_system,
                "batch_systems": args.batch_systems,
                "atoms": int(batch.num_nodes),
                "capacity": 256,
                "cutoff": 0.6,
                "measurement": {
                    "scope": "public compute_neighbors API wall-clock including framework selection, executor allocation, cell-list build/query/topology, and Batch write-back",
                    "cold": "one first call in a fresh process; includes runtime/extension load and first-call allocation; source compilation is excluded when a prebuilt cache is supplied",
                    "warmup": args.warmup,
                    "samples": args.samples,
                    "synchronization": "torch.cuda.synchronize before and after each host sample",
                },
                "cold_ms": cold_ms,
                "warm": {**_summary(samples), "samples_path": str(sample_path)},
                "correctness": correctness,
                "hook_boundary": hook,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
