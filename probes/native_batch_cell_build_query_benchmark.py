#!/usr/bin/env python3
"""Benchmark the isolated HIP build plus existing Torch Batch query.

The scope is pre-warmed GPU work measured with Torch HIP events.  It compares
the native HIP build composition, the Torch CSR build composition, the shared
Torch query, and their complete build+query compositions.  JIT, allocation,
Python wall time, and runtime dispatch are outside the measured interval.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Callable

import torch


Call = Callable[[], None]


def _sample_ms(call: Call) -> float:
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    call()
    stop.record()
    stop.synchronize()
    value = float(start.elapsed_time(stop))
    if value <= 0.0:
        raise RuntimeError(f"HIP event returned a non-positive sample: {value}")
    return value


def _summary(samples: list[float]) -> dict[str, float]:
    med = median(samples)
    deviation = pstdev(samples)
    return {
        "max_ms": max(samples),
        "mean_ms": mean(samples),
        "median_ms": med,
        "min_ms": min(samples),
        "relative_pstdev_percent": deviation / med * 100.0,
    }


def _write_samples(path: Path, samples: list[float]) -> None:
    path.write_text("".join(f"{sample:.9g}\n" for sample in samples), encoding="utf-8")


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
        torch.empty((atoms, 3), dtype=torch.int32, device=device),
        torch.empty((atoms, 3), dtype=torch.int32, device=device),
        torch.empty(atoms, dtype=torch.int32, device=device),
        torch.empty(cells, dtype=torch.int32, device=device),
        torch.empty(cells, dtype=torch.int32, device=device),
        torch.empty(atoms, dtype=torch.int32, device=device),
        torch.empty(cells, dtype=torch.int32, device=device),
    )


def _query_outputs(atoms: int, capacity: int, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        torch.full((atoms, capacity), atoms, dtype=torch.int32, device=device),
        torch.zeros((atoms, capacity, 3), dtype=torch.int32, device=device),
        torch.zeros(atoms, dtype=torch.int32, device=device),
    )


def _assert_build_parity(
    actual: tuple[torch.Tensor, ...],
    expected: tuple[torch.Tensor, ...],
    *,
    label: str,
) -> None:
    for name, lhs, rhs in zip(
        ("shifts", "mapping", "keys", "counts", "starts"),
        actual[:5],
        expected[:5],
        strict=True,
    ):
        if not torch.equal(lhs, rhs):
            raise AssertionError(f"{label}: {name} differs")
    if not torch.equal(actual[6], actual[3]):
        raise AssertionError(f"{label}: native cursor differs from final counts")
    # Validate cell membership without requiring the atomic list's order to
    # match the stable Torch argsort oracle.
    expected_cell_keys = torch.repeat_interleave(
        torch.arange(actual[3].numel(), dtype=torch.int32, device=actual[3].device),
        actual[3].to(torch.long),
    )
    listed_keys = actual[2][actual[5].to(torch.long)]
    if not torch.equal(listed_keys, expected_cell_keys):
        raise AssertionError(f"{label}: native atom list has wrong cell membership")


def _assert_query_parity(
    actual: tuple[torch.Tensor, ...],
    expected: tuple[torch.Tensor, ...],
    *,
    label: str,
) -> None:
    for name, lhs, rhs in zip(
        ("neighbor_matrix", "pair_shifts", "num_neighbors"), actual, expected, strict=True
    ):
        if not torch.equal(lhs, rhs):
            raise AssertionError(f"{label}: {name} differs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atoms-per-system", type=int, default=4096)
    parser.add_argument("--batch-systems", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--capacity", type=int, default=256)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/native-batch-cell-build-query-benchmark"),
    )
    args = parser.parse_args()
    if min(
        args.atoms_per_system,
        args.batch_systems,
        args.warmup,
        args.samples,
        args.capacity,
    ) <= 0:
        raise ValueError("all size, batch, warmup, samples, and capacity arguments must be positive")
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("HIP PyTorch and a visible HCU are required")

    root = Path(__file__).resolve().parent
    for variable, directory in (
        ("NVALCHEMI_HIP_BATCH_CELL_KEY_COUNT_BUILD_DIR", "native-hip-batch-cell-key-count"),
        ("NVALCHEMI_HIP_CELL_SCAN_BUILD_DIR", "native-hip-cell-start-scan"),
        ("NVALCHEMI_HIP_CELL_FILL_BUILD_DIR", "native-hip-cell-fill"),
    ):
        os.environ.setdefault(variable, str(root.parent / "artifacts" / directory))

    from nvalchemiops._cell_list_abi import (  # noqa: PLC0415
        build_batch_cell_key_counts_reference_into,
        build_cell_atom_list_reference_into,
        build_cell_starts_reference_into,
    )
    from nvalchemiops._hip_batch_cell_build import (  # noqa: PLC0415
        build_batch_cell_csr_hip_into,
    )
    from nvalchemiops.torch_reference_cell_list import batch_query_cell_list  # noqa: PLC0415

    device = torch.device("cuda")
    dtype = torch.float32
    dimension_templates = torch.tensor(
        [[32, 32, 32], [24, 24, 24]], dtype=torch.int32, device=device
    )
    cell_templates = torch.tensor(
        [
            [[16.0, 0.0, 0.0], [0.5, 16.0, 0.0], [0.3, 0.4, 16.0]],
            [[12.0, 0.0, 0.0], [0.4, 12.0, 0.0], [0.2, 0.3, 12.0]],
        ],
        dtype=dtype,
        device=device,
    )
    pbc_templates = torch.tensor(
        [[True, False, True], [False, True, False]], device=device
    )
    template_indices = torch.arange(args.batch_systems, device=device) % 2
    dimensions = dimension_templates[template_indices]
    cells = cell_templates[template_indices]
    pbc = pbc_templates[template_indices]
    inverse_cells = torch.linalg.inv(cells)
    radius = torch.full(
        (args.batch_systems, 3), 2, dtype=torch.int32, device=device
    )
    cells_per_system = dimensions.to(torch.int64).prod(dim=1)
    offsets = (cells_per_system.cumsum(0) - cells_per_system).to(torch.int32)
    total_cells = int(cells_per_system.sum().item())
    atoms = args.batch_systems * args.atoms_per_system
    batch_idx = torch.arange(
        args.batch_systems, dtype=torch.int32, device=device
    ).repeat_interleave(
        args.atoms_per_system
    )
    generator = torch.Generator(device=device).manual_seed(20260919)
    uniform = torch.rand((atoms, 3), dtype=dtype, device=device, generator=generator)
    clustered = 0.375 + 0.25 * torch.rand(
        (atoms, 3), dtype=dtype, device=device, generator=generator
    )
    workloads = {"uniform": uniform, "clustered": clustered}
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {}

    for workload_name, fractional in workloads.items():
        positions = torch.bmm(
            fractional.unsqueeze(1), cells[batch_idx.to(torch.long)]
        ).squeeze(1)
        native_build = _build_outputs(atoms, total_cells, device)
        torch_build = _build_outputs(atoms, total_cells, device)
        native_workspace = _workspace(native_build[3], native_build[4])

        def build_native() -> None:
            build_batch_cell_csr_hip_into(
                positions,
                inverse_cells,
                dimensions,
                pbc,
                batch_idx,
                offsets,
                *native_build,
                native_workspace,
            )

        def build_torch() -> None:
            build_batch_cell_key_counts_reference_into(
                positions,
                inverse_cells,
                dimensions,
                pbc,
                batch_idx,
                offsets,
                *torch_build[:4],
            )
            build_cell_starts_reference_into(torch_build[3], torch_build[4])
            build_cell_atom_list_reference_into(
                torch_build[2],
                torch_build[3],
                torch_build[4],
                torch_build[5],
                torch_build[6],
            )

        native_query = _query_outputs(atoms, args.capacity, dtype, device)
        torch_query = _query_outputs(atoms, args.capacity, dtype, device)

        def query_native() -> None:
            batch_query_cell_list(
                positions,
                cells,
                pbc,
                0.6,
                batch_idx,
                dimensions,
                radius,
                native_build[0],
                native_build[1],
                native_build[3],
                native_build[4],
                native_build[5],
                *native_query,
            )

        def query_torch() -> None:
            batch_query_cell_list(
                positions,
                cells,
                pbc,
                0.6,
                batch_idx,
                dimensions,
                radius,
                torch_build[0],
                torch_build[1],
                torch_build[3],
                torch_build[4],
                torch_build[5],
                *torch_query,
            )

        def full_native() -> None:
            build_native()
            query_native()

        def full_torch() -> None:
            build_torch()
            query_torch()

        build_native()
        build_torch()
        torch.cuda.synchronize()
        _assert_build_parity(native_build, torch_build, label=f"{workload_name} build")
        query_native()
        query_torch()
        torch.cuda.synchronize()
        _assert_query_parity(native_query, torch_query, label=f"{workload_name} query")

        for _ in range(args.warmup):
            full_native()
            full_torch()
        torch.cuda.synchronize()
        samples: dict[str, list[float]] = {
            "build_native": [],
            "build_torch": [],
            "query_native": [],
            "query_torch": [],
            "full_native": [],
            "full_torch": [],
        }
        calls = (
            ("build_native", build_native),
            ("build_torch", build_torch),
            ("query_native", query_native),
            ("query_torch", query_torch),
            ("full_native", full_native),
            ("full_torch", full_torch),
        )
        for sample_index in range(args.samples):
            for name, call in calls if sample_index % 2 == 0 else reversed(calls):
                samples[name].append(_sample_ms(call))
        torch.cuda.synchronize()
        _assert_build_parity(native_build, torch_build, label=f"{workload_name} after timing")
        _assert_query_parity(native_query, torch_query, label=f"{workload_name} after timing")

        summaries: dict[str, object] = {}
        for name, values in samples.items():
            path = output_dir / f"{workload_name}-{name}-ms.txt"
            _write_samples(path, values)
            summaries[name] = {
                **_summary(values),
                "samples_path": str(path),
            }
        summaries["full_native_over_torch_factor"] = (
            summaries["full_torch"]["median_ms"] / summaries["full_native"]["median_ms"]  # type: ignore[index]
        )
        results[workload_name] = summaries

    print(
        json.dumps(
            {
                "atoms": atoms,
                "atoms_per_system": args.atoms_per_system,
                "batch_systems": args.batch_systems,
                "capacity": args.capacity,
                "cells": total_cells,
                "device": torch.cuda.get_device_name(),
                "dtype": str(dtype),
                "measurement": {
                    "scope": "pre-warmed HIP event device work; fixed grid metadata; JIT/allocation/host time excluded",
                    "samples": args.samples,
                    "warmup": args.warmup,
                },
                "pbc": pbc.tolist(),
                "results": results,
                "torch_hip": torch.version.hip,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
