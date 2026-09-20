#!/usr/bin/env python3
"""Break down the isolated native HIP Batch cell-list build candidate.

The timed native stages are fused geometry/PBC/key/count, rocPRIM CSR-start
scan, and atomic CSR fill.  Each stage uses the same preallocated buffers as
the complete build.  A Torch CSR build on the identical inputs is kept as the
correctness oracle and end-to-end build baseline.

JIT/module load, buffer/workspace allocation, and fixed grid metadata are
outside the timed regions.  This probe does not select a runtime backend.
"""

from __future__ import annotations

import argparse
import json
import os
import time
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


def _sample_api_wall_ms(call: Call) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    call()
    torch.cuda.synchronize()
    value = (time.perf_counter() - start) * 1000.0
    if value <= 0.0:
        raise RuntimeError(f"API wall-clock returned a non-positive sample: {value}")
    return value


def _summary(samples: list[float]) -> dict[str, float]:
    med = median(samples)
    return {
        "max_ms": max(samples),
        "mean_ms": mean(samples),
        "median_ms": med,
        "min_ms": min(samples),
        "relative_pstdev_percent": pstdev(samples) / med * 100.0,
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


def _assert_build_parity(
    actual: tuple[torch.Tensor, ...],
    expected: tuple[torch.Tensor, ...],
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
    expected_cell_keys = torch.repeat_interleave(
        torch.arange(actual[3].numel(), dtype=torch.int32, device=actual[3].device),
        actual[3].to(torch.long),
    )
    listed_keys = actual[2][actual[5].to(torch.long)]
    if not torch.equal(listed_keys, expected_cell_keys):
        raise AssertionError(f"{label}: native atom list has wrong cell membership")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atoms-per-system", type=int, default=46)
    parser.add_argument("--batch-systems", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/native-batch-cell-build-breakdown-benchmark"),
    )
    args = parser.parse_args()
    if min(args.atoms_per_system, args.batch_systems, args.warmup, args.samples) <= 0:
        raise ValueError("atoms-per-system, batch-systems, warmup, and samples must be positive")
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
    from nvalchemiops._hip_batch_cell_key_count import (  # noqa: PLC0415
        build_batch_cell_key_counts_hip_into,
    )
    from nvalchemiops._hip_cell_fill import build_cell_atom_list_hip_into  # noqa: PLC0415
    from nvalchemiops._hip_cell_scan import build_cell_starts_hip_into  # noqa: PLC0415

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
    cells_per_system = dimensions.to(torch.int64).prod(dim=1)
    offsets = (cells_per_system.cumsum(0) - cells_per_system).to(torch.int32)
    total_cells = int(cells_per_system.sum().item())
    atoms = args.batch_systems * args.atoms_per_system
    batch_idx = torch.arange(
        args.batch_systems, dtype=torch.int32, device=device
    ).repeat_interleave(args.atoms_per_system)
    generator = torch.Generator(device=device).manual_seed(20260919)
    workloads = {
        "uniform": torch.rand((atoms, 3), dtype=dtype, device=device, generator=generator),
        "clustered": 0.375
        + 0.25 * torch.rand((atoms, 3), dtype=dtype, device=device, generator=generator),
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {}

    for workload_name, fractional in workloads.items():
        positions = torch.bmm(
            fractional.unsqueeze(1), cells[batch_idx.to(torch.long)]
        ).squeeze(1)
        native = _build_outputs(atoms, total_cells, device)
        torch_reference = _build_outputs(atoms, total_cells, device)
        workspace = _workspace(native[3], native[4])

        def key_count() -> None:
            build_batch_cell_key_counts_hip_into(
                positions,
                inverse_cells,
                dimensions,
                pbc,
                batch_idx,
                offsets,
                *native[:4],
            )

        def scan() -> None:
            build_cell_starts_hip_into(native[3], native[4], workspace)

        def fill() -> None:
            build_cell_atom_list_hip_into(
                native[2], native[3], native[4], native[5], native[6]
            )

        def build_native() -> None:
            key_count()
            scan()
            fill()

        def build_torch() -> None:
            build_batch_cell_key_counts_reference_into(
                positions,
                inverse_cells,
                dimensions,
                pbc,
                batch_idx,
                offsets,
                *torch_reference[:4],
            )
            build_cell_starts_reference_into(torch_reference[3], torch_reference[4])
            build_cell_atom_list_reference_into(
                torch_reference[2],
                torch_reference[3],
                torch_reference[4],
                torch_reference[5],
                torch_reference[6],
            )

        build_native()
        build_torch()
        torch.cuda.synchronize()
        _assert_build_parity(native, torch_reference, f"{workload_name} before timing")

        for _ in range(args.warmup):
            build_native()
            build_torch()
        torch.cuda.synchronize()

        event_samples: dict[str, list[float]] = {
            "key_count_native": [],
            "scan_native": [],
            "fill_native": [],
            "build_native": [],
            "build_torch": [],
        }
        calls = (
            ("key_count_native", key_count),
            ("scan_native", scan),
            ("fill_native", fill),
            ("build_native", build_native),
            ("build_torch", build_torch),
        )
        for sample_index in range(args.samples):
            for name, call in calls if sample_index % 2 == 0 else reversed(calls):
                if name == "scan_native":
                    key_count()
                elif name == "fill_native":
                    key_count()
                    scan()
                event_samples[name].append(_sample_ms(call))
        torch.cuda.synchronize()

        api_samples: dict[str, list[float]] = {
            "key_count_native_api_wall": [],
            "scan_native_api_wall": [],
            "fill_native_api_wall": [],
            "build_native_api_wall": [],
            "build_torch_api_wall": [],
        }
        api_calls = (
            ("key_count_native_api_wall", key_count),
            ("scan_native_api_wall", scan),
            ("fill_native_api_wall", fill),
            ("build_native_api_wall", build_native),
            ("build_torch_api_wall", build_torch),
        )
        for sample_index in range(args.samples):
            for name, call in api_calls if sample_index % 2 == 0 else reversed(api_calls):
                if name == "scan_native_api_wall":
                    key_count()
                elif name == "fill_native_api_wall":
                    key_count()
                    scan()
                api_samples[name].append(_sample_api_wall_ms(call))

        build_native()
        build_torch()
        torch.cuda.synchronize()
        _assert_build_parity(native, torch_reference, f"{workload_name} after timing")

        summaries: dict[str, object] = {}
        for name, values in {**event_samples, **api_samples}.items():
            path = output_dir / f"{workload_name}-{name}-ms.txt"
            _write_samples(path, values)
            summaries[name] = {**_summary(values), "samples_path": str(path)}
        native_sum = sum(
            summaries[name]["median_ms"]  # type: ignore[index]
            for name in ("key_count_native", "scan_native", "fill_native")
        )
        summaries["native_stage_sum_ms"] = native_sum
        summaries["native_stage_sum_over_full_factor"] = (
            native_sum / summaries["build_native"]["median_ms"]  # type: ignore[index]
        )
        summaries["build_native_over_torch_factor"] = (
            summaries["build_native"]["median_ms"]  # type: ignore[index]
            / summaries["build_torch"]["median_ms"]  # type: ignore[index]
        )
        results[workload_name] = summaries

    print(
        json.dumps(
            {
                "atoms": atoms,
                "atoms_per_system": args.atoms_per_system,
                "batch_systems": args.batch_systems,
                "cells": total_cells,
                "device": torch.cuda.get_device_name(),
                "dtype": str(dtype),
                "measurement": {
                    "scope": "pre-warmed HIP event and explicit-sync API wall clock; fixed metadata, buffers, workspaces and JIT excluded",
                    "stage_preparation": "key/count runs before isolated scan; key/count plus scan run before isolated fill",
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
