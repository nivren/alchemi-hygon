#!/usr/bin/env python3
"""Measure explicit Torch and HIP cell-key count APIs on one HCU stream.

The scope includes the public count API's ABI validation, output overwrite, and
count work.  It excludes JIT compilation, key generation, output comparison,
and host wall-clock time.  Raw HIP-event samples are emitted one value per line
for the DCU performance comparison helper.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Callable

import torch


CountCall = Callable[[], None]


def _device_sample_ms(call: CountCall, repeats: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        call()
    stop.record()
    stop.synchronize()
    elapsed = float(start.elapsed_time(stop)) / repeats
    if elapsed <= 0.0:
        raise RuntimeError(f"HIP event returned a non-positive sample: {elapsed}")
    return elapsed


def _samples_ms(
    torch_call: CountCall,
    hip_call: CountCall,
    *,
    warmup: int,
    repeats: int,
    samples: int,
) -> tuple[list[float], list[float]]:
    for _ in range(warmup):
        torch_call()
        hip_call()
    torch.cuda.synchronize()

    torch_samples: list[float] = []
    hip_samples: list[float] = []
    for sample_index in range(samples):
        # Alternate order so a monotonic thermal or shared-device change is not
        # systematically attributed to one implementation.
        calls = (
            ((torch_samples, torch_call), (hip_samples, hip_call))
            if sample_index % 2 == 0
            else ((hip_samples, hip_call), (torch_samples, torch_call))
        )
        for target, call in calls:
            target.append(_device_sample_ms(call, repeats))
    return torch_samples, hip_samples


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atoms", type=int, default=32768)
    parser.add_argument("--cells", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/cell-key-count-benchmark"),
    )
    args = parser.parse_args()
    if min(args.atoms, args.cells, args.warmup, args.repeats, args.samples) <= 0:
        raise ValueError("atoms, cells, warmup, repeats, and samples must be positive")
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
        build_cell_counts_hip_into,
    )

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260919)
    uniform = torch.randint(
        args.cells,
        (args.atoms,),
        device=device,
        dtype=torch.int32,
        generator=generator,
    )
    workloads = {"uniform": uniform, "single_cell": torch.zeros_like(uniform)}
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {}

    for name, keys in workloads.items():
        torch_counts = torch.empty(args.cells, dtype=torch.int32, device=device)
        hip_counts = torch.empty_like(torch_counts)
        build_cell_counts_reference_into(keys, torch_counts)
        build_cell_counts_hip_into(keys, hip_counts)
        torch.cuda.synchronize()
        if not torch.equal(torch_counts, hip_counts):
            raise AssertionError(f"{name}: count output differs before timing")

        torch_samples, hip_samples = _samples_ms(
            lambda: build_cell_counts_reference_into(keys, torch_counts),
            lambda: build_cell_counts_hip_into(keys, hip_counts),
            warmup=args.warmup,
            repeats=args.repeats,
            samples=args.samples,
        )
        torch.cuda.synchronize()
        if not torch.equal(torch_counts, hip_counts):
            raise AssertionError(f"{name}: count output differs after timing")
        torch_path = output_dir / f"torch-{name}-ms.txt"
        hip_path = output_dir / f"hip-{name}-ms.txt"
        _write_samples(torch_path, torch_samples)
        _write_samples(hip_path, hip_samples)
        torch_median = median(torch_samples)
        hip_median = median(hip_samples)
        results[name] = {
            "correct": True,
            "hip": _summary(hip_samples),
            "hip_samples_path": str(hip_path),
            "scope": "explicit count API device time",
            "torch": _summary(torch_samples),
            "torch_over_hip_median": torch_median / hip_median,
            "torch_samples_path": str(torch_path),
        }

    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "atoms": args.atoms,
                "cells": args.cells,
                "device": torch.cuda.get_device_name(),
                "measurement": {
                    "repeats_per_sample": args.repeats,
                    "samples": args.samples,
                    "warmup": args.warmup,
                },
                "results": results,
                "torch_hip": torch.version.hip,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
