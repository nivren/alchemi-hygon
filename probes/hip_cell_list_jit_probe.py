#!/usr/bin/env python3
"""Build and run the first isolated HIP cell-list module.

This is a feasibility probe for the stage-two build/binning boundary.  It is
not imported by the product package and is not a production backend.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from statistics import mean, median

import torch
from torch.utils.cpp_extension import load


def _reference(
    positions: torch.Tensor,
    inverse_cell: torch.Tensor,
    dimensions: torch.Tensor,
    pbc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fractional = positions @ inverse_cell
    shifts = torch.where(pbc, torch.floor(fractional), torch.zeros_like(fractional))
    wrapped = fractional - shifts
    mapping = torch.floor(wrapped * dimensions.to(positions.dtype)).to(torch.int32)
    mapping = torch.minimum(
        torch.maximum(mapping, torch.zeros_like(mapping)), dimensions - 1
    )
    keys = mapping[:, 0] + dimensions[0] * (
        mapping[:, 1] + dimensions[1] * mapping[:, 2]
    )
    return shifts.to(torch.int32), mapping, keys


def _device_samples_ms(
    function: object,
    *,
    warmup: int,
    repeats_per_sample: int,
    samples: int,
) -> list[float]:
    for _ in range(warmup):
        function()  # type: ignore[operator]
    torch.cuda.synchronize()
    timings = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats_per_sample):
            function()  # type: ignore[operator]
        stop.record()
        stop.synchronize()
        timings.append(float(start.elapsed_time(stop)) / repeats_per_sample)
    return timings


def main() -> None:
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("HIP PyTorch and a visible HCU are required")
    root = Path(__file__).resolve().parent
    source_root = root / "hip_cell_list_jit"
    build_root = Path(
        os.environ.get(
            "NVALCHEMI_HIP_JIT_BUILD_DIR",
            root.parent / "artifacts" / "hip-cell-list-jit",
        )
    ).resolve()
    build_root.mkdir(parents=True, exist_ok=True)
    compile_root = build_root / "sources"
    compile_root.mkdir(parents=True, exist_ok=True)
    for filename in ("cell_keys.cpp", "cell_keys.cu"):
        shutil.copy2(source_root / filename, compile_root / filename)
    extension = load(
        name="nvalchemi_cell_keys_hip_probe",
        sources=[str(compile_root / "cell_keys.cpp"), str(compile_root / "cell_keys.cu")],
        build_directory=str(build_root),
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O2"],
        verbose=False,
    )

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(20260919)
    cell = torch.tensor(
        [[4.0, 0.0, 0.0], [0.7, 3.5, 0.0], [0.4, 0.6, 4.2]],
        dtype=torch.float64,
        device=device,
    )
    dimensions = torch.tensor([7, 6, 8], dtype=torch.int32, device=device)
    pbc = torch.tensor([True, True, False], dtype=torch.bool, device=device)
    results: dict[str, bool] = {}
    for dtype in (torch.float32, torch.float64):
        fractional = torch.rand((257, 3), generator=generator, device=device, dtype=dtype)
        fractional = fractional * 2.5 - 0.75
        positions = fractional @ cell.to(dtype)
        inverse = torch.linalg.inv(cell.to(dtype))
        actual = extension.cell_keys(positions, inverse, dimensions, pbc)
        expected = _reference(positions, inverse, dimensions, pbc)
        results[str(dtype)] = all(torch.equal(a, e) for a, e in zip(actual, expected))
    timings: dict[str, dict[str, float]] = {}
    for atoms, warmup, repeats in ((368, 20, 50), (32768, 10, 20)):
        positions = (
            torch.rand((atoms, 3), generator=generator, device=device, dtype=torch.float32)
            * 2.5
            - 0.75
        ) @ cell.to(torch.float32)
        inverse = torch.linalg.inv(cell.to(torch.float32))
        hip_samples = _device_samples_ms(
            lambda: extension.cell_keys(positions, inverse, dimensions, pbc),
            warmup=warmup,
            repeats_per_sample=repeats,
            samples=10,
        )
        torch_samples = _device_samples_ms(
            lambda: _reference(positions, inverse, dimensions, pbc),
            warmup=warmup,
            repeats_per_sample=repeats,
            samples=10,
        )
        timings[str(atoms)] = {
            "hip_samples_ms": hip_samples,
            "torch_samples_ms": torch_samples,
            "hip_median_ms": median(hip_samples),
            "torch_median_ms": median(torch_samples),
            "hip_mean_ms": mean(hip_samples),
            "torch_mean_ms": mean(torch_samples),
            "torch_over_hip_median": median(torch_samples) / median(hip_samples),
        }
    torch.cuda.synchronize()
    if not all(results.values()):
        raise AssertionError(f"HIP cell-key output mismatch: {results}")
    print(
        json.dumps(
            {
                "device": torch.cuda.get_device_name(),
                "torch_hip": torch.version.hip,
                "atoms": 257,
                "mixed_pbc": [True, True, False],
                "dtype_parity": results,
                "device_loop_timings": timings,
                "build_directory": str(build_root),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
