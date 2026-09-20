#!/usr/bin/env python3
"""Compare native Batch fusion with the explicit native key/count composition.

The comparison measures pre-warmed HIP-event device time for the same flattened
two-system Batch output contract: shifts, mappings, global keys, and global
cell counts.  It excludes JIT, Python-side validation, scan/fill/query, and
end-to-end neighbor work.
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


def _sample_ms(call: Call, repeats: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        call()
    stop.record()
    stop.synchronize()
    value = float(start.elapsed_time(stop)) / repeats
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
    path.write_text("".join(f"{value:.9g}\n" for value in samples), encoding="utf-8")


def _outputs(atoms: int, cells: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        torch.empty((atoms, 3), dtype=torch.int32, device=device),
        torch.empty((atoms, 3), dtype=torch.int32, device=device),
        torch.empty(atoms, dtype=torch.int32, device=device),
        torch.empty(cells, dtype=torch.int32, device=device),
    )


def _assert_equal(
    actual: tuple[torch.Tensor, ...], expected: tuple[torch.Tensor, ...], label: str
) -> None:
    for name, lhs, rhs in zip(
        ("shifts", "mapping", "keys", "counts"), actual, expected, strict=True
    ):
        if not torch.equal(lhs, rhs):
            raise AssertionError(f"{label}: {name} differs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atoms-per-system", type=int, default=16384)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/batch-cell-key-count-fusion-benchmark"),
    )
    args = parser.parse_args()
    if min(args.atoms_per_system, args.warmup, args.repeats, args.samples) <= 0:
        raise ValueError("atoms-per-system, warmup, repeats, and samples must be positive")
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("HIP PyTorch and a visible HCU are required")

    root = Path(__file__).resolve().parent
    os.environ.setdefault(
        "NVALCHEMI_HIP_CELL_KEY_BUILD_DIR",
        str(root.parent / "artifacts" / "native-hip-cell-key"),
    )
    os.environ.setdefault(
        "NVALCHEMI_HIP_CELL_COUNT_BUILD_DIR",
        str(root.parent / "artifacts" / "native-hip-cell-key-count"),
    )
    os.environ.setdefault(
        "NVALCHEMI_HIP_BATCH_CELL_KEY_COUNT_BUILD_DIR",
        str(root.parent / "artifacts" / "native-hip-batch-cell-key-count"),
    )
    from nvalchemiops._hip_batch_cell_key_count import (  # noqa: PLC0415
        build_batch_cell_key_counts_hip_into,
    )
    from nvalchemiops._hip_cell_count import build_cell_counts_hip_into  # noqa: PLC0415
    from nvalchemiops._hip_cell_key import build_cell_keys_hip_into  # noqa: PLC0415

    device = torch.device("cuda")
    dtype = torch.float32
    dimensions = torch.tensor([[16, 16, 8], [8, 16, 16]], dtype=torch.int32, device=device)
    cells_per_system = dimensions.to(torch.int64).prod(dim=1)
    offsets = (cells_per_system.cumsum(0) - cells_per_system).to(torch.int32)
    total_cells = int(cells_per_system.sum().item())
    systems = dimensions.shape[0]
    atoms = systems * args.atoms_per_system
    system_slices = tuple(
        (
            system * args.atoms_per_system,
            (system + 1) * args.atoms_per_system,
            int(offsets[system].item()),
            int((offsets[system] + cells_per_system[system]).item()),
        )
        for system in range(systems)
    )
    batch_idx = torch.arange(systems, dtype=torch.int32, device=device).repeat_interleave(
        args.atoms_per_system
    )
    pbc = torch.tensor([[True, False, True], [False, True, False]], device=device)
    cells = torch.tensor(
        [
            [[4.0, 0.0, 0.0], [0.7, 3.5, 0.0], [0.4, 0.6, 4.2]],
            [[3.0, 0.0, 0.0], [0.2, 2.5, 0.0], [0.1, 0.3, 3.1]],
        ],
        dtype=dtype,
        device=device,
    )
    inverse_cells = torch.linalg.inv(cells)
    generator = torch.Generator(device=device).manual_seed(20260919)
    uniform_fractional = torch.rand((atoms, 3), dtype=dtype, device=device, generator=generator)
    concentrated_fractional = torch.full_like(uniform_fractional, 1.0 / 64.0)
    workloads = {
        "uniform": uniform_fractional,
        "single_cell": concentrated_fractional,
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {}

    for name, fractional in workloads.items():
        positions = torch.bmm(
            fractional.unsqueeze(1), cells[batch_idx.to(torch.long)]
        ).squeeze(1)
        fused_outputs = _outputs(atoms, total_cells, device)
        composed_outputs = _outputs(atoms, total_cells, device)

        def fused() -> None:
            build_batch_cell_key_counts_hip_into(
                positions,
                inverse_cells,
                dimensions,
                pbc,
                batch_idx,
                offsets,
                *fused_outputs,
            )

        def composed() -> None:
            for system, (start, end, cell_start, cell_end) in enumerate(system_slices):
                shifts, mapping, keys, counts = composed_outputs
                build_cell_keys_hip_into(
                    positions[start:end],
                    inverse_cells[system],
                    dimensions[system],
                    pbc[system],
                    shifts[start:end],
                    mapping[start:end],
                    keys[start:end],
                )
                build_cell_counts_hip_into(keys[start:end], counts[cell_start:cell_end])
                keys[start:end].add_(cell_start)

        fused()
        composed()
        torch.cuda.synchronize()
        _assert_equal(fused_outputs, composed_outputs, f"{name} before timing")

        for _ in range(args.warmup):
            fused()
            composed()
        torch.cuda.synchronize()
        fused_samples: list[float] = []
        composed_samples: list[float] = []
        for index in range(args.samples):
            calls = (
                ((fused_samples, fused), (composed_samples, composed))
                if index % 2 == 0
                else ((composed_samples, composed), (fused_samples, fused))
            )
            for target, call in calls:
                target.append(_sample_ms(call, args.repeats))
        torch.cuda.synchronize()
        _assert_equal(fused_outputs, composed_outputs, f"{name} after timing")

        fused_path = output_dir / f"fused-{name}-ms.txt"
        composed_path = output_dir / f"composed-{name}-ms.txt"
        _write_samples(fused_path, fused_samples)
        _write_samples(composed_path, composed_samples)
        fused_median = median(fused_samples)
        composed_median = median(composed_samples)
        results[name] = {
            "composed": _summary(composed_samples),
            "composed_samples_path": str(composed_path),
            "correct": True,
            "fused": _summary(fused_samples),
            "fused_samples_path": str(fused_path),
            "fused_lower_percent": (1.0 - fused_median / composed_median) * 100.0,
            "scope": "pre-warmed native build-half GPU event time",
            "composed_over_fused_median": composed_median / fused_median,
        }

    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "atoms": atoms,
                "atoms_per_system": args.atoms_per_system,
                "cell_offsets": offsets.tolist(),
                "cells": total_cells,
                "device": torch.cuda.get_device_name(),
                "dtype": str(dtype),
                "measurement": {
                    "repeats_per_sample": args.repeats,
                    "samples": args.samples,
                    "warmup": args.warmup,
                },
                "results": results,
                "systems": systems,
                "torch_hip": torch.version.hip,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
