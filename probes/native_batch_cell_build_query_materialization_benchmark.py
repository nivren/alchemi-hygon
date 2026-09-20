#!/usr/bin/env python3
"""Benchmark the complete isolated Batch cell-list neighbor pipeline.

Each timed call includes cell-list build, native candidate enumeration,
topology materialization, and distance/vector materialization.  The three
paths are:

* ``full_torch``: Torch cell-list build and Torch query/geometry reference;
* ``full_native_torch_geometry``: public native HIP build/query/topology and
  Torch geometry, which is the checked hybrid path;
* ``full_trusted_torch_geometry``: prevalidated native HIP build plan/query/
  topology and Torch geometry, which is the lifecycle candidate;
* ``full_native_native_geometry``: the same native path with the forward-only
  native HIP geometry candidate.

Grid metadata, output buffers, native workspaces, and JIT compilation are
outside the measured interval.  This is an isolated performance probe; it
does not change runtime dispatch or the default framework path.
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


def _query_outputs(
    atoms: int,
    capacity: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    return (
        torch.full((atoms, capacity), atoms, dtype=torch.int32, device=device),
        torch.zeros((atoms, capacity, 3), dtype=torch.int32, device=device),
        torch.zeros(atoms, dtype=torch.int32, device=device),
        torch.zeros((atoms, capacity), dtype=dtype, device=device),
        torch.zeros((atoms, capacity, 3), dtype=dtype, device=device),
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
    # Native atomic fill has unspecified order.  Check cell membership instead
    # of requiring its atom list to match the stable Torch argsort order.
    expected_cell_keys = torch.repeat_interleave(
        torch.arange(actual[3].numel(), dtype=torch.int32, device=actual[3].device),
        actual[3].to(torch.long),
    )
    listed_keys = actual[2][actual[5].to(torch.long)]
    if not torch.equal(listed_keys, expected_cell_keys):
        raise AssertionError(f"{label}: native atom list has wrong cell membership")


def _assert_geometry_close(
    actual: tuple[torch.Tensor, ...],
    expected: tuple[torch.Tensor, ...],
    label: str,
) -> None:
    names = ("neighbor_matrix", "pair_shifts", "num_neighbors", "distances", "vectors")
    for name, lhs, rhs in zip(names, actual, expected, strict=True):
        if lhs.is_floating_point():
            tolerance = 1e-6 if lhs.dtype == torch.float32 else 1e-12
            torch.testing.assert_close(
                lhs, rhs, atol=tolerance, rtol=tolerance, msg=f"{label}: {name}"
            )
        elif not torch.equal(lhs, rhs):
            raise AssertionError(f"{label}: discrete output {name} differs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atoms-per-system", type=int, default=46)
    parser.add_argument("--batch-systems", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--capacity", type=int, default=256)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/native-batch-cell-build-query-materialization-benchmark"),
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
        ("NVALCHEMI_HIP_BATCH_CELL_QUERY_BUILD_DIR", "native-hip-batch-cell-query"),
        (
            "NVALCHEMI_HIP_BATCH_QUERY_MATERIALIZE_BUILD_DIR",
            "native-hip-batch-query-materialize",
        ),
        ("NVALCHEMI_HIP_BATCH_QUERY_GEOMETRY_BUILD_DIR", "native-hip-batch-query-geometry"),
    ):
        os.environ.setdefault(variable, str(root.parent / "artifacts" / directory))

    from nvalchemiops._cell_list_abi import (  # noqa: PLC0415
        build_batch_cell_key_counts_reference_into,
        build_cell_atom_list_reference_into,
        build_cell_starts_reference_into,
    )
    from nvalchemiops._hip_batch_cell_build import (  # noqa: PLC0415
        _TrustedBatchCellBuildPlan,
        build_batch_cell_csr_hip_into,
    )
    from nvalchemiops._hip_batch_cell_query import (  # noqa: PLC0415
        batch_query_cell_list_hip_into,
    )
    from nvalchemiops._hip_batch_query_geometry import (  # noqa: PLC0415
        _load_jit_extension as load_hip_geometry_extension,
    )
    from nvalchemiops._hip_batch_query_materialize import (  # noqa: PLC0415
        _composite_key_layout,
        _load_jit_extension as load_hip_topology_extension,
        allocate_batch_query_topology_composite_workspace,
    )
    from nvalchemiops._torch_batch_query_materialization import (  # noqa: PLC0415
        materialize_batch_query_topology_geometry_into,
    )
    from nvalchemiops.torch_reference_cell_list import batch_query_cell_list  # noqa: PLC0415

    device = torch.device("cuda")
    dtype = torch.float32
    cutoff = 0.6
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
    radius = torch.full((args.batch_systems, 3), 2, dtype=torch.int32, device=device)
    cells_per_system = dimensions.to(torch.int64).prod(dim=1)
    offsets = (cells_per_system.cumsum(0) - cells_per_system).to(torch.int32)
    total_cells = int(cells_per_system.sum().item())
    atoms = args.batch_systems * args.atoms_per_system
    batch_idx = torch.arange(
        args.batch_systems, dtype=torch.int32, device=device
    ).repeat_interleave(args.atoms_per_system)
    generator = torch.Generator(device=device).manual_seed(20260919)
    uniform = torch.rand((atoms, 3), dtype=dtype, device=device, generator=generator)
    clustered = 0.375 + 0.25 * torch.rand(
        (atoms, 3), dtype=dtype, device=device, generator=generator
    )
    workloads = {"uniform": uniform, "clustered": clustered}
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {}
    topology_extension = load_hip_topology_extension()
    geometry_extension = load_hip_geometry_extension()
    composite_row_bits, composite_total_bits = _composite_key_layout(atoms, 8, 128)

    for workload_name, fractional in workloads.items():
        positions = torch.bmm(
            fractional.unsqueeze(1), cells[batch_idx.to(torch.long)]
        ).squeeze(1)
        native_build = _build_outputs(atoms, total_cells, device)
        torch_build = _build_outputs(atoms, total_cells, device)
        native_workspace = _workspace(native_build[3], native_build[4])
        torch_public = _query_outputs(atoms, args.capacity, device, dtype)
        native_torch_geometry_public = _query_outputs(atoms, args.capacity, device, dtype)
        trusted_torch_geometry_public = _query_outputs(atoms, args.capacity, device, dtype)
        native_native_geometry_public = _query_outputs(atoms, args.capacity, device, dtype)
        native_candidate = _query_outputs(atoms, args.capacity, device, dtype)[:3]

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

        # Initialize once through the checked public path.  Subsequent full
        # trusted calls reuse this exact plan and its caller-owned buffers.
        trusted_plan = _TrustedBatchCellBuildPlan.initialize(
            positions,
            inverse_cells,
            dimensions,
            pbc,
            batch_idx,
            offsets,
            *native_build,
            native_workspace,
        )

        def build_trusted() -> None:
            trusted_plan.run()

        def enumerate_native() -> None:
            batch_query_cell_list_hip_into(
                positions,
                cells,
                pbc,
                cutoff,
                batch_idx,
                dimensions,
                radius,
                offsets,
                native_build[0],
                native_build[1],
                native_build[3],
                native_build[4],
                native_build[5],
                *native_candidate,
            )

        # The checked trusted-plan initialization above performed the first
        # native build before scratch allocation and timed calls.
        build_torch()
        torch.cuda.synchronize()
        _assert_build_parity(native_build, torch_build, f"{workload_name} build")
        enumerate_native()
        torch.cuda.synchronize()
        topology_workspace = allocate_batch_query_topology_composite_workspace(
            native_candidate[0], native_candidate[2]
        )

        def materialize_native_topology(public: tuple[torch.Tensor, ...]) -> None:
            topology_extension.batch_query_materialize_topology_composite_workspace_into(
                native_candidate[0],
                native_candidate[1],
                native_candidate[2],
                public[0],
                public[1],
                public[2],
                topology_workspace.keys_a,
                topology_workspace.keys_b,
                topology_workspace.order_a,
                topology_workspace.order_b,
                topology_workspace.row_starts,
                topology_workspace.sort_workspace,
                topology_workspace.scan_workspace,
                atoms,
                composite_row_bits,
                composite_total_bits,
                8,
                128,
            )

        def materialize_torch_geometry(public: tuple[torch.Tensor, ...]) -> None:
            materialize_batch_query_topology_geometry_into(
                positions,
                cells,
                batch_idx,
                public[0],
                public[1],
                public[2],
                distances=public[3],
                vectors=public[4],
            )

        def materialize_native_geometry(public: tuple[torch.Tensor, ...]) -> None:
            geometry_extension.batch_query_geometry_into(
                positions,
                cells,
                batch_idx,
                public[0],
                public[1],
                public[2],
                public[3],
                public[4],
            )

        def query_torch() -> None:
            batch_query_cell_list(
                positions,
                cells,
                pbc,
                cutoff,
                batch_idx,
                dimensions,
                radius,
                torch_build[0],
                torch_build[1],
                torch_build[3],
                torch_build[4],
                torch_build[5],
                *torch_public[:3],
                return_distances=True,
                return_vectors=True,
                neighbor_distances=torch_public[3],
                neighbor_vectors=torch_public[4],
            )

        def full_torch() -> None:
            build_torch()
            query_torch()

        def full_native_torch_geometry() -> None:
            build_native()
            enumerate_native()
            materialize_native_topology(native_torch_geometry_public)
            materialize_torch_geometry(native_torch_geometry_public)

        def full_trusted_torch_geometry() -> None:
            build_trusted()
            enumerate_native()
            materialize_native_topology(trusted_torch_geometry_public)
            materialize_torch_geometry(trusted_torch_geometry_public)

        def full_native_native_geometry() -> None:
            build_native()
            enumerate_native()
            materialize_native_topology(native_native_geometry_public)
            materialize_native_geometry(native_native_geometry_public)

        # Correctness is checked before collecting timing samples and again
        # after timing to catch stale-buffer or repeated-call errors.
        query_torch()
        full_native_torch_geometry()
        full_trusted_torch_geometry()
        full_native_native_geometry()
        torch.cuda.synchronize()
        _assert_geometry_close(
            native_torch_geometry_public,
            torch_public,
            f"{workload_name} native build/query Torch geometry",
        )
        _assert_geometry_close(
            native_torch_geometry_public,
            torch_public,
            f"{workload_name} native build/query Torch geometry",
        )
        _assert_geometry_close(
            trusted_torch_geometry_public,
            torch_public,
            f"{workload_name} trusted plan/query Torch geometry",
        )
        _assert_geometry_close(
            native_native_geometry_public,
            torch_public,
            f"{workload_name} native build/query native geometry",
        )

        for _ in range(args.warmup):
            full_torch()
            full_native_torch_geometry()
            full_trusted_torch_geometry()
            full_native_native_geometry()
        torch.cuda.synchronize()

        event_samples: dict[str, list[float]] = {
            "build_native": [],
            "build_torch": [],
            "full_torch": [],
            "full_native_torch_geometry": [],
            "full_trusted_torch_geometry": [],
            "full_native_native_geometry": [],
        }
        calls = (
            ("build_native", build_native),
            ("build_torch", build_torch),
            ("full_torch", full_torch),
            ("full_native_torch_geometry", full_native_torch_geometry),
            ("full_trusted_torch_geometry", full_trusted_torch_geometry),
            ("full_native_native_geometry", full_native_native_geometry),
        )
        for sample_index in range(args.samples):
            ordered = calls if sample_index % 2 == 0 else reversed(calls)
            for name, call in ordered:
                event_samples[name].append(_sample_ms(call))
        torch.cuda.synchronize()

        api_calls = (
            ("full_torch_api_wall", full_torch),
            ("full_native_torch_geometry_api_wall", full_native_torch_geometry),
            ("full_trusted_torch_geometry_api_wall", full_trusted_torch_geometry),
            ("full_native_native_geometry_api_wall", full_native_native_geometry),
        )
        api_samples: dict[str, list[float]] = {name: [] for name, _ in api_calls}
        for sample_index in range(args.samples):
            ordered = api_calls if sample_index % 2 == 0 else reversed(api_calls)
            for name, call in ordered:
                api_samples[name].append(_sample_api_wall_ms(call))

        _assert_build_parity(native_build, torch_build, f"{workload_name} build after timing")
        _assert_geometry_close(
            native_torch_geometry_public,
            torch_public,
            f"{workload_name} native Torch geometry after timing",
        )
        _assert_geometry_close(
            trusted_torch_geometry_public,
            torch_public,
            f"{workload_name} trusted plan Torch geometry after timing",
        )
        _assert_geometry_close(
            native_native_geometry_public,
            torch_public,
            f"{workload_name} native geometry after timing",
        )

        summaries: dict[str, object] = {}
        for name, values in {**event_samples, **api_samples}.items():
            path = output_dir / f"{workload_name}-{name}-ms.txt"
            _write_samples(path, values)
            summaries[name] = {**_summary(values), "samples_path": str(path)}
        summaries["full_native_torch_geometry_over_torch_factor"] = (
            summaries["full_native_torch_geometry"]["median_ms"]  # type: ignore[index]
            / summaries["full_torch"]["median_ms"]  # type: ignore[index]
        )
        summaries["full_trusted_torch_geometry_over_torch_factor"] = (
            summaries["full_trusted_torch_geometry"]["median_ms"]  # type: ignore[index]
            / summaries["full_torch"]["median_ms"]  # type: ignore[index]
        )
        summaries["full_native_native_geometry_over_torch_factor"] = (
            summaries["full_native_native_geometry"]["median_ms"]  # type: ignore[index]
            / summaries["full_torch"]["median_ms"]  # type: ignore[index]
        )
        summaries["full_native_torch_geometry_api_over_torch_api_factor"] = (
            summaries["full_native_torch_geometry_api_wall"]["median_ms"]  # type: ignore[index]
            / summaries["full_torch_api_wall"]["median_ms"]  # type: ignore[index]
        )
        summaries["full_trusted_torch_geometry_api_over_torch_api_factor"] = (
            summaries["full_trusted_torch_geometry_api_wall"]["median_ms"]  # type: ignore[index]
            / summaries["full_torch_api_wall"]["median_ms"]  # type: ignore[index]
        )
        summaries["full_native_native_geometry_api_over_torch_api_factor"] = (
            summaries["full_native_native_geometry_api_wall"]["median_ms"]  # type: ignore[index]
            / summaries["full_torch_api_wall"]["median_ms"]  # type: ignore[index]
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
                "capacity": args.capacity,
                "cells": total_cells,
                "cutoff": cutoff,
                "device": torch.cuda.get_device_name(),
                "dtype": str(dtype),
                "measurement": {
                    "scope": "pre-warmed full build/query/topology/geometry; fixed metadata, buffers, workspaces and JIT excluded",
                    "event_samples": args.samples,
                    "warmup": args.warmup,
                    "api_wall_clock": "explicit torch.cuda.synchronize before and after each full call",
                    "native_geometry": "forward-only candidate; Torch geometry remains training/gradient path",
                    "trusted_build": "one checked public plan initialization; subsequent full calls use plan.run()",
                    "composite_key": {
                        "shift_bits": 8,
                        "shift_bias": 128,
                        "row_bits": composite_row_bits,
                        "total_bits": composite_total_bits,
                    },
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
