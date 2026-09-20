#!/usr/bin/env python3
"""Benchmark native query enumeration with Torch materialization.

The measured scopes are pre-warmed HIP-event device work:

* ``enumeration_native``: native HIP pair enumeration only;
* ``query_native_torch_materialize``: native enumeration followed by Torch
  canonical sorting/scatter into the public matrix/count/shift surface;
* ``query_torch``: the existing Torch reference query on the same CSR data;
* optional ``--breakdown`` scopes isolate Torch topology canonicalization/scatter,
  geometry materialization, and the complete helper on one fixed candidate.
* optional ``--hip-topology`` adds the isolated native HIP topology candidate
  beside the Torch topology stage.
* optional ``--hip-composite-topology`` adds the isolated one-sort int64
  composite-key candidate beside both topology stages.
* optional ``--hip-composite-workspace`` adds the same composite candidate
  with caller-owned scratch reused across calls.
* optional ``--hip-hybrid`` measures native query + composite topology + Torch
  geometry against the complete Torch query/materialization path.
* optional ``--hip-native-geometry`` measures the same hybrid path with the
  isolated native HIP forward geometry candidate.

JIT, allocation, Python wall time, and runtime selection are excluded from the
event regions.  The native candidate keeps its explicit overflow check; its
host-side error-check synchronization is not represented by HIP-event elapsed
time, so these are device-timeline measurements rather than API wall time.
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


def _query_outputs(
    atoms: int,
    capacity: int,
    device: torch.device,
    *,
    dtype: torch.dtype | None = None,
    geometry: bool = False,
) -> tuple[torch.Tensor, ...]:
    if geometry and dtype is None:
        raise ValueError("dtype is required when geometry outputs are requested")
    outputs: tuple[torch.Tensor, ...] = (
        torch.full((atoms, capacity), atoms, dtype=torch.int32, device=device),
        torch.zeros((atoms, capacity, 3), dtype=torch.int32, device=device),
        torch.zeros(atoms, dtype=torch.int32, device=device),
    )
    if geometry:
        outputs += (
            torch.zeros((atoms, capacity), dtype=dtype, device=device),
            torch.zeros((atoms, capacity, 3), dtype=dtype, device=device),
        )
    return outputs


def _materialize_candidate_with_torch(
    candidate: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    public: tuple[torch.Tensor, ...],
    positions: torch.Tensor,
    cells: torch.Tensor,
    batch_idx: torch.Tensor,
    *,
    geometry: bool,
) -> None:
    from nvalchemiops._torch_batch_query_materialization import (  # noqa: PLC0415
        materialize_batch_query_candidate_into,
    )

    kwargs = {}
    if geometry:
        kwargs = {"distances": public[3], "vectors": public[4]}
    materialize_batch_query_candidate_into(
        positions,
        cells,
        batch_idx,
        candidate[0],
        candidate[1],
        candidate[2],
        public[0],
        public[1],
        public[2],
        **kwargs,
    )


def _assert_equal(
    actual: tuple[torch.Tensor, ...], expected: tuple[torch.Tensor, ...], label: str
) -> None:
    names = ("neighbor_matrix", "pair_shifts", "num_neighbors", "distances", "vectors")[: len(actual)]
    for name, lhs, rhs in zip(names, actual, expected, strict=True):
        if not torch.equal(lhs, rhs):
            mismatch = torch.nonzero(lhs != rhs, as_tuple=False)
            first = tuple(int(value) for value in mismatch[0].tolist())
            row_detail = ""
            if lhs.ndim >= 2:
                row = first[0]
                row_detail = (
                    f"; actual_row={lhs[row].tolist()[:8]} "
                    f"expected_row={rhs[row].tolist()[:8]}"
                )
            raise AssertionError(
                f"{label}: {name} differs at {first}; "
                f"actual={lhs[first].item()} expected={rhs[first].item()}{row_detail}"
            )


def _assert_geometry_close(
    actual: tuple[torch.Tensor, ...], expected: tuple[torch.Tensor, ...], label: str
) -> None:
    names = ("neighbor_matrix", "pair_shifts", "num_neighbors", "distances", "vectors")[: len(actual)]
    for name, lhs, rhs in zip(names, actual, expected, strict=True):
        if lhs.is_floating_point():
            tolerance = 1e-6 if lhs.dtype == torch.float32 else 1e-12
            torch.testing.assert_close(lhs, rhs, atol=tolerance, rtol=tolerance, msg=f"{label}: {name}")
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
        "--geometry",
        action="store_true",
        help="include Torch materialization of distances and vectors",
    )
    parser.add_argument(
        "--breakdown",
        action="store_true",
        help="profile topology and geometry materialization stages on a fixed candidate",
    )
    parser.add_argument(
        "--hip-topology",
        action="store_true",
        help="measure the isolated native HIP topology candidate",
    )
    parser.add_argument(
        "--hip-composite-topology",
        action="store_true",
        help="measure the isolated one-sort int64 HIP topology candidate",
    )
    parser.add_argument(
        "--hip-composite-workspace",
        action="store_true",
        help="reuse caller-owned scratch for the composite topology candidate",
    )
    parser.add_argument(
        "--hip-hybrid",
        action="store_true",
        help="measure native query/composite topology plus Torch geometry",
    )
    parser.add_argument(
        "--hip-native-geometry",
        action="store_true",
        help="measure native query/composite topology plus native HIP geometry",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/native-batch-cell-query-materialization-benchmark"),
    )
    args = parser.parse_args()
    if min(
        args.atoms_per_system,
        args.batch_systems,
        args.warmup,
        args.samples,
        args.capacity,
    ) <= 0:
        raise ValueError("all size, batch, samples, warmup, and capacity arguments must be positive")
    if args.breakdown and not args.geometry:
        raise ValueError("--breakdown requires --geometry")
    if args.hip_topology and not args.breakdown:
        raise ValueError("--hip-topology requires --breakdown")
    if args.hip_composite_topology and not args.breakdown:
        raise ValueError("--hip-composite-topology requires --breakdown")
    if args.hip_composite_workspace and not args.hip_composite_topology:
        raise ValueError("--hip-composite-workspace requires --hip-composite-topology")
    if args.hip_hybrid and not (
        args.geometry and args.hip_composite_topology and args.hip_composite_workspace
    ):
        raise ValueError(
            "--hip-hybrid requires --geometry, --hip-composite-topology, "
            "and --hip-composite-workspace"
        )
    if args.hip_native_geometry and not (
        args.geometry and args.hip_composite_topology and args.hip_composite_workspace
    ):
        raise ValueError(
            "--hip-native-geometry requires --geometry, --hip-composite-topology, "
            "and --hip-composite-workspace"
        )
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("HIP PyTorch and a visible HCU are required")

    root = Path(__file__).resolve().parent
    for variable, directory in (
        ("NVALCHEMI_HIP_BATCH_CELL_KEY_COUNT_BUILD_DIR", "native-hip-batch-cell-key-count"),
        ("NVALCHEMI_HIP_CELL_SCAN_BUILD_DIR", "native-hip-cell-start-scan"),
        ("NVALCHEMI_HIP_CELL_FILL_BUILD_DIR", "native-hip-cell-fill"),
        ("NVALCHEMI_HIP_BATCH_CELL_QUERY_BUILD_DIR", "native-hip-batch-cell-query"),
        ("NVALCHEMI_HIP_BATCH_QUERY_GEOMETRY_BUILD_DIR", "native-hip-batch-query-geometry"),
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
    from nvalchemiops._hip_batch_cell_query import (  # noqa: PLC0415
        batch_query_cell_list_hip_into,
    )
    from nvalchemiops._hip_batch_query_materialize import (  # noqa: PLC0415
        _composite_key_layout,
        _load_jit_extension as load_hip_topology_extension,
        allocate_batch_query_topology_composite_workspace,
    )
    from nvalchemiops._hip_batch_query_geometry import (  # noqa: PLC0415
        _load_jit_extension as load_hip_geometry_extension,
    )
    from nvalchemiops._torch_batch_query_materialization import (  # noqa: PLC0415
        _canonicalize_candidate,
        _materialize_canonical_geometry,
        materialize_batch_query_topology_geometry_into,
        _reset_geometry_outputs,
        _reset_topology_outputs,
        _scatter_canonical_candidate,
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

        native_candidate = _query_outputs(atoms, args.capacity, device)
        candidate_public = _query_outputs(
            atoms, args.capacity, device, dtype=dtype, geometry=args.geometry
        )
        torch_public = _query_outputs(
            atoms, args.capacity, device, dtype=dtype, geometry=args.geometry
        )
        breakdown_topology_public: tuple[torch.Tensor, ...] | None = None
        breakdown_geometry_public: tuple[torch.Tensor, ...] | None = None
        canonical_candidate = None
        if args.breakdown:
            breakdown_topology_public = _query_outputs(atoms, args.capacity, device)
            breakdown_geometry_public = _query_outputs(
                atoms, args.capacity, device, dtype=dtype, geometry=True
            )
        hip_topology_public: tuple[torch.Tensor, ...] | None = None
        composite_topology_public: tuple[torch.Tensor, ...] | None = None
        hybrid_public: tuple[torch.Tensor, ...] | None = None
        native_geometry_public: tuple[torch.Tensor, ...] | None = None
        hip_topology_extension = None
        hip_geometry_extension = None
        composite_topology_workspace = None
        if args.hip_topology or args.hip_composite_topology:
            hip_topology_public = _query_outputs(atoms, args.capacity, device)
            if args.hip_composite_topology:
                composite_topology_public = _query_outputs(atoms, args.capacity, device)
                if args.hip_hybrid:
                    hybrid_public = _query_outputs(
                        atoms, args.capacity, device, dtype=dtype, geometry=True
                    )
                if args.hip_native_geometry:
                    native_geometry_public = _query_outputs(
                        atoms, args.capacity, device, dtype=dtype, geometry=True
                    )
            hip_topology_extension = load_hip_topology_extension()
        if args.hip_native_geometry:
            hip_geometry_extension = load_hip_geometry_extension()
        composite_row_bits, composite_total_bits = _composite_key_layout(atoms, 8, 128)

        def enumeration_native() -> None:
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

        def materialize_candidate() -> None:
            _materialize_candidate_with_torch(
                native_candidate,
                candidate_public,
                positions,
                cells,
                batch_idx,
                geometry=args.geometry,
            )

        def query_native_torch_materialize() -> None:
            enumeration_native()
            materialize_candidate()

        def materialize_topology_stage() -> None:
            assert breakdown_topology_public is not None
            _reset_topology_outputs(
                breakdown_topology_public[0],
                breakdown_topology_public[1],
                breakdown_topology_public[2],
                atoms,
            )
            topology = _canonicalize_candidate(*native_candidate)
            _scatter_canonical_candidate(
                topology,
                breakdown_topology_public[0],
                breakdown_topology_public[1],
                breakdown_topology_public[2],
            )

        def materialize_geometry_stage() -> None:
            assert breakdown_geometry_public is not None
            if canonical_candidate is None:
                raise RuntimeError("breakdown candidate has not been canonicalized")
            _reset_geometry_outputs(
                breakdown_geometry_public[3], breakdown_geometry_public[4]
            )
            _materialize_canonical_geometry(
                positions,
                cells,
                batch_idx,
                canonical_candidate,
                breakdown_geometry_public[3],
                breakdown_geometry_public[4],
            )

        def materialize_geometry_native_hip_stage() -> None:
            if (
                composite_topology_public is None
                or native_geometry_public is None
                or hip_geometry_extension is None
            ):
                raise RuntimeError("native HIP geometry benchmark was not initialized")
            hip_geometry_extension.batch_query_geometry_into(
                positions,
                cells,
                batch_idx,
                composite_topology_public[0],
                composite_topology_public[1],
                composite_topology_public[2],
                native_geometry_public[3],
                native_geometry_public[4],
            )

        def materialize_topology_hip_stage() -> None:
            if hip_topology_public is None or hip_topology_extension is None:
                raise RuntimeError("native HIP topology benchmark was not initialized")
            hip_topology_extension.batch_query_materialize_topology_into(
                native_candidate[0],
                native_candidate[1],
                native_candidate[2],
                hip_topology_public[0],
                hip_topology_public[1],
                hip_topology_public[2],
                atoms,
            )

        def materialize_topology_composite_hip_stage() -> None:
            if composite_topology_public is None or hip_topology_extension is None:
                raise RuntimeError(
                    "native HIP composite topology benchmark was not initialized"
                )
            hip_topology_extension.batch_query_materialize_topology_composite_into(
                native_candidate[0],
                native_candidate[1],
                native_candidate[2],
                composite_topology_public[0],
                composite_topology_public[1],
                composite_topology_public[2],
                atoms,
                composite_row_bits,
                composite_total_bits,
                8,
                128,
            )

        def materialize_topology_composite_hip_reused_stage() -> None:
            if (
                composite_topology_public is None
                or composite_topology_workspace is None
                or hip_topology_extension is None
            ):
                raise RuntimeError(
                    "reused native HIP composite topology benchmark was not initialized"
                )
            hip_topology_extension.batch_query_materialize_topology_composite_workspace_into(
                native_candidate[0],
                native_candidate[1],
                native_candidate[2],
                composite_topology_public[0],
                composite_topology_public[1],
                composite_topology_public[2],
                composite_topology_workspace.keys_a,
                composite_topology_workspace.keys_b,
                composite_topology_workspace.order_a,
                composite_topology_workspace.order_b,
                composite_topology_workspace.row_starts,
                composite_topology_workspace.sort_workspace,
                composite_topology_workspace.scan_workspace,
                atoms,
                composite_row_bits,
                composite_total_bits,
                8,
                128,
            )

        def query_native_hybrid() -> None:
            if (
                hybrid_public is None
                or hip_topology_extension is None
                or composite_topology_workspace is None
            ):
                raise RuntimeError(
                    "hybrid benchmark requires composite topology workspace"
                )
            enumeration_native()
            hip_topology_extension.batch_query_materialize_topology_composite_workspace_into(
                native_candidate[0],
                native_candidate[1],
                native_candidate[2],
                hybrid_public[0],
                hybrid_public[1],
                hybrid_public[2],
                composite_topology_workspace.keys_a,
                composite_topology_workspace.keys_b,
                composite_topology_workspace.order_a,
                composite_topology_workspace.order_b,
                composite_topology_workspace.row_starts,
                composite_topology_workspace.sort_workspace,
                composite_topology_workspace.scan_workspace,
                atoms,
                composite_row_bits,
                composite_total_bits,
                8,
                128,
            )
            materialize_batch_query_topology_geometry_into(
                positions,
                cells,
                batch_idx,
                hybrid_public[0],
                hybrid_public[1],
                hybrid_public[2],
                distances=hybrid_public[3],
                vectors=hybrid_public[4],
            )

        def query_native_hybrid_native_geometry() -> None:
            if (
                native_geometry_public is None
                or hip_topology_extension is None
                or hip_geometry_extension is None
                or composite_topology_workspace is None
            ):
                raise RuntimeError(
                    "native HIP geometry benchmark requires composite topology workspace"
                )
            enumeration_native()
            hip_topology_extension.batch_query_materialize_topology_composite_workspace_into(
                native_candidate[0],
                native_candidate[1],
                native_candidate[2],
                native_geometry_public[0],
                native_geometry_public[1],
                native_geometry_public[2],
                composite_topology_workspace.keys_a,
                composite_topology_workspace.keys_b,
                composite_topology_workspace.order_a,
                composite_topology_workspace.order_b,
                composite_topology_workspace.row_starts,
                composite_topology_workspace.sort_workspace,
                composite_topology_workspace.scan_workspace,
                atoms,
                composite_row_bits,
                composite_total_bits,
                8,
                128,
            )
            hip_geometry_extension.batch_query_geometry_into(
                positions,
                cells,
                batch_idx,
                native_geometry_public[0],
                native_geometry_public[1],
                native_geometry_public[2],
                native_geometry_public[3],
                native_geometry_public[4],
            )

        def query_torch() -> None:
            if args.geometry:
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
            else:
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
                    *torch_public,
                )

        build_native()
        build_torch()
        torch.cuda.synchronize()
        enumeration_native()
        if args.hip_composite_workspace:
            composite_topology_workspace = allocate_batch_query_topology_composite_workspace(
                native_candidate[0], native_candidate[2]
            )
        materialize_candidate()
        query_torch()
        torch.cuda.synchronize()
        if args.hip_topology:
            torch.cuda.synchronize()
            materialize_topology_hip_stage()
            torch.cuda.synchronize()
        if args.hip_composite_topology:
            torch.cuda.synchronize()
            materialize_topology_composite_hip_stage()
            torch.cuda.synchronize()
        if args.hip_composite_workspace:
            torch.cuda.synchronize()
            materialize_topology_composite_hip_reused_stage()
            torch.cuda.synchronize()
        if args.hip_hybrid:
            torch.cuda.synchronize()
            query_native_hybrid()
            torch.cuda.synchronize()
        if args.hip_native_geometry:
            torch.cuda.synchronize()
            query_native_hybrid_native_geometry()
            torch.cuda.synchronize()
        _assert_equal(candidate_public, torch_public, f"{workload_name} candidate materialization")
        if args.breakdown:
            canonical_candidate = _canonicalize_candidate(*native_candidate)
            materialize_topology_stage()
            materialize_geometry_stage()
            assert breakdown_topology_public is not None
            assert breakdown_geometry_public is not None
            _assert_equal(
                breakdown_topology_public,
                candidate_public[:3],
                f"{workload_name} topology breakdown",
            )
            _assert_equal(
                breakdown_geometry_public[3:],
                candidate_public[3:],
                f"{workload_name} geometry breakdown",
            )
        if args.hip_topology:
            assert hip_topology_public is not None
            _assert_equal(
                hip_topology_public,
                candidate_public[:3],
                f"{workload_name} native HIP topology",
            )
        if args.hip_composite_topology:
            assert composite_topology_public is not None
            _assert_equal(
                composite_topology_public,
                candidate_public[:3],
                f"{workload_name} native HIP composite topology",
            )
        if args.hip_hybrid:
            assert hybrid_public is not None
            _assert_equal(
                hybrid_public,
                torch_public,
                f"{workload_name} native HIP/Torch hybrid",
            )
        if args.hip_native_geometry:
            assert native_geometry_public is not None
            _assert_geometry_close(
                native_geometry_public,
                torch_public,
                f"{workload_name} native HIP geometry hybrid",
            )
            _assert_geometry_close(
                native_geometry_public,
                candidate_public,
                f"{workload_name} native HIP geometry candidate",
            )

        for _ in range(args.warmup):
            enumeration_native()
            query_native_torch_materialize()
            query_torch()
            if args.hip_hybrid:
                query_native_hybrid()
            if args.hip_native_geometry:
                query_native_hybrid_native_geometry()
        torch.cuda.synchronize()
        if args.breakdown:
            for _ in range(args.warmup):
                materialize_topology_stage()
                materialize_geometry_stage()
                materialize_candidate()
                if args.hip_topology:
                    torch.cuda.synchronize()
                    materialize_topology_hip_stage()
                if args.hip_composite_topology:
                    torch.cuda.synchronize()
                    materialize_topology_composite_hip_stage()
                if args.hip_composite_workspace:
                    torch.cuda.synchronize()
                    materialize_topology_composite_hip_reused_stage()
                if args.hip_native_geometry:
                    materialize_geometry_native_hip_stage()
            torch.cuda.synchronize()
        samples: dict[str, list[float]] = {
            "enumeration_native": [],
            "query_native_torch_materialize": [],
            "query_torch": [],
        }
        if args.breakdown:
            samples.update(
                {
                    "materialize_topology": [],
                    "materialize_geometry": [],
                    "materialize_complete": [],
                }
            )
        if args.hip_topology:
            samples["materialize_topology_hip"] = []
        if args.hip_composite_topology:
            samples["materialize_topology_composite_hip"] = []
        if args.hip_composite_workspace:
            samples["materialize_topology_composite_hip_reused"] = []
        if args.hip_hybrid:
            samples["query_native_hybrid"] = []
        if args.hip_native_geometry:
            samples["query_native_hybrid_native_geometry"] = []
            samples["materialize_geometry_native_hip"] = []
        calls = (
            ("enumeration_native", enumeration_native),
            ("query_native_torch_materialize", query_native_torch_materialize),
            ("query_torch", query_torch),
        )
        if args.hip_hybrid:
            calls += (("query_native_hybrid", query_native_hybrid),)
        if args.hip_native_geometry:
            calls += (
                ("query_native_hybrid_native_geometry", query_native_hybrid_native_geometry),
            )
        breakdown_calls = (
            ("materialize_topology", materialize_topology_stage),
            ("materialize_geometry", materialize_geometry_stage),
            ("materialize_complete", materialize_candidate),
        )
        if args.hip_topology:
            breakdown_calls += (("materialize_topology_hip", materialize_topology_hip_stage),)
        if args.hip_composite_topology:
            breakdown_calls += (
                (
                    "materialize_topology_composite_hip",
                    materialize_topology_composite_hip_stage,
                ),
            )
        if args.hip_composite_workspace:
            breakdown_calls += (
                (
                    "materialize_topology_composite_hip_reused",
                    materialize_topology_composite_hip_reused_stage,
                ),
            )
        if args.hip_native_geometry:
            breakdown_calls += (
                ("materialize_geometry_native_hip", materialize_geometry_native_hip_stage),
            )
        for sample_index in range(args.samples):
            for name, call in calls if sample_index % 2 == 0 else reversed(calls):
                samples[name].append(_sample_ms(call))
            if args.breakdown:
                for name, call in (
                    breakdown_calls
                    if sample_index % 2 == 0
                    else reversed(breakdown_calls)
                ):
                    if name in {
                        "materialize_topology_hip",
                        "materialize_topology_composite_hip",
                        "materialize_topology_composite_hip_reused",
                    }:
                        torch.cuda.synchronize()
                    samples[name].append(_sample_ms(call))
        torch.cuda.synchronize()
        materialize_candidate()
        query_torch()
        torch.cuda.synchronize()
        if args.hip_topology:
            torch.cuda.synchronize()
            materialize_topology_hip_stage()
            torch.cuda.synchronize()
        if args.hip_composite_topology:
            torch.cuda.synchronize()
            materialize_topology_composite_hip_stage()
            torch.cuda.synchronize()
        if args.hip_composite_workspace:
            torch.cuda.synchronize()
            materialize_topology_composite_hip_reused_stage()
            torch.cuda.synchronize()
        if args.hip_hybrid:
            torch.cuda.synchronize()
            query_native_hybrid()
            torch.cuda.synchronize()
        if args.hip_native_geometry:
            torch.cuda.synchronize()
            query_native_hybrid_native_geometry()
            torch.cuda.synchronize()
        _assert_equal(candidate_public, torch_public, f"{workload_name} after timing")
        if args.breakdown:
            canonical_candidate = _canonicalize_candidate(*native_candidate)
            materialize_topology_stage()
            materialize_geometry_stage()
            assert breakdown_topology_public is not None
            assert breakdown_geometry_public is not None
            _assert_equal(
                breakdown_topology_public,
                candidate_public[:3],
                f"{workload_name} topology breakdown after timing",
            )
            _assert_equal(
                breakdown_geometry_public[3:],
                candidate_public[3:],
                f"{workload_name} geometry breakdown after timing",
            )
        if args.hip_topology:
            assert hip_topology_public is not None
            _assert_equal(
                hip_topology_public,
                candidate_public[:3],
                f"{workload_name} native HIP topology after timing",
            )
        if args.hip_composite_topology:
            assert composite_topology_public is not None
            _assert_equal(
                composite_topology_public,
                candidate_public[:3],
                f"{workload_name} native HIP composite topology after timing",
            )
        if args.hip_hybrid:
            assert hybrid_public is not None
            _assert_equal(
                hybrid_public,
                torch_public,
                f"{workload_name} native HIP/Torch hybrid after timing",
            )
        if args.hip_native_geometry:
            assert native_geometry_public is not None
            _assert_geometry_close(
                native_geometry_public,
                torch_public,
                f"{workload_name} native HIP geometry hybrid after timing",
            )
            _assert_geometry_close(
                native_geometry_public,
                candidate_public,
                f"{workload_name} native HIP geometry candidate after timing",
            )

        api_wall_samples: dict[str, list[float]] = {}
        if args.hip_composite_workspace:
            api_wall_calls = (
                (
                    "materialize_topology_composite_hip_api_wall",
                    materialize_topology_composite_hip_stage,
                ),
                (
                    "materialize_topology_composite_hip_reused_api_wall",
                    materialize_topology_composite_hip_reused_stage,
                ),
            )
            if args.hip_hybrid:
                api_wall_calls += (
                    ("query_native_hybrid_api_wall", query_native_hybrid),
                    ("query_torch_api_wall", query_torch),
                )
            if args.hip_native_geometry:
                api_wall_calls += (
                    ("materialize_geometry_native_hip_api_wall", materialize_geometry_native_hip_stage),
                    ("materialize_geometry_torch_api_wall", materialize_geometry_stage),
                    (
                        "query_native_hybrid_native_geometry_api_wall",
                        query_native_hybrid_native_geometry,
                    ),
                    ("query_torch_native_geometry_api_wall", query_torch),
                )
            api_wall_samples = {name: [] for name, _ in api_wall_calls}
            for sample_index in range(args.samples):
                ordered_calls = (
                    api_wall_calls
                    if sample_index % 2 == 0
                    else reversed(api_wall_calls)
                )
                for name, call in ordered_calls:
                    api_wall_samples[name].append(_sample_api_wall_ms(call))

        summaries: dict[str, object] = {}
        for name, values in samples.items():
            path = output_dir / f"{workload_name}-{name}-ms.txt"
            _write_samples(path, values)
            summaries[name] = {**_summary(values), "samples_path": str(path)}
        for name, values in api_wall_samples.items():
            path = output_dir / f"{workload_name}-{name}-ms.txt"
            _write_samples(path, values)
            summaries[name] = {**_summary(values), "samples_path": str(path)}
        summaries["query_native_torch_materialize_over_torch_factor"] = (
            summaries["query_torch"]["median_ms"]
            / summaries["query_native_torch_materialize"]["median_ms"]  # type: ignore[index]
        )
        if args.breakdown:
            summaries["geometry_over_complete_factor"] = (
                summaries["materialize_geometry"]["median_ms"]  # type: ignore[index]
                / summaries["materialize_complete"]["median_ms"]  # type: ignore[index]
            )
            summaries["topology_over_complete_factor"] = (
                summaries["materialize_topology"]["median_ms"]  # type: ignore[index]
                / summaries["materialize_complete"]["median_ms"]  # type: ignore[index]
            )
        if args.hip_topology:
            summaries["hip_topology_over_torch_topology_factor"] = (
                summaries["materialize_topology_hip"]["median_ms"]  # type: ignore[index]
                / summaries["materialize_topology"]["median_ms"]  # type: ignore[index]
            )
        if args.hip_composite_topology:
            summaries["hip_composite_topology_over_torch_topology_factor"] = (
                summaries["materialize_topology_composite_hip"]["median_ms"]  # type: ignore[index]
                / summaries["materialize_topology"]["median_ms"]  # type: ignore[index]
            )
            if args.hip_topology:
                summaries["hip_composite_topology_over_hip_topology_factor"] = (
                    summaries["materialize_topology_composite_hip"]["median_ms"]  # type: ignore[index]
                    / summaries["materialize_topology_hip"]["median_ms"]  # type: ignore[index]
                )
        if args.hip_composite_workspace:
            summaries["hip_composite_reused_over_composite_factor"] = (
                summaries["materialize_topology_composite_hip_reused"]["median_ms"]  # type: ignore[index]
                / summaries["materialize_topology_composite_hip"]["median_ms"]  # type: ignore[index]
            )
            summaries["hip_composite_reused_api_over_composite_api_factor"] = (
                    summaries["materialize_topology_composite_hip_reused_api_wall"]["median_ms"]  # type: ignore[index]
                    / summaries["materialize_topology_composite_hip_api_wall"]["median_ms"]  # type: ignore[index]
            )
        if args.hip_hybrid:
            summaries["query_native_hybrid_over_torch_factor"] = (
                summaries["query_native_hybrid"]["median_ms"]  # type: ignore[index]
                / summaries["query_torch"]["median_ms"]  # type: ignore[index]
            )
            summaries["query_native_hybrid_api_over_torch_api_factor"] = (
                summaries["query_native_hybrid_api_wall"]["median_ms"]  # type: ignore[index]
                / summaries["query_torch_api_wall"]["median_ms"]  # type: ignore[index]
            )
        if args.hip_native_geometry:
            summaries["native_geometry_over_torch_geometry_factor"] = (
                summaries["materialize_geometry_native_hip"]["median_ms"]  # type: ignore[index]
                / summaries["materialize_geometry"]["median_ms"]  # type: ignore[index]
            )
            summaries["native_geometry_api_over_torch_geometry_api_factor"] = (
                summaries["materialize_geometry_native_hip_api_wall"]["median_ms"]  # type: ignore[index]
                / summaries["materialize_geometry_torch_api_wall"]["median_ms"]  # type: ignore[index]
            )
            summaries["query_native_hybrid_native_geometry_over_torch_factor"] = (
                summaries["query_native_hybrid_native_geometry"]["median_ms"]  # type: ignore[index]
                / summaries["query_torch"]["median_ms"]  # type: ignore[index]
            )
            summaries["query_native_hybrid_native_geometry_api_over_torch_api_factor"] = (
                summaries["query_native_hybrid_native_geometry_api_wall"]["median_ms"]  # type: ignore[index]
                / summaries["query_torch_native_geometry_api_wall"]["median_ms"]  # type: ignore[index]
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
                    "scope": "pre-warmed HIP event device work; fixed grid/CSR metadata; JIT/allocation/host time excluded",
                    "native_topology_stream_boundary": "torch.cuda.synchronize before native scope; excluded from event",
                    "geometry_outputs": args.geometry,
                    "breakdown": args.breakdown,
                    "native_topology": args.hip_topology,
                    "native_composite_topology": args.hip_composite_topology,
                    "native_composite_workspace_reused": args.hip_composite_workspace,
                    "hybrid_native_query_composite_topology_torch_geometry": args.hip_hybrid,
                    "hybrid_native_query_composite_topology_native_hip_geometry": args.hip_native_geometry,
                    "native_geometry_stage": args.hip_native_geometry,
                    "api_wall_clock": args.hip_composite_workspace,
                    "composite_key": {
                        "shift_bits": 8,
                        "shift_bias": 128,
                        "row_bits": composite_row_bits,
                        "total_bits": composite_total_bits,
                    },
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
