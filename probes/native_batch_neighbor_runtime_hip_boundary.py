#!/usr/bin/env python3
"""Exercise the explicit native-HIP Batch neighbor runtime wrapper.

The probe uses the existing Torch cell-list build/query as an independent
oracle and invokes only the new ops-side wrapper for the native hybrid path:

    trusted build plan -> native candidate query -> composite topology
    -> Torch differentiable geometry

It is a correctness boundary probe, not a performance benchmark and does not
register ``backend='hip'``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch


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
    atoms: int, capacity: int, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, ...]:
    return (
        torch.full((atoms, capacity), atoms, dtype=torch.int32, device=device),
        torch.zeros((atoms, capacity, 3), dtype=torch.int32, device=device),
        torch.zeros(atoms, dtype=torch.int32, device=device),
        torch.zeros((atoms, capacity), dtype=dtype, device=device),
        torch.zeros((atoms, capacity, 3), dtype=dtype, device=device),
    )


def _workspace(counts: torch.Tensor, starts: torch.Tensor) -> torch.Tensor:
    from nvalchemiops._hip_cell_scan import cell_starts_hip_workspace_size

    counts.zero_()
    return torch.empty(
        cell_starts_hip_workspace_size(counts, starts),
        dtype=torch.uint8,
        device=counts.device,
    )


def _assert_equal(lhs: torch.Tensor, rhs: torch.Tensor, name: str) -> None:
    if not torch.equal(lhs, rhs):
        raise AssertionError(f"{name} differs")


def _assert_outputs_close(
    actual: tuple[torch.Tensor, ...], expected: tuple[torch.Tensor, ...]
) -> None:
    names = ("matrix", "shifts", "counts", "distances", "vectors")
    for name, lhs, rhs in zip(names, actual, expected, strict=True):
        if lhs.is_floating_point():
            tolerance = 1e-6 if lhs.dtype == torch.float32 else 1e-12
            torch.testing.assert_close(lhs, rhs, atol=tolerance, rtol=tolerance)
        else:
            _assert_equal(lhs, rhs, name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atoms-per-system", type=int, default=46)
    parser.add_argument("--batch-systems", type=int, default=32)
    parser.add_argument("--capacity", type=int, default=256)
    parser.add_argument(
        "--check-gradients",
        action="store_true",
        help="verify first- and second-order position gradients through Torch geometry",
    )
    args = parser.parse_args()
    if min(args.atoms_per_system, args.batch_systems, args.capacity) <= 0:
        raise ValueError("atoms, batch, and capacity must be positive")
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("HIP PyTorch and a visible HCU are required")

    root = Path(__file__).resolve().parent
    for variable, directory in (
        ("NVALCHEMI_HIP_BATCH_CELL_KEY_COUNT_BUILD_DIR", "runtime-cell-key-count"),
        ("NVALCHEMI_HIP_CELL_SCAN_BUILD_DIR", "runtime-cell-scan"),
        ("NVALCHEMI_HIP_CELL_FILL_BUILD_DIR", "runtime-cell-fill"),
        ("NVALCHEMI_HIP_BATCH_CELL_QUERY_BUILD_DIR", "runtime-cell-query"),
        ("NVALCHEMI_HIP_BATCH_QUERY_MATERIALIZE_BUILD_DIR", "runtime-topology"),
    ):
        os.environ.setdefault(variable, str(root.parent / "artifacts" / directory))

    from nvalchemiops._cell_list_abi import (  # noqa: PLC0415
        build_batch_cell_key_counts_reference_into,
        build_cell_atom_list_reference_into,
        build_cell_starts_reference_into,
    )
    from nvalchemiops._hip_batch_cell_build import _TrustedBatchCellBuildPlan  # noqa: PLC0415
    from nvalchemiops._hip_batch_cell_query import (  # noqa: PLC0415
        batch_query_cell_list_hip_into,
    )
    from nvalchemiops._hip_batch_neighbor_runtime import (  # noqa: PLC0415
        run_native_hip_batch_neighbor_into,
    )
    from nvalchemiops._hip_batch_query_materialize import (  # noqa: PLC0415
        allocate_batch_query_topology_composite_workspace,
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
    generator = torch.Generator(device=device).manual_seed(20260920)
    fractional = torch.rand((atoms, 3), dtype=dtype, device=device, generator=generator)
    positions = torch.bmm(
        fractional.unsqueeze(1), cells[batch_idx.to(torch.long)]
    ).squeeze(1)
    if args.check_gradients:
        positions = positions.detach().requires_grad_(True)

    native_build = _build_outputs(atoms, total_cells, device)
    torch_build = _build_outputs(atoms, total_cells, device)
    native_workspace = _workspace(native_build[3], native_build[4])
    plan = _TrustedBatchCellBuildPlan.initialize(
        positions,
        inverse_cells,
        dimensions,
        pbc,
        batch_idx,
        offsets,
        *native_build,
        native_workspace,
    )

    torch_query = _query_outputs(atoms, args.capacity, device, dtype)
    candidate = _query_outputs(atoms, args.capacity, device, dtype)[:3]
    public = _query_outputs(atoms, args.capacity, device, dtype)

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
        torch_build[2], torch_build[3], torch_build[4], torch_build[5], torch_build[6]
    )
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
        *torch_query[:3],
        return_distances=True,
        return_vectors=True,
        neighbor_distances=torch_query[3],
        neighbor_vectors=torch_query[4],
    )

    # Populate candidate counts once so workspace sizing sees the real active
    # candidate shape.  The wrapper repeats this query on its own current stream.
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
        *candidate,
    )
    topology_workspace = allocate_batch_query_topology_composite_workspace(
        candidate[0], candidate[2]
    )

    run_native_hip_batch_neighbor_into(
        plan,
        cells,
        cutoff,
        radius,
        *candidate,
        *public,
        topology_workspace,
    )
    torch.cuda.synchronize()

    for name, lhs, rhs in zip(
        ("shifts", "mapping", "keys", "counts", "starts"),
        native_build[:5],
        torch_build[:5],
        strict=True,
    ):
        _assert_equal(lhs, rhs, f"build {name}")
    _assert_equal(native_build[6], native_build[3], "build cursor")
    _assert_outputs_close(public, torch_query)

    gradient_result = {"first_order": False, "second_order": False}
    if args.check_gradients:
        distance_sum = public[3].sum()
        first = torch.autograd.grad(distance_sum, positions, create_graph=True)[0]
        if not bool(torch.isfinite(first).all()):
            raise AssertionError("first-order position gradient contains non-finite values")
        second = first.square().sum()
        second.backward()
        if positions.grad is None or not bool(torch.isfinite(positions.grad).all()):
            raise AssertionError("second-order position gradient contains non-finite values")
        gradient_result = {"first_order": True, "second_order": True}

    print(
        json.dumps(
            {
                "status": "passed",
                "device": torch.cuda.get_device_name(),
                "torch_hip": torch.version.hip,
                "atoms_per_system": args.atoms_per_system,
                "batch_systems": args.batch_systems,
                "atoms": atoms,
                "cells": total_cells,
                "capacity": args.capacity,
                "dtype": str(dtype),
                "pipeline": "trusted_build -> native_query -> composite_topology -> torch_geometry",
                "scope": "explicit wrapper correctness boundary; no dispatcher registration or performance timing",
                "gradient_check": gradient_result,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
