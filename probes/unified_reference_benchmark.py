#!/usr/bin/env python3
"""Unified Torch-reference neighbor and fixed-cell dynamics benchmark.

The neighbor cases use the same CIF coordinates in periodic and no-PBC modes
so that the output schema and timing boundaries are directly comparable.  The
no-PBC cases are an algorithm-boundary measurement, not a claim that the
current MACE production lane is non-periodic.  The optional end-to-end case is
the production-shaped periodic ``[46, 92]`` MACE/FIRE2 path.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import torch
from ase.io import read

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import FIRE2
from nvalchemi.dynamics.base import ConvergenceHook, DynamicsStage
from nvalchemi.hooks import NeighborListHook, StageTimingHook
from nvalchemi.models.base import NeighborListFormat
from nvalchemi.models.mace import MACEWrapper
from nvalchemi.neighbors import compute_neighbors


DEFAULT_SCALE_ROOTS = {
    46: Path("/data/csp_data/perf_46"),
    92: Path("/data/csp_data/perf_92"),
    184: Path("/data/csp_data/perf_184"),
    368: Path("/data/csp_data/perf_368"),
}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(device: torch.device, function: Any) -> tuple[float, Any]:
    _sync(device)
    started = time.perf_counter()
    result = function()
    _sync(device)
    return time.perf_counter() - started, result


def _files(root: Path, count: int) -> list[Path]:
    paths = sorted(root.expanduser().resolve().glob("*.cif"))
    if len(paths) < count:
        raise ValueError(f"{root}: requested {count} CIFs, found {len(paths)}")
    return paths[:count]


def _load_structures(paths: list[Path]) -> list[Any]:
    structures = [read(path) for path in paths]
    if not structures:
        raise ValueError("benchmark case must contain at least one structure")
    if not all(bool(atoms.pbc.all()) for atoms in structures):
        raise ValueError(f"benchmark inputs must have 3D PBC: {paths}")
    return structures


def _structure_metadata(
    paths: list[Path],
    structures: list[Any] | None = None,
    *,
    effective_periodic: bool | None = None,
) -> dict[str, Any]:
    if structures is None:
        structures = _load_structures(paths)
    source_pbc = [atoms.pbc.tolist() for atoms in structures]
    effective_pbc = source_pbc
    if effective_periodic is not None:
        effective_pbc = [[effective_periodic] * 3 for _ in structures]
    return {
        "paths": [str(path) for path in paths],
        "atom_counts": [len(atoms) for atoms in structures],
        "source_pbc": source_pbc,
        "effective_pbc": effective_pbc,
        "cell_volumes": [float(atoms.cell.volume) for atoms in structures],
    }


def _make_batch(
    paths: list[Path],
    device: torch.device,
    *,
    periodic: bool,
) -> Batch:
    data_list = []
    for atoms in _load_structures(paths):
        data = AtomicData.from_atoms(atoms)
        if not periodic:
            # compute_neighbors explicitly drops cell/pbc when all PBC flags
            # are false. Keeping the fields preserves the same Batch schema.
            data.pbc = torch.zeros_like(data.pbc)
        data_list.append(data)
    return Batch.from_data_list(data_list).to(device)


def _assert_neighbor_contract(batch: Batch, *, periodic: bool) -> dict[str, Any]:
    edges = batch.neighbor_list
    if edges is None:
        raise AssertionError("COO neighbor output is missing")
    source = edges[:, 0].long()
    target = edges[:, 1].long()
    batch_idx = batch.batch_idx.long()
    cross_system = int((batch_idx[source] != batch_idx[target]).sum().item())
    if cross_system:
        raise AssertionError(f"neighbor list crossed systems: {cross_system}")
    shifts = getattr(batch, "neighbor_list_shifts", None)
    if periodic and shifts is None:
        raise AssertionError("periodic neighbor output is missing image shifts")
    if not periodic and shifts is not None and shifts.numel() != 0:
        raise AssertionError("no-PBC neighbor output unexpectedly has image shifts")
    counts = torch.bincount(source, minlength=batch.num_nodes)
    return {
        "edges": int(edges.shape[0]),
        "edge_shape": list(edges.shape),
        "batch_ptr": batch.batch_ptr.detach().cpu().tolist(),
        "max_neighbors": int(counts.max().item()) if counts is not None and counts.numel() else 0,
        "cross_system_edges": cross_system,
        "shift_shape": list(shifts.shape) if shifts is not None else None,
    }


def _run_neighbor_case(
    label: str,
    paths: list[Path],
    model: MACEWrapper,
    device: torch.device,
    *,
    periodic: bool,
    half_list: bool,
    method: str | None,
    warmup: int,
    steady: int,
) -> dict[str, Any]:
    structures = _load_structures(paths)
    batch = _make_batch(paths, device, periodic=periodic)
    config = model.model_config.neighbor_config
    if config is None:
        raise RuntimeError("MACE model did not provide a neighbor configuration")

    def build() -> None:
        compute_neighbors(
            batch,
            cutoff=config.cutoff,
            format=NeighborListFormat.COO,
            half_list=half_list,
            backend="torch_reference",
            method=method,
        )

    cold_s, _ = _timed(device, build)
    contract = _assert_neighbor_contract(batch, periodic=periodic)
    warmup_s: list[float] = []
    for _ in range(warmup):
        elapsed, _ = _timed(device, build)
        warmup_s.append(elapsed)
    steady_s: list[float] = []
    for _ in range(steady):
        elapsed, _ = _timed(device, build)
        steady_s.append(elapsed)
    return {
        "case": label,
        "status": "passed",
        "path": "periodic" if periodic else "no_pbc",
        "half_list": half_list,
        "method": method or "dense",
        **_structure_metadata(paths, structures, effective_periodic=periodic),
        "num_nodes": batch.num_nodes,
        "cold_s": cold_s,
        "warmup_s": warmup_s,
        "steady_s": steady_s,
        "steady_mean_s": mean(steady_s),
        "steady_stdev_s": stdev(steady_s) if len(steady_s) > 1 else 0.0,
        **contract,
    }


def _make_dynamics_batch(paths: list[Path], device: torch.device) -> Batch:
    data_list = []
    for atoms in _load_structures(paths):
        data = AtomicData.from_atoms(atoms)
        data.add_node_property("velocities", torch.zeros(len(atoms), 3))
        data.forces = torch.zeros(len(atoms), 3)
        data.energy = torch.zeros(1, 1)
        data_list.append(data)
    batch = Batch.from_data_list(data_list).to(device)
    batch.status = torch.zeros(batch.num_graphs, 1, dtype=torch.long, device=device)
    return batch


def _run_end_to_end(
    paths: list[Path],
    model: MACEWrapper,
    device: torch.device,
    *,
    steps: int,
    skin: float,
    method: str | None,
) -> dict[str, Any]:
    structures = _load_structures(paths)
    batch = _make_dynamics_batch(paths, device)
    convergence = ConvergenceHook.from_fmax(
        threshold=-1.0, source_status=0, target_status=1
    )
    timing = StageTimingHook(
        profiled_stages="detailed",
        enable_nvtx=False,
        timer_backend="auto",
    )
    dynamics = FIRE2(
        model=model,
        dt=0.01,
        n_steps=steps,
        convergence_hook=convergence,
        hooks=[convergence, timing],
        backend="torch_reference",
    )
    dynamics.register_hook(
        NeighborListHook(
            model.model_config.neighbor_config,
            skin=skin,
            backend="torch_reference",
            method=method,
            stage=DynamicsStage.BEFORE_COMPUTE,
        )
    )
    _sync(device)
    started = time.perf_counter()
    dynamics.run(batch)
    _sync(device)
    elapsed = time.perf_counter() - started
    timing_summary = timing.summary()
    timing.close()
    if dynamics.step_count != steps:
        raise AssertionError(
            f"end-to-end run stopped at {dynamics.step_count}, expected {steps}"
        )
    return {
        "case": "periodic_mixed_46_92_e2e",
        "status": "passed",
        "path": "periodic",
        **_structure_metadata(paths, structures, effective_periodic=True),
        "steps": steps,
        "skin": skin,
        "neighbor_method": method or "dense",
        "dt": 0.01,
        "elapsed_s": elapsed,
        "steady_step_s": elapsed / steps,
        "batch_ptr": batch.batch_ptr.detach().cpu().tolist(),
        "neighbor_edges_last_step": int(batch.neighbor_list.shape[0]),
        "stage_timing": timing_summary,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--scale-sizes", type=int, nargs="+", default=[46, 92, 184, 368])
    parser.add_argument("--batch-root", type=Path, default=DEFAULT_SCALE_ROOTS[92])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steady", type=int, default=2)
    parser.add_argument("--e2e-steps", type=int, default=100)
    parser.add_argument("--e2e-skin", type=float, default=0.5)
    parser.add_argument(
        "--neighbor-method", choices=("dense", "cell_list"), default="dense"
    )
    parser.add_argument("--skip-scales", action="store_true")
    parser.add_argument("--skip-batches", action="store_true")
    parser.add_argument("--skip-e2e", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.warmup < 0 or args.steady < 1 or args.e2e_steps < 1:
        raise ValueError("warmup must be non-negative; steady/e2e-steps must be positive")
    if args.e2e_skin < 0 or any(size < 1 for size in args.scale_sizes + args.batch_sizes):
        raise ValueError("sizes must be positive and e2e-skin must be non-negative")
    if args.device == "cuda" and (not torch.version.hip or not torch.cuda.is_available()):
        raise RuntimeError("HCU device requested but HIP Torch is unavailable")

    device = torch.device(args.device)
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    process_started = time.perf_counter()
    model_load_s, model = _timed(
        device,
        lambda: MACEWrapper.from_checkpoint(str(checkpoint), device=device, dtype=torch.float32),
    )
    model.eval()
    print(
        json.dumps(
            {
                "stage": "model_loaded",
                "status": "passed",
                "device": args.device,
                "device_name": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
                "checkpoint": str(checkpoint),
                "model_load_s": model_load_s,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    cases: list[dict[str, Any]] = []
    method = None if args.neighbor_method == "dense" else args.neighbor_method
    if not args.skip_scales:
        for size in args.scale_sizes:
            root = DEFAULT_SCALE_ROOTS.get(size)
            if root is None:
                raise ValueError(f"no default scale root is registered for {size} atoms")
            paths = _files(root, 1)
            cases.append(
                _run_neighbor_case(
                    f"periodic_scale_{size}", paths, model, device,
                    periodic=True, half_list=False, method=method,
                    warmup=args.warmup, steady=args.steady,
                )
            )
            if method == "cell_list":
                cases.append(
                    _run_neighbor_case(
                        f"periodic_scale_{size}_half", paths, model, device,
                        periodic=True, half_list=True, method=method,
                        warmup=args.warmup, steady=args.steady,
                    )
                )
            cases.append(
                _run_neighbor_case(
                    f"no_pbc_scale_{size}", paths, model, device,
                    periodic=False, half_list=False, method=method,
                    warmup=args.warmup, steady=args.steady,
                )
            )
            cases.append(
                _run_neighbor_case(
                    f"no_pbc_scale_{size}_half", paths, model, device,
                    periodic=False, half_list=True, method=method,
                    warmup=args.warmup, steady=args.steady,
                )
            )

    if not args.skip_batches:
        for size in args.batch_sizes:
            paths = _files(args.batch_root, size)
            cases.append(
                _run_neighbor_case(
                    f"periodic_batch_{size}", paths, model, device,
                    periodic=True, half_list=False, method=method,
                    warmup=args.warmup, steady=args.steady,
                )
            )
            cases.append(
                _run_neighbor_case(
                    f"no_pbc_batch_{size}", paths, model, device,
                    periodic=False, half_list=False, method=method,
                    warmup=args.warmup, steady=args.steady,
                )
            )

    e2e: dict[str, Any] | None = None
    if not args.skip_e2e:
        e2e_paths = [_files(DEFAULT_SCALE_ROOTS[46], 1)[0], _files(DEFAULT_SCALE_ROOTS[92], 1)[0]]
        e2e = _run_end_to_end(
            e2e_paths,
            model,
            device,
            steps=args.e2e_steps,
            skin=args.e2e_skin,
            method=method,
        )

    print(
        json.dumps(
            {
                "stage": "summary",
                "status": "passed",
                "device": args.device,
                "backend": "torch_reference",
                "neighbor_method": args.neighbor_method,
                "elapsed_since_process_start_s": time.perf_counter() - process_started,
                "warmup": args.warmup,
                "steady": args.steady,
                "neighbor_cases": cases,
                "end_to_end": e2e,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
