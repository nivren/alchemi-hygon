"""Collect a pre-Tier-1 baseline for Torch-reference neighbor construction.

The probe uses real periodic CIFs and records only ``compute_neighbors``.  It
does not run MACE forward or dynamics, so the measurements isolate the current
reference neighbor path (including its Python assembly step).  Each case emits
JSON lines for model load, cold construction, warmup, and steady samples.
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
from nvalchemi.models.mace import MACEWrapper
from nvalchemi.neighbors import compute_neighbors


DEFAULT_SCALE_ROOTS = {
    "n46": Path("/data/csp_data/perf_46"),
    "n92": Path("/data/csp_data/perf_92"),
    "n184": Path("/data/csp_data/perf_184"),
    "n368": Path("/data/csp_data/perf_368"),
    "n736": Path("/data/csp_data/perf_736"),
}


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _timed(device: torch.device, function: Any) -> tuple[float, Any]:
    _sync(device)
    start = time.perf_counter()
    result = function()
    _sync(device)
    return time.perf_counter() - start, result


def _files(root: Path, count: int) -> list[Path]:
    paths = sorted(root.expanduser().resolve().glob("*.cif"))
    if len(paths) < count:
        raise ValueError(f"{root}: requested {count} CIFs, found {len(paths)}")
    return paths[:count]


def _load_structures(paths: list[Path]) -> list[Any]:
    structures = [read(path) for path in paths]
    counts = [len(atoms) for atoms in structures]
    if not all(bool(atoms.pbc.all()) for atoms in structures):
        raise ValueError(f"all baseline inputs must have 3D PBC, got {paths}")
    if len(set(counts)) == 0:
        raise ValueError("baseline case must contain at least one structure")
    return structures


def _make_batch(structures: list[Any], device: torch.device) -> Batch:
    data = [AtomicData.from_atoms(atoms) for atoms in structures]
    return Batch.from_data_list(data).to(device)


def _run_case(
    label: str,
    paths: list[Path],
    model: MACEWrapper,
    config: Any,
    device: torch.device,
    warmup: int,
    steady: int,
    process_start: float,
) -> dict[str, Any]:
    structures = _load_structures(paths)
    batch = _make_batch(structures, device)

    def build() -> None:
        compute_neighbors(batch, config=config, backend="torch_reference")

    def emit(stage: str, **extra: Any) -> None:
        print(
            json.dumps(
                {
                    "stage": stage,
                    "case": label,
                    "elapsed_since_process_start_s": time.perf_counter() - process_start,
                    **extra,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    atom_counts = [len(atoms) for atoms in structures]
    emit(
        "case_start",
        paths=[str(path) for path in paths],
        atom_counts=atom_counts,
        num_systems=batch.num_graphs,
        num_nodes=batch.num_nodes,
        batch_ptr=batch.batch_ptr.detach().cpu().tolist(),
    )

    cold_s, _ = _timed(device, build)
    edges = int(batch.neighbor_list.shape[0])
    emit("cold", seconds=cold_s, edges=edges)

    warmup_samples: list[float] = []
    for index in range(warmup):
        duration_s, _ = _timed(device, build)
        warmup_samples.append(duration_s)
        emit("warmup", index=index + 1, seconds=duration_s, edges=int(batch.neighbor_list.shape[0]))

    steady_samples: list[float] = []
    for index in range(steady):
        duration_s, _ = _timed(device, build)
        steady_samples.append(duration_s)
        emit("steady", index=index + 1, seconds=duration_s, edges=int(batch.neighbor_list.shape[0]))

    summary: dict[str, Any] = {
        "case": label,
        "status": "passed",
        "atom_counts": atom_counts,
        "num_systems": batch.num_graphs,
        "num_nodes": batch.num_nodes,
        "batch_ptr": batch.batch_ptr.detach().cpu().tolist(),
        "edges": int(batch.neighbor_list.shape[0]),
        "cold_s": cold_s,
        "warmup_s": warmup_samples,
        "steady_s": steady_samples,
        "steady_mean_s": mean(steady_samples),
        "steady_stdev_s": stdev(steady_samples) if len(steady_samples) > 1 else 0.0,
        "edges_per_atom": int(batch.neighbor_list.shape[0]) / max(batch.num_nodes, 1),
    }
    emit("case_summary", **summary)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--batch-root", type=Path, default=DEFAULT_SCALE_ROOTS["n92"])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument(
        "--scale-root",
        type=Path,
        action="append",
        dest="scale_roots",
        help="periodic CIF root; may be repeated. Defaults to 46/92/184/368/736 roots",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steady", type=int, default=3)
    parser.add_argument("--skip-scales", action="store_true")
    parser.add_argument("--skip-mixed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.warmup < 0 or args.steady < 1 or any(value < 1 for value in args.batch_sizes):
        raise ValueError("warmup must be non-negative; steady and batch sizes must be positive")
    if args.device == "cuda" and (not torch.version.hip or not torch.cuda.is_available()):
        raise RuntimeError("HCU device requested but HIP Torch is unavailable")

    device = torch.device(args.device)
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    process_start = time.perf_counter()
    load_s, model = _timed(
        device,
        lambda: MACEWrapper.from_checkpoint(str(checkpoint), device=device, dtype=torch.float32),
    )
    model.eval()
    config = model.model_config.neighbor_config
    print(
        json.dumps(
            {
                "stage": "model_loaded",
                "status": "passed",
                "device": args.device,
                "torch_version": torch.__version__,
                "torch_hip": torch.version.hip,
                "device_name": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
                "checkpoint": str(checkpoint),
                "cutoff": config.cutoff,
                "format": str(config.format),
                "model_load_s": load_s,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    results: list[dict[str, Any]] = []
    if not args.skip_scales:
        if args.scale_roots:
            scale_roots = [
                (path.expanduser().resolve().name, path) for path in args.scale_roots
            ]
        else:
            scale_roots = list(DEFAULT_SCALE_ROOTS.items())
        for label, root in scale_roots:
            paths = _files(root, 1)
            results.append(
                _run_case(
                    label,
                    paths,
                    model,
                    config,
                    device,
                    args.warmup,
                    args.steady,
                    process_start,
                )
            )

    for size in args.batch_sizes:
        paths = _files(args.batch_root, size)
        results.append(
            _run_case(
                f"batch_{size}", paths, model, config, device, args.warmup, args.steady, process_start
            )
        )

    if not args.skip_mixed:
        mixed_paths = [
            _files(DEFAULT_SCALE_ROOTS["n46"], 1)[0],
            _files(DEFAULT_SCALE_ROOTS["n92"], 1)[0],
            _files(DEFAULT_SCALE_ROOTS["n184"], 1)[0],
            _files(DEFAULT_SCALE_ROOTS["n368"], 1)[0],
        ]
        results.append(
            _run_case(
                "mixed_46_92_184_368",
                mixed_paths,
                model,
                config,
                device,
                args.warmup,
                args.steady,
                process_start,
            )
        )
    print(
        json.dumps(
            {
                "stage": "summary",
                "status": "passed",
                "device": args.device,
                "backend": "torch_reference",
                "cases": results,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
