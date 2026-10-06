# SPDX-License-Identifier: Apache-2.0
"""M5 three-path comparisons on identical evolving Batch snapshots."""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path

import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.base import NeighborListFormat
from nvalchemi.neighbors import compute_neighbors
from nvalchemiops.executor import clear_entrypoint_cache
import nvalchemiops._hip_neighbor_executor as executor


def load_baseline(path):
    spec = importlib.util.spec_from_file_location("m5_baseline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def signature(batch):
    edges = batch.neighbor_list.long()
    assert torch.equal(batch.batch_idx[edges[:, 0]], batch.batch_idx[edges[:, 1]])
    rows = torch.cat((edges, batch.neighbor_list_shifts.long()), -1)
    for column in reversed(range(5)):
        rows = rows[torch.argsort(rows[:, column], stable=True)]
    return rows


def summary(values):
    return {"median_ms": statistics.median(values), "mean_ms": statistics.mean(values),
            "std_ms": statistics.pstdev(values), "min_ms": min(values), "max_ms": max(values)}


class ThreePaths:
    def __init__(self, baseline):
        self.baseline = baseline.neighbor_list
        self.optimized = executor.neighbor_list

    def select(self, path):
        executor.neighbor_list = self.baseline if path == "hip_original" else self.optimized
        clear_entrypoint_cache()
        return "torch_reference" if path == "torch_reference" else "hip"

    def neighbor(self, batch, path, cutoff, capacity=256):
        compute_neighbors(batch, cutoff=cutoff, max_neighbors=capacity,
                          format=NeighborListFormat.COO, method="cell_list",
                          backend="torch_reference" if path == "torch_reference" else "hip")


PATHS = ("torch_reference", "hip_original", "hip_optimized")


def compare_snapshot(snapshot, paths, name, root, *, cutoff=6.0, rounds=3, samples=3, warmup=1, model=None):
    outputs, first, memory = {}, {}, {}
    calls = {}
    for path in PATHS:
        batch = snapshot.clone()
        calls[path] = batch
        paths.select(path)
        torch.cuda.synchronize()
        start = time.perf_counter()
        paths.neighbor(batch, path, cutoff)
        torch.cuda.synchronize()
        first[path] = (time.perf_counter() - start) * 1000
        outputs[path] = signature(batch)
        if model is not None:
            from mace_mpa0_cpu_baseline import compare
            values = {k: v.detach() for k, v in model(batch).items() if k in ("energy", "forces", "stress")}
            if path == PATHS[0]:
                expected = values
            else:
                compare(values, expected, snapshot.positions.dtype)
    assert all(torch.equal(outputs[path], outputs[PATHS[0]]) for path in PATHS)
    # Correctness and canonicalization are outside measured samples.
    del outputs
    modes = ("neighbor",) if model is None else ("neighbor", "model", "neighbor_model")
    measured = {}
    for mode in modes:
        raw = {path: [] for path in PATHS}
        for path in PATHS:
            paths.select(path)
            batch = calls[path]
            def call():
                if mode != "model":
                    paths.neighbor(batch, path, cutoff)
                if mode != "neighbor":
                    model(batch)
            for _ in range(warmup):
                call()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
            call()
            torch.cuda.synchronize()
            memory[f"{path}:{mode}"] = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                      "incremental_peak_bytes": torch.cuda.max_memory_allocated() - before}
        for run in range(rounds):
            values = {path: [] for path in PATHS}
            for iteration in range(samples):
                offset = (run + iteration) % len(PATHS)
                for path in PATHS[offset:] + PATHS[:offset]:
                    paths.select(path)
                    batch = calls[path]
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    if mode != "model":
                        paths.neighbor(batch, path, cutoff)
                    if mode != "neighbor":
                        model(batch)
                    torch.cuda.synchronize()
                    values[path].append((time.perf_counter() - start) * 1000)
            for path in PATHS:
                raw[path].append(values[path])
                (root / f"{name}-{mode}-{path}-round{run}-ms.txt").write_text("".join(f"{v:.9f}\n" for v in values[path]))
        measured[mode] = {path: {"rounds": raw[path], **summary([v for group in raw[path] for v in group])}
                          for path in PATHS}
    paths.select("hip_optimized")
    batch = calls["hip_optimized"]
    report = {"name": name, "N": snapshot.num_nodes, "B": snapshot.num_graphs,
              "ptr": snapshot.batch_ptr.cpu().tolist(), "source_ids": snapshot.source_index.flatten().cpu().tolist(),
              "edges": batch.neighbor_list.shape[0], "max_neighbors": int(torch.bincount(batch.neighbor_list[:, 0].long(), minlength=batch.num_nodes).max()),
              "volume_range_A3": [float(torch.linalg.det(snapshot.cell).min()), float(torch.linalg.det(snapshot.cell).max())],
              "first_api_ms": first, "memory": memory, "timings": measured, "topology_equal": True}
    print(name, snapshot.num_nodes, snapshot.num_graphs,
          {p: round(measured["neighbor"][p]["median_ms"], 3) for p in PATHS}, flush=True)
    return report


def make_systems(sizes, dtype, *, start_id=0, seed=20261006):
    generator = torch.Generator(device="cuda").manual_seed(seed + start_id)
    systems = []
    for i, n in enumerate(sizes):
        side = (n / .04) ** (1 / 3)
        cell = torch.tensor([[1, 0, 0], [.04, .99, 0], [-.03, .02, 1.01]], dtype=dtype, device="cuda") * side
        r = torch.rand(n, 3, device="cuda", dtype=dtype, generator=generator) @ cell
        data = AtomicData(positions=r, atomic_numbers=torch.ones(n, device="cuda", dtype=torch.long),
                          cell=cell[None], pbc=torch.ones(1, 3, device="cuda", dtype=torch.bool))
        data.add_system_property("source_index", torch.tensor([[start_id + i]], device="cuda"))
        systems.append(data)
    return systems


def evolve(batch, transform):
    result = batch.clone()
    matrix = batch.cell.new_tensor(transform)
    result.positions = batch.positions @ matrix
    result.cell = batch.cell @ matrix
    return result


def snapshots(mean_size, count, dtype, *, seed=20261006):
    sizes = [int(mean_size * (.8 if i % 2 == 0 else 1.2)) for i in range(count)]
    batch = Batch.from_data_list(make_systems(sizes, dtype, seed=seed), device="cuda")
    yield "initial", batch
    batch = evolve(batch, [[.94, 0, 0], [0, .94, 0], [0, 0, .94]])
    yield "compression", batch
    batch = evolve(batch, [[1.08, 0, 0], [0, 1.08, 0], [0, 0, 1.08]])
    yield "expansion", batch
    batch = evolve(batch, [[1, .11, 0], [.04, 1, -.06], [0, .03, 1]])
    yield "shear", batch
    arrivals = make_systems([int(mean_size * v) for v in (.7, .9, 1.1)], dtype, start_id=count, seed=seed)
    # Geometry-only fixture of actual exit/refill layout; dynamics lifecycle
    # is separately validated by the public M3 gate.
    batch = Batch.from_data_list(batch.to_data_list()[2:] + arrivals, device="cuda")
    yield "exit_refill", batch
    batch = evolve(batch, [[.98, .03, 0], [0, .98, 0], [0, 0, .98]])
    yield "refill_compression_shear", batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case", choices=("small", "many", "medium", "few"), default="many")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    assert torch.version.hip and torch.cuda.is_available()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = load_baseline(args.baseline)
    paths = ThreePaths(baseline)
    sizes = {"small": (46, 32), "many": (100, 200), "medium": (500, 40), "few": (2000, 10)}
    mean_size, count = sizes[args.case]
    evidence = []
    for name, snapshot in snapshots(mean_size, count, getattr(torch, args.dtype), seed=args.seed):
        if args.profile_only:
            paths.select("hip_optimized")
            paths.neighbor(snapshot, "hip_optimized", 6.0)
            torch.cuda.synchronize()
            break
        report = compare_snapshot(snapshot, paths, name, args.output_dir, rounds=args.rounds, samples=args.samples)
        evidence.append(report)
        (args.output_dir / "summary.json").write_text(json.dumps({"status": "running", "case": args.case,
            "dtype": args.dtype, "seed": args.seed, "snapshots": evidence}, indent=2) + "\n")
    (args.output_dir / "summary.json").write_text(json.dumps({"status": "passed", "case": args.case,
        "dtype": args.dtype, "seed": args.seed, "snapshots": evidence, "environment": {"torch": torch.__version__,
        "hip": torch.version.hip, "device": torch.cuda.get_device_name()}}, indent=2) + "\n")


if __name__ == "__main__":
    main()
