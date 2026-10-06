# SPDX-License-Identifier: Apache-2.0
"""Same-input M4 wall timings; stage synchronization is diagnostic only."""
import argparse
import json
import statistics
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch

from framework_neighbor_hip_wallclock import _make_data, _wall_ms
from hip_coo_mace_snapshot import evaluate
from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.base import NeighborListFormat
from nvalchemi.neighbors import compute_neighbors


def signature(data):
    rows = torch.cat((data.neighbor_list.long(), data.neighbor_list_shifts.long()), -1)
    for column in reversed(range(5)):
        rows = rows[torch.argsort(rows[:, column], stable=True)]
    return rows


def summarize(samples):
    return {"median_ms": statistics.median(samples), "mean_ms": statistics.mean(samples),
            "std_ms": statistics.pstdev(samples), "min_ms": min(samples), "max_ms": max(samples)}


def stage_times(call):
    import nvalchemiops._hip_neighbor_executor as executor
    import nvalchemiops._hip_batch_cell_query as query
    import nvalchemiops._hip_batch_query_materialize as materialize
    stages = {}

    def wrap(name, fn):
        def wrapped(*args, **kwargs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = fn(*args, **kwargs)
            torch.cuda.synchronize()
            stages.setdefault(name, []).append((time.perf_counter() - start) * 1000)
            return result
        return wrapped

    targets = [(executor, "_current_cell_metadata"), (executor, "_allocate_outputs"),
               (executor, "cell_starts_hip_workspace_size"),
               (executor, "allocate_batch_query_topology_composite_workspace"),
               (executor, "_result"), (executor._TrustedBatchCellBuildPlan, "initialize"),
               (query, "batch_query_cell_list_hip_into"),
               (materialize, "materialize_batch_query_topology_composite_hip_into_with_workspace")]
    with ExitStack() as stack:
        for obj, name in targets:
            stack.enter_context(patch.object(obj, name, wrap(name, getattr(obj, name))))
        _wall_ms(call)
    return stages


def measure(label, call, root, rounds, warmup, samples, *, first_api_ms=None):
    cold = _wall_ms(call) if first_api_ms is None else first_api_ms
    raw = []
    for run in range(rounds):
        for _ in range(warmup):
            call()
        values = [_wall_ms(call) for _ in range(samples)]
        (root / f"{label}-round{run}-ms.txt").write_text("".join(f"{v:.9f}\n" for v in values))
        raw.append({"samples_ms": values, **summarize(values)})
    return {"label": label, "first_api_ms": cold, "rounds": raw, "combined": summarize([
        value for run in raw for value in run["samples_ms"]])}


def measure_pair(label, call, root, rounds, warmup, samples, baseline, candidate):
    import nvalchemiops._hip_neighbor_executor as executor
    from nvalchemiops.executor import clear_entrypoint_cache
    functions = {"baseline": baseline, "candidate": candidate}
    collected = {mode: [] for mode in functions}
    def select(mode):
        executor.neighbor_list = functions[mode]
        clear_entrypoint_cache()
    for mode in functions:
        select(mode)
        for _ in range(warmup):
            call()
    for run in range(rounds):
        raw = {mode: [] for mode in functions}
        for iteration in range(samples):
            order = ("baseline", "candidate") if (run + iteration) % 2 == 0 else ("candidate", "baseline")
            for mode in order:
                select(mode)  # Switching overhead is outside the timed API call.
                raw[mode].append(_wall_ms(call))
        for mode, values in raw.items():
            (root / f"{label}-{mode}-round{run}-ms.txt").write_text("".join(f"{v:.9f}\n" for v in values))
            collected[mode].append({"samples_ms": values, **summarize(values)})
    select("candidate")
    return {"label": label, "paired": {mode: {"rounds": runs, "combined": summarize([
        v for r in runs for v in r["samples_ms"]])} for mode, runs in collected.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--executor-source", type=Path, help="load saved baseline executor for paired measurements")
    parser.add_argument("--paired-baseline", type=Path, help="alternate saved baseline and candidate in one process")
    parser.add_argument("--candidate-executor-source", type=Path, help="replay an archived experiment in this probe only")
    parser.add_argument("--workspace-source", type=Path, help="scratch module for archived experiment replay")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args()
    if min(args.rounds, args.samples) <= 0 or args.warmup < 0:
        parser.error("rounds/samples must be positive and warmup nonnegative")
    assert torch.version.hip and torch.cuda.is_available()
    if args.workspace_source or args.candidate_executor_source:
        import importlib.util
        import sys
        import nvalchemiops._hip_neighbor_executor as executor
        if args.workspace_source:
            name = "nvalchemiops._hip_neighbor_workspace"
            spec = importlib.util.spec_from_file_location(name, args.workspace_source)
            scratch_module = importlib.util.module_from_spec(spec)
            sys.modules[name] = scratch_module
            spec.loader.exec_module(scratch_module)
        if args.candidate_executor_source:
            spec = importlib.util.spec_from_file_location("m4_candidate_executor", args.candidate_executor_source)
            candidate_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(candidate_module)
            executor.neighbor_list = candidate_module.neighbor_list
    candidate = baseline = None
    if args.executor_source or args.paired_baseline:
        import importlib.util
        import nvalchemiops._hip_neighbor_executor as executor
        spec = importlib.util.spec_from_file_location("m4_saved_executor", args.executor_source or args.paired_baseline)
        saved = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(saved)
        candidate, baseline = executor.neighbor_list, saved.neighbor_list
        if args.executor_source:
            executor.neighbor_list = baseline
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for dtype in (torch.float32, torch.float64):
        for n, b in ((46, 32), (92, 64)):
            data = _make_data(n, b, torch.device("cuda"), dtype)
            for item in data:
                item.pbc.fill_(True)
            batch = Batch.from_data_list(data)
            kwargs = dict(cutoff=3.0, format=NeighborListFormat.COO, max_neighbors=128, method="cell_list")
            call = lambda: compute_neighbors(batch, backend="hip", **kwargs)
            label = f"{n}x{b}-{str(dtype).split('.')[-1]}-coo"
            if args.profile_only:
                for _ in range(2):
                    call()
                torch.cuda.synchronize()
                return
            first_api_ms = _wall_ms(call)
            oracle = Batch.from_data_list(data)
            compute_neighbors(oracle, backend="torch_reference", **kwargs)
            assert torch.equal(signature(batch), signature(oracle))
            if args.paired_baseline:
                report = measure_pair(label, call, args.output_dir, args.rounds, args.warmup, args.samples,
                                      baseline, candidate)
                report["first_api_ms"] = first_api_ms
            else:
                report = measure(label, call, args.output_dir, args.rounds, args.warmup, args.samples,
                                 first_api_ms=first_api_ms)
                report["diagnostic_stages_ms"] = {} if args.executor_source else stage_times(call)
            results.append(report)
            summary = report.get("combined") or {mode: values["combined"] for mode, values in report["paired"].items()}
            print(label, summary, flush=True)
    if args.checkpoint:
        import hashlib
        from ase.io import read
        from mace_mpa0_cpu_baseline import CHECKPOINT_SHA256, SAMPLES
        from nvalchemi.models.mace import MACEWrapper
        assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == CHECKPOINT_SHA256
        torch.set_default_dtype(torch.float32)
        model = MACEWrapper.from_checkpoint(args.checkpoint, device=torch.device("cuda"), dtype=torch.float32,
                                            enable_cueq=False, compile_model=False)
        model.set_config("active_outputs", {"energy", "forces", "stress"})
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        batch = Batch.from_data_list([AtomicData.from_atoms(read(args.data_root / f), dtype=torch.float32)
                                     for f in SAMPLES], device="cuda")
        expected, reference_edges, _ = evaluate(model, batch, backend="torch_reference")
        actual, edges, _ = evaluate(model, batch, backend="hip")
        assert torch.equal(edges, reference_edges)
        from mace_mpa0_cpu_baseline import compare
        compare(actual, expected, torch.float32)
        def neighbor():
            compute_neighbors(batch, config=model.model_config.neighbor_config, backend="hip",
                              method="cell_list", max_neighbors=128)
        neighbor()
        def combined():
            neighbor()
            model(batch)
        for label, call in (("mpa0-neighbor", neighbor), ("mpa0-model", lambda: model(batch)),
                            ("mpa0-neighbor-model", combined)):
            if args.paired_baseline:
                results.append(measure_pair(label, call, args.output_dir, args.rounds, args.warmup, args.samples,
                                            baseline, candidate))
            else:
                results.append(measure(label, call, args.output_dir, args.rounds, args.warmup, args.samples))
            summary = results[-1].get("combined") or {mode: values["combined"] for mode, values in results[-1]["paired"].items()}
            print(label, summary, flush=True)
    (args.output_dir / "summary.json").write_text(json.dumps({"status": "passed", "results": results,
        "environment": {"torch": torch.__version__, "hip": torch.version.hip,
                        "device": torch.cuda.get_device_name()}, "args": str(args)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
