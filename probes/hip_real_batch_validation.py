# SPDX-License-Identifier: Apache-2.0
"""M7 controlled two-backend real MPA-0 variable-cell validation."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
import time
from pathlib import Path

import torch
from ase.io import read

from hip_batch_metadata_benchmark import signature, summary
from hip_variable_cell_mace_workflow import optimizer as make_optimizer, prime
from mace_mpa0_cpu_baseline import CHECKPOINT_SHA256, compare
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import FIRE2VariableCell, FIREVariableCell, FusedStage
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.sampler import SizeAwareSampler
from nvalchemi.dynamics.sinks import HostMemory
from nvalchemi.neighbors import compute_neighbors
from nvalchemi.models.mace import MACEWrapper

PATHS = ("torch_reference", "hip")


def optimizer(*args, **kwargs):
    dynamics, hook = make_optimizer(*args, **kwargs)
    hook.method = "cell_list"
    return dynamics, hook


def neighbor(data, model, backend):
    compute_neighbors(data, config=model.model_config.neighbor_config,
                      method="cell_list", backend=backend)


def values(model, data):
    result = {k: v.detach().clone() for k, v in model(data).items()
              if k in ("energy", "forces", "stress")}
    result["energy"] = result["energy"].reshape(-1)
    for value in result.values():
        assert value.dtype == data.positions.dtype and value.device == data.positions.device
        assert bool(torch.isfinite(value).all())
    return result


def timed(call):
    torch.cuda.synchronize()
    start = time.perf_counter()
    call()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000


def state_check(actual, expected, dynamics, reference):
    dtype = actual.positions.dtype
    rtol, atol = (2e-8, 2e-9) if dtype == torch.float64 else (2e-4, 2e-5)
    assert torch.equal(signature(actual), signature(expected))
    result = compare({k: getattr(actual, k).reshape(-1) for k in ("energy", "forces", "stress")},
                     {k: getattr(expected, k).reshape(-1) for k in ("energy", "forces", "stress")}, dtype)
    pairs = [(k, getattr(actual, k), getattr(expected, k)) for k in dynamics._mutable_fields]
    pairs += [("state." + k, v, getattr(reference._state, k)) for k, v in dynamics._state]
    result["state_max_abs_errors"] = {}
    for key, a, b in pairs:
        if a.is_floating_point():
            torch.testing.assert_close(a, b, rtol=rtol, atol=atol, msg=lambda message: key + ": " + message)
            assert bool(torch.isfinite(a).all())
            result["state_max_abs_errors"][key] = float((a - b).abs().max()) if a.numel() else 0
        else:
            assert torch.equal(a, b), key
    assert dynamics.step_count == reference.step_count
    assert torch.equal(actual.batch_ptr, expected.batch_ptr)
    assert torch.equal(actual.source_index, expected.source_index)
    return result


def memory():
    return {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved()}


def snapshots(model, frame, root, name):
    batches = {path: frame.clone() for path in PATHS}
    outputs, first, topology = {}, {}, {}
    for path, data in batches.items():
        first[path] = timed(lambda: neighbor(data, model, path))
        topology[path] = signature(data)
        outputs[path] = values(model, data)
    assert torch.equal(topology["hip"], topology["torch_reference"])
    parity = compare(outputs["hip"], outputs["torch_reference"], frame.positions.dtype)
    del topology, outputs
    modes, mem = {}, {}
    for mode in ("neighbor", "model", "neighbor_model"):
        def call(path):
            if mode != "model":
                neighbor(batches[path], model, path)
            if mode != "neighbor":
                model(batches[path])
        for path in PATHS:
            call(path)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
            call(path)
            torch.cuda.synchronize()
            mem[path + ":" + mode] = {**memory(), "allocated_before_bytes": before,
                "incremental_peak_bytes": torch.cuda.max_memory_allocated() - before}
        raw = {path: [] for path in PATHS}
        for run in range(3):
            groups = {path: [] for path in PATHS}
            for sample in range(2):
                order = PATHS if (run + sample) % 2 == 0 else PATHS[::-1]
                for path in order:
                    groups[path].append(timed(lambda: call(path)))
            for path in PATHS:
                raw[path].append(groups[path])
                (root / f"{name}-{mode}-{path}-round{run}-ms.txt").write_text(
                    "".join(f"{v:.9f}\n" for v in groups[path]))
        modes[mode] = {path: {"rounds": raw[path], **summary([v for g in raw[path] for v in g])}
                       for path in PATHS}
    return {"name": name, "N": frame.num_nodes, "B": frame.num_graphs,
            "edges": batches["hip"].neighbor_list.shape[0], "topology_equal": True,
            "same_geometry": parity, "first_neighbor_api_ms": first, "memory": mem, "timings": modes}


def trajectories(model, initial, cls, root):
    checks, raw, mem, final = [], {p: [] for p in PATHS}, {}, None
    # A separate three-step trajectory warms every measured step/control branch.
    for path in PATHS:
        warm = initial.clone()
        opt, _ = optimizer(cls, model, path)
        prime(opt, warm)
        for _ in range(3):
            opt.step(warm)
        torch.cuda.synchronize()
        del warm, opt
    for run in range(3):
        batches = {p: initial.clone() for p in PATHS}
        opts = {p: optimizer(cls, model, p)[0] for p in PATHS}
        for path in PATHS:
            prime(opts[path], batches[path])
        checks.append(state_check(batches["hip"], batches["torch_reference"], opts["hip"], opts["torch_reference"]))
        group = {p: [] for p in PATHS}
        for step in range(3):
            order = PATHS if (run + step) % 2 == 0 else PATHS[::-1]
            for path in order:
                torch.cuda.reset_peak_memory_stats()
                before = torch.cuda.memory_allocated()
                group[path].append(timed(lambda: opts[path].step(batches[path])))
                mem[f"round{run}:step{step}:{path}"] = {**memory(), "allocated_before_bytes": before,
                    "incremental_peak_bytes": torch.cuda.max_memory_allocated() - before}
            try:
                checks.append(state_check(batches["hip"], batches["torch_reference"], opts["hip"], opts["torch_reference"]))
            except AssertionError:
                torch.save({"round": run, "step": step + 1,
                    "paths": {p: {k: getattr(batches[p], k).detach().cpu() for k in
                        ("positions", "cell", "forces", "velocities", "energy", "stress", "source_index", "batch_ptr", "neighbor_list", "neighbor_list_shifts")} for p in PATHS}}, root / "failure-snapshot.pt")
                raise
            print(f"trajectory round={run} step={step + 1} passed", flush=True)
        for path in PATHS:
            raw[path].append(group[path])
            (root / f"complete_step-{path}-round{run}-ms.txt").write_text(
                "".join(f"{v:.9f}\n" for v in group[path]))
        final = batches["hip"].clone()
        del batches, opts
    movement = {k: float((getattr(final, k) - getattr(initial, k)).abs().max()) for k in ("positions", "cell")}
    assert all(v > 0 for v in movement.values())
    return {"steps_per_round": 3, "rounds": 3, "checks": checks, "memory": mem,
            "movement_A": movement, "timings": {p: {"rounds": raw[p], **summary([v for g in raw[p] for v in g])}
                                                for p in PATHS}}, final


class Dataset:
    def __init__(self, systems):
        self.systems = systems

    def __len__(self):
        return len(self.systems)

    def get_metadata(self, index):
        return self.systems[index].num_nodes, 0

    def __getitem__(self, index):
        return self.systems[index].clone(), {}


def lifecycle(model, systems, cls, max_atoms):
    reports, stored_batches = {}, {}
    budgets = {i: 1 + i % 3 for i in range(len(systems))}
    for path in PATHS:
        counts, events = {}, []
        class Priming:
            stage = DynamicsStage.BEFORE_PRE_UPDATE
            frequency = 1
            def __call__(self, ctx, stage):
                data = ctx.batch
                ref = data.clone()
                neighbor(ref, model, "torch_reference")
                assert torch.equal(signature(data), signature(ref))
                expected = values(model, ref)
                actual = {k: getattr(data, k).reshape_as(v) for k, v in expected.items()}
                parity = compare(actual, expected, data.positions.dtype)
                events.append({"source_ids": data.source_index.flatten().cpu().tolist(),
                               "system_ids": data.system_id.flatten().cpu().tolist(),
                               "ptr": data.batch_ptr.cpu().tolist(), "parity": parity})
        class Budget:
            stage = DynamicsStage.AFTER_STEP
            frequency = 1
            def __call__(self, ctx, stage):
                for row, source in enumerate(ctx.batch.source_index.flatten().cpu().tolist()):
                    counts[source] = counts.get(source, 0) + 1
                    if counts[source] >= budgets[source]:
                        ctx.batch.status[row] = 2
        opt, hook = optimizer(cls, model, path, extra_hooks=[Priming(), Budget()], shared_compute=True)
        sampler = SizeAwareSampler(Dataset(systems), max_atoms=max_atoms)
        sink = HostMemory(capacity=len(systems))
        stage = FusedStage(sub_stages=[(0, opt)], hooks=[hook], sampler=sampler,
                           sinks=[sink], refill_frequency=1, device_type="cuda")
        data = sampler.build_initial_batch().to("cuda")
        opt._ensure_state_initialized(data)
        opt._call_hooks(DynamicsStage.BEFORE_STEP, data)
        stage.run(batch=data, n_steps=12)
        stored = sink.read()
        assert len(sink) == len(systems) and counts == budgets
        assert sorted(stored.source_index.flatten().tolist()) == list(budgets)
        assert sorted(stored.system_id.flatten().tolist()) == list(budgets)
        assert bool((stored.status == 2).all())
        seen_initial = set(events[0]["source_ids"])
        assert any(set(e["source_ids"]) - seen_initial for e in events[1:]), "no arrivals"
        reports[path] = {"counts": counts, "events": events, "steps": stage.step_count,
                         "budget_exit": True, "arrival_priming": True}
        stored_batches[path] = {int(d.source_index.item()): d for d in stored.to_data_list()}
        del stage, data, opt
    assert reports["hip"]["counts"] == reports["torch_reference"]["counts"]
    assert [{k: e[k] for k in ("source_ids", "system_ids", "ptr")} for e in reports["hip"]["events"]] == [
        {k: e[k] for k in ("source_ids", "system_ids", "ptr")} for e in reports["torch_reference"]["events"]]
    rtol, atol = (2e-8, 2e-9) if systems[0].positions.dtype == torch.float64 else (2e-4, 2e-5)
    moving = 0
    for source in budgets:
        a, b = stored_batches["hip"][source], stored_batches["torch_reference"][source]
        for key in ("positions", "cell", "velocities", "energy", "forces", "stress"):
            torch.testing.assert_close(getattr(a, key), getattr(b, key), rtol=rtol, atol=atol)
        moving += int(not torch.equal(a.cell, systems[source].cell))
    assert moving == len(systems)
    return {"paths": reports, "final_state_parity": True, "moving_cells": moving}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--atoms", type=int, choices=(1000, 2000), required=True)
    parser.add_argument("--dtype", choices=("float64", "float32"), required=True)
    parser.add_argument("--optimizer", choices=("fire", "fire2"), required=True)
    parser.add_argument("--lifecycle", action="store_true")
    args = parser.parse_args()
    assert torch.version.hip and torch.cuda.is_available()
    assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == CHECKPOINT_SHA256
    dtype = getattr(torch, args.dtype)
    torch.set_default_dtype(dtype)
    model = MACEWrapper.from_checkpoint(args.checkpoint, device=torch.device("cuda"), dtype=dtype,
                                        enable_cueq=False, compile_model=False)
    model.set_config("active_outputs", {"energy", "forces", "stress"})
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    systems, sources, count = [], [], 0
    manifest = json.loads(args.manifest.read_text())
    for i, item in enumerate(manifest):
        path = args.data_root / item["filename"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        a = read(path)
        assert len(a) == item["atoms"] and a.pbc.all() and a.get_volume() > 0
        data = AtomicData.from_atoms(a, dtype=dtype)
        data.velocities, data.forces = torch.zeros_like(data.positions), torch.zeros_like(data.positions)
        data.energy, data.stress = torch.zeros(1, 1, dtype=dtype), torch.zeros(1, 3, 3, dtype=dtype)
        data.add_system_property("source_index", torch.tensor([[i]], dtype=torch.long))
        systems.append(data)
        sources.append(item)
        count += len(a)
        if count >= args.atoms:
            break
    assert count >= args.atoms
    args.output_dir.mkdir(parents=True, exist_ok=True)
    initial = Batch.from_data_list(systems, device="cuda")
    cls = FIRE2VariableCell if args.optimizer == "fire2" else FIREVariableCell
    report = {"status": "running", "dtype": args.dtype, "optimizer": cls.__name__,
              "N": count, "B": len(systems), "sources": sources, "checkpoint_sha256": CHECKPOINT_SHA256,
              "environment": {"torch": torch.__version__, "hip": torch.version.hip,
                "device": torch.cuda.get_device_name(), "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory}}
    def save():
        (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    save()
    report["snapshots"] = [snapshots(model, initial, args.output_dir, "initial")]
    save()
    print("initial same-geometry and timing passed", count, len(systems), flush=True)
    report["trajectory"], final = trajectories(model, initial, cls, args.output_dir)
    save()
    report["snapshots"].append(snapshots(model, final, args.output_dir, "step3"))
    save()
    if args.lifecycle:
        # Additional real CIF arrivals from the same manifest, keeping batch near requested size.
        extra = []
        for i, item in enumerate(manifest[len(systems):len(systems) + 8], start=len(systems)):
            p = args.data_root / item["filename"]
            assert hashlib.sha256(p.read_bytes()).hexdigest() == item["sha256"]
            d = AtomicData.from_atoms(read(p), dtype=dtype)
            d.velocities, d.forces = torch.zeros_like(d.positions), torch.zeros_like(d.positions)
            d.energy, d.stress = torch.zeros(1, 1, dtype=dtype), torch.zeros(1, 3, 3, dtype=dtype)
            d.add_system_property("source_index", torch.tensor([[i]], dtype=torch.long))
            extra.append(d)
        assert extra
        report["arrival_sources"] = manifest[len(systems):len(systems) + len(extra)]
        report["lifecycle"] = lifecycle(model, systems + extra, cls, count)
    report["status"] = "passed"
    save()
    print("passed", args.output_dir, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if "--output-dir" in sys.argv:
            root = Path(sys.argv[sys.argv.index("--output-dir") + 1])
            target = root / "summary.json"
            if target.exists():
                report = json.loads(target.read_text())
                report.update(status="failed", error=str(error), traceback=traceback.format_exc())
                target.write_text(json.dumps(report, indent=2) + "\n")
        raise
