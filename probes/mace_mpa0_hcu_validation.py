# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
"""Bounded real MPA-0 CPU/HCU parity and native M4 relaxation acceptance.

This evidence harness reuses the application and public Hooks. It adds no
optimizer/filter/backend implementation. Each invocation uses one dtype.
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

# Measure expensive imports as startup, separately from the validation run.
PROBE_STARTED = time.perf_counter()

import torch
from ase.io import read

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.models.mace import MACEWrapper
from nvalchemi.neighbors import compute_neighbors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples import relax_mace_mpa0 as application


def load_model(checkpoint, device, dtype):
    model = MACEWrapper.from_checkpoint(
        checkpoint, device=device, dtype=dtype, enable_cueq=False, compile_model=False
    )
    model.set_config("active_outputs", {"energy", "forces", "stress"})
    model.eval()
    return model


def batch_from_paths(paths, device, dtype):
    return Batch.from_data_list(
        [AtomicData.from_atoms(read(p, format="cif"), dtype=dtype) for p in paths]
    ).to(device)


def check_edges(batch):
    edges = batch.neighbor_list.long()
    assert torch.equal(batch.batch_idx[edges[:, 0]], batch.batch_idx[edges[:, 1]])
    signature = torch.cat((edges, batch.neighbor_list_shifts.long()), dim=1)
    for col in reversed(range(signature.shape[1])):
        signature = signature[torch.argsort(signature[:, col], stable=True)]
    return signature.detach().cpu()


def evaluate(model, batch):
    compute_neighbors(
        batch, config=model.model_config.neighbor_config, backend="torch_reference"
    )
    signature = check_edges(batch)
    outputs = model(batch)
    shapes = {
        "energy": (batch.num_graphs,),
        "forces": (batch.num_nodes, 3),
        "stress": (batch.num_graphs, 3, 3),
    }
    values = {}
    for key, shape in shapes.items():
        if key != "energy":
            assert outputs[key].shape == shape
        value = outputs[key].reshape(shape)
        assert value.dtype == batch.positions.dtype
        assert value.device == batch.positions.device
        assert torch.isfinite(value).all()
        values[key] = value.detach().cpu()
    return values, signature


def compare(actual, expected, dtype):
    rtol = 2e-8 if dtype == torch.float64 else 2e-4
    atols = (
        {"energy": 2e-7, "forces": 2e-7, "stress": 2e-9}
        if dtype == torch.float64
        else {"energy": 2e-3, "forces": 2e-4, "stress": 2e-6}
    )
    errors = {}
    for key, atol in atols.items():
        torch.testing.assert_close(actual[key], expected[key], rtol=rtol, atol=atol)
        errors[key] = float((actual[key] - expected[key]).abs().max())
    return {"rtol": rtol, "atol": atols, "max_abs_errors": errors}


def metrics(values):
    return {
        "energy_eV": float(values["energy"].item()),
        "fmax_eV_A": float(values["forces"].norm(dim=-1).max()),
        "max_abs_stress_GPa": float(values["stress"].abs().max()) * 160.21766208,
    }


def verify_exports(records, checkpoint, dtype, device):
    cpu = load_model(checkpoint, torch.device("cpu"), dtype)
    hcu = load_model(checkpoint, device, dtype)
    result = []
    for record in records:
        assert record["status"] in ("converged", "max_steps", "wall_timeout")
        assert application.sha256(record["source"]) == record["input_sha256"]
        assert application.sha256(record["final_cif"]) == record["final_cif_sha256"]
        paths = [record["final_cif"]]
        c, ce = evaluate(cpu, batch_from_paths(paths, "cpu", dtype))
        h, he = evaluate(hcu, batch_from_paths(paths, device, dtype))
        assert torch.equal(ce, he)
        parity = compare(h, c, dtype)
        measured = metrics(h)
        cpu_measured = metrics(c)
        scalar_atols = (
            (2e-7, 2e-7, 2e-9 * 160.21766208)
            if dtype == torch.float64
            else (2e-3, 2e-4, 2e-6 * 160.21766208)
        )
        for (key, value), atol in zip(measured.items(), scalar_atols, strict=True):
            assert abs(value - record[key]) <= atol, (key, value, record[key])
        for values in (measured, cpu_measured):
            passed = (
                values["fmax_eV_A"] <= record["fmax_tolerance_eV_A"]
                and values["max_abs_stress_GPa"] <= record["stress_tolerance_GPa"]
            )
            if record["status"] == "converged":
                assert passed, (record["source"], values)
        result.append(
            {
                "source_index": record["source_index"],
                "cpu": cpu_measured,
                "hcu": measured,
                "parity": parity,
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("parity", "relax"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--optimizer", choices=("fire", "fire2"), default="fire2")
    parser.add_argument("--max-batch-size", type=int, default=2)
    parser.add_argument("--max-atoms", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--max-wall-seconds", type=float, default=180)
    args = parser.parse_args()
    with args.checkpoint.open("rb") as stream:
        assert (
            hashlib.file_digest(stream, "sha256").hexdigest()
            == application.MODEL_SHA256
        )
    dtype, device = getattr(torch, args.dtype), torch.device(args.device)
    assert device.type == "cuda" and torch.version.hip and torch.cuda.is_available()
    torch.manual_seed(0)
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    evidence = {
        "mode": args.mode,
        "dtype": str(dtype),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "architecture": torch.cuda.get_device_properties(device).gcnArchName,
        "torch_version": str(torch.__version__),
        "hip_version": torch.version.hip,
        "model_sha256": application.MODEL_SHA256,
        "seed": 0,
        "startup_seconds": started - PROBE_STARTED,
        "inputs": [
            {
                "name": p.name,
                "sha256": application.sha256(p),
                "natoms": len(read(p, format="cif")),
            }
            for p in args.inputs
        ],
    }
    try:
        if args.mode == "parity":
            args.output_dir.mkdir(parents=True, exist_ok=False)
            cases = []
            for target in (torch.device("cpu"), device):
                before = time.perf_counter()
                model = load_model(args.checkpoint, target, dtype)
                load_seconds = time.perf_counter() - before
                batch = batch_from_paths(args.inputs, target, dtype)
                evaluate(model, batch)  # One warmup, not included in warm timing.
                if target.type == "cuda":
                    torch.cuda.synchronize(device)
                before = time.perf_counter()
                for _ in range(3):
                    values, edges = evaluate(model, batch)
                if target.type == "cuda":
                    torch.cuda.synchronize(device)
                cases.append((values, edges))
                evidence[str(target)] = {
                    "load_seconds": load_seconds,
                    "warm_evaluation_seconds_mean": (time.perf_counter() - before) / 3,
                    "scope": "neighbor rebuild + model + evidence transfer; 1 warmup/3 timed",
                }
            assert torch.equal(cases[0][1], cases[1][1])
            evidence["parity"] = compare(cases[1][0], cases[0][0], dtype)
        else:
            trace = []

            class Audit:
                stage = DynamicsStage.AFTER_COMPUTE
                frequency = 1

                def __call__(self, ctx, stage):
                    batch = ctx.batch
                    for key in ("positions", "cell", "energy", "forces", "stress"):
                        value = getattr(batch, key)
                        assert value.device == device and value.dtype == dtype
                        assert torch.isfinite(value).all()
                    check_edges(batch)
                    sources = batch.source_index.flatten().tolist()
                    ids = batch.system_id.flatten().tolist()
                    natoms = batch.batch_ptr.diff().tolist()
                    assert batch.num_graphs <= args.max_batch_size
                    assert batch.num_nodes <= args.max_atoms
                    assert len(set(sources)) == len(sources)
                    event = {"sources": sources, "ids": ids, "natoms": natoms}
                    if not trace or event != trace[-1]:
                        trace.append(event)

            class AuditedStage(application.FusedStage):
                def __init__(self, *a, **kw):
                    super().__init__(*a, **kw)
                    self.register_hook(Audit())

            with patch.object(application, "FusedStage", AuditedStage):
                records = application.relax(
                    args.inputs,
                    args.checkpoint,
                    args.output_dir,
                    optimizer=args.optimizer,
                    device=device,
                    dtype=dtype,
                    max_batch_size=args.max_batch_size,
                    max_atoms=args.max_atoms,
                    max_steps=args.max_steps,
                    max_wall_seconds=args.max_wall_seconds,
                )
            torch.cuda.synchronize(device)
            evidence["relaxation_seconds"] = time.perf_counter() - started
            evidence["relaxation_peak_allocated_bytes"] = (
                torch.cuda.max_memory_allocated(device)
            )
            evidence["relaxation_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(
                device
            )
            evidence["budgets"] = {
                "max_batch_size": args.max_batch_size,
                "max_atoms": args.max_atoms,
                "max_steps_per_system": args.max_steps,
                "max_wall_seconds": args.max_wall_seconds,
            }
            assert sorted(r["source_index"] for r in records) == list(
                range(len(args.inputs))
            )
            ids = [r["system_id"] for r in records]
            assert len(set(ids)) == len(ids) and sorted(ids) == list(
                range(len(records))
            )
            source_ids = {r["source_index"]: r["system_id"] for r in records}
            for event in trace:
                assert all(
                    source_ids[s] == i
                    for s, i in zip(event["sources"], event["ids"], strict=True)
                )
            first_sources = set(trace[0]["sources"])
            later_sources = (
                set().union(*(set(t["sources"]) for t in trace)) - first_sources
            )
            if len(args.inputs) > args.max_batch_size:
                assert later_sources, "required refill was not observed"
            evidence.update(
                optimizer=args.optimizer,
                admissions=trace,
                refill_observed=bool(later_sources),
                results=[
                    {k: r[k] for k in ("source_index", "system_id", "status", "steps")}
                    for r in records
                ],
                exports=verify_exports(records, args.checkpoint, dtype, device),
            )
        torch.cuda.synchronize(device)
        evidence.update(
            status="passed",
            elapsed_seconds=time.perf_counter() - started,
            probe_total_seconds=time.perf_counter() - PROBE_STARTED,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        )
        (args.output_dir / "validation.json").write_text(
            json.dumps(evidence, indent=2, allow_nan=False, sort_keys=True) + "\n"
        )
        print(json.dumps(evidence, allow_nan=False, sort_keys=True), flush=True)
    finally:
        torch.set_default_dtype(previous_dtype)


if __name__ == "__main__":
    main()
