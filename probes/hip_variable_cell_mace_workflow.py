# SPDX-License-Identifier: Apache-2.0
"""Bounded MPA-0 HIP-neighbor variable-cell trajectories and budget refill."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from ase.io import read

from hip_coo_mace_snapshot import evaluate
from mace_mpa0_cpu_baseline import CHECKPOINT_SHA256, SAMPLES, compare
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import FIRE2VariableCell, FIREVariableCell, FusedStage
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.hooks.cell_align import AlignCellHook
from nvalchemi.dynamics.sampler import SizeAwareSampler
from nvalchemi.dynamics.sinks import HostMemory
from nvalchemi.hooks import NeighborListHook
from nvalchemi.models.mace import MACEWrapper


def signature(data):
    rows = torch.cat((data.neighbor_list.long(), data.neighbor_list_shifts.long()), -1)
    for column in reversed(range(5)):
        rows = rows[torch.argsort(rows[:, column], stable=True)]
    edges = data.neighbor_list.long()
    assert torch.equal(data.batch_idx[edges[:, 0]], data.batch_idx[edges[:, 1]])
    return rows


def audit(model, data, dtype):
    expected, edges, topology = evaluate(model, data, backend="torch_reference")
    assert torch.equal(signature(data), edges)
    actual = {key: getattr(data, key).reshape_as(value) for key, value in expected.items()}
    assert bool(torch.all(torch.linalg.det(data.cell) > 0))
    for key in ("positions", "cell", "velocities", "energy", "forces", "stress"):
        value = getattr(data, key)
        assert value.device.type == "cuda" and value.dtype == dtype
        assert bool(torch.isfinite(value).all()), key
    return {"parity": compare(actual, expected, dtype), "topology": topology}


def optimizer(cls, model, backend, *, extra_hooks=(), shared_compute=False):
    neighbor = NeighborListHook(
        model.model_config.neighbor_config, stage=DynamicsStage.BEFORE_COMPUTE,
        skin=0, backend=backend, method="cell_list" if backend == "hip" else None,
    )
    hooks = [] if shared_compute else [neighbor]
    if cls is FIRE2VariableCell:
        hooks.append(AlignCellHook(backend="torch_reference"))
    hooks.extend(extra_hooks)
    kwargs = {"dt_min": 0.0} if cls is FIREVariableCell else {}
    return cls(model=model, dt=0.02, backend="torch_reference", hooks=hooks, **kwargs), neighbor


def prime(dynamics, data):
    dynamics._ensure_state_initialized(data)
    dynamics._call_hooks(DynamicsStage.BEFORE_STEP, data)
    dynamics._call_hooks(DynamicsStage.BEFORE_COMPUTE, data)
    dynamics.compute(data)


def trajectory(cls, model, systems, dtype):
    data = Batch.from_data_list(systems, device="cuda")
    oracle = data.clone()
    dynamics, _ = optimizer(cls, model, "hip")
    reference, _ = optimizer(cls, model, "torch_reference")
    prime(dynamics, data)
    prime(reference, oracle)
    initial_r, initial_h = data.positions.clone(), data.cell.clone()
    checks = [audit(model, data, dtype)]
    maxima = {}
    rtol, atol = (2e-8, 2e-9) if dtype == torch.float64 else (2e-4, 2e-5)
    for step in range(5):
        dynamics.step(data)
        reference.step(oracle)
        assert torch.equal(signature(data), signature(oracle))
        compare({key: getattr(data, key).reshape(-1) for key in ("energy", "forces", "stress")},
                {key: getattr(oracle, key).reshape(-1) for key in ("energy", "forces", "stress")}, dtype)
        pairs = [(key, getattr(data, key), getattr(oracle, key)) for key in dynamics._mutable_fields]
        pairs += [("state." + key, value, getattr(reference._state, key)) for key, value in dynamics._state]
        for key, actual, expected in pairs:
            if actual.is_floating_point():
                torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
                maxima[key] = max(maxima.get(key, 0), float((actual - expected).abs().max()))
            else:
                assert torch.equal(actual, expected), key
        checks.append(audit(model, data, dtype))
        print(f"passed trajectory {cls.__name__} {dtype} step {step + 1}", flush=True)
    movement = {"positions_max_abs_A": float((data.positions - initial_r).abs().max()),
                "cell_max_abs_A": float((data.cell - initial_h).abs().max())}
    assert all(value > 0 for value in movement.values())
    return {"optimizer": cls.__name__, "dtype": str(dtype), "steps": 5,
            "batch_ptr": data.batch_ptr.cpu().tolist(), "state_max_abs_errors": maxima,
            "movement": movement, "current_geometry_checks": checks}


class Dataset:
    def __init__(self, systems):
        self.systems = [data.clone() for data in systems]
        for i, data in enumerate(self.systems):
            data.add_system_property("source_index", torch.tensor([[i]], dtype=torch.long))

    def __len__(self):
        return len(self.systems)

    def get_metadata(self, i):
        return self.systems[i].num_nodes, 0

    def __getitem__(self, i):
        return self.systems[i].clone(), {}


def refill(cls, model, systems, dtype):
    counts, checks, ptrs = {}, [], []
    budgets = {0: 1, 1: 3, 2: 2}

    class Priming:
        stage = DynamicsStage.BEFORE_PRE_UPDATE
        frequency = 1

        def __call__(self, ctx, stage):
            checks.append(audit(model, ctx.batch, dtype))
            ptrs.append(ctx.batch.batch_ptr.cpu().tolist())

    class Budget:
        stage = DynamicsStage.AFTER_STEP
        frequency = 1

        def __call__(self, ctx, stage):
            for row, source in enumerate(ctx.batch.source_index.flatten().cpu().tolist()):
                counts[source] = counts.get(source, 0) + 1
                if counts[source] >= budgets[source]:
                    ctx.batch.status[row] = 2

    dynamics, neighbor = optimizer(cls, model, "hip", extra_hooks=[Priming(), Budget()], shared_compute=True)
    # FusedStage owns the shared forward pass and its BEFORE_COMPUTE hook.
    sink = HostMemory(capacity=3)
    sampler = SizeAwareSampler(Dataset(systems), max_atoms=14, max_batch_size=2)
    stage = FusedStage(sub_stages=[(0, dynamics)], hooks=[neighbor],
                       sampler=sampler,
                       sinks=[sink], refill_frequency=1, device_type="cuda")
    initial = sampler.build_initial_batch().to("cuda")
    dynamics._ensure_state_initialized(initial)
    dynamics._call_hooks(DynamicsStage.BEFORE_STEP, initial)
    assert stage.run(batch=initial, n_steps=5) is None
    stored = sink.read()
    assert len(sink) == 3 and counts == budgets
    assert sorted(stored.source_index.flatten().tolist()) == [0, 1, 2]
    assert sorted(stored.system_id.flatten().tolist()) == [0, 1, 2]
    assert stored.status.flatten().tolist() == [2, 2, 2]
    assert stage.step_count == dynamics.step_count == 3
    assert [0, 4, 14] in ptrs and [0, 10, 14] in ptrs
    print(f"passed budget refill {cls.__name__} {dtype}", flush=True)
    return {"optimizer": cls.__name__, "dtype": str(dtype), "steps": stage.step_count,
            "budget_updates_per_source": counts, "ptrs_before_kick": ptrs,
            "exit_status": "budget only; no physical convergence claim", "priming_checks": checks}


def run(checkpoint, root):
    assert torch.version.hip and torch.cuda.is_available()
    started = time.monotonic()
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest == CHECKPOINT_SHA256
    atoms, sources = [], []
    for filename in SAMPLES:
        path = root / filename
        atoms.append(read(path))
        sources.append({"filename": filename, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "atoms": len(atoms[-1])})
    trajectories, lifecycles = [], []
    for dtype in (torch.float64, torch.float32):
        torch.set_default_dtype(dtype)
        model = MACEWrapper.from_checkpoint(checkpoint, device=torch.device("cuda"), dtype=dtype,
                                             enable_cueq=False, compile_model=False)
        model.set_config("active_outputs", {"energy", "forces", "stress"})
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        systems = [AtomicData.from_atoms(a, dtype=dtype) for a in atoms]
        for data in systems:
            data.velocities = torch.zeros_like(data.positions)
            data.forces = torch.zeros_like(data.positions)
            data.energy = torch.zeros(1, 1, dtype=dtype)
            data.stress = torch.zeros(1, 3, 3, dtype=dtype)
        for cls in (FIRE2VariableCell, FIREVariableCell):
            trajectories.append(trajectory(cls, model, systems, dtype))
            lifecycles.append(refill(cls, model, systems, dtype))
        del model, systems
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return {"status": "passed", "scope": "M3 finite trajectories and budget refill; no convergence/performance claim",
            "checkpoint_sha256": digest, "sources": sources, "trajectories": trajectories,
            "lifecycles": lifecycles, "environment": {"torch": torch.__version__, "hip": torch.version.hip,
            "device": torch.cuda.get_device_name()}, "elapsed_seconds": time.monotonic() - started}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    previous = torch.get_default_dtype()
    try:
        report = run(args.checkpoint, args.data_root)
    finally:
        torch.set_default_dtype(previous)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "elapsed_seconds": report["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
