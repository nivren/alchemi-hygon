# SPDX-License-Identifier: Apache-2.0
"""Isolate M7 strict FP32 state checks using two Torch-only trajectories."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from ase.io import read

from hip_real_batch_validation import neighbor, optimizer, prime, state_check, values
from mace_mpa0_cpu_baseline import CHECKPOINT_SHA256, compare
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import FIRE2VariableCell
from nvalchemi.models.mace import MACEWrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--failure-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == CHECKPOINT_SHA256
    torch.set_default_dtype(torch.float32)
    model = MACEWrapper.from_checkpoint(args.checkpoint, device=torch.device("cuda"), dtype=torch.float32,
                                        enable_cueq=False, compile_model=False)
    model.set_config("active_outputs", {"energy", "forces", "stress"})
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    systems, n = [], 0
    for i, item in enumerate(json.loads(args.manifest.read_text())):
        path = args.data_root / item["filename"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        data = AtomicData.from_atoms(read(path), dtype=torch.float32)
        data.velocities, data.forces = torch.zeros_like(data.positions), torch.zeros_like(data.positions)
        data.energy, data.stress = torch.zeros(1, 1), torch.zeros(1, 3, 3)
        data.add_system_property("source_index", torch.tensor([[i]], dtype=torch.long))
        systems.append(data)
        n += data.num_nodes
        if n >= 1000:
            break
    initial = Batch.from_data_list(systems, device="cuda")
    saved = torch.load(args.failure_snapshot, weights_only=True)["paths"]["torch_reference"]
    fixed = initial.clone()
    fixed.positions, fixed.cell = saved["positions"].cuda(), saved["cell"].cuda()
    neighbor(fixed, model, "torch_reference")
    outputs = [values(model, fixed) for _ in range(12)]
    errors = []
    for item in outputs[1:]:
        check = compare(item, outputs[0], torch.float32)
        delta = (item["forces"] - outputs[0]["forces"]).abs()
        check["strict_force_failed_elements"] = int((delta > 2e-5 + 2e-4 * outputs[0]["forces"].abs()).sum())
        errors.append(check)
    report = {"scope": "diagnostic control, not an acceptance gate", "N": n,
              "fixed_geometry_torch_repeats": errors, "torch_only_trajectories": []}
    for run in range(3):
        a, b = initial.clone(), initial.clone()
        da, _ = optimizer(FIRE2VariableCell, model, "torch_reference")
        db, _ = optimizer(FIRE2VariableCell, model, "torch_reference")
        prime(da, a)
        prime(db, b)
        for step in range(3):
            da.step(a)
            db.step(b)
            try:
                check = state_check(a, b, da, db)
                error = None
            except AssertionError as failure:
                check, error = None, str(failure)
            entry = {"round": run, "step": step + 1, "strict_state_passed": error is None,
                "error": error, "state_check": check,
                "physical_outputs": compare({k: getattr(a, k).reshape(-1) for k in ("energy", "forces", "stress")},
                    {k: getattr(b, k).reshape(-1) for k in ("energy", "forces", "stress")}, torch.float32),
                "max_abs": {k: float((getattr(a, k) - getattr(b, k)).abs().max()) for k in
                    ("positions", "cell", "velocities", "forces")}}
            report["torch_only_trajectories"].append(entry)
            print(run, step + 1, error or "passed", flush=True)
        del a, b, da, db
    torch.save(outputs, args.output.with_suffix(".pt"))
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
