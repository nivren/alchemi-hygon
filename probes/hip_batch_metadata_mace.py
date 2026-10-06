# SPDX-License-Identifier: Apache-2.0
"""Three neighbor paths on identical real variable-cell FIRE snapshots."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from ase.io import read

from hip_batch_metadata_benchmark import ThreePaths, compare_snapshot, load_baseline
from hip_variable_cell_mace_workflow import optimizer, prime
from mace_mpa0_cpu_baseline import CHECKPOINT_SHA256, SAMPLES
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import FIRE2VariableCell, FIREVariableCell
from nvalchemi.models.mace import MACEWrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float32", "float64"), required=True)
    parser.add_argument("--optimizer", choices=("fire", "fire2"), required=True)
    args = parser.parse_args()
    assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == CHECKPOINT_SHA256
    dtype = getattr(torch, args.dtype)
    torch.set_default_dtype(dtype)
    model = MACEWrapper.from_checkpoint(args.checkpoint, device=torch.device("cuda"), dtype=dtype,
                                       enable_cueq=False, compile_model=False)
    model.set_config("active_outputs", {"energy", "forces", "stress"})
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    systems, sources = [], []
    for i, filename in enumerate(SAMPLES):
        path = args.data_root / filename
        data = AtomicData.from_atoms(read(path), dtype=dtype)
        data.velocities = torch.zeros_like(data.positions)
        data.forces = torch.zeros_like(data.positions)
        data.energy = torch.zeros(1, 1, dtype=dtype)
        data.stress = torch.zeros(1, 3, 3, dtype=dtype)
        data.add_system_property("source_index", torch.tensor([[i]], dtype=torch.long))
        systems.append(data)
        sources.append({"filename": filename, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    data = Batch.from_data_list(systems, device="cuda")
    cls = FIRE2VariableCell if args.optimizer == "fire2" else FIREVariableCell
    dynamics, _ = optimizer(cls, model, "hip")
    prime(dynamics, data)
    frames = [data.clone()]
    for _ in range(5):
        dynamics.step(data)
        frames.append(data.clone())
    assert not torch.equal(frames[0].cell, frames[-1].cell)
    assert not torch.equal(frames[0].positions, frames[-1].positions)
    paths = ThreePaths(load_baseline(args.baseline))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for step, frame in enumerate(frames):
        reports.append(compare_snapshot(frame, paths, f"step{step}", args.output_dir,
                                        cutoff=model.cutoff, model=model))
        (args.output_dir / "summary.json").write_text(json.dumps({"status": "running", "snapshots": reports}, indent=2) + "\n")
    (args.output_dir / "summary.json").write_text(json.dumps({"status": "passed", "dtype": args.dtype,
        "optimizer": cls.__name__, "steps": 5, "sources": sources, "checkpoint_sha256": CHECKPOINT_SHA256,
        "snapshots": reports, "movement_A": {"cell": float((frames[-1].cell - frames[0].cell).abs().max()),
        "positions": float((frames[-1].positions - frames[0].positions).abs().max())}}, indent=2) + "\n")


if __name__ == "__main__":
    main()
