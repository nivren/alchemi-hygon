# SPDX-License-Identifier: Apache-2.0
"""M3 CPU-only, three-step native variable-cell MPA-0 wiring on M1 CIFs.

Not an application runner or convergence gate. M4 handles workflow lifecycle;
M5 handles the real target model on HCU. No model or input downloads/writes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import torch
from ase.io import read

from mace_mpa0_cpu_baseline import CHECKPOINT_SHA256, SAMPLES, compare, evaluate
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import FIRE2VariableCell, FIREVariableCell
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.hooks.cell_align import AlignCellHook
from nvalchemi.hooks._context import DynamicsContext
from nvalchemi.hooks.neighbor_list import NeighborListHook
from nvalchemi.models.mace import MACEWrapper


def run(checkpoint, root):
    started = time.monotonic()
    model_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if model_hash != CHECKPOINT_SHA256:
        raise ValueError("checkpoint does not match the M1 MPA-0 SHA256")
    sources = [
        {
            "filename": name,
            "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest(),
        }
        for name in SAMPLES
    ]
    cases = []
    for dtype in (torch.float64, torch.float32):
        # MACE ZBL integer-Z power uses the process default dtype (M1 evidence).
        torch.set_default_dtype(dtype)
        model = MACEWrapper.from_checkpoint(
            checkpoint,
            device=torch.device("cpu"),
            dtype=dtype,
            enable_cueq=False,
            compile_model=False,
        )
        model.set_config("active_outputs", {"energy", "forces", "stress"})
        model.eval()
        for cls in (FIRE2VariableCell, FIREVariableCell):
            data = [
                AtomicData.from_atoms(read(root / name), dtype=dtype)
                for name in SAMPLES
            ]
            b = Batch.from_data_list(data)
            # Dynamics copies outputs into existing slots; from_atoms may omit them.
            b.forces = torch.zeros_like(b.positions)
            b.velocities = torch.zeros_like(b.positions)
            b.stress = torch.zeros_like(b.cell)
            b.energy = torch.zeros(b.num_graphs, 1, dtype=dtype)
            neighbor = NeighborListHook(
                model.model_config.neighbor_config,
                skin=0.0,
                stage=DynamicsStage.BEFORE_COMPUTE,
                backend="torch_reference",
            )
            hooks = [neighbor]
            if cls is FIRE2VariableCell:
                hooks.append(AlignCellHook())
            d = cls(model=model, dt=0.02, backend="torch_reference", hooks=hooks)
            d._init_state(b)
            alignment = None
            # Align before the initial model call; normal subsequent calls use hooks.
            if cls is FIRE2VariableCell:
                unrotated, _ = evaluate(model, b)
                old_h = b.cell.clone()
                c, s = math.cos(0.37), math.sin(0.37)
                rotation = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=dtype)
                b.positions.copy_(b.positions @ rotation)
                b.cell.copy_(b.cell @ rotation)
                hooks[-1](
                    DynamicsContext(batch=b, workflow=d), DynamicsStage.BEFORE_STEP
                )
            neighbor(DynamicsContext(batch=b, workflow=d), DynamicsStage.BEFORE_COMPUTE)
            d.compute(b)
            if cls is FIRE2VariableCell:
                transform = torch.linalg.solve(b.cell, old_h)
                expected_frame = {
                    "energy": unrotated["energy"],
                    "forces": torch.bmm(
                        transform[b.batch_idx.long()], unrotated["forces"][..., None]
                    ).squeeze(-1),
                    "stress": transform @ unrotated["stress"] @ transform.mT,
                }
                alignment = compare(
                    {
                        "energy": b.energy.flatten(),
                        "forces": b.forces,
                        "stress": b.stress,
                    },
                    expected_frame,
                    dtype,
                )
            initial = {
                "energy_eV": b.energy.flatten().tolist(),
                "volume_A3": torch.linalg.det(b.cell).abs().tolist(),
            }
            initial_r, initial_h = b.positions.clone(), b.cell.clone()
            d.run(b, n_steps=3)
            actual = {
                key: getattr(b, key).detach().clone()
                for key in ("energy", "forces", "stress")
            }
            actual["energy"] = actual["energy"].flatten()
            expected, edges = evaluate(model, b)
            consistency = compare(actual, expected, dtype)
            if torch.equal(initial_r, b.positions) or torch.equal(initial_h, b.cell):
                raise AssertionError("atomic and cell updates were not both exercised")
            if torch.any(torch.linalg.det(b.cell) <= 0):
                raise AssertionError("nonpositive cell volume")
            if any(
                v.device.type != "cpu"
                or v.dtype != dtype
                or not torch.isfinite(v).all()
                for v in actual.values()
            ):
                raise AssertionError("invalid output/device/dtype")
            selected = (
                d._backend_selection if cls is FIRE2VariableCell else d._fire_selection
            )
            cases.append(
                {
                    "optimizer": cls.__name__,
                    "dtype": str(dtype),
                    "steps": d.step_count,
                    "initial": initial,
                    "final_energy_eV": actual["energy"].tolist(),
                    "final_volume_A3": torch.linalg.det(b.cell).abs().tolist(),
                    "max_abs_stress_eV_A3": actual["stress"]
                    .abs()
                    .amax((-1, -2))
                    .tolist(),
                    "position_change_max_A": float(
                        (b.positions - initial_r).abs().max()
                    ),
                    "cell_change_max_A": float((b.cell - initial_h).abs().max()),
                    "batch_ptr": b.batch_ptr.tolist(),
                    "neighbors": len(edges),
                    "cross_system_edges": 0,
                    "final_single_point_consistency": consistency,
                    "rotated_alignment_covariance": alignment,
                    "selection": selected.as_dict(),
                    "convergence_status": "not_evaluated",
                }
            )
    return {
        "status": "passed",
        "device": "cpu",
        "checkpoint_sha256": model_hash,
        "torch": torch.__version__,
        "sources": sources,
        "cases": cases,
        "elapsed_s": time.monotonic() - started,
        "HCU_MPA0": "not_run",
        "scope": "three-step wiring only; not converged relaxation or batch/refill application",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    previous_dtype = torch.get_default_dtype()
    try:
        report = run(args.checkpoint.expanduser().resolve(), args.data_root.resolve())
    finally:
        torch.set_default_dtype(previous_dtype)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "elapsed_s": report["elapsed_s"],
                "output": str(args.output),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
