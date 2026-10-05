# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded M2 real MPA-0 single-point parity; no relaxation or downloads."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import torch
from ase.io import read

from mace_mpa0_cpu_baseline import CHECKPOINT_SHA256, SAMPLES, compare
from nvalchemi.data import AtomicData, Batch
from nvalchemi.hooks import NeighborListHook
from nvalchemi.hooks._context import HookContext
from nvalchemi.models.mace import MACEWrapper
from nvalchemi.neighbors import compute_neighbors


def evaluate(model, snapshot, *, backend, hook=None):
    data = snapshot.clone()
    before = data.positions.clone(), data.cell.clone()
    if hook is None:
        compute_neighbors(
            data, config=model.model_config.neighbor_config, backend=backend,
            method="cell_list" if backend == "hip" else None,
        )
    else:
        hook(HookContext(batch=data), None)
    edges = data.neighbor_list.long()
    if not torch.equal(data.batch_idx[edges[:, 0]], data.batch_idx[edges[:, 1]]):
        raise AssertionError("neighbor edges cross system boundaries")
    signature = torch.cat((edges, data.neighbor_list_shifts.long()), dim=1)
    for column in reversed(range(5)):
        signature = signature[torch.argsort(signature[:, column], stable=True)]
    output = model(data)  # MACE's force/strain autograd remains enabled.
    values = {key: output[key].detach().clone() for key in ("energy", "forces", "stress")}
    values["energy"] = values["energy"].reshape(-1)
    shapes = {"energy": (data.num_graphs,), "forces": (data.num_nodes, 3), "stress": (data.num_graphs, 3, 3)}
    for key, value in values.items():
        assert value.shape == shapes[key] and bool(torch.isfinite(value).all()), key
        assert value.device == data.positions.device and value.dtype == data.positions.dtype, key
    torch.testing.assert_close(data.positions, before[0], rtol=0, atol=0)
    torch.testing.assert_close(data.cell, before[1], rtol=0, atol=0)
    return values, signature, {
        "edges": edges.shape[0],
        "edges_per_system": data._storage.groups["edges"].segment_lengths.cpu().tolist(),
        "counts": torch.bincount(edges[:, 0], minlength=data.num_nodes).cpu().tolist(),
        "nonzero_shift_components": int(torch.count_nonzero(data.neighbor_list_shifts)),
    }


def finite_differences(model, snapshot, initial, signature):
    eps = 1e-5
    chosen = int(initial["forces"].norm(dim=1).argmax())
    energies = []
    for sign in (-1, 1):
        perturbed = snapshot.clone()
        perturbed.positions[chosen, 0] += sign * eps
        values, edges, _ = evaluate(model, perturbed, backend="hip")
        assert torch.equal(edges, signature), "atomic difference changed topology"
        energies.append(values["energy"].sum())
    force_fd = -(energies[1] - energies[0]) / (2 * eps)
    torch.testing.assert_close(force_fd, initial["forces"][chosen, 0], rtol=3e-4, atol=3e-6)
    direction = snapshot.cell.new_tensor([[0, 0.5, 0], [0.5, 0, 0], [0, 0, 0]])
    energies = []
    for sign in (-1, 1):
        perturbed = snapshot.clone()
        transform = torch.eye(3, device=direction.device, dtype=direction.dtype) + sign * eps * direction
        perturbed.positions = snapshot.positions @ transform
        perturbed.cell = snapshot.cell @ transform
        values, edges, _ = evaluate(model, perturbed, backend="hip")
        assert torch.equal(edges, signature), "strain difference changed topology"
        energies.append(values["energy"])
    stress_fd = (energies[1] - energies[0]) / (2 * eps) / torch.linalg.det(snapshot.cell).abs()
    stress_projection = (initial["stress"] * direction).sum(dim=(-1, -2))
    torch.testing.assert_close(stress_fd, stress_projection, rtol=3e-4, atol=3e-7)
    return {
        "step": eps, "selected_atom": chosen,
        "force_max_abs_error_eV_A": float((force_fd - initial["forces"][chosen, 0]).abs()),
        "stress_max_abs_error_eV_A3": float((stress_fd - stress_projection).abs().max()),
        "unchanged_topology": True,
    }


def run(checkpoint, root):
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("a visible allocated HIP device is required")
    started = time.monotonic()
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != CHECKPOINT_SHA256:
        raise ValueError("checkpoint differs from locked MPA-0 SHA256")
    sources, atoms = [], []
    for filename in SAMPLES:
        path = root / filename
        system = read(path)
        assert system.pbc.all() and system.get_volume() > 0
        atoms.append(system)
        sources.append({"filename": filename, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "atoms": len(system)})
    evidence = []
    for dtype in (torch.float64, torch.float32):
        torch.set_default_dtype(dtype)
        model = MACEWrapper.from_checkpoint(checkpoint, device=torch.device("cuda"), dtype=dtype, enable_cueq=False, compile_model=False)
        model.set_config("active_outputs", {"energy", "forces", "stress"})
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        assert model.cutoff == 6.0
        batch = Batch.from_data_list([AtomicData.from_atoms(a, dtype=dtype) for a in atoms], device="cuda")
        hook = NeighborListHook(model.model_config.neighbor_config, backend="hip", method="cell_list")
        cases = []
        for name in ("initial", "affine_shear", "wrap"):
            snapshot = batch.clone()
            if name == "affine_shear":
                transform = batch.cell.new_tensor([[0.98, 0.03, 0], [0.01, 1.01, 0.02], [0, 0.01, 0.99]])
                snapshot.positions = batch.positions @ transform
                snapshot.cell = batch.cell @ transform
            elif name == "wrap":
                snapshot.positions[0] += batch.cell[0, 0] - batch.cell[0, 1]
            candidate, edges, topology = evaluate(model, snapshot, backend="hip")
            reference, reference_edges, _ = evaluate(model, snapshot, backend="torch_reference")
            assert torch.equal(edges, reference_edges), name
            case = {"name": name, "topology": topology, "parity": compare(candidate, reference, dtype)}
            hooked, hook_edges, _ = evaluate(model, snapshot, backend="hip", hook=hook)
            assert torch.equal(hook_edges, edges), name
            case["hook_parity"] = compare(hooked, candidate, dtype)
            if name == "initial" and dtype == torch.float64:
                case["finite_differences"] = finite_differences(model, snapshot, candidate, edges)
            cases.append(case)
            print(f"passed {dtype} {name}", flush=True)
        evidence.append({"dtype": str(dtype), "batch_ptr": batch.batch_ptr.cpu().tolist(), "cases": cases})
        del model, batch, hook, snapshot, candidate, reference, hooked
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return {
        "status": "passed", "scope": "M2 HIP MATRIX plus Torch COO / MPA-0 single points; no relaxation or performance claim",
        "checkpoint_sha256": digest, "sources": sources, "evidence": evidence,
        "environment": {"torch": torch.__version__, "hip": torch.version.hip, "device": torch.cuda.get_device_name(),
                        "mace": importlib.metadata.version("mace-torch"), "e3nn": importlib.metadata.version("e3nn")},
        "elapsed_seconds": time.monotonic() - started,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    previous_dtype = torch.get_default_dtype()
    try:
        report = run(args.checkpoint, args.data_root)
    finally:
        torch.set_default_dtype(previous_dtype)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "elapsed_seconds": report["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
