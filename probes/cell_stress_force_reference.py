# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded CPU/HCU stress conversion and periodic MACE energy-derivative probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from ase.build import molecule

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics._ops.npt_nph import stress_to_cell_force
from nvalchemi.models.mace import MACEWrapper
from nvalchemi.neighbors import compute_neighbors


def conversion_parity(device: torch.device) -> list[dict]:
    """Compare five independent stress regimes and their gradients on CPU/HCU."""
    cases = []
    for dtype in (torch.float32, torch.float64):
        h = torch.tensor(
            [[2.0, 0.0, 0.0], [0.6, 2.4, 0.0], [-0.3, 0.5, 2.8]], dtype=dtype
        ).repeat(5, 1, 1)
        s = torch.tensor(
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.2]],
                [[-0.2, 0.0, 0.0], [0.0, -0.2, 0.0], [0.0, 0.0, -0.2]],
                [[0.0, 0.1, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.1, 0.03, -0.02], [0.03, -0.2, 0.05], [-0.02, 0.05, 0.3]],
            ],
            dtype=dtype,
        )
        v = torch.linalg.det(h).abs()
        for aligned in (False, True):
            cpu = tuple(x.clone().requires_grad_() for x in (s, h, v))
            actual_inputs = tuple(
                x.to(device).clone().requires_grad_() for x in (s, h, v)
            )
            expected = stress_to_cell_force(*cpu, aligned, backend="torch_reference")
            actual = stress_to_cell_force(
                *actual_inputs, aligned, backend="torch_reference"
            )
            expected_grads = torch.autograd.grad(expected.square().sum(), cpu)
            actual_grads = torch.autograd.grad(actual.square().sum(), actual_inputs)
            rtol, atol = (3e-5, 3e-6) if dtype == torch.float32 else (2e-11, 2e-12)
            torch.testing.assert_close(actual.cpu(), expected, rtol=rtol, atol=atol)
            for a, b in zip(actual_grads, expected_grads, strict=True):
                torch.testing.assert_close(a.cpu(), b, rtol=rtol, atol=atol)
            cases.append(
                {
                    "dtype": str(dtype),
                    "keep_aligned": aligned,
                    "max_abs_error": float((actual.cpu() - expected).abs().max()),
                    "stress_regimes": [
                        "zero",
                        "tension",
                        "compression",
                        "shear",
                        "mixed",
                    ],
                    "first_backward_parity": True,
                }
            )
        if device.type == "cuda":
            with torch.cuda.stream(torch.cuda.Stream()):
                stream_inputs = tuple(x.to(device) * 1.01 for x in (s, h, v))
                stream_result = stress_to_cell_force(
                    *stream_inputs, False, backend="torch_reference"
                )
            torch.cuda.synchronize()
            expected_stream = stress_to_cell_force(
                s * 1.01, h * 1.01, v * 1.01, False, backend="torch_reference"
            )
            torch.testing.assert_close(
                stream_result.cpu(), expected_stream, rtol=rtol, atol=atol
            )
            try:
                stress_to_cell_force(
                    s.to(device), h, v.to(device), backend="torch_reference"
                )
            except ValueError as exc:
                assert "same device" in str(exc)
            else:
                raise AssertionError("mixed devices were silently accepted")
    return cases


def mace_evidence(device: torch.device, checkpoint: Path, dtype: torch.dtype) -> dict:
    """Finite-difference actual model energy while deforming coordinates and cells."""
    model = MACEWrapper.from_checkpoint(str(checkpoint), device=device, dtype=dtype)
    model.set_config("active_outputs", {"energy", "forces", "stress"})
    model.eval()
    systems = []
    for name, h in [
        ("H2O", [[4.2, 0.0, 0.0], [0.4, 4.7, 0.0], [0.2, -0.3, 4.9]]),
        ("CH4", [[4.6, 0.2, -0.1], [0.6, 4.3, 0.3], [-0.2, 0.4, 5.37]]),
    ]:
        atoms = molecule(name)
        atoms.set_cell(h)
        atoms.center()
        atoms.pbc = True
        systems.append(AtomicData.from_atoms(atoms))
    batch = Batch.from_data_list(systems).to(device)
    batch.positions = batch.positions.to(dtype)
    batch.cell = batch.cell.to(dtype)
    positions, cell = batch.positions.clone(), batch.cell.clone()

    def evaluate(r, h):
        data = batch.clone()
        data.positions, data.cell = r, h
        compute_neighbors(
            data, config=model.model_config.neighbor_config, backend="torch_reference"
        )
        output = model(data)
        signature = torch.cat(
            (data.neighbor_list.long(), data.neighbor_list_shifts.long()), dim=1
        )
        # Compare edge/image sets: deformation can reorder equal-distance edges.
        for column in reversed(range(signature.shape[1])):
            signature = signature[torch.argsort(signature[:, column], stable=True)]
        return output, signature

    output, signature = evaluate(positions, cell)
    stress = output["stress"]
    v = torch.linalg.det(cell).abs()
    force = stress_to_cell_force(stress, cell, v, False, backend="torch_reference")
    assert torch.isfinite(stress).all() and torch.isfinite(force).all()
    assert stress.abs().max() > 1e-6, "fixture must have nonzero stress"
    assert signature[:, 2:].count_nonzero() > 0, (
        "fixture must exercise periodic image shifts"
    )
    assert torch.equal(batch.positions, positions) and torch.equal(batch.cell, cell)
    forward = {
        "dtype": str(dtype),
        "batch_ptr": batch.batch_ptr.cpu().tolist(),
        "neighbor_edges": len(signature),
        "nonzero_image_shift_entries": int(signature[:, 2:].count_nonzero()),
        "energy_eV": output["energy"].detach().cpu().tolist(),
        "stress_eV_A3": stress.detach().cpu().tolist(),
        "cell_force_eV_A": force.detach().cpu().tolist(),
        "unchanged_input_cell_positions": True,
        "finite_difference_checked": dtype == torch.float64,
    }
    if dtype == torch.float32:
        # Total energies include large atomic reference offsets. Subtracting
        # nearby FP32 totals is too noisy for a stress derivative oracle;
        # verify FP32 through CPU/HCU parity, and use FP64 for finite differences.
        return forward
    eps = 1e-5
    fd = torch.zeros_like(stress)
    for i, j in [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]:
        delta = torch.zeros_like(cell)
        delta[:, i, j] = eps
        if i != j:
            delta[:, j, i] = eps
        energies = []
        for sign in (1.0, -1.0):
            deformation = torch.eye(3, dtype=dtype, device=device) + sign * delta
            deformed_r = torch.einsum(
                "ni,nij->nj", positions, deformation[batch.batch_idx.long()]
            )
            deformed_h = cell @ deformation
            values, edges = evaluate(deformed_r, deformed_h)
            if not torch.equal(signature, edges):
                before = set(map(tuple, signature.cpu().tolist()))
                after = set(map(tuple, edges.cpu().tolist()))
                raise AssertionError(
                    f"FD topology changed at {(i, j, sign)}: removed={before - after}, added={after - before}"
                )
            energies.append(values["energy"].detach().reshape(-1))
        fd[:, i, j] = (energies[0] - energies[1]) / (2 * eps * v) / (2 if i != j else 1)
        fd[:, j, i] = fd[:, i, j]
    rtol, atol = 3e-4, 3e-7
    torch.testing.assert_close(stress, fd, rtol=rtol, atol=atol)
    # Two direct row-cell perturbations additionally check the conversion.
    frac = torch.linalg.solve(
        cell[batch.batch_idx.long()].mT, positions[..., None]
    ).squeeze(-1)
    errors = []
    for i, j in [(1, 0), (2, 1)]:
        energies = []
        for sign in (1.0, -1.0):
            h = cell.clone()
            h[:, i, j] += sign * eps
            r = torch.einsum("ni,nij->nj", frac, h[batch.batch_idx.long()])
            values, edges = evaluate(r, h)
            assert torch.equal(signature, edges)
            energies.append(values["energy"].detach().reshape(-1))
        fd_force = -(energies[0] - energies[1]) / (2 * eps)
        torch.testing.assert_close(
            force[:, i, j], fd_force, rtol=rtol, atol=atol * float(v.max())
        )
        errors.append(float((force[:, i, j] - fd_force).abs().max()))
    assert torch.equal(batch.positions, positions) and torch.equal(batch.cell, cell)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {
        **forward,
        "max_stress_fd_abs_error": float((stress - fd).abs().max()),
        "row_cell_force_fd_abs_errors": errors,
        "fd_step": eps,
        "rtol": rtol,
        "stress_atol": atol,
        "unchanged_input_cell_positions": True,
    }


def main() -> None:
    """Run only one requested device; emit a durable, bounded evidence artifact."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-reference", type=Path)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        assert torch.version.hip and torch.cuda.is_available(), "HIP HCU required"
        assert torch.cuda.device_count() == 1, "probe requires one visible device"
    checkpoint = args.checkpoint.expanduser().resolve()
    parity = conversion_parity(device)
    print(json.dumps({"stage": "conversion", "cases": parity}), flush=True)
    mace = []
    for dtype in (torch.float32, torch.float64):
        result = mace_evidence(device, checkpoint, dtype)
        mace.append(result)
        print(json.dumps({"stage": "mace", **result}), flush=True)
    mace_parity = []
    if args.cpu_reference:
        reference = json.loads(args.cpu_reference.read_text())
        assert (
            reference["checkpoint_sha256"]
            == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        )
        for actual, expected in zip(mace, reference["mace"], strict=True):
            assert actual["dtype"] == expected["dtype"]
            assert actual["neighbor_edges"] == expected["neighbor_edges"]
            rtol, atol = (
                (2e-4, 2e-5) if actual["dtype"] == "torch.float32" else (2e-9, 2e-10)
            )
            errors = {}
            for key in ("stress_eV_A3", "cell_force_eV_A"):
                a, b = (
                    torch.tensor(actual[key], dtype=torch.float64),
                    torch.tensor(expected[key], dtype=torch.float64),
                )
                torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
                errors[key] = float((a - b).abs().max())
            mace_parity.append(
                {
                    "dtype": actual["dtype"],
                    "max_abs_errors": errors,
                    "rtol": rtol,
                    "atol": atol,
                }
            )
    evidence = {
        "status": "passed",
        "device": str(device),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
        "device_name": torch.cuda.get_device_name() if device.type == "cuda" else "CPU",
        "architecture": getattr(
            torch.cuda.get_device_properties(0), "gcnArchName", None
        )
        if device.type == "cuda"
        else None,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "conversion": parity,
        "mace": mace,
        "mace_cpu_reference_parity": mace_parity,
        "scope": "stress generation and cell-force conversion only; no cell update or dynamics",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({"status": "passed", "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
