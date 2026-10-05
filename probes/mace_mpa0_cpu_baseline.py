# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only MPA-0 CIF single-point, Batch and energy-derivative evidence.

Uses existing public APIs and a trusted local checkpoint. This probe does not
relax structures, advance cells, download models, or claim HCU support.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import torch
from ase.io import read
from mace.calculators import MACECalculator

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics._ops.stress import stress_to_cell_force
from nvalchemi.models.mace import MACEWrapper
from nvalchemi.neighbors import compute_neighbors

SAMPLES = (
    "mp-1222651_Li2MgGe.cif",
    "mp-998591_LiSnCl3.cif",
    "mp-1096278_NaLiTl2.cif",
)
CHECKPOINT_SHA256 = "75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638"


def evaluate(model: MACEWrapper, batch: Batch) -> tuple[dict, torch.Tensor]:
    """Rebuild neighbors and evaluate an independent clone at current r/H."""
    data = batch.clone()
    compute_neighbors(
        data, config=model.model_config.neighbor_config, backend="torch_reference"
    )
    edges = data.neighbor_list.long()
    if not torch.equal(data.batch_idx[edges[:, 0]], data.batch_idx[edges[:, 1]]):
        raise AssertionError("neighbor edges cross system boundaries")
    signature = torch.cat((edges, data.neighbor_list_shifts.long()), dim=1)
    for column in reversed(range(signature.shape[1])):
        signature = signature[torch.argsort(signature[:, column], stable=True)]
    output = model(data)
    result = {
        key: output[key].detach().clone() for key in ("energy", "forces", "stress")
    }
    expected_shapes = {
        "energy": (data.num_graphs,),
        "forces": (data.num_nodes, 3),
        "stress": (data.num_graphs, 3, 3),
    }
    result["energy"] = result["energy"].reshape(-1)
    for key, value in result.items():
        if value.shape != expected_shapes[key] or not torch.isfinite(value).all():
            raise AssertionError(f"invalid {key}: shape={tuple(value.shape)}")
        if value.dtype != batch.positions.dtype or value.device.type != "cpu":
            raise AssertionError(f"{key} changed dtype/device")
    return result, signature


def compare(actual: dict, expected: dict, dtype: torch.dtype) -> dict:
    """Check separately stated tolerances in each physical quantity's units."""
    rtol = 2e-8 if dtype == torch.float64 else 2e-4
    tolerances = (
        {"energy": 2e-7, "forces": 2e-7, "stress": 2e-9}
        if dtype == torch.float64
        else {"energy": 2e-3, "forces": 2e-4, "stress": 2e-6}
    )
    errors = {}
    for key, atol in tolerances.items():
        torch.testing.assert_close(actual[key], expected[key], rtol=rtol, atol=atol)
        errors[key] = float((actual[key] - expected[key]).abs().max())
    return {"max_abs_errors": errors, "rtol": rtol, "atol": tolerances}


def derivatives(model: MACEWrapper, batch: Batch, output: dict, signature) -> dict:
    """Check selected atomic partials, all symmetric strains and row-cell forces."""
    eps = 1e-5
    r, h = batch.positions.clone(), batch.cell.clone()
    volumes = torch.linalg.det(h).abs()

    def energy(positions, cell):
        displaced = batch.clone()
        displaced.positions, displaced.cell = positions, cell
        values, edges = evaluate(model, displaced)
        if not torch.equal(signature, edges):
            raise AssertionError(
                "finite-difference perturbation changed neighbor topology"
            )
        return values["energy"]

    # Test XYZ for the highest-force atom of each system, not only symmetric sites.
    chosen = []
    for start, stop in zip(batch.batch_ptr[:-1], batch.batch_ptr[1:], strict=True):
        chosen.append(
            int(start)
            + int(output["forces"][int(start) : int(stop)].norm(dim=1).argmax())
        )
    chosen_tensor = torch.tensor(chosen, dtype=torch.long)
    fd_force = torch.empty(batch.num_graphs, 3, dtype=torch.float64)
    for axis in range(3):
        delta = torch.zeros_like(r)
        delta[chosen_tensor, axis] = eps
        fd_force[:, axis] = -(energy(r + delta, h) - energy(r - delta, h)) / (2 * eps)
    expected_force = output["forces"][chosen_tensor]
    torch.testing.assert_close(expected_force, fd_force, rtol=3e-4, atol=3e-6)

    # A reproducible all-atom direction also exercises the complete force vector.
    direction = torch.sin(torch.arange(r.numel(), dtype=r.dtype).reshape_as(r) + 0.37)
    expected_directional = torch.zeros(batch.num_graphs, dtype=r.dtype)
    expected_directional.index_add_(
        0, batch.batch_idx.long(), -(output["forces"] * direction).sum(dim=1)
    )
    fd_directional = (
        energy(r + eps * direction, h) - energy(r - eps * direction, h)
    ) / (2 * eps)
    torch.testing.assert_close(
        expected_directional, fd_directional, rtol=3e-4, atol=3e-6
    )

    fd_stress = torch.zeros_like(h)
    for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)):
        delta = torch.zeros_like(h)
        delta[:, i, j] = eps
        if i != j:
            delta[:, j, i] = eps
        energies = []
        for sign in (1.0, -1.0):
            transform = torch.eye(3, dtype=h.dtype) + sign * delta
            rp = torch.einsum("ni,nij->nj", r, transform[batch.batch_idx.long()])
            energies.append(energy(rp, h @ transform))
        fd_stress[:, i, j] = (
            (energies[0] - energies[1]) / (2 * eps * volumes) / (2 if i != j else 1)
        )
        fd_stress[:, j, i] = fd_stress[:, i, j]
    torch.testing.assert_close(output["stress"], fd_stress, rtol=3e-4, atol=3e-7)

    raw_cell_force = stress_to_cell_force(
        output["stress"], h, volumes, False, backend="torch_reference"
    )
    frac = torch.linalg.solve(h[batch.batch_idx.long()].mT, r[..., None]).squeeze(-1)
    cell_errors = []
    for i, j in ((1, 0), (2, 1)):
        energies = []
        for sign in (1.0, -1.0):
            hp = h.clone()
            hp[:, i, j] += sign * eps
            rp = torch.einsum("ni,nij->nj", frac, hp[batch.batch_idx.long()])
            energies.append(energy(rp, hp))
        fd = -(energies[0] - energies[1]) / (2 * eps)
        torch.testing.assert_close(raw_cell_force[:, i, j], fd, rtol=3e-4, atol=3e-6)
        cell_errors.append(float((raw_cell_force[:, i, j] - fd).abs().max()))
    assert torch.equal(batch.positions, r) and torch.equal(batch.cell, h)
    return {
        "step": eps,
        "atomic_partial_count": 3 * batch.num_graphs,
        "selected_global_atom_indices": chosen,
        "force_max_abs_error_eV_A": float((expected_force - fd_force).abs().max()),
        "all_atom_directional_error_eV_A": float(
            (expected_directional - fd_directional).abs().max()
        ),
        "strain_component_count_per_system": 6,
        "stress_max_abs_error_eV_A3": float((output["stress"] - fd_stress).abs().max()),
        "cell_components": [[1, 0], [2, 1]],
        "cell_force_max_abs_errors_eV_A": cell_errors,
        "unchanged_topology": True,
        "unchanged_input_positions_cell": True,
        "rtol": 3e-4,
        "force_cell_atol_eV_A": 3e-6,
        "stress_atol_eV_A3": 3e-7,
    }


def run(checkpoint: Path, root: Path) -> dict:
    """Use actual CIFs with explicit precision; leave the input files untouched."""
    started = time.monotonic()
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if checkpoint_hash != CHECKPOINT_SHA256:
        raise ValueError("checkpoint does not match the locked MPA-0 SHA256")
    atoms_list, sources = [], []
    for filename in SAMPLES:
        path = root / filename
        atoms = read(path)
        occupancy = atoms.info.get("occupancy", {})
        if any(
            len(site) != 1 or any(abs(float(v) - 1) > 1e-8 for v in site.values())
            for site in occupancy.values()
        ):
            raise ValueError(f"partial/mixed occupancy: {filename}")
        if (
            not atoms.pbc.all()
            or not np.isfinite(atoms.positions).all()
            or not np.isfinite(atoms.cell).all()
            or atoms.get_volume() <= 0
        ):
            raise ValueError(f"invalid periodic structure: {filename}")
        atoms_list.append(atoms)
        sources.append(
            {
                "filename": filename,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "num_atoms": len(atoms),
                "formula": atoms.get_chemical_formula(),
                "volume_A3": atoms.get_volume(),
                "cell_parameters": atoms.cell.cellpar().tolist(),
            }
        )
    evidence, model_info = [], None
    for dtype in (torch.float64, torch.float32):
        # ASE's calculator changes the process default dtype. MACE 0.3.15's
        # ZBL integer-Z power promotion also depends on it. Configure each
        # isolated probe case explicitly; do not cast model outputs to hide it.
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
        supported = set(model.model.atomic_numbers.tolist())
        unknown = {int(z) for atoms in atoms_list for z in atoms.numbers} - supported
        if unknown:
            raise ValueError(f"unsupported atomic numbers: {sorted(unknown)}")
        data_list = [AtomicData.from_atoms(atoms, dtype=dtype) for atoms in atoms_list]
        batch = Batch.from_data_list(data_list)
        positions, cell = batch.positions.clone(), batch.cell.clone()
        output, signature = evaluate(model, batch)
        if signature[:, 2:].count_nonzero() == 0:
            raise AssertionError("fixtures do not exercise periodic images")
        singles = [
            evaluate(model, Batch.from_data_list([data]))[0] for data in data_list
        ]
        expected = {key: torch.cat([value[key] for value in singles]) for key in output}
        item = {
            "dtype": str(dtype),
            "process_default_dtype": str(torch.get_default_dtype()),
            "batch_ptr": batch.batch_ptr.tolist(),
            "neighbor_edges": len(signature),
            "cross_system_edges": 0,
            "nonzero_image_shift_entries": int(signature[:, 2:].count_nonzero()),
            "batch_single_comparison": compare(output, expected, dtype),
            "energy_eV": output["energy"].tolist(),
            "fmax_eV_A": [
                float(value["forces"].norm(dim=1).max()) for value in singles
            ],
            "stress_eV_A3": output["stress"].tolist(),
        }
        if dtype == torch.float64:
            calculator = MACECalculator(
                models=model.model,
                device="cpu",
                default_dtype="float64",
                enable_cueq=False,
            )
            ase_values = {key: [] for key in output}
            for atoms in atoms_list:
                reference = atoms.copy()
                reference.calc = calculator
                ase_values["energy"].append(
                    torch.tensor([reference.get_potential_energy()], dtype=dtype)
                )
                ase_values["forces"].append(
                    torch.tensor(reference.get_forces(), dtype=dtype)
                )
                ase_values["stress"].append(
                    torch.tensor(reference.get_stress(voigt=False)[None], dtype=dtype)
                )
            ase_output = {key: torch.cat(values) for key, values in ase_values.items()}
            item["ase_calculator_comparison"] = compare(output, ase_output, dtype)
            item["finite_difference"] = derivatives(model, batch, output, signature)
            model_info = {
                "type": type(model.model).__name__,
                "cutoff_A": float(model.model.r_max),
                "atomic_numbers": sorted(supported),
                "heads": model.model.heads,
                "pair_repulsion": bool(model.model.pair_repulsion),
                "interaction_types": [
                    type(block).__name__ for block in model.model.interactions
                ],
            }
        assert torch.equal(batch.positions, positions) and torch.equal(batch.cell, cell)
        evidence.append(item)
        print(
            json.dumps(
                {"completed_dtype": str(dtype), "elapsed_s": time.monotonic() - started}
            ),
            flush=True,
        )
    return {
        "status": "passed",
        "scope": "CPU single points only; no optimizer or cell propagation",
        "device": "cpu",
        "checkpoint_sha256": checkpoint_hash,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("torch", "mace-torch", "e3nn", "ase", "numpy")
        },
        "backend": "torch_reference",
        "enable_cueq": False,
        "compile_model": False,
        "model": model_info,
        "sources": sources,
        "results": evidence,
        "elapsed_s": time.monotonic() - started,
        "HCU_MPA0": "not_run",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    previous_dtype = torch.get_default_dtype()
    try:
        evidence = run(
            args.checkpoint.expanduser().resolve(),
            args.data_root.expanduser().resolve(),
        )
    finally:
        torch.set_default_dtype(previous_dtype)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": evidence["status"],
                "output": str(args.output),
                "elapsed_s": evidence["elapsed_s"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
