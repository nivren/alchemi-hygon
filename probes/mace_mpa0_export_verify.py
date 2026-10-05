# SPDX-License-Identifier: Apache-2.0
"""Independently verify saved CIF metrics; this does not run optimization."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from ase.io import read

from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.mace import MACEWrapper
from nvalchemi.neighbors import compute_neighbors

MODEL_SHA256 = "75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    with args.checkpoint.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == MODEL_SHA256
    previous_dtype = torch.get_default_dtype()
    cases = []
    try:
        for dtype in (torch.float64, torch.float32):
            torch.set_default_dtype(dtype)
            model = MACEWrapper.from_checkpoint(
                args.checkpoint,
                device=torch.device("cpu"),
                dtype=dtype,
                enable_cueq=False,
                compile_model=False,
            )
            model.set_config("active_outputs", {"energy", "forces", "stress"})
            model.eval()
            for path in args.results:
                records = list(map(json.loads, path.read_text().splitlines()))
                for record in records:
                    if record["dtype"] != str(dtype) or "energy_eV" not in record:
                        continue
                    atoms = read(record["final_cif"], format="cif")
                    batch = Batch.from_data_list(
                        [AtomicData.from_atoms(atoms, dtype=dtype)]
                    )
                    compute_neighbors(
                        batch,
                        config=model.model_config.neighbor_config,
                        backend="torch_reference",
                    )
                    outputs = model(batch)
                    energy = float(outputs["energy"].detach().item())
                    force = float(outputs["forces"].detach().norm(dim=-1).max())
                    stress = (
                        float(outputs["stress"].detach().abs().max()) * 160.21766208
                    )
                    passed = (
                        force <= record["fmax_tolerance_eV_A"]
                        and stress <= record["stress_tolerance_GPa"]
                    )
                    energy_error = abs(energy - record["energy_eV"])
                    force_error = abs(force - record["fmax_eV_A"])
                    assert energy_error <= (2e-7 if dtype == torch.float64 else 2e-3)
                    assert force_error <= (2e-7 if dtype == torch.float64 else 2e-4)
                    if record["status"] == "converged":
                        assert passed, (path, record["source_index"], force, stress)
                    cases.append(
                        {
                            "results": str(path),
                            "source_index": record["source_index"],
                            "optimizer": record["optimizer"],
                            "dtype": str(dtype),
                            "recorded_status": record["status"],
                            "steps": record["steps"],
                            "energy_eV": energy,
                            "fmax_eV_A": force,
                            "max_abs_stress_GPa": stress,
                            "final_thresholds_passed": passed,
                            "energy_error_eV": energy_error,
                            "fmax_error_eV_A": force_error,
                        }
                    )
    finally:
        torch.set_default_dtype(previous_dtype)
    assert cases
    print(
        json.dumps(
            {
                "status": "passed",
                "device": "cpu",
                "model_sha256": MODEL_SHA256,
                "scope": "independent final CIF read/model/topology and thresholds",
                "cases": cases,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
