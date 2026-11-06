# SPDX-License-Identifier: Apache-2.0
"""Evaluate energy, forces and stress for periodic structures with MACE."""

import argparse
import json

import torch
from ase.io import read

from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.mace import MACEWrapper
from nvalchemi.neighbors import compute_neighbors

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True, help="local MACE checkpoint")
parser.add_argument("--cif", nargs="+", required=True, help="one or more periodic structures")
parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
args = parser.parse_args()

torch.set_default_dtype(torch.float64)
structures = [read(path) for path in args.cif]
if any(not atoms.pbc.all() for atoms in structures):
    parser.error("inputs must be periodic in all three directions")
model = MACEWrapper.from_checkpoint(
    args.checkpoint, device=torch.device(args.device), dtype=torch.float64,
    enable_cueq=False, compile_model=False,
)
supported = set(model.model.atomic_numbers.cpu().tolist())
for path, atoms in zip(args.cif, structures):
    missing = set(atoms.numbers) - supported
    if missing:
        parser.error(f"{path}: model does not support atomic numbers {sorted(missing)}")
model.set_config("active_outputs", {"energy", "forces", "stress"})
model.eval()
batch = Batch.from_data_list(
    [AtomicData.from_atoms(atoms, dtype=torch.float64) for atoms in structures]
).to(args.device)
compute_neighbors(batch, config=model.model_config.neighbor_config, backend="torch_reference")
result = model(batch)

ptr = batch.batch_ptr.cpu().tolist()
for i, path in enumerate(args.cif):
    print(json.dumps({
        "structure": path,
        "energy_eV": result["energy"][i].item(),
        "forces_eV_per_A": result["forces"][ptr[i]:ptr[i + 1]].detach().cpu().tolist(),
        "stress_eV_per_A3": result["stress"][i].detach().cpu().tolist(),
    }))
