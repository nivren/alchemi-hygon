# SPDX-License-Identifier: Apache-2.0
"""Save geometry and BFGS state together, or resume a trusted local checkpoint."""

import argparse
import hashlib
from pathlib import Path

import torch
from ase import Atoms
from ase.io import read, write

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import ASECompatibleBFGSVariableCell
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.hooks import NeighborListHook
from nvalchemi.models.mace import MACEWrapper

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True, help="MACE model file")
start = parser.add_mutually_exclusive_group(required=True)
start.add_argument("--cif", help="start a new optimization")
start.add_argument("--resume", type=Path, help="trusted restart.pt made by this example")
parser.add_argument("--output-dir", type=Path, required=True, help="new output directory")
parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
parser.add_argument("--backend", choices=("torch_reference", "hip.bfgs-ase-jacobi-streams-v1"), default="torch_reference")
parser.add_argument("--steps", type=int, default=3, help="maximum additional steps")
args = parser.parse_args()
if args.steps < 1 or args.output_dir.exists():
    parser.error("--steps must be positive and --output-dir must not exist")

torch.set_default_dtype(torch.float64)
with args.checkpoint.open("rb") as stream:
    model_hash = hashlib.file_digest(stream, "sha256").hexdigest()
saved = None
if args.resume:
    # Batch is a project object: only load a trusted local file, in the same environment.
    saved = torch.load(args.resume, map_location=args.device, weights_only=False)
    if saved["format_version"] != 1 or saved["model_sha256"] != model_hash or saved["backend"] != args.backend:
        parser.error("restart format, model hash or optimizer backend does not match")
    batch = saved["batch"]
else:
    atoms = read(args.cif)
    if not atoms.pbc.all():
        parser.error("input must be periodic in three directions")
    data = AtomicData.from_atoms(atoms, dtype=torch.float64)
    data.forces = torch.zeros_like(data.positions)
    data.energy = torch.zeros(1, 1, dtype=torch.float64)
    data.stress = torch.zeros(1, 3, 3, dtype=torch.float64)
    batch = Batch.from_data_list([data]).to(args.device)
model = MACEWrapper.from_checkpoint(
    args.checkpoint, device=torch.device(args.device), dtype=torch.float64,
    enable_cueq=False, compile_model=False,
)
missing = set(batch.atomic_numbers.cpu().tolist()) - set(model.model.atomic_numbers.cpu().tolist())
if missing:
    parser.error(f"model does not support atomic numbers {sorted(missing)}")
model.set_config("active_outputs", {"energy", "forces", "stress"})
model.eval()
optimizer = ASECompatibleBFGSVariableCell(model=model, backend=args.backend, fmax=0.05)
optimizer.register_hook(
    NeighborListHook(model.model_config.neighbor_config, skin=0.0, backend="torch_reference"),
    stage=DynamicsStage.BEFORE_COMPUTE,
)
if saved is not None:
    optimizer.load_state_dict(saved["optimizer"], device=args.device)
optimizer.run(batch, n_steps=args.steps)

args.output_dir.mkdir(parents=True, exist_ok=False)
torch.save(dict(format_version=1, batch=batch.clone(), optimizer=optimizer.state_dict(),
                model_sha256=model_hash, backend=args.backend), args.output_dir / "restart.pt")
atoms = Atoms(numbers=batch.atomic_numbers.cpu().numpy(),
              positions=batch.positions.detach().cpu().numpy(),
              cell=batch.cell[0].detach().cpu().numpy(), pbc=batch.pbc[0].cpu().numpy())
write(args.output_dir / "structure.cif", atoms)
generalized, _ = optimizer.filter_state.generalized_forces(batch.forces, batch.stress[0], batch.cell[0])
print(f"total steps: {optimizer.step_count}")
print(f"energy: {batch.energy.item():.8f} eV")
print(f"converged: {generalized.norm(dim=-1).max().item() < 0.05}")
print(f"restart: {args.output_dir / 'restart.pt'}")
