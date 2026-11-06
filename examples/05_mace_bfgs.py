# SPDX-License-Identifier: Apache-2.0
"""Relax a periodic structure with MACE and ASE-compatible variable-cell BFGS."""

import argparse

import torch
from ase.io import read, write
from ase.units import GPa

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import ASECompatibleBFGSVariableCell
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.hooks import NeighborListHook
from nvalchemi.models.mace import MACEWrapper

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--cif", required=True)
parser.add_argument("--output", required=True, help="output CIF")
parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
parser.add_argument("--backend", choices=("torch_reference", "hip.bfgs-ase-jacobi-streams-v1"), default="torch_reference")
parser.add_argument("--steps", type=int, default=100)
args = parser.parse_args()
if args.steps < 1:
    parser.error("--steps must be positive")

torch.set_default_dtype(torch.float64)
atoms = read(args.cif)
if not atoms.pbc.all():
    parser.error("input must be periodic in all three directions")
model = MACEWrapper.from_checkpoint(
    args.checkpoint, device=torch.device(args.device), dtype=torch.float64,
    enable_cueq=False, compile_model=False,
)
missing = set(atoms.numbers) - set(model.model.atomic_numbers.cpu().tolist())
if missing:
    parser.error(f"model does not support atomic numbers {sorted(missing)}")
model.set_config("active_outputs", {"energy", "forces", "stress"})
model.eval()
data = AtomicData.from_atoms(atoms, dtype=torch.float64)
data.forces = torch.zeros_like(data.positions)
data.energy = torch.zeros(1, 1, dtype=torch.float64)
data.stress = torch.zeros(1, 3, 3, dtype=torch.float64)
batch = Batch.from_data_list([data]).to(args.device)
optimizer = ASECompatibleBFGSVariableCell(
    model=model, backend=args.backend, fmax=0.05, maxstep=0.2,
)
optimizer.register_hook(
    NeighborListHook(model.model_config.neighbor_config, skin=0.0, backend="torch_reference"),
    stage=DynamicsStage.BEFORE_COMPUTE,
)
optimizer.run(batch, n_steps=args.steps)

generalized, _ = optimizer.filter_state.generalized_forces(
    batch.forces, batch.stress[0], batch.cell[0],
)
gmax = generalized.norm(dim=-1).max().item()
atoms.set_cell(batch.cell[0].detach().cpu().numpy(), scale_atoms=False)
atoms.set_positions(batch.positions.detach().cpu().numpy())
write(args.output, atoms)
print(f"steps: {optimizer.step_count}")
print(f"energy: {batch.energy.item():.8f} eV")
print(f"max physical force: {batch.forces.norm(dim=-1).max().item():.8f} eV/Å")
print(f"max absolute stress: {batch.stress.abs().max().item() / GPa:.8f} GPa")
print(f"max generalized force: {gmax:.8f} (limit < 0.05)")
print(f"converged: {gmax < 0.05}")
print(f"structure: {args.output}")
