# SPDX-License-Identifier: Apache-2.0
"""Two-stage variable-cell relaxation: a FIRE2 budget followed by BFGS."""

import argparse

import torch
from ase.io import read, write
from ase.units import GPa

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import ASECompatibleBFGSVariableCell, FIRE2VariableCell
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.hooks.cell_align import AlignCellHook
from nvalchemi.hooks import NeighborListHook
from nvalchemi.models.mace import MACEWrapper

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--cif", required=True)
parser.add_argument("--output-prefix", required=True, help="prefix for .fire2.cif and .bfgs.cif")
parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
parser.add_argument("--backend", choices=("torch_reference", "hip.bfgs-ase-jacobi-streams-v1"), default="torch_reference")
parser.add_argument("--fire-steps", type=int, default=10)
parser.add_argument("--bfgs-steps", type=int, default=100)
args = parser.parse_args()
if args.fire_steps < 1 or args.bfgs_steps < 1:
    parser.error("--fire-steps and --bfgs-steps must be positive")

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
data.velocities = torch.zeros_like(data.positions)
data.forces = torch.zeros_like(data.positions)
data.energy = torch.zeros(1, 1, dtype=torch.float64)
data.stress = torch.zeros(1, 3, 3, dtype=torch.float64)
batch = Batch.from_data_list([data]).to(args.device)
# First stage: a fixed budget, without a convergence claim.
fire = FIRE2VariableCell(
    model=model, dt=0.02, backend="torch_reference",
    hooks=[AlignCellHook(backend="torch_reference")],
)
fire.register_hook(
    NeighborListHook(model.model_config.neighbor_config, skin=0.0, backend="torch_reference"),
    stage=DynamicsStage.BEFORE_COMPUTE,
)
fire.run(batch, n_steps=args.fire_steps)
atoms.set_cell(batch.cell[0].detach().cpu().numpy(), scale_atoms=False)
atoms.set_positions(batch.positions.detach().cpu().numpy())
write(f"{args.output_prefix}.fire2.cif", atoms)
print(f"FIRE2 steps: {fire.step_count}")
print(f"FIRE2 energy: {batch.energy.item():.8f} eV")

# Bind a fresh BFGS history and reference cell at the handoff geometry.
optimizer = ASECompatibleBFGSVariableCell(
    model=model, backend=args.backend, fmax=0.05, maxstep=0.2,
)
optimizer.register_hook(
    NeighborListHook(model.model_config.neighbor_config, skin=0.0, backend="torch_reference"),
    stage=DynamicsStage.BEFORE_COMPUTE,
)
optimizer.run(batch, n_steps=args.bfgs_steps)

generalized, _ = optimizer.filter_state.generalized_forces(
    batch.forces, batch.stress[0], batch.cell[0],
)
gmax = generalized.norm(dim=-1).max().item()
atoms.set_cell(batch.cell[0].detach().cpu().numpy(), scale_atoms=False)
atoms.set_positions(batch.positions.detach().cpu().numpy())
write(f"{args.output_prefix}.bfgs.cif", atoms)
print(f"BFGS steps: {optimizer.step_count}")
print(f"energy: {batch.energy.item():.8f} eV")
print(f"max physical force: {batch.forces.norm(dim=-1).max().item():.8f} eV/Å")
print(f"max absolute stress: {batch.stress.abs().max().item() / GPa:.8f} GPa")
print(f"max generalized force: {gmax:.8f} (limit < 0.05)")
print(f"converged: {gmax < 0.05}")
print(f"structure: {args.output_prefix}.bfgs.cif")
