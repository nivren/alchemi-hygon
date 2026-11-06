# SPDX-License-Identifier: Apache-2.0
"""Relax a periodic structure and its cell with MACE and FIRE2."""

import argparse

import torch
from ase.io import read, write
from ase.units import GPa

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import FIRE2VariableCell
from nvalchemi.dynamics.base import ConvergenceHook, DynamicsStage
from nvalchemi.dynamics.hooks.cell_align import AlignCellHook
from nvalchemi.hooks import NeighborListHook
from nvalchemi.models.mace import MACEWrapper


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--cif", required=True, help="input periodic structure")
parser.add_argument("--checkpoint", required=True, help="local MACE checkpoint")
parser.add_argument("--output", required=True, help="output structure (e.g. relaxed.cif)")
parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
parser.add_argument("--steps", type=int, default=100, help="maximum FIRE2 steps")
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
model.set_config("active_outputs", {"energy", "forces", "stress"})
model.eval()
data = AtomicData.from_atoms(atoms, dtype=torch.float64)
data.velocities = torch.zeros_like(data.positions)
data.forces = torch.zeros_like(data.positions)
data.energy = torch.zeros(1, 1, dtype=torch.float64)
data.stress = torch.zeros(1, 3, 3, dtype=torch.float64)
batch = Batch.from_data_list([data]).to(args.device)

fmax_limit = 0.01  # eV/Å
stress_limit = 0.1 * GPa  # 0.1 GPa in eV/Å³
convergence = ConvergenceHook(criteria=[
    {"key": "forces", "threshold": fmax_limit, "reduce_op": "norm", "reduce_dims": -1},
    {"key": "stress", "threshold": stress_limit,
     "custom_op": lambda stress: stress.abs().amax((-1, -2)) <= stress_limit},
])
fire = FIRE2VariableCell(
    model=model, dt=0.02, n_steps=args.steps, backend="torch_reference",
    hooks=[AlignCellHook(backend="torch_reference")], convergence_hook=convergence,
)
fire.register_hook(
    NeighborListHook(model.model_config.neighbor_config, skin=0.0, backend="torch_reference"),
    stage=DynamicsStage.BEFORE_COMPUTE,
)
fire.run(batch)

fmax = batch.forces.norm(dim=-1).max().item()
max_stress = batch.stress.abs().max().item()
atoms.set_cell(batch.cell[0].detach().cpu().numpy(), scale_atoms=False)
atoms.set_positions(batch.positions.detach().cpu().numpy())
write(args.output, atoms)
print(f"steps: {fire.step_count}")
print(f"energy: {batch.energy.item():.8f} eV")
print(f"max force: {fmax:.8f} eV/Å (limit {fmax_limit})")
print(f"max absolute stress: {max_stress / GPa:.8f} GPa (limit 0.1)")
print(f"converged: {fmax <= fmax_limit and max_stress <= stress_limit}")
print(f"structure: {args.output}")
