# SPDX-License-Identifier: Apache-2.0
"""Short fixed-cell MACE NVE trajectory; timestep in fs."""

import argparse

import numpy as np
import torch
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import NVE
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.hooks import NeighborListHook
from nvalchemi.models.mace import MACEWrapper
from nvalchemi.neighbors import compute_neighbors

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--cif", required=True)
parser.add_argument("--output", required=True, help="ASE .traj output (includes initial frame)")
parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
parser.add_argument("--steps", type=int, default=10)
parser.add_argument("--dt", type=float, default=0.5, help="timestep in fs")
parser.add_argument("--temperature", type=float, default=300, help="initial sampling temperature in K")
args = parser.parse_args()
if args.steps < 1 or not np.isfinite(args.dt) or args.dt <= 0:
    parser.error("--steps and --dt must be positive")
if not np.isfinite(args.temperature) or args.temperature <= 0:
    parser.error("--temperature must be positive")

torch.set_default_dtype(torch.float64)
atoms = read(args.cif)
if not atoms.pbc.all() or len(atoms) < 2:
    parser.error("input must have at least two atoms and full three-dimensional periodicity")
model = MACEWrapper.from_checkpoint(
    args.checkpoint, device=torch.device(args.device), dtype=torch.float64,
    enable_cueq=False, compile_model=False,
)
missing = set(atoms.numbers) - set(model.model.atomic_numbers.cpu().tolist())
if missing:
    parser.error(f"model does not support atomic numbers {sorted(missing)}")
model.set_config("active_outputs", {"energy", "forces"})
model.eval()
MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature, rng=np.random.default_rng(42))
Stationary(atoms, preserve_temperature=False)
data = AtomicData.from_atoms(atoms, dtype=torch.float64)
data.velocities = torch.tensor(atoms.get_velocities(), dtype=torch.float64)
batch = Batch.from_data_list([data]).to(args.device)
# Velocity Verlet needs the force at the initial geometry before its first kick.
compute_neighbors(batch, config=model.model_config.neighbor_config, backend="torch_reference")
initial = model(batch)
batch.forces = initial["forces"].detach()
batch.energy = initial["energy"].detach().reshape(1, 1)
nve = NVE(model=model, dt=args.dt, backend="torch_reference")
nve.register_hook(
    NeighborListHook(model.model_config.neighbor_config, skin=0.0, backend="torch_reference"),
    stage=DynamicsStage.BEFORE_COMPUTE,
)

print("step,time_fs,potential_eV,kinetic_eV,total_eV")
with Trajectory(args.output, "w", atoms) as trajectory:
    for step in range(args.steps + 1):
        if step:
            nve.step(batch)
        atoms.set_positions(batch.positions.detach().cpu().numpy())
        atoms.set_velocities(batch.velocities.detach().cpu().numpy())
        potential = batch.energy.item()
        kinetic = atoms.get_kinetic_energy()
        atoms.calc = SinglePointCalculator(
            atoms, energy=potential, forces=batch.forces.detach().cpu().numpy(),
        )
        trajectory.write(atoms)
        print(f"{step},{step * args.dt:.6f},{potential:.12f},{kinetic:.12f},{potential + kinetic:.12f}")
