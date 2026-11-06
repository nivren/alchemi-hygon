# SPDX-License-Identifier: Apache-2.0
"""Variable-cell BFGS with a small live batch and automatic refill."""

import argparse
import json
from pathlib import Path

import torch
from ase.io import read, write

from nvalchemi.data import AtomicData
from nvalchemi.dynamics import ASECompatibleBFGSVariableCell, FusedStage, SizeAwareSampler
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.sinks import HostMemory
from nvalchemi.hooks import NeighborListHook
from nvalchemi.models.mace import MACEWrapper


class StructureDataset:
    def __init__(self, systems):
        self.systems = systems

    def __len__(self):
        return len(self.systems)

    def get_metadata(self, index):
        return self.systems[index].num_nodes, 0

    def __getitem__(self, index):
        return self.systems[index].clone(), {}


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--cif", nargs="+", required=True)
parser.add_argument("--output-dir", type=Path, required=True, help="new output directory")
parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
parser.add_argument("--backend", choices=("torch_reference", "hip.bfgs-ase-jacobi-streams-v1"), default="torch_reference")
parser.add_argument("--batch-size", type=int, default=2)
parser.add_argument("--max-atoms", type=int, default=64)
parser.add_argument("--steps", type=int, default=100, help="global fused-step limit")
args = parser.parse_args()
if min(args.batch_size, args.max_atoms, args.steps) < 1:
    parser.error("batch size, max atoms and steps must be positive")

torch.set_default_dtype(torch.float64)
structures = [read(path) for path in args.cif]
if any(not atoms.pbc.all() for atoms in structures):
    parser.error("all inputs must be periodic in three directions")
model = MACEWrapper.from_checkpoint(
    args.checkpoint, device=torch.device(args.device), dtype=torch.float64,
    enable_cueq=False, compile_model=False,
)
model.set_config("active_outputs", {"energy", "forces", "stress"})
model.eval()
supported = set(model.model.atomic_numbers.cpu().tolist())
systems = []
for index, atoms in enumerate(structures):
    if set(atoms.numbers) - supported:
        parser.error(f"{args.cif[index]} contains elements not supported by the model")
    data = AtomicData.from_atoms(atoms, dtype=torch.float64)
    data.forces = torch.zeros_like(data.positions)
    data.energy = torch.zeros(1, 1, dtype=torch.float64)
    data.stress = torch.zeros(1, 3, 3, dtype=torch.float64)
    data.add_system_property("source_index", torch.tensor([[index]], dtype=torch.long))
    systems.append(data)
sampler = SizeAwareSampler(
    StructureDataset(systems), max_atoms=args.max_atoms, max_batch_size=args.batch_size,
)
sink = HostMemory(capacity=len(systems))
optimizer = ASECompatibleBFGSVariableCell(
    model=model, backend=args.backend, fmax=0.05, device_type=args.device,
)
workflow = FusedStage(
    [(0, optimizer)], sampler=sampler, sinks=[sink], refill_frequency=1,
    device_type=args.device,
)
workflow.register_hook(
    NeighborListHook(model.model_config.neighbor_config, skin=0.0, backend="torch_reference"),
    stage=DynamicsStage.BEFORE_COMPUTE,
)
initial = sampler.build_initial_batch().to(args.device)
print("initial sources:", initial.source_index.flatten().tolist(), flush=True)
print("initial atom counts:", initial.num_nodes_per_graph.tolist(), flush=True)
remaining = workflow.run(batch=initial, n_steps=args.steps)
if remaining is not None or len(sink) != len(systems):
    raise RuntimeError(f"global step limit reached: {len(sink)}/{len(systems)} completed; increase --steps")
results = sink.read()
assert sorted(results.source_index.flatten().tolist()) == list(range(len(systems)))
assert sorted(results.system_id.flatten().tolist()) == list(range(len(systems)))
assert bool((results.status == 1).all())
args.output_dir.mkdir(parents=True, exist_ok=False)
summary = []
for data in results.to_data_list():
    source = int(data.source_index.item())
    atoms = structures[source].copy()
    atoms.set_cell(data.cell[0].numpy(), scale_atoms=False)
    atoms.set_positions(data.positions.numpy())
    filename = f"{source:03d}.cif"
    write(args.output_dir / filename, atoms)
    summary.append(dict(source_index=source, system_id=int(data.system_id.item()),
                        input=args.cif[source], output=filename, energy_eV=data.energy.item(),
                        max_force_eV_per_A=data.forces.norm(dim=-1).max().item(), status="converged"))
(args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(f"completed: {len(sink)} systems in {workflow.step_count} fused steps")
