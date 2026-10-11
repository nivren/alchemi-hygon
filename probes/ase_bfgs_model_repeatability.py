# SPDX-License-Identifier: Apache-2.0
"""Same-geometry MPA-0 force/stress repeatability diagnostic, separate from solver gates."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from ase.io import read
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import ASECompatibleBFGSVariableCell
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.hooks import NeighborListHook
from nvalchemi.models.mace import MACEWrapper

p = argparse.ArgumentParser()
p.add_argument("--manifest", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument(
    "--checkpoint",
    type=Path,
    default=Path("/home/wangleping/.cache/mace/mace-mpa-0-medium.model"),
)
a = p.parse_args()
device = torch.device("cuda", torch.cuda.current_device())
inputs = json.loads(a.manifest.read_text())["li"]
rows = []
for i, row in enumerate(inputs):
    data = AtomicData.from_atoms(read(row["path"]), dtype=torch.float64).to(device)
    data.add_system_property("system_id", torch.tensor([[i]], device=device))
    rows.append(data)
batch = Batch.from_data_list(rows)
batch.forces = torch.zeros_like(batch.positions)
batch.energy = torch.zeros(batch.num_graphs, 1, device=device, dtype=torch.float64)
batch.stress = torch.zeros(batch.num_graphs, 3, 3, device=device, dtype=torch.float64)
model = MACEWrapper.from_checkpoint(
    a.checkpoint,
    device=device,
    dtype=torch.float64,
    enable_cueq=False,
    compile_model=False,
)
model.set_config("active_outputs", {"energy", "forces", "stress"})
model.eval()
for parameter in model.parameters():
    parameter.requires_grad_(False)
d = ASECompatibleBFGSVariableCell(
    model, backend="hip.bfgs-ase-jacobi-streams-v1", fmax=1e-10
)
d.register_hook(
    NeighborListHook(
        model.model_config.neighbor_config,
        skin=0,
        backend="torch_reference",
        stage=DynamicsStage.BEFORE_COMPUTE,
    )
)
# Establish neighbor data once. Repeated model calls below hold all geometry fixed.
d.step(batch)
saved_q, saved_cell = batch.positions.clone(), batch.cell.clone()
values = []
alternate = torch.cuda.Stream()
alternate.wait_stream(torch.cuda.current_stream())
for i in range(12):
    stream = torch.cuda.current_stream() if i % 2 == 0 else alternate
    with torch.cuda.stream(stream):
        out = model(batch)
        row = {key: out[key].clone() for key in ("energy", "forces", "stress")}
        done = torch.cuda.Event()
        done.record()
    torch.cuda.current_stream().wait_event(done)
    values.append(row)
assert torch.equal(saved_q, batch.positions) and torch.equal(saved_cell, batch.cell)
differences = {
    key: max(float((r[key] - values[0][key]).abs().max()) for r in values)
    for key in values[0]
}
result = dict(
    status="passed_diagnostic",
    samples=12,
    unchanged_geometry=True,
    alternate_streams=True,
    checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
    max_absolute_difference=differences,
    scope="Same geometry and neighbor data, repeated HCU model calls; identifies possible model variation but does not prove the cause of any long trajectory difference",
)
a.output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
