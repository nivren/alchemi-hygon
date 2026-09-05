"""Real MACE checkpoint smoke. Only load a trusted user-supplied checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import torch

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--device', choices=['cpu','cuda'], required=True)
a = p.parse_args()
if a.device == 'cuda': assert torch.version.hip and torch.cuda.is_available()
# mace.data creates the graph through MACE's own neighbor construction.
import mace
from mace import data
from mace.tools import AtomicNumberTable
from mace.tools.torch_geometric.batch import Batch
from ase import Atoms
model = torch.load(a.checkpoint, map_location=a.device, weights_only=False).double()
model.train()
for param in model.parameters(): param.requires_grad_(True)
config = data.config_from_atoms(Atoms('H2O', positions=[[0.,0.,0.],[0.96,0.,0.],[-0.24,0.93,0.]]))
graph = data.AtomicData.from_config(config, z_table=AtomicNumberTable([int(z) for z in model.atomic_numbers]), cutoff=float(model.r_max))
batch = Batch.from_data_list([graph]).to(a.device)
for key, value in batch.to_dict().items():
    if isinstance(value, torch.Tensor) and value.is_floating_point(): batch[key] = value.double()
out = model(batch.to_dict(), training=True, compute_force=True)
assert torch.isfinite(out['energy']).all() and torch.isfinite(out['forces']).all()
loss = out['forces'].square().mean()
grads = torch.autograd.grad(loss, tuple(model.parameters()), allow_unused=True)
nonzero = [g for g in grads if g is not None and g.abs().max().item() > 0]
assert nonzero and all(torch.isfinite(g).all() for g in nonzero), 'force loss must reach model parameters'
if a.device == 'cuda': torch.cuda.synchronize()
print(json.dumps({'status':'passed','device':a.device,'mace_file':mace.__file__, 'checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(), 'energy':out['energy'].tolist(),'force_loss':loss.item(),'nonzero_parameter_gradients':len(nonzero),'scope':'MACE direct model; toolkit wrapper not tested'}))
