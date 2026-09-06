"""Run a cached MACE checkpoint through the framework wrapper and Torch neighbors."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from ase import Atoms
from ase.io import read

from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.mace import MACEWrapper
from nvalchemi.neighbors import compute_neighbors


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--cif", type=Path, nargs="+")
parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
args = parser.parse_args()

if args.device == "cuda":
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError(
            "HCU device requested but the HIP Torch device is unavailable"
        )

device = torch.device(args.device)
checkpoint = args.checkpoint.expanduser().resolve()
if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)

if args.cif is None:
    structures = [
        Atoms(
            "H2O",
            positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.0, 0.96, 0.0]],
        )
    ]
    cifs = []
else:
    cifs = [path.expanduser().resolve() for path in args.cif]
    for cif in cifs:
        if not cif.is_file():
            raise FileNotFoundError(cif)
    structures = [read(cif) for cif in cifs]

wrapper = MACEWrapper.from_checkpoint(
    str(checkpoint), device=device, dtype=torch.float32
)
supported = {int(z) for z in wrapper.model.atomic_numbers.detach().cpu().tolist()}
observed = {int(z) for atoms in structures for z in atoms.numbers.tolist()}
unknown = sorted(observed - supported)
if unknown:
    raise ValueError(
        f"input contains atomic numbers outside checkpoint coverage: {unknown}"
    )

batch = Batch.from_data_list([AtomicData.from_atoms(atoms) for atoms in structures]).to(
    device
)
compute_neighbors(
    batch,
    config=wrapper.model_config.neighbor_config,
    backend="torch_reference",
)
wrapper.eval()
output = wrapper(batch)
energy = output["energy"].detach()
forces = output["forces"].detach()
if not torch.isfinite(energy).all() or not torch.isfinite(forces).all():
    raise FloatingPointError("MACE wrapper returned a non-finite energy or force")
edge_src = batch.neighbor_list[:, 0].long()
edge_dst = batch.neighbor_list[:, 1].long()
cross_system_edges = int(
    (batch.batch_idx[edge_src] != batch.batch_idx[edge_dst]).sum().item()
)
if cross_system_edges:
    raise AssertionError(
        f"neighbor list contains {cross_system_edges} cross-system edges"
    )
force_sums = []
for graph_index in range(batch.num_graphs):
    node_mask = batch.batch_idx == graph_index
    force_sums.append(forces[node_mask].sum(dim=0).cpu().tolist())
if args.device == "cuda":
    torch.cuda.synchronize()

print(
    json.dumps(
        {
            "status": "passed",
            "device": args.device,
            "torch_device_name": (
                torch.cuda.get_device_name() if args.device == "cuda" else "cpu"
            ),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "cifs": [str(cif) for cif in cifs],
            "cif_sha256": [
                hashlib.sha256(cif.read_bytes()).hexdigest() for cif in cifs
            ],
            "natoms_per_system": [len(atoms) for atoms in structures],
            "num_systems": batch.num_graphs,
            "atomic_numbers": sorted(observed),
            "pbc_per_system": [
                [bool(value) for value in atoms.pbc] for atoms in structures
            ],
            "neighbor_edges": int(batch.neighbor_list.shape[0]),
            "cross_system_edges": cross_system_edges,
            "batch_ptr": batch.batch_ptr.cpu().tolist(),
            "energy": energy.cpu().tolist(),
            "force_norm": torch.linalg.vector_norm(forces).item(),
            "sum_force_per_system": force_sums,
            "scope": "MACEWrapper + compute_neighbors(torch_reference); no dynamics",
        }
    )
)
