"""Print stage timing for a MACE wrapper Batch execution."""

import argparse
import json
import time
from pathlib import Path

import torch
from ase.io import read

from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.mace import MACEWrapper
from nvalchemi.neighbors import compute_neighbors


parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--cif", type=Path, nargs="+", required=True)
parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
args = parser.parse_args()

if args.device == "cuda" and (not torch.version.hip or not torch.cuda.is_available()):
    raise RuntimeError("HCU device requested but the HIP Torch device is unavailable")

device = torch.device(args.device)
checkpoint = args.checkpoint.expanduser().resolve()
cifs = [path.expanduser().resolve() for path in args.cif]
if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)
for cif in cifs:
    if not cif.is_file():
        raise FileNotFoundError(cif)

start = time.monotonic()


def mark(stage: str, **extra: object) -> None:
    print(
        json.dumps({"stage": stage, "elapsed_s": time.monotonic() - start, **extra}),
        flush=True,
    )


wrapper = MACEWrapper.from_checkpoint(
    str(checkpoint), device=device, dtype=torch.float32
)
mark(
    "model_loaded",
    device=(torch.cuda.get_device_name() if args.device == "cuda" else "cpu"),
)
structures = [read(cif) for cif in cifs]
batch = Batch.from_data_list([AtomicData.from_atoms(atoms) for atoms in structures]).to(
    device
)
mark(
    "batch_built",
    num_systems=batch.num_graphs,
    batch_ptr=batch.batch_ptr.cpu().tolist(),
)
compute_neighbors(
    batch,
    config=wrapper.model_config.neighbor_config,
    backend="torch_reference",
)
mark("neighbors_built", edges=int(batch.neighbor_list.shape[0]))
wrapper.eval()
model_inputs = wrapper.adapt_input(batch)
mark(
    "input_adapted",
    edges=int(model_inputs["edge_index"].shape[1]),
    positions=list(model_inputs["positions"].shape),
)
raw_output = wrapper.model.forward(
    model_inputs,
    compute_force=True,
    compute_stress=False,
    compute_displacement=False,
    training=wrapper.training,
)
mark("raw_forward", keys=sorted(raw_output))
output = wrapper.adapt_output(raw_output, batch)
mark("output_adapted", energy=output["energy"].detach().cpu().tolist())
if args.device == "cuda":
    torch.cuda.synchronize()
mark("synchronized")
