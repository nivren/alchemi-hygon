# SPDX-FileCopyrightText: Copyright (c) 2026 alchemi-hygon contributors
# SPDX-License-Identifier: Apache-2.0
"""构造 Batch 并计算周期邻居；可传入 CIF 文件。"""

import argparse

import torch
from ase import Atoms
from ase.io import read

from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.base import NeighborListFormat
from nvalchemi.neighbors import compute_neighbors

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--cif", nargs="+", help="一个或多个 CIF；不传时使用两个小体系")
parser.add_argument("--cutoff", type=float, default=0.5, help="邻居截断半径，单位 Å")
parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
parser.add_argument(
    "--backend", choices=("torch_reference", "hip"), default="torch_reference"
)
args = parser.parse_args()

if args.cif:
    structures = [read(path) for path in args.cif]
else:
    structures = [
        Atoms("H2", positions=[[0.1, 0, 0], [1.9, 0, 0]], cell=[2, 10, 10], pbc=True),
        Atoms("H2", positions=[[0.2, 0, 0], [1.8, 0, 0]], cell=[2, 10, 10], pbc=True),
    ]

if not all(atoms.pbc.all() for atoms in structures):
    raise ValueError("此示例需要三个方向均周期的结构")

# 复用上游 ASE 转换、Batch 构造和一次性邻居计算接口。
data = [
    AtomicData.from_atoms(atoms, device=args.device, dtype=torch.float64)
    for atoms in structures
]
batch = Batch.from_data_list(data, device=args.device)
compute_neighbors(
    batch, cutoff=args.cutoff, format=NeighborListFormat.COO,
    backend=args.backend, method="cell_list",
)

# COO 原子索引位于整个 Batch 中；每条边和周期位移的行一一对应。
edges = batch.neighbor_list.long()
shifts = batch.neighbor_list_shifts
system = batch.batch_idx.long()
assert torch.equal(system[edges[:, 0]], system[edges[:, 1]]), "邻居不应跨体系"
counts = torch.bincount(system[edges[:, 0]], minlength=batch.num_graphs)
print(f"device={batch.device}, backend={args.backend}, cutoff={args.cutoff} Å")
print(f"systems={batch.num_graphs}, atoms={batch.num_nodes}, batch_ptr={batch.batch_ptr.tolist()}")
print(f"每个体系的有向邻居边数：{counts.tolist()}")
print(f"前 8 条边 [中心原子, 邻居原子]：{edges[:8].tolist()}")
print(f"对应的周期位移 shift：{shifts[:8].tolist()}")
