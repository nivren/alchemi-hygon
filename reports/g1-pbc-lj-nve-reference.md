# G1 periodic LJ and short NVE reference slice

日期：2026-09-06  
上游锁定：framework `4dfe3723def34df3fadb245981081ccf8c94c257`，ops `26dbceb61e30cca80e1a5805eebeb51d7dc68fd1`

本步验证 Torch reference 的周期 pair geometry、LJ energy/force 和一个独立 velocity-Verlet 短轨迹。探针复用 framework 的 `Batch`、`NeighborListHook` 与 `LennardJonesModelWrapper`，但不导入或修改 Warp-backed `nvalchemi.dynamics.NVE`。

周期 LJ 约定为：

```text
r_ij = r_i - r_j - shift @ cell
```

支持范围固定为 full-list、无 switching、无 virial/stress、无 skin/rebuild。周期 half-list、跨体系 active pair、不可逆 cell 和重叠 active pair 都显式失败。

CPU 单元回归：

```bash
PYTHONPATH=packages/ops \
  /home/wangleping/codes/nvalchemi-toolkit/.venv/bin/python -m pytest -q \
  packages/ops/test/torch/test_torch_reference_backend.py

PYTHONPATH=packages/framework:packages/ops \
  /home/wangleping/codes/nvalchemi-toolkit/.venv/bin/python -m pytest -q \
  packages/framework/test/models/test_neighbors_torch_reference.py \
  packages/framework/test/hooks/test_neighbor_list_torch_reference.py \
  packages/framework/test/models/test_lj_torch_reference.py
```

结果分别为 ops `10 passed`、framework `16 passed`。周期 LJ 测试使用独立 FP64 pair 公式，对能量、signed-shift 力、总力守恒和 energy gradient 做比较。

短 NVE 探针：

```bash
PYTHONPATH=packages/framework:packages/ops \
  /home/wangleping/codes/nvalchemi-toolkit/.venv/bin/python \
  probes/pbc_lj_nve_reference.py --device cpu

source /opt/dtk-26.04/env.sh
PYTHONPATH=packages/framework:packages/ops HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 timeout 90 \
  /home/wangleping/codes/nvalchemi-toolkit/.venv/bin/python \
  probes/pbc_lj_nve_reference.py --device cuda
```

CPU 与 HCU 都通过：`BW200, UBB BW1000`，1200 步、`dt=1e-4`，第 1079 步发生周期 wrap，最大总能漂移 `5.373538503050668e-09`，最终总力 `[0, 0, 0]`。HCU 进程退出码为 0；stderr 中的 `clang++`/`hipconfig` 提示来自 DTK 环境探测，不影响本次 Torch 运行。

同一探针增加 `--skin 0.5` 后，CPU 与 BW200 HCU 的能量、周期 wrap 和总力结果完全一致。reference Hook 在 raw Cartesian displacement 超过 `skin/2`（或 cell/batch 结构变化）时重建整批邻居，并将 `cutoff + skin` 中超出实际 cutoff 的 pair 在 LJ reference 中过滤掉。该路径用于验证语义，不代表上游 Warp 的 per-system rebuild、预分配 scratch 或性能实现。

该证据是 G1 的小型数值闭环，不代表完整 `nvalchemi.dynamics.NVE`、周期 half-list、Verlet skin/rebuild、switching、stress、性能后端或生产 Warp API 已在 HCU 支持。
