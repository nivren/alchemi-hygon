# G1 邻居→LJ Torch reference 探针

日期：2026-09-05

本探针不导入 `nvalchemiops`，因为锁定上游的邻居和 LJ 模块仍会在导入链中初始化 Warp。它定义一个小型 Torch reference oracle，固定第一条纵向链的语义：两个 batch system、无 PBC、dense neighbor matrix、full/half 拓扑、LJ 能量和位置负梯度力。

## 契约覆盖

- `positions`: `(N, 3)`，`float64`；`batch_ptr=[0, 3, 5]`，禁止跨体系配对。
- 邻居矩阵使用全局原子索引和 `N` padding sentinel；`num_neighbors` 为每行有效数。
- full list 包含 `(i,j)` 与 `(j,i)`；half list 仅保留 `i < j`。
- LJ 使用 `V(r)=4*epsilon*((sigma/r)^12-(sigma/r)^6)`，full list 按二分之一计数，half list 按一次计数。
- 力由 `-autograd.grad(energy, positions)` 得到；每个体系的总力为零。
- 本探针暂不覆盖 PBC、cell-list、容量扩容、switching、virial 或生产 `nvalchemiops` API。

## 可重跑命令

CPU：

```bash
OMP_NUM_THREADS=1 .venv/bin/python probes/neighbor_lj_reference.py --device cpu
```

单卡 HCU（先确认共享资源和设备节点）：

```bash
source /opt/dtk-26.04/env.sh
HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 timeout 60 \
  .venv/bin/python probes/neighbor_lj_reference.py --device cuda
```

实测 CPU 与 HCU 输出一致：

```text
energy_expected = -0.6236757013081533
energy_full     = -0.6236757013081533
energy_half     = -0.6236757013081533
force_norm      = 23.859458411523782
pairs_full      = [[0,1], [1,0], [3,4], [4,3]]
pairs_half      = [[0,1], [3,4]]
status          = passed
```

原始 stdout、stderr、退出码：

- `artifacts/g1/neighbor_lj_reference_cpu.json`
- `artifacts/g1/neighbor_lj_reference_cuda0.json`
- `artifacts/g1/neighbor_lj_reference_cpu.stderr`
- `artifacts/g1/neighbor_lj_reference_cuda0.stderr`
- `artifacts/g1/neighbor_lj_reference_cpu.exit`
- `artifacts/g1/neighbor_lj_reference_cuda0.exit`

这只证明 Torch reference 在当前 CPU 和 gfx936 目标 HCU 上的该小输入成立，不等同于上游邻居/LJ API 或生产算子已经移植完成。
