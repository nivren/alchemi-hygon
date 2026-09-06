# G1 Neighbor staging capacity reference

日期：2026-09-06 UTC  
代码基线：`4449c30` 之后的 N1 工作树  
目标设备：BW200 / UBB BW1000，gfx936  
后端：显式 `torch_reference`

## 契约

`NeighborListHook` 的 reference staging 矩阵独立管理 K 维容量。算子仍在收到过小的 `max_neighbors` 时抛出 `NeighborOverflowError`；Hook 在结构事件或局部 skin rebuild 中发现容量不足时，按 `round_16(int(actual * 1.5))` 扩容并重跑。实际邻居数低于容量一半且大于零时，按 `round_16(actual * 2)` trim，且不低于 `max_neighbors` override。收缩不重跑，扩容会保留未变化 system 的缓存并重新执行变化 system。

## 可重跑命令

```bash
source scripts/activate_hygon_env.sh project
PYTHONPATH=packages/framework:packages/ops \
  .venv/bin/python probes/neighbor_capacity_reference.py --device cpu

source scripts/activate_hygon_env.sh project
HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  PYTHONPATH=packages/framework:packages/ops timeout 120 \
  .venv/bin/python probes/neighbor_capacity_reference.py --device cuda
```

两次退出码均为 `0`。HCU 输出设备为 `BW200, UBB BW1000`。

## 结果

```json
{
  "grow": {"initial_override": 1, "final_capacity": 16, "counts": [3, 3, 3, 3]},
  "mixed_batch": {"batch_ptr": [0, 4, 6], "capacity": 16, "counts": [3, 3, 3, 3, 1, 1]},
  "shrink": {"initial_capacity": 32, "final_capacity": 16, "max_count": 1},
  "floor": {"override": 24, "final_capacity": 24, "max_count": 1},
  "trajectory": {"capacities": [32, 16, 32, 16, 32, 16], "max_count": 1}
}
```

framework reference Hook 回归为 `16 passed`；framework neighbors/LJ 回归为 `10 passed`；ops reference backend 回归为 `10 passed`。已有 per-system skin Batch HCU probe 仍以退出码 `0` 通过，未变化 system 的矩阵和 reference positions 保持不变。

## 限制

这是 eager Torch reference 的 staging 契约证据，输入规模为小型合成 Batch。尚未验证 PBC 容量压力、周期 half-list、cell-list 生产路径、异步 rebuild 或规模化性能；算子层显式 overflow 仍然保留。
