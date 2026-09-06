# G1 Torch reference per-system skin/rebuild batching

日期：2026-09-06 UTC  
范围：两体系 `Batch`、`skin=0.5`、MATRIX、显式 `backend="torch_reference"`；只移动第 0 个 system，检查第 1 个 system 的缓存和全局索引。

可重跑命令：

```bash
source scripts/activate_hygon_env.sh project
python probes/skin_rebuild_batch_reference.py --device cpu

source scripts/activate_hygon_env.sh project
HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 timeout 90 \
  python probes/skin_rebuild_batch_reference.py --device cuda
```

CPU 和 BW200/gfx936 HCU 均退出码 `0`，输出的关键字段一致：

```text
batch_ptr=[0, 2, 4]
neighbor_matrix=[[1], [0], [3], [2]]
updated_reference_system=true
unchanged_reference_system=true
unchanged_neighbor_matrix_system=true
```

这证明 reference Hook 在 Batch 中只更新变化 system 的 reference 坐标和邻居行，并恢复局部邻居的全局 batch 偏移。首次构建、Batch 布局变化仍走整批路径；容量不足仍显式抛出 `NeighborOverflowError`，尚未实现动态容量扩展、cell-list 或异步 rebuild。
