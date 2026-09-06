# 异构 `[46,92]` MACE/FIRE2 轨迹组合

日期：2026-09-06  
范围：一个 46 原子周期 CIF 与一个 92 原子周期 CIF 放入同一个 Batch，使用本地 `MACE-OFF23_small.model`、reference NeighborListHook、公共 `FIRE2(backend="torch_reference")` 固定晶胞运行 3 步，并通过 `SnapshotHook → HostMemory` 写出每一步的完整体系快照。

该探针只验证最小轨迹编排和异构 Batch 边界，不宣称优化收敛、checkpoint/restart、GPUBuffer/ZarrData 或生产性能后端已完成。

## 断言

- 单个 Batch 的 `batch_ptr` 为 `[0,46,138]`，读回 3 个 frame 后为 `[0,46,138,184,276,322,414]`。
- 每个 frame 的体系顺序为 46 原子体系、92 原子体系；`batch_idx` 非递减。
- `trajectory_step` 保持 `[1,1,2,2,3,3]`，说明 HostMemory 快照不受后续 live Batch 更新污染。
- 读回的邻居边无跨体系边。

## CPU 证据

环境：项目 `.venv`，DTK 环境脚本已加载，设备 `cpu`。

```bash
source scripts/activate_hygon_env.sh project
PYTHONPATH=packages/framework:packages/ops timeout 90 \
  .venv/bin/python probes/mace_fire2_heterogeneous_trajectory.py \
  --device cpu --checkpoint /home/wangleping/.cache/mace/MACE-OFF23_small.model \
  --steps 3 --max-wall-seconds 80
```

结果：退出码 `0`；`status=passed`；`stored_frames=3`；`stored_batch_ptr=[0,46,138,184,276,322,414]`；`neighbor_edges_last_frame=3284`；`cross_system_edges=0`；耗时 `18.541438532993197 s`。

## HCU 证据

环境：`source scripts/activate_hygon_env.sh project`，DTK 26.04，`HIP_VISIBLE_DEVICES=0`，BW200/UBB BW1000（gfx936），项目 `.venv`。

```bash
source scripts/activate_hygon_env.sh project
HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
PYTHONPATH=packages/framework:packages/ops timeout 90 \
  .venv/bin/python probes/mace_fire2_heterogeneous_trajectory.py \
  --device cuda --checkpoint /home/wangleping/.cache/mace/MACE-OFF23_small.model \
  --steps 3 --max-wall-seconds 80
```

结果：退出码 `0`；`status=passed`；`stored_frames=3`；`stored_batch_ptr=[0,46,138,184,276,322,414]`；`neighbor_edges_last_frame=3284`；`cross_system_edges=0`；最终两个体系 `fmax=[0.7451172471046448,0.8051751852035522]`；设备识别为 `BW200, UBB BW1000`；耗时 `17.940327855059877 s`。

## 限制

这是 3 步短轨迹接线证据，不能替代 32×92 长弛豫，也不能证明 restart/checkpoint。NeighborListHook 仍是 full-list reference；周期 half-list、switching、virial/stress、cell-list、Triton/HIP 后端和变胞优化保持独立排期。
