# 真实 MACE/FIRE2 异构 inflight 补位

日期：2026-09-06  
范围：四个周期 CIF `[46,92,46,92]`，`SizeAwareSampler(max_atoms=138, max_batch_size=2)`，公共固定晶胞 `FIRE2(backend="torch_reference")`，`FusedStage(refill_frequency=1)`，`HostMemory` 收集毕业体系。

这个探针验证编排语义，不是收敛性能基准。收敛阈值临时设为 `10.0`，使初始两个体系在第一轮毕业、触发一次批量补位，第二轮完成剩余两个体系；不据此评价物理收敛质量。

## 上游补位语义

`FusedStage.run` 每隔 `refill_frequency` 调用 `refill_check`。只要有任意体系毕业，就移除毕业项，并按剩余 Batch 的原子/边/数量预算调用 `request_replacements_budget`；不要求整批毕业。初始 `[46,92]` 同时毕业时，138 原子和 2 个槽位一次请求两个 replacement。采样器的 budget 实现按大 atom bin 优先，因此实际顺序为 `[92,46]`，读回指针为 `[0,46,138,230,276]`。

## CPU

```bash
source scripts/activate_hygon_env.sh project
PYTHONPATH=packages/framework:packages/ops timeout 90 \
  .venv/bin/python probes/mace_fire2_inflight_reference.py \
  --device cpu --checkpoint /home/wangleping/.cache/mace/MACE-OFF23_small.model \
  --max-wall-seconds 80
```

退出码 `0`；`status=passed`；`system_ids=[0,1,2,3]`，每个只写入一次；`trajectory_steps=[1,1,2,2]`；`cross_system_edges=0`；`neighbor_edges=6568`；耗时 `18.766980774933472 s`。

## HCU

```bash
source scripts/activate_hygon_env.sh project
HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
PYTHONPATH=packages/framework:packages/ops timeout 90 \
  .venv/bin/python probes/mace_fire2_inflight_reference.py \
  --device cuda --checkpoint /home/wangleping/.cache/mace/MACE-OFF23_small.model \
  --max-wall-seconds 80
```

退出码 `0`；`status=passed`；设备 BW200/UBB BW1000（gfx936）；`system_ids=[0,1,2,3]`，`trajectory_steps=[1,1,2,2]`，`graduated_batch_ptr=[0,46,138,230,276]`，`cross_system_edges=0`；耗时 `18.604425920872018 s`。

## 设备与限制

采样器不会替任意 dataset 自动把 `AtomicData` 迁移到目标设备；HCU 探针的数据集因此显式在 HCU 创建设备张量，replacement 与初始 Batch 保持一致。这是 dataset/device 契约，不是静默 CPU 回退。

本证据覆盖真实 MACE 的异构补位、体系 ID、Batch 偏移、一次性 sink 写入和邻居边界；`refill_frequency=1` 的检查开销尚未 profile。长轨迹、混合收敛阈值下的 occupancy、checkpoint/restart、NVT/Langevin 和多卡 ownership 仍未验证。
