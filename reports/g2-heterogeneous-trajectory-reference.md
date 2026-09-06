# G2 异构 Batch 最小轨迹契约

日期：2026-09-06  
目标设备：海光 BW200 / UBB BW1000，gfx936  
范围：`SnapshotHook` → `HostMemory` 的最小轨迹写出；验证异构 `[3, 5]` Batch 的体系边界、字段快照和 frame 顺序。该报告不把轨迹写出扩展为 checkpoint/restart 支持。

## 契约

- 两个体系原子数分别为 3 和 5；两帧连续写入同一个 `HostMemory(capacity=4)`。
- 写入后修改 live Batch 的位置、力、速度、能量和动态系统字段 `trajectory_step`，第一帧必须保持原值。
- 读回 Batch 的 `batch_ptr` 为 `[0, 3, 8, 11, 16]`，`batch_idx` 非递减，体系 ID 顺序为 `[101, 202, 101, 202]`。
- 每个 frame 的 energy、positions、forces、velocities、status 和 `trajectory_step` 都按体系边界保持一致，且读回字段不与 live Batch 共享可变存储。
- 该 slice 不覆盖 GPUBuffer、ZarrData、checkpoint/restart、重启状态恢复或完整生产 dynamics API。

## 实现修正

`HostMemory.write` 原先只调用 `data.to("cpu")`。当输入已经在 CPU 时，设备转换可以返回仍与 live Batch 共享的张量；第二帧的 in-place 更新因此会改写第一帧的动态字段。现在设备转换后显式调用 `AtomicData.clone()`，使 sink 持有独立快照。该修改登记为 `docs/UPSTREAM.md` 的 LP-008。

## 验证

项目环境加载：

```bash
source scripts/activate_hygon_env.sh project
```

CPU：

```bash
PYTHONPATH=packages/framework:packages/ops \
  .venv/bin/python -m pytest -q \
  packages/framework/test/compatibility/test_heterogeneous_trajectory_reference.py -k cpu
```

结果：退出码 `0`，`1 passed, 1 deselected`，Python `3.12.13`，pytest `8.4.2`；覆盖位置、力、速度和能量的跨帧副本隔离断言。

HCU：

```bash
source scripts/activate_hygon_env.sh project
HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
PYTHONPATH=packages/framework:packages/ops timeout 60 \
  .venv/bin/python -m pytest -q \
  packages/framework/test/compatibility/test_heterogeneous_trajectory_reference.py -k cuda
```

结果：退出码 `0`，`1 passed, 1 deselected`，耗时 `18.35 s`，设备为 BW200/UBB BW1000（gfx936）；覆盖位置、力、速度和能量的跨帧副本隔离断言。

上游 sink 回归（CPU HostMemory/Drain/DataSink 子集）：

```bash
PYTHONPATH=packages/framework:packages/ops timeout 90 \
  .venv/bin/python -m pytest -q packages/framework/test/dynamics/test_sinks.py \
  -k 'HostMemory or DrainMethod or DataSinkABC'
```

结果：退出码 `0`，`17 passed, 39 deselected`。完整 `test_sinks.py` 中还包含依赖设备或 Zarr 的其他组，本次不将未完整运行的组计入通过证据。

## 已知边界

这是最小轨迹输出契约，不是完整的轨迹性能基线，也不证明 restart/checkpoint。`test_observer_hooks.py` 的 `TestLoggingHook.test_snapshot_decouples_energy_from_batch[cpu]` 仍在 `_segmented_max` 的 Warp 边界失败，属于 logging/kinetics 的未隔离依赖，不影响本报告的 SnapshotHook + HostMemory 断言，且未被 skip 或静默回退。
