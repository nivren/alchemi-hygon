# 上游 SizeAwareSampler reference 回归

日期：2026-09-06  
范围：锁定 framework 上游 `SizeAwareSampler` 测试。该轮只验证 inflight batching 的容量预算、异构样本分箱、初始批次和 replacement 选择，不进行性能测量，也不改变 sampler 实现。

## CPU

```bash
PYTHONPATH=packages/framework:packages/ops timeout 90 \
  .venv/bin/python -m pytest -q packages/framework/test/dynamics/test_sampler.py
```

退出码 `0`，`51 passed`，约 `0.40 s`。

## HCU

```bash
source /opt/dtk-26.04/env.sh
PYTHONPATH=packages/framework:packages/ops HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  timeout 90 .venv/bin/python -m pytest -q packages/framework/test/dynamics/test_sampler.py
```

退出码 `0`，`51 passed`，约 `0.13 s`。设备为 DTK 26.04 的 BW200/UBB BW1000（gfx936）。测试主要使用 CPU synthetic `AtomicData`，HCU 结果说明项目环境下 sampler 的设备无关控制流和 Batch 构造路径可收集运行；不构成邻居或模型 kernel 的 HCU 性能证据。

## 覆盖与限制

测试覆盖：`max_atoms`/`max_edges`/`max_batch_size` 约束，diverse-size packing，round-robin 与 bin width，消费标记，兼容 replacement、耗尽和预算 replacement largest-first；也覆盖空数据和非法预算的显式错误。

该回归不替代真实 MACE/FIRE2 inflight。真实 HCU 短 inflight 证据仍见 `reports/g2-mace-fire2-inflight-reference.md`；长轨迹 occupancy、动态邻居容量与 checkpoint/restart 仍未验证。
