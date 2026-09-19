# Torch reference Langevin controlled statistics

日期：2026-09-19
分支：`codex/test-torch-nvt-langevin-stat`

## 范围

这是固定晶胞 Torch reference 的最小统计验收，不使用 MACE、ASE 或生产模型。
测试包含两个独立谐振子 Batch，每个 system 128 个三维自由度，解析力为
`F = -k*x`，其中 `k = 1 eV/A^2`。目标温度为 300 K 和 600 K，friction
分别为 `0.2` 和 `0.5`，使用 float64、2500 步、前 500 步 burn-in。

统计 oracle 为简谐势的 canonical 分布：

- `mean(x^2) = kT / k`
- `mean(v^2) = kT / m`

测试使用每步递增的 deterministic seed，只比较采样矩，不比较随机轨迹逐步相等。

## CPU 结果

命令：

```bash
PYTHONPATH=packages/framework:packages/ops \
  .venv/bin/python -m pytest -q \
  packages/framework/test/compatibility/test_dynamics_reference_langevin.py \
  packages/framework/test/compatibility/test_dynamics_reference_langevin_statistics.py
```

结果：`13 passed`，退出码 `0`。统计测试单独运行约 3 秒；位置和速度二阶矩均在
15% 有限样本容差内通过。

## HCU 结果

命令在主机可见的 HCU 0 上运行，测试根据运行时设备可见性自动将全部状态张量放到
`cuda`：

```bash
source scripts/activate_hygon_env.sh project
HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  PYTHONPATH=packages/framework:packages/ops timeout 120 \
  .venv/bin/python -m pytest -q \
  packages/framework/test/compatibility/test_dynamics_reference_langevin_statistics.py
```

运行时确认 `torch.cuda.is_available()=True`、`device_count=1`、设备为
`BW200, UBB BW1000`；结果为 `1 passed`，退出码 `0`，耗时 19.60 秒。

## 边界

本报告只证明当前 reference 算子在 CPU 和单卡 HCU 上通过一个短谐势统计门，不等于
完整上游 Langevin/NVT 统计套件、MACE 长轨迹、restart、inflight、分布式 ownership
或生产 Triton/HIP 实现已完成。
