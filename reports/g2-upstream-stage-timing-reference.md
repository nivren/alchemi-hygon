# 上游 StageTimingHook reference 回归

日期：2026-09-06
范围：锁定上游 `test/hooks/test_stage_timing_hook.py` 全文件。此步验证 Hook 的
阶段计时、频率门控、设备事件、CSV 输出和生命周期，不采集生产性能基线。

## 验证

```bash
source scripts/activate_hygon_env.sh project
PYTHONPATH=packages/framework:packages/ops HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
timeout 120 .venv/bin/python -u -m pytest -q \
  packages/framework/test/hooks/test_stage_timing_hook.py --disable-warnings
```

在 DTK 26.04、BW200/UBB BW1000（gfx936）主机权限环境中退出码为 `0`，
`42 passed, 1 skipped`，约 `31.27 s`。CPU 与 HCU 参数化用例均实际执行，覆盖：

- dynamics/training stage preset、注册和组合；
- CPU `perf_counter` 与 HCU `cuda_event` 的自动选择；
- 频率门控、正时差、reset、设备上下文和异常 flush；
- CSV、console 输出以及 DemoDynamics 完整循环。

唯一 skip 是 `nvtx` 未安装（`test_nvtx_push_pop_called`）；`enable_nvtx=False`
路径通过。原始 stdout 和退出码保存在
`artifacts/g2/upstream_stage_timing_hcu0.*`。

## 边界

这份证据只说明 StageTimingHook 的功能语义在 CPU/HCU 可运行，不构成冷启动、稳态
吞吐、规模阶梯或后端性能结论。PhysicsNeMo profiler、完整 CUDA profiling 和
生产 Triton/HIP kernel 仍独立排期。
