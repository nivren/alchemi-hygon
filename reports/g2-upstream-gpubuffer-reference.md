# 上游 sink reference 回归

日期：2026-09-06  
范围：锁定 framework 上游 `test/dynamics/test_sinks.py` 全文件，重点补齐
GPUBuffer 与 ZarrData 的基础边界。该步只验证 sink 的功能语义，不构成吞吐或
多卡通信性能结论。

## HCU

```bash
source scripts/activate_hygon_env.sh project
PYTHONPATH=packages/framework:packages/ops HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
timeout 90 .venv/bin/python -u -m pytest -q \
  packages/framework/test/dynamics/test_sinks.py --disable-warnings
```

在 DTK 26.04、BW200/UBB BW1000（gfx936）主机权限环境中退出码为 `0`，结果为
`56 passed`，约 `1.37 s`。其中 GPUBuffer 的 23 项覆盖初始化、Batch 写读、容量满、
最大原子数、mask、zero/drain、设备属性和非法 CPU 设备；ZarrData 的 16 项参数化
与错误边界也全部通过。CPU/HCU 测试均在同一进程实际执行。
原始 stdout、stderr 和退出码保存在
`artifacts/g2/upstream_sinks_full_hcu0.{stdout,stderr,exit}`。

## CPU

```bash
source scripts/activate_hygon_env.sh project
HIP_VISIBLE_DEVICES= PYTHONPATH=packages/framework:packages/ops \
timeout 90 .venv/bin/python -u -m pytest -q \
  packages/framework/test/dynamics/test_sinks.py --disable-warnings
```

退出码为 `0`，结果为 `33 passed, 23 skipped`，约 `1.00 s`。跳过项全部是
GPUBuffer 的 CUDA 设备用例；DataSink、Drain、HostMemory 和 ZarrData 的 CPU
用例全部通过。原始 stdout、stderr 和退出码保存在
`artifacts/g2/upstream_sinks_full_cpu.{stdout,stderr,exit}`。

## 边界

这份证据只覆盖 GPUBuffer/ZarrData 的基础 sink 语义。受限沙箱中曾观察到
`zarr.open(..., mode="w")` 事件循环不唤醒，独立最小复现同样阻塞；主机权限环境
完整回归正常，因此该现象是测试载体隔离问题，不是项目依赖或 HCU 能力结论。
HostMemory 的 CPU/HCU 快照证据仍见
`reports/g2-upstream-fusedstage-state-sinks-reference.md` 和
`reports/g2-heterogeneous-trajectory-reference.md`。完整 stream 生命周期、
checkpoint/restart、长轨迹容量和性能仍未验证。
