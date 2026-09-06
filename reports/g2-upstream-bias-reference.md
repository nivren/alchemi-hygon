# 上游 BiasedPotentialHook reference 回归

日期：2026-09-06  
锁定上游 `test_bias_hook.py` 全文件在项目 `.venv` CPU 运行：

```bash
source scripts/activate_hygon_env.sh project
PYTHONPATH=packages/framework:packages/ops timeout 120 \
  .venv/bin/python -m pytest -q packages/framework/test/dynamics/test_bias_hook.py
```

退出码 `0`，`12 passed, 8 skipped`，约 `8.30 s`。覆盖能量/力 bias 的逐图
shape、原地 mutation、零 bias、阶段/频率、NaNDetector 组合和 CPU
`torch.compile` smoke；skip 为无 GPU 时的 CUDA 参数化。

在设备节点可见的主机权限终端使用：

```bash
source scripts/activate_hygon_env.sh project
PYTHONPATH=packages/framework:packages/ops HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
timeout 120 .venv/bin/python -u -m pytest -q \
  packages/framework/test/dynamics/test_bias_hook.py --disable-warnings
```

退出码 `0`，CPU/CUDA 参数化共 `22 passed`（与 safety/freeze 合并执行总计约
`8.90 s`）；CUDA compile smoke 实际运行。设备为 DTK 26.04、BW200/UBB BW1000
（gfx936），`HIP_VISIBLE_DEVICES=0`。
这只覆盖纯 Torch bias hook，不代表 MACE、邻居或完整 Hook 生产后端。受限沙箱中
出现 `No HIP GPUs are available` 时，应按设备节点隔离处理。与 safety/freeze 合并
执行的原始输出和退出码保存在 `artifacts/g2/upstream_safety_freeze_bias_hcu0.*`。
