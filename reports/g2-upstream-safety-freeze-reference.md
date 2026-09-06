# 上游 safety/freeze hook reference 回归

日期：2026-09-06  
范围：锁定上游 `test_safety_hooks.py` 与 `test_freeze_hook.py`。两组 hook 只依赖
Torch 张量和 Batch 状态，不涉及 Warp、热浴或变胞积分器；本步用于补齐固定胞
reference workflow 的安全/冻结行为证据。

## CPU

```bash
source scripts/activate_hygon_env.sh project
NVALCHEMI_TEST_BACKEND=torch_reference \
PYTHONPATH=packages/framework:packages/ops timeout 120 \
  .venv/bin/python -m pytest -q \
  packages/framework/test/dynamics/test_safety_hooks.py \
  packages/framework/test/dynamics/test_freeze_hook.py
```

退出码 `0`，`45 passed, 14 skipped`，约 `11.98 s`。覆盖 NaN/Inf 检测、逐图错误
报告、最大力 clamp、原地 mutation、冻结位置/速度/力、异构 Batch、频率/阶段、
Hook 协议和 CPU `torch.compile` smoke；skip 为上游 CUDA 参数化在无 GPU 进程中的
既有条件跳过。

## HCU

使用同一项目环境脚本在设备节点可见的主机权限终端重跑：

```bash
source scripts/activate_hygon_env.sh project
PYTHONPATH=packages/framework:packages/ops HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
timeout 120 .venv/bin/python -u -m pytest -q \
  packages/framework/test/dynamics/test_safety_hooks.py \
  packages/framework/test/dynamics/test_freeze_hook.py --disable-warnings
```

退出码 `0`；CPU/CUDA 参数化共 `57 passed`，与 bias 合并执行总计约 `8.90 s`。
CUDA 参数均实际运行，
包括 `cudagraphs` compile smoke；没有因设备不可见而 skip。设备为 DTK 26.04、
BW200/UBB BW1000（gfx936），`HIP_VISIBLE_DEVICES=0`。原始输出和退出码保存在
`artifacts/g2/upstream_safety_freeze_bias_hcu0.*`（与 bias 测试合并执行）。

受限沙箱仍可能让同一命令跳过 CUDA 参数或报告 `No HIP GPUs are available`；这只
表示设备节点隔离，不改变上述主机权限证据。

## 限制

这只是 safety/freeze hook 的行为回归，不等于完整 Hook 生命周期、异步轨迹、
checkpoint/restart 或 Warp 默认路径验证。
