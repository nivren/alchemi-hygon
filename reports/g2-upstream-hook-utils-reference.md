# 上游 hook utility 的 Torch reference 回归

日期：2026-09-06  
范围：`test_hook_utils.py` 与 `test_periodic_hook.py` 的逐图 reduction、动能/温度、
周期坐标包裹和 `torch.compile` smoke。此步用于清除 observer/trajectory 载体的隐式 Warp
收集阻断，不启动性能 profile。

## 实现边界

`nvalchemi.hooks.periodic` 不再在模块导入时加载 Warp；Warp 和
`nvalchemiops.dynamics.utils` 只在默认 `None`/`"warp"` 分支导入。
`wrap_positions_into_cell(..., backend="torch_reference")` 委派到顶层
`nvalchemi._dynamics_reference.periodic` 的 Torch custom op，保留原地更新、
逐体系 `batch_idx` 选择 cell、triclinic fractional wrapping 和逐维 PBC 语义。
`WrapPeriodicHook(compute_backend=...)` 可显式指定，未指定时继承
`ctx.workflow.backend`。

锁定上游的 `test_hook_utils.py` 只增加 `NVALCHEMI_TEST_BACKEND` 测试载体，
设置 `torch_reference` 时将其传给 reduction、kinetics 和 periodic helper；未
设置时仍保留上游 Warp 默认路径。

## CPU

```bash
source scripts/activate_hygon_env.sh project
NVALCHEMI_TEST_BACKEND=torch_reference \
PYTHONPATH=packages/framework:packages/ops timeout 120 \
  .venv/bin/python -m pytest -q packages/framework/test/dynamics/test_hook_utils.py
```

退出码 `0`，`30 passed, 5 skipped`，约 `14.75 s`。skip 为上游 compile 测试
在无 CUDA 设备时的既有参数化条件。

可重跑 probe：

```bash
source scripts/activate_hygon_env.sh project
PYTHONPATH=packages/framework:packages/ops timeout 60 \
  .venv/bin/python probes/hooks_utils_reference.py --device cpu
```

退出码 `0`；四种 reduction、异构两图 KE/temperature、周期 cell wrapping 均
通过，输出 `warp_loaded: false`。

周期 hook 上游行为回归：

```bash
source scripts/activate_hygon_env.sh project
NVALCHEMI_TEST_BACKEND=torch_reference \
PYTHONPATH=packages/framework:packages/ops timeout 120 \
  .venv/bin/python -m pytest -q packages/framework/test/dynamics/test_periodic_hook.py
```

退出码 `0`，`15 passed, 11 skipped`，约 `1.16 s`。覆盖正交/三斜晶胞、逐维
PBC、异构 Batch、原地 mutation、维度 squeeze 和 CPU `fullgraph=True`；严格
compile 前先 materialize lazy `batch_idx`，避免上游 Batch 的
`repeat_interleave` 动态形状限制干扰周期算子本身。

## HCU

项目 HCU 命令必须在设备节点可见、且先加载 DTK 的主机权限终端运行。标准入口
`source scripts/activate_hygon_env.sh project` 已验证会在 bash 加载
`/opt/dtk-26.04/env.sh`，在 zsh 加载 `/opt/dtk-26.04/env.zsh`；两者均将
`DTKROOT=/opt/dtk-26.04` 和项目 `.venv` 配置到当前 shell。

基础运行时复核：

```bash
source scripts/activate_hygon_env.sh project
HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 timeout 60 \
  .venv/bin/python -u probes/torch_probe.py --device cuda
```

退出码 `0`；`torch_version=2.9.0`、`cuda_available=True`、`device_count=1`，
`torch.version.hip=6.3.26093`，并完成 FP64 梯度、二阶梯度、segment 和 FFT
检查。这个结果确认当前共享 HCU 可用。

reference probe：

```bash
source scripts/activate_hygon_env.sh project
PYTHONPATH=packages/framework:packages/ops HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  timeout 60 .venv/bin/python probes/hooks_utils_reference.py --device cuda
```

退出码 `0`；四种逐图 reduction、异构两图 KE/temperature 和周期 wrapping 均
完成，JSON 中 `warp_loaded=false`。

上游工具与周期 hook reference 回归：

```bash
source scripts/activate_hygon_env.sh project
NVALCHEMI_TEST_BACKEND=torch_reference \
PYTHONPATH=packages/framework:packages/ops HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
timeout 120 .venv/bin/python -u -m pytest -q \
  packages/framework/test/dynamics/test_hook_utils.py \
  packages/framework/test/dynamics/test_periodic_hook.py --disable-warnings
```

退出码 `0`，`61 passed`，约 `17.17 s`；包含 CPU/HCU 参数化用例、周期三斜胞
和部分 PBC、原地 mutation，以及两种编译 smoke。最后的
`test_wrap_positions_compiles[cuda]` 单独复跑同样为 `1 passed`。原始输出和退出码
保存在 `artifacts/g2/torch_probe_hcu0.*`、`artifacts/g2/hooks_utils_reference_hcu0.*`
和 `artifacts/g2/upstream_hook_utils_periodic_hcu0.*`。

在不具备设备节点的普通沙箱中运行同一 probe 仍会报告
`RuntimeError: No HIP GPUs are available`；该结果是沙箱隔离信号，不作为 HCU
能力或 reference 算法失败证据。

## 默认路径反向验证

未设置 `NVALCHEMI_TEST_BACKEND` 时，单个 reduction 测试仍在
`_segmented_max` 内部的 `import warp` 边界失败（退出码 `1`）。这确认新分支
没有把默认 Warp 路径静默替换成 Torch。

## 限制

本步覆盖 Torch reference 的 eager、CPU/HCU compile smoke；未宣称 Warp
生产周期 hook、GPUBuffer/ZarrData、异步轨迹、checkpoint/restart 或性能支持。
Langevin/Nose-Hoover/NPT、变胞 stress 和 Triton/HIP 后端继续独立排期。
