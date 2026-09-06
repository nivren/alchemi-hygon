# 上游 FusedStage/inflight reference 子集

日期：2026-09-06  
范围：直接复用锁定上游 `packages/framework/test/dynamics/test_inflight.py` 的 `TestFusedStageInflight`，仅为其中使用公共 FIRE 的状态收缩用例增加已有 `NVALCHEMI_TEST_BACKEND` 测试环境选择。未设置变量时仍保留 Warp 默认路径。

## 原有 FusedStage 子集结果

CPU：

```bash
source scripts/activate_hygon_env.sh project
NVALCHEMI_TEST_BACKEND=torch_reference \
PYTHONPATH=packages/framework:packages/ops timeout 90 \
  .venv/bin/python -m pytest -q packages/framework/test/dynamics/test_inflight.py \
  -k 'TestFusedStageInflight'
```

退出码 `0`：`10 passed, 3 skipped, 20 deselected`。通过项覆盖 sampler 初始 Batch、毕业/补位、状态清理、send path 的 per-system state 收缩、数据集耗尽和 HostMemory sink 写入。

HCU：

```bash
source scripts/activate_hygon_env.sh project
NVALCHEMI_TEST_BACKEND=torch_reference HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
PYTHONPATH=packages/framework:packages/ops timeout 90 \
  .venv/bin/python -m pytest -q packages/framework/test/dynamics/test_inflight.py \
  -k 'TestFusedStageInflight'
```

退出码 `0`：`13 passed, 20 deselected`。设备为 BW200/UBB BW1000（gfx936），项目 `.venv`、DTK 26.04。

## 完整 inflight 文件回归

在不改变测试载体的前提下，进一步运行该文件的全部 33 项：

```bash
source scripts/activate_hygon_env.sh project
NVALCHEMI_TEST_BACKEND=torch_reference \
PYTHONPATH=packages/framework:packages/ops timeout 90 \
  .venv/bin/python -m pytest -q packages/framework/test/dynamics/test_inflight.py
```

CPU 退出码 `0`：`30 passed, 3 skipped`（约 `3.34 s`）。3 项 skip 是参数化 CUDA
用例在无 GPU 进程中的既有条件跳过。

```bash
source scripts/activate_hygon_env.sh project
NVALCHEMI_TEST_BACKEND=torch_reference HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
PYTHONPATH=packages/framework:packages/ops timeout 90 \
  .venv/bin/python -m pytest -q packages/framework/test/dynamics/test_inflight.py
```

HCU 退出码 `0`：`33 passed`（约 `0.52 s`），设备为 DTK 26.04 的
BW200/UBB BW1000（gfx936）。除原 FusedStageInflight 外，还覆盖
`refill_check` 的 partial replacement、bookkeeping/system_id 保留、多阶段状态，
收敛触发补位以及 step convergence 返回值。

## 默认路径反向验证

未设置 `NVALCHEMI_TEST_BACKEND` 的相同 CPU 选择为 `10 passed, 1 failed, 3 skipped, 20 deselected`；唯一失败为 FIRE send path 调用 `_ops.fire` 时缺少 Warp。该失败发生在 Warp 边界，证明测试环境变量没有把产品默认路径静默改成 reference。

## 边界

本证据只覆盖 FusedStage/inflight 的 reference 状态与 sampler/sink 编排，不覆盖完整真实 MLIP inflight 弛豫。`test_state_management.py` 中依赖 `NVTLangevin` 的 5 个 FusedStage 状态测试仍在 `_ops.langevin` 的 Warp 导入边界失败；NVT/Langevin reference 另行排期。也未将 `test_pipeline_stage.py` 的分布式流水线测试混入本结论。
