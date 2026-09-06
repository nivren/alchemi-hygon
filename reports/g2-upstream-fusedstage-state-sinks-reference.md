# 上游 FusedStage、状态和 sink reference 子集

日期：2026-09-06  
范围：锁定 framework 上游测试中可在当前 Warp-free reference 环境运行的编排、状态工厂和 HostMemory 子集。此轮不实现 Langevin/Nose-Hoover/NPT，也不测性能。

## FusedStage 与 Hook 行为

运行上游测试载体：

```bash
PYTHONPATH=packages/framework:packages/ops \
  .venv/bin/python -m pytest -q \
  packages/framework/test/dynamics/test_single_loop.py::TestConvergenceHook \
  packages/framework/test/dynamics/test_single_loop.py::TestFusedStage \
  packages/framework/test/dynamics/test_single_loop.py::TestFusedStageSubstageHooks
```

结果：退出码 `0`，`55 passed`，约 `2.92 s`。覆盖共享 forward、逐体系 status 迁移、masked update、FusedStage 编排、Hook 阶段/频率/空 mask 和收敛生命周期。

这些测试使用上游文件中的 `BaseDynamics`/非保守 DemoModel 测试替身，Batch 在 CPU 构造；它们验证的是编排语义，不是 HCU kernel。真实 HCU FusedStage/inflight 证据仍见 `reports/g2-upstream-inflight-reference.md`。

同一 FusedStage/Hook 55 项与下述 17 项 sink 测试合并运行时，在加载 DTK 26.04、`HIP_VISIBLE_DEVICES=0` 的 BW200/gfx936 HCU 上退出码为 `0`，结果 `72 passed`（约 `0.70 s`）；其中 FusedStage/Hook 子集未出现失败。

该 HCU 命令是在上面两个 selector 列表前加如下环境前缀后合并执行：

```bash
source /opt/dtk-26.04/env.sh
PYTHONPATH=packages/framework:packages/ops HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  timeout 90 .venv/bin/python -m pytest -q \
  packages/framework/test/dynamics/test_single_loop.py::TestConvergenceHook \
  packages/framework/test/dynamics/test_single_loop.py::TestFusedStage \
  packages/framework/test/dynamics/test_single_loop.py::TestFusedStageSubstageHooks \
  packages/framework/test/dynamics/test_sinks.py::TestDataSinkABC \
  packages/framework/test/dynamics/test_sinks.py::TestDrainMethod \
  packages/framework/test/dynamics/test_sinks.py::TestHostMemory
```

另外，将设备默认值、通信 stream 生命周期、FusedStage stream 传播和 pipeline 组合的上游测试单独运行：

```bash
source /opt/dtk-26.04/env.sh
PYTHONPATH=packages/framework:packages/ops HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  timeout 90 .venv/bin/python -m pytest -q \
  packages/framework/test/dynamics/test_single_loop.py::TestFusedStageDeviceValidation \
  packages/framework/test/dynamics/test_single_loop.py::TestCommunicationMixinStreamContext \
  packages/framework/test/dynamics/test_single_loop.py::TestFusedStageStreamContext \
  packages/framework/test/dynamics/test_single_loop.py::TestDistributedPipelineComposition
```

HCU 退出码为 `0`，结果 `31 passed`（约 `1.13 s`）。同一命令在无 GPU 的 CPU 进程中为 `27 passed, 4 failed`：1 项失败是上游断言无 GPU 时默认 `device_type` 仍为 `cuda`，3 项 stream 失败是测试在构造 CUDA 子阶段后才打补丁，FusedStage 已在真实 CPU 环境初始化为 `cpu`。这是测试环境/补丁时序差异；未修改生产默认设备选择，也未将其计为 HCU 失败。

## DataSink 与 HostMemory 行为

```bash
PYTHONPATH=packages/framework:packages/ops \
  .venv/bin/python -m pytest -q \
  packages/framework/test/dynamics/test_sinks.py::TestDataSinkABC \
  packages/framework/test/dynamics/test_sinks.py::TestDrainMethod \
  packages/framework/test/dynamics/test_sinks.py::TestHostMemory
```

结果：退出码 `0`，`17 passed`，约 `0.17 s`。覆盖 sink 抽象协议、drain/清空、容量、mask 写入和 HostMemory CPU 快照读回。

同一 17 项在 DTK 26.04、BW200/gfx936 HCU 进程中退出码为 `0`，结果 `17 passed`（与 CPU 一样，测试数据在 HostMemory 中验证快照语义）。

## 状态管理与固定晶胞 reference 接线

```bash
NVALCHEMI_TEST_BACKEND=torch_reference \
  PYTHONPATH=packages/framework:packages/ops \
  .venv/bin/python -m pytest -q \
  packages/framework/test/dynamics/test_state_management.py::TestStateLazyInit::test_nve_state_initialized_on_first_step \
  packages/framework/test/dynamics/test_state_management.py::TestStateLazyInit::test_fire_state_initialized_on_first_step \
  packages/framework/test/dynamics/test_state_management.py::TestStateLazyInit::test_fire2_state_initialized_on_first_step \
  packages/framework/test/dynamics/test_state_management.py::TestStateLazyInit::test_second_step_does_not_reinitialize \
  'packages/framework/test/dynamics/test_state_management.py::TestStateShapes::test_nve_shapes[1]' \
  'packages/framework/test/dynamics/test_state_management.py::TestStateShapes::test_nve_shapes[3]' \
  'packages/framework/test/dynamics/test_state_management.py::TestStateShapes::test_fire_shapes[1]' \
  'packages/framework/test/dynamics/test_state_management.py::TestStateShapes::test_fire_shapes[3]' \
  'packages/framework/test/dynamics/test_state_management.py::TestStateShapes::test_fire2_shapes[1]' \
  'packages/framework/test/dynamics/test_state_management.py::TestStateShapes::test_fire2_shapes[3]' \
  packages/framework/test/dynamics/test_state_management.py::TestStateInvariant \
  'packages/framework/test/dynamics/test_state_management.py::TestPipelineStateMutation::test_partial_removal_is_safe_for_next_step[sender]' \
  'packages/framework/test/dynamics/test_state_management.py::TestPipelineStateMutation::test_partial_removal_is_safe_for_next_step[final]' \
  'packages/framework/test/dynamics/test_state_management.py::TestMakeNewState::test_nve_make_new_state[1]' \
  'packages/framework/test/dynamics/test_state_management.py::TestMakeNewState::test_nve_make_new_state[4]' \
  'packages/framework/test/dynamics/test_state_management.py::TestMakeNewState::test_fire_make_new_state[1]' \
  'packages/framework/test/dynamics/test_state_management.py::TestMakeNewState::test_fire_make_new_state[3]' \
  'packages/framework/test/dynamics/test_state_management.py::TestMakeNewState::test_fire2_make_new_state[1]' \
  'packages/framework/test/dynamics/test_state_management.py::TestMakeNewState::test_fire2_make_new_state[2]'
```

为固定晶胞构造器增加 `NVALCHEMI_TEST_BACKEND` 选择后，结果为退出码 `0`、`20 passed`，约 `2.59 s`。覆盖 NVE/FIRE/FIRE2 的 lazy init、state shape、Batch invariant、partial removal 和新状态工厂；未设置变量时仍传入 `None`，保留 Warp 默认行为。

同一精确 20 项在 DTK 26.04、BW200/gfx936 HCU 上退出码为 `0`，结果 `20 passed`（约 `0.44 s`）。

命令只需在上述固定晶胞 selector 前加 `source /opt/dtk-26.04/env.sh`、
`NVALCHEMI_TEST_BACKEND=torch_reference HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1`，
并以项目 `.venv/bin/python` 执行；selector 列表保持不变，未包含 Langevin、Nose-Hoover、NPT 或变胞测试。

反向检查未设置 backend 的三个 lazy-init 测试：退出码 `1`、`3 failed`，分别在 velocity-Verlet/FIRE/FIRE2 的 Warp custom-op 边界报告 `ModuleNotFoundError: No module named 'warp'`。这证明新增选择没有把默认路径静默改成 reference。

同一文件中涉及 NVTLangevin、NVTNoseHoover 和 NPT 的状态测试仍未接线；它们在导入 `_ops.langevin` 或 `_ops.nose_hoover` 时会触发相同的 Warp import boundary，不属于本轮通过范围。

## 结论与后续

当前上游 FusedStage/Hook、固定晶胞状态管理和 HostMemory 行为子集已在项目 `.venv` CPU 与单卡 HCU 通过，可作为后续 reference 回归载体；未移植热浴/压强积分器的测试继续保留为明确阻断。Langevin reference 可以另行规划，不与本轮固定晶胞接线混在一起。

## DemoDynamics 与 FusedStage 集成补充

锁定上游 `test_demo_dynamics.py` 全文件在项目环境 CPU 运行：

```bash
source scripts/activate_hygon_env.sh project
PYTHONPATH=packages/framework:packages/ops timeout 120 \
  .venv/bin/python -m pytest -q packages/framework/test/dynamics/test_demo_dynamics.py
```

退出码 `0`，`34 passed, 2 skipped`，约 `8.13 s`。覆盖 DemoDynamics 首步和
多图独立演化、位置/速度/能量/力接口、Hook 顺序、FusedStage/DistributedPipeline
组合、收敛 hook 参数及 `n_steps` 契约；skip 为无 GPU 时的两个设备参数。
该文件的 DemoDynamics 是上游用于编排测试的纯 Torch 替身，不代表 MACE 或生产
reference 邻居后端性能。

在设备节点可见的主机权限 HCU 上使用相同 selector 和环境入口：

```bash
source scripts/activate_hygon_env.sh project
PYTHONPATH=packages/framework:packages/ops HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
timeout 120 .venv/bin/python -u -m pytest -q \
  packages/framework/test/dynamics/test_demo_dynamics.py --disable-warnings
```

退出码 `0`，CPU/CUDA 参数化共 `36 passed`，约 `8.83 s`；CUDA 设备参数实际
执行。设备为 DTK 26.04、BW200/UBB BW1000（gfx936）。该 HCU 结果仍只覆盖
纯 Torch 编排替身，不代表 MACE、reference 邻居或生产后端性能。
原始输出和退出码保存在 `artifacts/g2/upstream_demo_dynamics_hcu0.*`。
