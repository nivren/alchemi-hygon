# ASE-compatible BFGS M1：本地新增 Ops 与数值基线

日期：2026-10-09。分支 `wangleping/feat-ase-bfgs-m1`，从干净 develop
`cfd22ad11fe0e1704dff48f53abee69135f8db99` 创建。实现、测试、文档待用户审阅；
未提交、未合并、未推送。用户已确认 M1–M4 范围及 M1 开发目标；下一阶段须另确认。

## 实现与来源

**这是 NVIDIA 上游没有提供的本地新增功能，不是已有 BFGS 的移植或上游 bugfix。**
架构参考 Framework 0.3.0-rc/Ops 0.5.0-rc L-BFGS 的显式状态和 prepare/step 分层，
不复制有限历史算法或新框架依赖。数学独立实现，逐步对齐 ASE 3.29.0 无线搜索 BFGS；
冻结 SHA、源码路径、ASE SHA256 与许可证见
[合同](../docs/TORCH_ASE_BFGS_CONTRACT.md)。没有 ASE/Warp/Framework 运行依赖。
产品锁仍为 Framework 0.2.0/Ops 0.4.1，未改 external、Torch、DTK 或默认 FIRE/FIRE2。

新增 `nvalchemiops.bfgs_ase`：

- `ASEBFGSState`：独立 H、更新前坐标/力、体系 ID、步数、初始化状态、alpha/maxstep。
- `bfgs_ase_prepare_state` / `bfgs_ase_step`：FP64、单体系 `[N,3]`、固定晶胞；默认
  `alpha=70, maxstep=.2`；原位更新坐标和 State，力保持不变，返回独立诊断张量。
- `state_dict/from_state_dict`：schema/algorithm 校验、克隆快照、显式设备恢复；无
  ASE JSON 互读要求。输入、快照和候选数值错误在提交更新前明确失败。
- 数值核心 `bfgs_ase_reference`：Hessian 更新、eigh、abs 谱方向、全体系统一缩放、
  坐标更新在输入设备；无阻尼、曲率跳过、线搜索、谱截断或隐藏精度转换。
- catalog 登记 `torch_reference.bfgs-ase-v1`，使用现有通用 executor；没有修改共享
  backend/executor/dispatcher。替换 fixture executor 的实际调用通过。
- 新 API 默认 `torch_reference`；None/warp 无上游 BFGS 而报错，auto 未准入。
  缓存 selection 仍校验实际 device/dtype/features/gradient。

代码采用本地 contributor 的 Apache-2.0 SPDX/完整许可证头；保留既有 NVIDIA 文件
来源。独立数学实现未复制 ASE LGPL 代码。上游旧 license checker 硬编码 NVIDIA
copyright，不能用它证明本地新增文件的归属；本轮逐文件校验本地 SPDX 和完整许可证
正文，不修改该检查器，也不声称全包 license/pre-commit gate 已通过。

## CPU 验收

环境：项目 .venv；PyTorch 2.9.0/HIP 6.3.26093；ASE 3.29.0；OMP_NUM_THREADS=1。
通过 `HIP_VISIBLE_DEVICES=''` 隔离 HCU。

```sh
HIP_VISIBLE_DEVICES='' bash scripts/check_cpu_reference.sh
source scripts/activate_hygon_env.sh project
HIP_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q \
  packages/ops/test/torch/test_ase_bfgs.py
HIP_VISIBLE_DEVICES='' .venv/bin/python probes/ase_bfgs_reference.py \
  --device cpu --output artifacts/ase-bfgs-m1-2026-10-09/cpu-numerics.json
```

三个命令退出码均 0。专项最终复测68 passed；最后审阅将 None/warp 的错误从通用
capability 拒绝明确为“上游没有 legacy BFGS”，不改数学或既有后端。CPU reference
gate 五个独立 pytest 进程分别：

| 范围 | 结果 |
|---|---|
| Ops reference（包含 BFGS 68 项） | 224 passed, 3 skipped |
| Framework 独立导入入口 | 1 passed |
| Framework 兼容/模型/Hooks/动态工作流 | 222 passed, 1 deselected |
| 既有状态生命周期 | 22 passed, 34 deselected |
| 既有 VV/FIRE/FIRE2 Ops | 24 passed, 73 deselected |

合计 **493 passed, 3 skipped**；跳过项是既有 NHC 的三项 Warp 对照，环境无 Warp；
不是新增 BFGS skip。compileall、git diff --check 通过。这是项目指定 reference gate，
不是全上游测试、全 HCU gate 或 NVIDIA/Warp 验收。

BFGS 68 项包括：独立二次/四次势轨迹，公开 ASE `step()` + Calculator、prepare/
determine 的 H/谱/方向/统一步长/坐标/历史；位移 1e-7 阈值两侧；零力/负谱/简并谱/
近零非零谱；重启前/首步后/多步后；快照序列化/无 alias；shape/stride/FP32/device/
gradient/identity 错误；精确零分母/零谱/溢出时不改变状态；后端范围、缓存 selection；
第二 metadata executor；新进程阻断 ASE/Warp/Framework 后仍可执行。

开发中修正了三个测试夹具/清单问题：ASE prepare 返回最大 norm 标量，近零谱夹具
须显式 FP64；新增 catalog ID 需加入既有精确清单；ASE 公开 step 的旧传参入口将
参数视为力，因此公开 API 对照改用真实解析 Calculator。未修改算法或放宽容差。

## HCU 实测

hy-smi 只读检查 HCU4：0% 显存/利用率、可用 65390 MiB；有无卡归属且显存为0的
KFD 进程，没有操作它。仅使用物理 HCU4，`HIP_VISIBLE_DEVICES=4` 后进程设备为
`cuda:0`；BW1000 64G；DTK 26.04；限时 120 秒。

```sh
source scripts/activate_hygon_env.sh project
HIP_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 timeout 120 \
  .venv/bin/python probes/ase_bfgs_reference.py --device cuda --profile \
  --output artifacts/ase-bfgs-m1-2026-10-09/hcu4-numerics.json
```

退出码 0。1/3/12 原子（3/9/36 自由度）各 8 步与 ASE CPU Oracle 对照；中途
device-local Restart 与连续运行逐值一致；36 自由度案例在非默认 stream 执行，
使用正确 stream wait；显式 CPU→HCU 快照恢复通过。
作业结束后 hy-smi 显示各卡显存使用0%，无 KFD PIDs；已释放设备资源。

DeviceGuard 覆盖实际 step/状态验证，3 个案例各观察到 8 次 `aten._linalg_eigh`；
返回张量都在 HCU、所有浮点返回均 FP64，无 Tensor CPU 回退。标量 guard 允许同步。
profiler 独立记录一步 120 个 GPU events，含 `set_tridiag<double...>` 求解器 kernel
以及 `aten::linalg_eigh/_linalg_eigh`。这支持当前设备张量路径及 GPU 求解器运行，
不宣称数学库内部不存在任何 host 控制、整个步骤无同步或已有端到端性能收益。

三规模八步中的最大绝对误差（与各自 ASE 轨迹比较）：

| 量 | CPU | HCU4 |
|---|---:|---:|
| Hessian | 2.8422e-14 | 3.5527e-14 |
| 特征值 | 3.1974e-14 | 3.9080e-14 |
| 原始方向 | 8.3267e-15 | 6.8834e-15 |
| 实际步长 | 3.3307e-16 | 2.7756e-16 |
| 坐标 | 6.6614e-16 | 6.6614e-16 |
| 历史力 | 1.0658e-14 | 8.8818e-15 |

良态解析案例 CPU rtol1e-10/atol1e-12，HCU rtol1e-8/atol1e-10。
最大归一化特征分解残差分别 7.6753e-16 / 7.5723e-16；均通过 1e-12 门槛。
误差不是任意模型/病态大体系的通用保证。

## 记录、限制与下一阶段

原始证据：`artifacts/ase-bfgs-m1-2026-10-09/` 的 `cpu-reference.log/.exit`、
`cpu-bfgs-final.log/.exit`、
`cpu-numerics.json`、`cpu-probe.log/.exit`、`hcu-resource-before.log`、`hcu-resource-after.log`、
`hcu4-probe.log/.exit`、`hcu4-numerics.json`、`hcu4-numerics.trace.json`。
汇总及产品文件 SHA256 见同目录 `acceptance-summary.json`；Ruff 0.11.13 来自现有
uv cache，不安装/更新依赖，目标文件 format/check 通过。原始 trace 不提交。

M1 只交付单体系固定晶胞 FP64 eager Ops；没有 BaseDynamics 公共优化器、真实
MLIP BFGS 弛豫、约束/Filter、异构 Batch/inflight、DomainParallel、autograd/compile、
NVIDIA 硬件验证、大矩阵容量或性能准入。H/历史数值在 HCU，metadata 和标量检查有
host 控制；不能描述为整套框架均在设备内运行。精确奇异/非有限情况明确报错，ASE
对此可能产生 NaN/Inf；正常有限轨迹按声明容差逐步兼容。

M2 提案：单体系固定晶胞 `ASECompatibleBFGS` 接入现有 BaseDynamics，复用模型/
Hooks/收敛和生命周期；验证初始收敛及严格 `< fmax`、冻结/四入口/Hook 时序、最小
checkpoint；CPU/HCU 解析势和真实 MLIP 小体系；开始记录模型与优化步基础时间。
不提前实现 M3 Filter 或 M4 Batch。用户确认 M2 后才开始；M1 尚未提交/集成。
