# alchemi-hygon

海光 DCU 原子模拟框架开发项目，基于 NVIDIA nvalchemi-toolkit 和
nvalchemi-toolkit-ops 的锁定源码，保留 framework / ops 两个包与上游公共接口，
通过 Torch reference、HIP 等路径逐步适配。当前是开发基线，已验证功能均有明确范围；
不应将上游代码或接口的存在理解为已完成 DCU 验收。

## 从这里开始

| 想了解什么 | 入口 | 内容与关系 |
|---|---|---|
| 如何准备环境和输入 | [项目环境与输入](docs/DEVELOPMENT_ENVIRONMENT.md) | Hygon 环境、设备检查，以及外部模型和结构文件配置 |
| 当前有哪些能力 | [应用与模块能力](docs/IMPLEMENTATION_STATUS_MATRIX.md) | 按应用与模块查看验证范围、限制和证据；后端细节见能力矩阵与特性契约 |
| 当前进度和限制 | [当前状态](docs/STATUS.md) | 当前基线、近期验证和未覆盖范围；详细结果指向原报告 |
| 接下来做什么 | [当前任务与开发计划](docs/PARALLEL_DEVELOPMENT_PLAN.md) | 项目当前工作顺序、暂停方向和里程碑确认方式 |
| 项目如何组织与开发 | [架构](docs/DEVELOPER_ARCHITECTURE.md)、[交接](docs/PROJECT_HANDOFF.md)、[开发指南](docs/DEVELOPMENT_GUIDE.md) | 分别说明架构、代码与后端边界、开发与验证流程；Agent 规则见 [AGENTS.md](AGENTS.md) |
| 验证结论从哪里来 | [验证报告索引](reports/README.md) | 按主题查找指定设备、模型与输入下的报告；原始输出在 artifacts |
| 上游基线与后续跟踪 | [上游锁定](docs/UPSTREAM_LOCK.yaml)、[上游补丁](docs/UPSTREAM.md)、[上游跟踪流程](docs/UPSTREAM_SYNC.md) | 分别记录锁定版本、项目补丁和候选评估流程 |

建议先读环境与能力，再按需要查看当前状态、开发计划和原始证据。已验证范围以
应用矩阵、[后端能力矩阵](docs/BACKEND_CAPABILITY_MATRIX.md)、
[特性契约](docs/FEATURE_COMPATIBILITY.yaml)和实际报告共同限定；计划不等于已实现。

面向应用的精简示例正在整理，当前不要将 `packages/framework/examples/` 和
`packages/ops/examples/` 中保留的上游示例视为已在海光 DCU 全部验证。
现有[ASE-compatible BFGS 使用说明](docs/ASE_BFGS_USAGE.md)提供 CPU 代码和后端边界；
模型和结构文件可以由使用者从外部提供。

## 当前应用范围

- **模型与数据**：支持已有窄范围的 `AtomicData` / `Batch`、真实 MACE / MPA-0
  单点及异构 Batch；不同模型、精度和输入必须分别验证。
- **固定晶胞**：NVE、FIRE/FIRE2、Langevin 和 NHC 的 Torch reference 已有短流程或
  统计验证；长真实势轨迹、完整 checkpoint 和生产吞吐尚未普遍验收。
- **晶体弛豫**：FIRE/FIRE2 变胞流程和本地 ASE-compatible BFGS 已有小样本证据，
  包括显式 HIP 邻居或 BFGS 后端的指定范围；泛化到其他材料与规模需另行验证。
- **后续能力**：NPT/NPH、长程相互作用、单大体系 DomainParallel 和完整 MLIP 训练
  尚未形成本项目完整 DCU 应用验收。

逐项范围与证据见[应用能力矩阵](docs/IMPLEMENTATION_STATUS_MATRIX.md)；
[后端能力矩阵](docs/BACKEND_CAPABILITY_MATRIX.md)和
[特性契约](docs/FEATURE_COMPATIBILITY.yaml)记录更细的设备、布局和梯度条件。

## 快速检查已有项目环境

在已准备好项目 `.venv`、DTK 和海光 Torch 的主机上，从仓库根目录运行：

```bash
source scripts/activate_hygon_env.sh project
HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= scripts/check_cpu_reference.sh
```

单卡 HCU 验证前先确认已分配或空闲的设备，再显式选择设备：

```bash
HIP_VISIBLE_DEVICES=4 scripts/check_hcu_reference_smoke.sh
```

这里的 `4` 是本项目服务器的历史验证设备编号；其他环境应使用自己的可用设备。
CPU 基线与 HCU smoke 不能替代完整上游或具体应用验收。环境尚未准备好时，
先阅读[环境指南](docs/DEVELOPMENT_ENVIRONMENT.md)，不要用通用 CUDA Torch 替换海光版本。

## 后端与目录

对大多数已有上游操作，`backend=None` 保留上游默认语义（通常为 Warp）；海光工作流应
根据具体操作显式选择 `torch_reference` 或已准入 HIP 实现。ASE-compatible BFGS 是
本地新增接口，具有自己的默认与支持规则，见[使用说明](docs/ASE_BFGS_USAGE.md)。
HIP 邻居目前仅支持 periodic / Batch / full-list、MATRIX 与设备内 Torch COO 打包、
FP32/FP64、`skin=0`、调用间晶胞变化时完整重建；不进入 `auto`。

- `packages/framework/`、`packages/ops/`：正式源码，保持两个可独立构建的包。
- `external/`：本机只读上游 clone，不是产品安装源；锁定来源见 [UPSTREAM_LOCK.yaml](docs/UPSTREAM_LOCK.yaml)。
- `docs/`、`adr/`：环境、架构、特性契约及决策。
- `probes/`：验证探针；`reports/`：可提交的验证摘要；`artifacts/`：不入库的原始输出。
- 具体模型和材料应用可位于独立下游工程；不要将下游加速路径等同框架原生能力。

首次部署、初始化或建立上游参考 clone 时，查阅 [START_HERE.md](docs/START_HERE.md)
和 [UPSTREAM.md](docs/UPSTREAM.md)。已有工作树和环境无需重新初始化。
