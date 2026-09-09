# 基础开发版本 v0.1

状态：候选集成分支 `team/dev-baseline-v0.1`。该分支由人工审阅后合入 `develop`；agent
不得自动合并或推送。

本版本的目的不是宣称完整 DCU production backend，而是提供一个可供 2--3 人并行开发的
稳定起点：后端选择边界集中、三个可运行的 Torch-reference golden paths、可重复的 CPU
gate、以及显式的 HCU 批验证入口。

## 已收口的边界

| golden path | 已覆盖的操作与范围 | 主要代码与回归 |
|---|---|---|
| neighbors → LJ | dense periodic/no-PBC full-list、no-PBC cell-list strategy、LJ energy/force/二阶梯度窄 slice | `nvalchemiops.torch_reference*`；ops Torch tests、`test_lj_torch_reference.py` |
| fixed-cell dynamics | VV、FIRE、FIRE2、kinetics；异构 Batch 与 workflow state 窄 slice | `nvalchemi._dynamics_reference`；dynamics reference、state、public wrapper tests |
| periodic + observer | periodic wrap、segmented reduction、Logging/energy-drift observer reference | `nvalchemi._dynamics_reference.periodic`；hook utility、periodic、observer tests |

所有这些路径均通过 registry 解析精确 `BackendSelection` 后交给 dispatcher。`backend=None`
仍是上游 Warp legacy 路径；`backend="torch_reference"` 才是当前 reference 路径。不要把
HCU 可见性、`torch.cuda` 命名或缺 Warp 自动解释为切换后端的理由。

默认登记项在 `packages/ops/nvalchemiops/_backend_catalog/` 按操作族管理。该目录仅含
`Implementation` metadata 和 executor 字符串，不能导入 executor 或初始化 Warp；公共 registry
API 和默认登记顺序仍由 `nvalchemiops.backend` 保证。

## 本版本明确不做的事

- 不实现 PlatformFingerprint、BackendProfile、Frozen BackendPlan 或 M2 planner。
- 不新增 periodic cell-list、NVTLangevin/NVT Nose-Hoover、NPT/NPH 或变胞 dynamics。
- 不登记 Triton/HIP production 实现，不把任何既有 Torch reference 结果当作性能结论。
- 不引入 hosted CI、复杂审批流或额外插件。CPU gate 和定期 HCU 批验证是当前足够的协作纪律。

## 最小协作流程

1. 从候选集成分支或 `develop` 创建 `<开发者>/<类型>-<主题>` 分支；只拥有本任务的代码、
   测试和文档文件。
2. 每个提交前运行 `scripts/check_cpu_reference.sh`。它将 framework 与 ops 分进程运行，避免
   上游共同的顶层 `test` 包发生 import-path 冲突。
3. 由指定人员在已分配 HCU 的窗口运行
   `HIP_VISIBLE_DEVICES=<assigned> scripts/check_hcu_reference_smoke.sh`。没有这条新的真实
   设备记录，变更只能标为 CPU verified，不得标为 DCU verified。
4. 小提交使用 `git commit -s`，只显式暂存自己的路径；候选分支由人类审阅、合入 `develop`。

可并行的角色边界由负责人分派，而不是在代码里绑定个人姓名：一人拥有 neighbor contract 与
strategy，一人拥有 thermostat/integrator reference，一人拥有跨包回归、HCU 批记录和集成。
任何跨边界接口变更先在 issue/交接中写清契约，再切独立分支。

## 新 Torch operation 的最小完成定义

- 来源、shape/dtype/layout、单位、PBC/full-half、空输入/错误、mutation、stream、确定性和
  梯度等级写入契约。
- CPU reference 与针对正确性/失败行为的最小测试存在；不以示例替代回归。
- registry metadata、framework 的单次 `BackendSelection` 传递、dispatcher 精确 ID 校验均已接线。
- `backend=None` legacy 反向守护未变；未知和未登记请求明确失败。
- `scripts/check_cpu_reference.sh` 通过；HCU 批次按实际运行结果追加证据，未运行时明确 pending。
- `FEATURE_COMPATIBILITY.yaml`、`STATUS.md` 和 report 只登记实际覆盖的切片与限制。

完整步骤见 [ADD_TORCH_OPERATION.md](ADD_TORCH_OPERATION.md)。面向第一次参与项目开发者的
手把手积分器教程见
[TUTORIAL_TORCH_INTEGRATOR_FOR_BEGINNERS.md](TUTORIAL_TORCH_INTEGRATOR_FOR_BEGINNERS.md)，
其中以 `TORCH-NVT-LANGEVIN` 为实际练习案例，并用已完成的 VV 作为对照。

## 基线后的独立任务队列

| ID | 任务 | 前提与验收 |
|---|---|---|
| `TORCH-NEIGHBOR-PBC-CELL` | 周期 full-list cell-list reference | 先写 periodic cell、image shift、batch/overflow 契约；与 dense CPU FP64 和 HCU 对照；未通过不登记 strategy capability |
| `TORCH-NVT-LANGEVIN` | 固定晶胞 NVTLangevin Torch reference | 与 upstream 随机数、状态、温度统计和 restart 语义逐项确认；先 CPU 再 HCU，不改 NPT/NPH |
| `TORCH-NVT-NHC` | 固定晶胞 NVT Nose-Hoover chain reference | 建立链状态、质量、能量/温控统计、Batch/inflight 和错误契约；独立于 Langevin 合入 |

当至少两个 operation 出现多个已验证实现，或需要可复现实验策略时，再启动
`BACKEND_PLATFORM_PIPELINE_PLAN.md` 的 M2 profile/planner；不要为当前单一 reference
实现提前制造配置层。
