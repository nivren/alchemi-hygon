# Upstream Sync D — 首次隔离升级评估

日期：2026-10-09。结论：**暂不整体升级；保留当前 Baseline，恢复正常移植开发。**
候选组合保持 Candidate，没有晋升为 Validated。D 已完成评估，用户已确认以本签名提交收口。

## 1. 来源与隔离

| 对象 | 精确 SHA / 路径 |
|---|---|
| 稳定产品源码 develop | `2e211ab93a4f1ad0c8c720eecb22129d846c7792` |
| C 签名收口 / D 工作起点 | `ad189987b94b8b97b3332ee8982c960a3ce6a5e6` |
| Framework Baseline | `4dfe3723def34df3fadb245981081ccf8c94c257` |
| Ops Baseline | `26dbceb61e30cca80e1a5805eebeb51d7dc68fd1` |
| Framework Candidate main | `f59e28dc6701583a13bb294e536076c2a7379e98` |
| Ops Candidate 0.5.0-rc | `5fffc0726b1e71bb8b68e10b5f87942064759f5b` |
| 独立分支 | `sync/upstream-2026-10` |
| worktree（主项目根目录下） | `artifacts/upstream-sync-2026-10/worktree` |
| 完整候选目录 | `artifacts/upstream-sync-2026-10/candidate/{framework,ops}` |
| 独立运行环境 | `artifacts/upstream-sync-2026-10/runtime-env` |

worktree 从 develop 建立，fast-forward 带入已确认的 A–C 文档/工具提交；两包产品源码
仍与稳定 develop 完全相同。官方 Git 对象以 `git archive` 完整导出，Framework 772 个、
Ops 608 个文件，保留许可证、测试和完整模块；没有修改 `external/`。

独立 venv 使用 `--without-pip` 创建，通过本环境自己的 `.pth` 只读复用项目已安装依赖。
不安装候选、不进行依赖升级；它是保持海光 Torch 2.9 的实验运行环境，
不是满足候选所有声明的正式安装环境。候选导入仅使用候选目录的 PYTHONPATH；
模块探针清空 PYTHONPATH，以显式文件路径加载三个完整且未修改的上游模块，核对 SHA256。
探针还检查未混入本地 nvalchemi/nvalchemiops 命名空间。

## 2. 版本组合与依赖阻塞

2026-10-09 的官方最新正式 release 元数据为
[Framework v0.2.0](https://github.com/NVIDIA/nvalchemi-toolkit/releases/tag/v0.2.0) 和
[Ops v0.4.1](https://github.com/NVIDIA/nvalchemi-toolkit-ops/releases/tag/v0.4.1)，
分别发布于 2026-08-07 与 2026-08-04；本轮经官方 release 页面和 REST GET 核对。
冻结 Framework 候选仍声明 0.2.0，Ops 候选声明 0.5.0。
`0.5.0-rc` 是分支名，不能当作正式发布或 PEP440 rc 版本。

| 检查 | 候选要求 | 当前环境/组合 | 结论 |
|---|---|---|---|
| Python | 两包 >=3.11,<3.15 | 3.12.13 | 满足声明 |
| 两包源码版本组合 | Framework Ops>=0.5.0 | 候选 Ops 声明 0.5.0 | 版本声明满足；不证明运行兼容 |
| Framework Torch | >=2.8 | 2.9.0+das.opt1.dtk2604 | 满足此单项声明 |
| Ops Torch extra | >=2.10.0 | 2.9.0+das.opt1.dtk2604 | 不满足；不因不选 extra 就推断兼容 |
| Framework NumPy | >=2,<3 | 1.26.4 | 不满足 |
| Framework plotext | <6 | 6.0.0 | 不满足；reporting/CLI 需独立处理 |
| Framework PhysicsNeMo | >=2.2.0 | 未安装 | stock 候选依赖缺失 |
| Framework click | 无版本下限 | 未安装 | stock CLI 依赖缺失 |
| Ops Warp | >=1.16.0 | 未安装 | stock Ops 顶层导入阻塞 |
| jaxtyping/TensorDict/pydantic/plum/zarr/periodictable/rich/loguru | 见候选 pyproject | 当前安装版本 | 本轮直接声明检查满足 |

产品采用源码 PYTHONPATH 路径，未安装 nvalchemi-toolkit-ops distribution；
metadata 缺失与候选源码声明 0.5.0 的组合检查分别记录。
现有基线通过适配路径运行，不把 stock 依赖缺失描述为当前基线不可用。
没有执行在线完整依赖求解、换 Torch/NumPy、补装 Warp/PhysicsNeMo 或运行 NVIDIA extras。

实际导入：`from nvalchemi.data import Batch, GroupLayout` 和
`from nvalchemiops.torch.lbfgs import lbfgs_prepare_state` 都退出 1，错误为缺 Warp。
前者经 data.transforms→AtomicData→level_storage 引入 Warp；后者在 Ops 顶层 `import warp` 即失败。
这两次失败完整保留，不能将小模块成功当作全包 gate 通过。

## 3. 当前产品到候选的 API 与适配层

对当前产品 C 提交和冻结候选运行静态 AST 比较。完整签名/字段/导出/实现差异保存于
`static-evaluation.json`：Framework 1319、Ops 526 个变化条目，以及 121/81 个同签名实现变化。
这是 AST 条目数量，含新增/删除/注解/导出；不是正式公共 API 数量，也不是数值验收。
候选 Framework 的 118 条显式 Ops from-import 均有静态符号存在依据；
此检查不证明实参、返回值、动态导出或后端语义兼容。

| 模块/场景 | 具体变化和风险 | 预计需要修改的适配层 | 本轮决定 |
|---|---|---|---|
| GroupLayout | 新增分组映射、reduce_all/any、broadcast；依赖 Torch | Batch.set_group_layout、schema/cache 与 grouped state 接线 | 三模块实验之一可运行；可作为独立移植候选，尚未集成 |
| HookContext/Protocol | DynamicsContext 新 graduated_mask；Checkpointable/StatefulHook 与 dispatch 生命周期扩展 | Hook registry、admission/graduation、observer 时序和 checkpoint | 数据对象/直接调用可运行；完整 registry/生命周期待验证 |
| Batch / Storage | 可扩展 LevelSchema、groups/product levels；index_select 新 drop；UniformLevelStorage 构造器无本地 backend 参数 | Torch/Warp storage backend、segment offsets、schema 与 Batch 接线 | 需要整体闭包，不能直接替换文件 |
| Dynamics / FusedStage | BaseDynamics 新 by_group；私有 state 改为 Batch，priming/迁移/Hook 顺序扩展 | `_dynamics_reference`、ops executor、四入口 mask/state、refill 与 continuation | 保留 UBF；新版完整状态/生命周期适配延期 |
| Neighbors | NeighborListHook 构造器无本地 backend 参数；候选 prepared/Torch 包装仍可能调用 Warp | Framework.backend 显式传递、ops dispatcher/catalog/executor 与 HIP 当前晶胞路径 | 保留现有窄 HIP 准入；prepared/compile 独立规划 |
| Model Wrapper | 模型/Hessian/训练功能扩展；MACE 本地 checkpoint 修复仍须保留 | Wrapper 设备/输出约定、stress、邻居格式、训练梯度与可选依赖 | 不引入新模型族；现有 MACE 保持当前基线 |
| L-BFGS / NEB | grouped API 新增，Ops `torch/lbfgs.py` 直接 import Warp 并调用 Warp optimizer | Torch reference 的新算子、状态与 grouped 生命周期适配 | 先独立可行性评估，不声称支持 |
| UBF-008/009/011 | C 已确认核心对应，附加平台/边界逻辑仍有价值 | 保留本地 implementation/reference，继续维护科学回归 | retain；无需为重复获得核心修复整体升级 |

候选没有本地 Framework 14 个和 Ops 48 个命名空间文件，包括 `_backend`、
`_dynamics_reference`、storage backend、registry/catalog/dispatcher/executor、HIP kernels、
Torch NHC/cell/stress/reference 等。直接把当前 packages 替换为 stock 候选会丢失这些适配。
没有据此重构整个 Backend Registry；后续按模块接线并保留已验证能力。

## 4. 历史分叉与三方文本试验

| 组件 | merge-base | Baseline/Candidate 独有提交 | 两边修改交集 | 文本冲突 / 干净 |
|---|---|---|---|---|
| Framework | `bdc7fb68efad484062699f9725e8c33448c479e0` | 24 / 41 | 40 文件 | 21 / 19 |
| Ops | `200452c75797a5230c55cbbc88fef6c99f137500` | 4 / 42 | 8 文件 | 3 / 5 |

两套历史均 diverged。逐交集文件使用 `git merge-file -p --diff3`，明确三端为
本地 C 产品文件、锁定 Baseline 文件、Candidate 文件。只生成忽略的合并结果，
不改工作树产品，也没有执行 subtree pull、冲突解决或完整仓库 merge。
21/3 是这个明确基点的逐文件文本冲突结果；干净的文本合并仍需语义审查。
本试验不处理跨文件 rename、mode 或真实 subtree 的拓扑合并，因此不是正式合并保证。

Framework 冲突集中在 Batch/level_storage、BaseDynamics、cell_align/fire/npt_nph wrapper、
Langevin/NHC/FIRE2、Hook 导出/logging/monitors/periodic、DomainParallel、training、
pyproject 及状态/observer 测试。Ops 冲突为顶层 __init__、cell_filter、pyproject。
完整逐文件列表、退出码和三端 blob SHA 见 `static-evaluation.json` 与 `merge-file/`。

正式升级前须审查 Baseline 独有提交，明确保留哪些逻辑和补丁，再做隔离 subtree/补丁重放。
不能直接从 main 覆盖，也不能用 24 个文本冲突数量估算整个适配的工期。

## 5. 代表性源码同步实验与 CPU/HCU 验证

试验采用完整冻结候选源码组合，从中选择三个运行依赖闭合的原样模块：
`nvalchemi/data/group_layout.py`、`nvalchemi/hooks/_context.py`、`nvalchemi/hooks/_protocol.py`。
通过 [隔离探针](../probes/upstream_sync_v1.py) 按路径加载，源文件 SHA256 固定于探针；
原候选源码、公开包导入和主工作区产品均未改动，没有逐文件混入稳定产品。

| 检查 | CPU | HCU4 |
|---|---|---|
| GroupLayout 映射、group_ptr、node/group/graph 基数 | 通过 | 通过 |
| reduce_all/any；两精度 broadcast 与一阶梯度 | 通过 | 通过 |
| 标签规范化、非连续组拒绝、非法编号/dtype/mask/shape | 通过 | 通过 |
| 空输入、buffer 已占用行裁剪、frozen dataclass | 通过 | 通过 |
| DynamicsContext keyword-only/mask/identity；Hook protocol/直接调用 | 通过 | 通过 |
| 共计 probe checks / 进程退出码 | 11 / 0 | 11 / 0 |

使用明确标注的 ProbeBatch 最小输入夹具。没有执行真实 candidate Batch.set_group_layout、
schema/cache、FusedStage/grouped updates、真实模型、Hook registry/event dispatch、checkpoint、
compile、原生 Warp 或性能测试；Hook 直接调用不等于引擎时序验证。
孤立模块的 Torch 2.9 成功不解除候选 Ops Torch>=2.10 的全包声明阻塞。

运行日期 2026-10-09；Python 3.12.13、Torch distribution 2.9.0+das.opt1.dtk2604
（torch.__version__=2.9.0）、torch.version.hip=6.3.26093、NumPy 1.26.4、jaxtyping 0.3.11；
激活 DTK 26.04 env.zsh，OMP_NUM_THREADS=1。HCU4 为 BW1000 64G，
`gfx936:sramecc+:xnack-`，HIP_VISIBLE_DEVICES=4 对应进程 cuda:0。
探针逐张量断言设备，未使用静默 CPU fallback。
每次进程 timeout 90s。HCU4 前后只读检查均空闲，作业后没有 KFD PID。

探针 SHA256：`4370fe9993b717be4833eaff0c28228df0319a85e3ca25d9560ec0f366f3ace5`。
原始目录均相对主项目根目录 `artifacts/upstream-sync-2026-10/`：

- `candidate-manifest.json`：完整源码身份/1380 个文件哈希。
- `static-evaluation.json`、`static-evaluation.log`、`merge-file/`：声明/API/冲突分析。
- `candidate-imports.json`、两个 `*-import.log`：实际 stock 导入失败与退出码。
- `cpu-isolated-modules.{json,log}`、`hcu4-isolated-modules.{json,log}`：11/11 检查和环境。
- `hcu-*before*`、`hcu-resources-after.json`、`hcu-pids-after.log`：资源检查。
- `commands.json`、`validation-status.json`：命令、不变性和未覆盖范围。
- `releases/`：官方 latest release GET 元数据；没有发送消息或创建上游 issue/PR。

最终不变性核对通过：主项目 912 份基线源码/夹具、候选 1380 份原文件、3 份环境控制文件
与两套缓存引用均保持原值；D worktree 的 packages 和版本锁与 develop 相同，
能力契约与 C 起点相同（保留 A 已补入的验收证据）。`git diff --check` 退出 0。
环境核对包含控制文件和实际运行 metadata，没有声称对全部已安装依赖文件做完整哈希。

## 6. 决策与重新评估条件

1. **整体升级 defer**：依赖组合未满足，stock public import 阻塞，关键状态与 backend 接口
   需要适配，24 个交集文件出现文本冲突；没有全候选 CPU/HCU/L3 数值证据。
2. **当前 Baseline retain**：16 项 UBF 保留；C 新科学案例进入后续相关回归。
   初始 HIP M1–M9 和已验证 Torch reference 功能继续在固定版本上开发。
3. **独立 backport 候选**：GroupLayout/Hook 数据协议可作为小任务进一步评估；
   需先定义真实 Batch/生命周期接线和验收，不能由本 probe 自动发布支持。
4. **重新评估触发**：需要具体上游功能/修复、可接受的候选组合和独立环境方案、
   Backend/Storage/Dynamics 适配闭包及 L1/L2/L3 gate 明确后，再创建下一轮 sync 分支。
   不为追赶 main 自动重排全部开发，也不为此更换生产 Torch/DTK。

全阶段产品/锁/环境保持原值，候选继续 Candidate；本轮不合并或推送 develop。
这次评估允许正常功能开发继续，sync 分支保留本轮报告/探针，不成为长期开发主线。

## 7. 收口与后续交付

用户于 2026-10-09 确认 D 签名收口和 PR 整理；本提交归档评估成果。
将 A–D 的流程、Delta 工具、UBF 表和脱敏报告整理为独立可审阅 PR。
该 PR 不包含候选源码导出、实验环境或产品升级；整包升级保持 defer。
后续按用户确认进行集成/推送，并在稳定基线上选择下一项正常移植任务。
固定晶胞 ASE-compatible BFGS、官方 L-BFGS 评估、HIP 邻居性能和单节点 DomainParallel
仍是候选队列，本轮没有自动决定优先级或混入同一个功能 PR。
