# Upstream Bugfix Alignment C — 2026-10-09

## 1. 结果与范围

UBF-001～016 均已建立结构化跟踪，见
[UPSTREAM_BUGFIX_STATUS.yaml](../docs/UPSTREAM_BUGFIX_STATUS.yaml)。结论为：
3 项核心语义 equivalent、5 项 partial、1 项 different、7 项 unknown；16 项均 retain。
本轮没有删除补丁、同步产品源码、修改版本锁或升级生产环境。

新增 [官方科学回归案例](../scripts/test_upstream_bugfix_regressions.py)，将 Ops PR194 的
单键晶胞力有限差分和异构 FIRE2 晶胞运动案例接到本地 Torch reference API，6 项 CPU 通过。
既有 ops 111 passed/3 skipped、framework 兼容性 87 passed、storage 2 passed；
另有本地 checkpoint 隔离断言 1 passed。完整 MACE 测试模块的收集失败单独保留，
不能将隔离断言计为原模块通过。

C 在 `wangleping/docs-upstream-bugfix-alignment` 完成，已获用户确认收口；
B 已签名提交 `84b4ba95ac008d4022acfcff6ea076a5d7793a55`。本轮未合并或推送。
用户已授权 D 隔离评估。A–C 的记录收口后，日常移植可恢复，D 独立开展。

## 2. 冻结来源与核验方法

| 对象 | 精确 SHA | 用途 |
|---|---|---|
| develop | `2e211ab93a4f1ad0c8c720eecb22129d846c7792` | 已集成并经 A 验证的稳定产品源码 |
| 本地产品审查提交 | `84b4ba95ac008d4022acfcff6ea076a5d7793a55` | 包含 A/B 记录；packages 与 develop 相同 |
| Framework Baseline v0.2.0 | `4dfe3723def34df3fadb245981081ccf8c94c257` | 原因归属和本地补丁对照 |
| Ops Baseline v0.4.1 | `26dbceb61e30cca80e1a5805eebeb51d7dc68fd1` | 同上 |
| Framework Candidate main | `f59e28dc6701583a13bb294e536076c2a7379e98` | 冻结源码审查，未执行 |
| Ops Candidate 0.5.0-rc | `5fffc0726b1e71bb8b68e10b5f87942064759f5b` | 冻结源码审查，未执行 |

使用忽略的官方 Git 对象缓存 `artifacts/upstream-delta-cache/{framework,ops}.git`；
`external/` 未改动。对照基线、候选、本地实现和具体官方提交的 diff，
通过 GitHub REST GET 核对 PR merged 状态和 merge SHA，再检查其包含于冻结候选。
不以提交主题、相关 PR 或版本较新来推断完整修复。

45 份源码/测试快照的 Git blob、SHA256 和行数见
`artifacts/upstream-bugfix-alignment-2026-10-09/source-index.json`；官方元数据与提交 diff
在同目录 `github/`、`diffs/`。公开报告只引用官方链接和脱敏结论。
结构化表中的 `upstream_fix_commit` 可以是部分边界的具体实现提交；
必须结合 `equivalence_scope` 阅读。相关设计/弃用提交另列，不冒充算法修复。

## 3. 16 项对照

| UBF | 官方证据与候选源码判断 | equivalence | action | 必须保留的边界或价值 |
|---|---|---|---|---|
| 001 Yoshida 参数 | NHC pre/post 和 wrapper 仍未透传 Yoshida/nloops；无等价 fix 证据 | unknown | retain | Yoshida 3/5 与两次半步调用 |
| 002 Q/ndof refill | `_make_new_state` 仍用 dummy masses、Q.expand、平均原子数 | unknown | retain | 异构实际计数、scalar refill、vector arrivals 显式限制 |
| 003 inactive 温控链 | [PR181](https://github.com/NVIDIA/nvalchemi-toolkit/pull/181)/[182](https://github.com/NVIDIA/nvalchemi-toolkit/pull/182) 保护 FusedStage masked state；未显示普通 NHC 的同等保护 | partial | retain | 四入口、阶段归属、AFTER_* observer 前恢复 |
| 004 eta drift | 两个 ops 传播函数仍各两次使用 `half_dt=dt/2` 推进 eta | unknown | retain | dt/4 对称 drift、独立半步导数与扩展能量回归 |
| 005 inactive outputs | [PR204](https://github.com/NVIDIA/nvalchemi-toolkit/pull/204) 的 `_publish_model_output` 按 storage level 发布活动行；架构不同 | partial | retain | 冻结 E/F/stress、N=B 层级区分、观察者时序 |
| 006 HostMemory ownership | write→to_data_list/get_data→to(device) 未建立完整所有权等价证据；[PR190](https://github.com/NVIDIA/nvalchemi-toolkit/pull/190) 只针对 LoggingHook | unknown | retain | 直接 sink.write 的快照深复制、CPU 异构帧 |
| 007 local checkpoint | candidate from_checkpoint 仍无条件进入下载器 | unknown | retain | 本地文件直读与命名模型下载分支 |
| 008 uniform put | [Framework PR194](https://github.com/NVIDIA/nvalchemi-toolkit/pull/194) 复用初始 dest_mask 和首字段映射 | equivalent | retain | 核心槽位语义对应；Torch/Warp backend 和 segmented 合同继续保留 |
| 009 cell force | [Ops PR194](https://github.com/NVIDIA/nvalchemi-toolkit-ops/pull/194) 的 `-V H^-T sigma` 核心公式对应 | equivalent | retain | reference、两模式、dtype/strides/alias、梯度与 wrapper |
| 010 cell alignment | ops 仍先取负 H；framework 仍未显式写回 positions.contiguous 临时缓冲 | unknown | retain | 左手分数坐标、原 tensor strides 写回；offenders 检测不能替代修复 |
| 011 FIRE2 affine | 同一 Ops PR194 的行晶格仿射映射数学对应 | equivalent | retain | 核心公式对应；限幅、状态及 API 的整实现等价未证明 |
| 012 full FIRE | [PR218](https://github.com/NVIDIA/nvalchemi-toolkit/pull/218) 弃用 FIRE/FIREVariableCell，未采用本地联合算法 | different | retain | 兼容 API、应力 half-kick、九项晶胞自由度联合控制 |
| 013 Warp race | batch_idx kernel 仍在同一 launch 读取和写入共享控制量；无两阶段 fix 证据 | unknown | retain | 分离参数更新和速度混合；实际 Warp 竞态未运行验证 |
| 014 AlignCellHook | [PR206](https://github.com/NVIDIA/nvalchemi-toolkit/pull/206) 过滤子集并旋转 F/stress，未对应旋转 velocities | partial | retain | 速度协变、已对齐跳过、运动后 cell state 的 realign 拒绝 |
| 015 optimizer state | PR181/182 通用 state/FusedStage 机制覆盖部分边界，新 Batch/group state 契约不同 | partial | retain | 四入口、field level、identity、observer、refill/migration |
| 016 arrival priming | PR181/182 引入 forces_primed 和 admission/refill 生命周期 | partial | retain | E/F/stress 首 kick 前有效、active_batch、Hook/共享计算时序 |

unknown 表示没有建立官方等价修复证据；不表示官方已修复，也不表示已穷尽所有官方历史。
上述“仍有”仅针对表中的冻结候选源码。equivalent 的三项都限定在核心语义，
没有任何一项获得整个实现 identical 或可 retire 的结论。

### 行晶格公式与限幅

行晶格定义 `r = s H`，UBF-009 的晶胞力为 `F_H = -V H^-T sigma`。
UBF-011 的纯晶胞位移为 `delta r = r H^-1 delta H`；官方列向量 kernel 中使用
`transpose(H^-1 cell_raw)`，与本地行向量表达等价。
非正交案例能区分旧公式的乘法顺序，不能只测立方晶胞。

Ops PR194 同时调整 FIRE2 最大位移限制。新增本地案例验证“分数坐标保持”和
“位移不超过 maxstep”两条不变量，但没有比较两实现全部中间控制量、限幅比例或轨迹。
因此维持 UBF-009/011 的 retain；不顺带替换本地整条 FIRE2 更新。

### 生命周期和 storage

candidate `_save_state_fields/_restore_unmasked_state` 明确用于 masked FusedStage，
普通 `BaseDynamics.step` 仍直接调用 NHC pre/post。通用机制存在不证明四入口均已覆盖。
新发布 helper 通过 schema 区分 system/atoms/edges 输出，这是新依赖闭包的一部分；
本地 N=B、stress 和 observer 回归仍有保留价值。

UBF-008 是本轮新增的精确对应：`initial_dest_mask` 由 Framework
`bcc43f55748b6d1bbd928c660c1eaeb109fa604c` 引入。
两个实现复用首字段选定的行，并只发布一次占用更新；backend 路由与 segmented 合同需独立核验。

## 4. 官方回归的吸收与后续检查项

| 官方测试/设计 | 本轮处理 | 后续缺口 |
|---|---|---|
| Ops PR194 `test_stress_to_cell_force_matches_bond_energy_finite_difference` | 新增本地两精度×full/下三角案例，4 passed；原有多键谐势梯度测试保留 | 实际 Warp/NVIDIA 路径需另跑 |
| Ops PR194 `test_cell_only_motion_preserves_fractional_coordinates` | 新增 [3,6,5] 异构体系、uphill reset、受限晶胞运动，两精度 2 passed | 限幅全实现/控制量与原生 Warp 对照未完成 |
| Framework PR194 `test_put_aligns_multiple_fields_to_same_slots` | 对应现有 uniform 多字段/占用 mask 回归，本轮复验；segmented 另复验 | 新版 schema、dtype 与 grouped storage 不能逐文件移入旧基线 |
| candidate `test_masked_updates_preserve_inactive_sub_stage_state` | 已有本地四入口/observer 回归保留，记录为 D 验证项 | candidate 普通 NHC、grouped/segmented state 与 observer 完整验证 |
| candidate `test_refill_resets_force_priming_for_optimizer` | 本地 workflow 首 kick/refill 回归复验，登记设计对应 | candidate admission/runner/shared-compute Hook 时序 |
| PR190 LoggingHook snapshots | 登记为相关证据，未认定 HostMemory 等价 | sink 直接写入的张量所有权检查与实际复现 |

没有官方等价 fix 的 001/002/004/007/010/013 继续保留；科学问题和最小回归可作为反馈候选，
本轮没有向官方发送消息或创建 issue/PR。UBF-006 先完成直接 sink 所有权实证再决定反馈。

## 5. 运行证据、失败与限制

产品 SHA 为第 2 节的 `84b4ba95...`。新增回归文件来自 C 工作树，SHA256：
`88c1d2a012aa2ced0ca567f3a1f72e0e8b1e7d3d3252ef1c91e33f3251e94bb1`。
设备 CPU，`HIP_VISIBLE_DEVICES=''`，OMP_NUM_THREADS=1；使用
`source scripts/activate_hygon_env.sh project`（zsh 激活 `/opt/dtk-26.04/env.zsh`）。
Python 3.12.13、Torch 2.9.0+das.opt1.dtk2604、NumPy 1.26.4、pytest 8.4.2。
framework 和新增案例使用 `NVALCHEMI_TEST_BACKEND=torch_reference`；ops 直接导入 reference。

| 运行 | 结果 | 退出码 | 原始日志（同 artifacts 目录） |
|---|---|---|---|
| 本地 ops NHC/stress/cell tools/variable-cell FIRE | 111 passed/3 skipped | 0 | `cpu-ops.log` |
| 本地 framework 定向 compatibility | 87 passed | 0 | `cpu-framework-compatibility.log` |
| 本地 uniform/segmented 两个 storage 节点 | 2 passed | 0 | `cpu-framework-storage.log` |
| 本地 checkpoint 隔离断言与原 MockMACEModel | 1 passed/1 warning | 0 | `cpu-checkpoint-probe.log` |
| 新增 PR194 科学案例 | 6 passed | 0 | `cpu-official-regression-cases.log` |
| 初次 framework 合并收集 | PhysicsNeMo 缺失，未执行测试 | 4 | `cpu-framework.log` |

原始日志目录：`artifacts/upstream-bugfix-alignment-2026-10-09/`；完整命令与环境见
`commands.json`。MACE 原测试模块导入了无关训练类，触发分布式 PhysicsNeMo 导入；
隔离夹具保留原有本地路径断言和 MockMACEModel，避开这些训练导入，未新增生产 fallback。
此断言只证明路径分支，不能证明 checkpoint 权重数值或训练；真实模型证据引用
[A 报告](g2-integration-a-2026-10-09.md)，不是本轮重跑。
checkpoint warning 来自已存在的 e3nn 常量加载环境，不修改权重加载配置。

3 skipped 为缺 Warp 的 NHC parity 测试。源码/bridge stub/reference 的证据分别记录；
本轮无 HCU、候选运行、实际 Warp/NVIDIA、完整升级、长弛豫或性能回归。
历史 HCU gate 与 C 的静态等价审查相互独立；候选继续为 Candidate。

完整性核验退出 0：16 个稳定编号及必填字段、全部本地实现提交的祖先关系、
官方 PR merged SHA/候选包含性、45 份源码快照及引用路径通过；
912 份已有产品/模型/夹具哈希、3 份环境文件和缓存 refs 保持原值。
packages/版本锁/probes/configs 相对冻结提交无 diff。结果见 `validation-status.json`。

新增案例的重复运行入口：

```sh
source scripts/activate_hygon_env.sh project
HIP_VISIBLE_DEVICES='' NVALCHEMI_TEST_BACKEND=torch_reference OMP_NUM_THREADS=1 \
  .venv/bin/python -m pytest -q scripts/test_upstream_bugfix_regressions.py
```

## 6. 下一阶段 D，已获用户确认

从固定稳定源码建立 `sync/upstream-2026-10` 隔离 worktree，审查两候选组合的依赖交集、
Batch/Dynamics/Hooks/Neighbors/Storage/Model Wrapper API 闭包与 subtree 分叉合并范围。
先形成模块可接受、阻塞和修改量清单，再选择最小代表性闭包试验，按实际资源运行 CPU/HCU。
依赖不满足时记录阻塞，允许结论“暂不升级”，不把失败写成通过。

已有声明风险：Framework Ops>=0.5.0、NumPy>=2,<3；Ops Torch extras>=2.10，
当前 Torch 为 2.9。D 不升级生产 Torch/DTK 或原地重建 `.venv`，不自动合入 develop。
完整候选依赖闭包应在独立目录/环境评估，不能任意复制新版文件到旧产品当作升级成功。
交付模块兼容性/阻塞/适配范围和接受或延期决策报告；完成后再次等待用户确认。
