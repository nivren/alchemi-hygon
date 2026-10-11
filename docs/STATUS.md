# 当前开发状态

更新时间：2026-10-09。本文只维护当前状态、已验证范围、下一步和限制；完整历史交接保留在
[`docs/history/STATUS.md`](history/STATUS.md)。

## 当前基线

- 当前交付基线：`develop`，在既有基线 `fb05b61` 上以 fast-forward 集成 HIP 邻居 M1–M9 提交链（阶段末提交 `b549558`）及本次交接文档更新。既有 NHC、stress/cell-force、MPA-0 M1–M5、Batch 测量、LOCAL-004 和应用分离收口均保留。
- 集成前 CPU 聚焦回归：ops `110 passed/34 skipped`，framework 独立进程 `54 passed/28 skipped/1 xfailed`，probe 比较器 `3 passed`；当时未重跑 HCU。集成后 A 阶段指定 CPU/HCU 验收见下一节，各历史阶段性能结论不扩大为全库 gate。
- toolkit 应用分离提交为 `730a848`，由 `codex/fix-fire-zero-dt-min` 合入；LOCAL-004 实现提交为 `0e354f4`。完整移植审查见[报告](../reports/g2-relaxation-port-closure-review.md)。
- 独立应用工程 `/home/wangleping/codes/alchemi-app` 已在本地 `develop` 创建首提交 `9ae235c`；此工程拥有 MPA-0 CIF 应用入口和专属测试。
- 首轮 remote refs 核验已通过：toolkit 三个 remote 的 `develop` 均包含集成状态提交 `9552588`，应用工程三个 remote 的 `develop` 均包含首提交 `9ae235c`；本条状态记录随其后的 toolkit commit 继续同步。
- 上游 framework/ops 锁定 SHA 见 [`UPSTREAM_LOCK.yaml`](UPSTREAM_LOCK.yaml)；产品源码位于
  `packages/framework` 和 `packages/ops`，`external/` 只作只读参考。

## G2 集成验收 A 与上游跟踪（2026-10-09）

- 固定已集成的源码 `2e211ab93a4f1ad0c8c720eecb22129d846c7792`，在个人分支
  `wangleping/test-g2-integration-a` 完成指定验收；不改产品源码、探针、锁定版本或生产环境。
- CPU reference 脚本 `425 passed/3 skipped`；补充 HIP ops `67 passed/34 skipped`、
  framework `49 passed/28 skipped/1 xfailed`、比较器 `3 passed`。各组存在重叠，非全上游测试。
- HCU4 ops `101 passed`、framework `77 passed/1 xfailed`；固定晶胞公共 gate、COO/MPA-0
  两精度单点/FP64 差分、两精度 FIRE/FIRE2 短步/补位通过；较大真实 1029/2030 原子
  ×两精度×两优化器共 8/8 通过。共 14 个 HCU 任务均退出 0，结束后设备空闲。
- 保持 full-PBC/full-list/MATRIX+Torch COO/skin=0、当前晶胞完整重建、显式 HIP 选择的
  窄边界；实际 Warp/NVIDIA、长物理收敛、完整 compile 与其他未准入能力没有新增验证。
  FP32 FIRE2 两组分别 46/47、99/100 晶胞移动；未移动结构两后端晶胞相同，逐结构比较通过。
- [A 报告](../reports/g2-integration-a-2026-10-09.md)包含 SHA、环境、命令、退出码、数值和限制；
  长期开发流程写入 [UPSTREAM_SYNC.md](UPSTREAM_SYNC.md)，AGENTS/开发指南提供入口。
- 用户确认后 A 已签名收口 `132ff32`；B 已在 `wangleping/feat-upstream-delta-v1` 完成只读
  [差异工具](../scripts/upstream_delta.py)、[候选记录](UPSTREAM_CANDIDATES.yaml)和
  [首次报告](../reports/upstream-delta-2026-10.md)。14 项 CPU Git/AST 夹具验证通过。
  Framework/Ops 历史均分叉，467/322 个文件变化、40/8 个潜在交集；未运行候选源码或 HCU，
  尚非 Validated。B 已签名收口 `84b4ba9`。
  C 在 `wangleping/docs-upstream-bugfix-alignment` 完成
  [16 项 UBF 对照](UPSTREAM_BUGFIX_STATUS.yaml)：3 项核心 equivalent、5 partial、
  1 different、7 unknown；全部 retain。官方 PR194 科学案例 6 passed；本地 ops
  111 passed/3 skipped、framework compatibility 87 passed、storage 两节点通过，
  checkpoint 隔离断言 1 passed；原 MACE 模块收集失败单列于
  [C 报告](../reports/upstream-bugfix-alignment-2026-10.md)。候选/HCU/Warp 本轮未运行。
  C 已签名收口 `ad18998`。日常移植可恢复，D 独立开展。
  A–C 阶段未修改 packages/版本锁/生产环境；后续本地集成和三处交付见下节。

- D 在 `sync/upstream-2026-10` 独立 worktree 完成，见
  [隔离升级决策](../reports/upstream-sync-2026-10.md)。完整候选原样导出、依赖与 API 审查完成，
  48 个交集文件的三方文本试验出现 Framework21/Ops3 冲突；text clean 不代表语义兼容。
  未满足 Torch/NumPy/plotext 等声明，Warp/PhysicsNeMo/click 缺失；实际 stock data/L-BFGS
  导入因缺 Warp 退出1。候选三个原样独立模块（GroupLayout/HookContext/Protocol）
  CPU/HCU4 各11项 probe checks 通过；使用 ProbeBatch 最小夹具，非真实 Batch/grouped
  工作流、Hook 引擎时序、全包或原生 Warp gate。没有改变当前产品能力契约。
  结论为暂不整体升级，候选仍 Candidate；D 已签名收口 `a0a3763`。

## 2026-10-09：HIP 邻居 FIRE2 完整弛豫补测

用户确认的小规模完整弛豫已在 `wangleping/test-hip-fire2-convergence` 独立 worktree 完成，
产品起点 `ad18998`，packages 与 develop `2e211ab` 相同。见
[验收报告](../reports/g2-hip-fire2-convergence-2026-10-09.md)及
[契约](HIP_FIRE2_CONVERGENCE_VALIDATION.md)。CPU 定向59 passed/16既有设备skip，探针夹具4 passed；
HCU4 上 FP64/FP32 × HIP/Torch reference邻居四作业均退出0，三个Li/Na CIF各自双阈值收敛，
更新77/153/53次，实际容量2/14原子补位、来源/ID/status及priming核验通过。
12个结构运行（其中HIP6个）均满足fmax<=0.01 eV/Å且max(abs(stress))<=0.1 GPa；
终态同几何完整E/F/stress及CIF重读CPU/HCU重算通过。
优化器仍Torch reference；不新增任意材料/大Batch收敛、性能、Warp、compile或DD支持。
产品/上游锁/环境未变；补测已签名收口 `8306b7f`。

## 2026-10-09：A–D 与补测集成和三处交付

独立分支 `wangleping/integration-g2-upstream-sync-v1` 已保留 A–D 与补测的签名提交，
以 `8623517` 合并两条成果支线；三个文档冲突均保留双方证据。
CPU 复跑：工作流59 passed/16设备skip、Delta夹具14 passed、官方科学案例6 passed、
补测夹具4 passed、候选独立模块11 checks；均退出0。
[收口报告](../reports/g2-upstream-sync-v1-closure-2026-10-09.md)包含交付范围、证据复用和用户授权记录。
用户随后明确取消本轮 PR，授权直接合并本地 develop 并推送三处。
本地 develop 已从 `2e211ab` 快进集成到成果提交 `68ceb00`，本节签名文档提交随后交付至
`github-origin`、`gitee-origin`、`local-origin` 的 develop；实际 SHA/退出码核验保存于
`artifacts/g2-upstream-sync-delivery-2026-10-09/verification.json`。
packages/产品版本锁/生产环境保持原值，候选仍 Candidate；本次未新增测试或 HCU 作业。

## 2026-10-09：Upstream Sync V1 最小维护收口（4/4）

维护分支 `wangleping/fix-upstream-sync-v1-closure` 从 develop `82f7841` 建立；
修复已签名提交 `34dcf9f`、文档提交 `4daed66`，4/4 验收完成。
用户已明确授权直接合并并推送三处；本地 develop 已快进集成到 `4daed66`。
交付状态文档提交随成果推送至 `github-origin`、`gitee-origin`、`local-origin` 的 develop；
实际交付 SHA、退出码及远端核验见
`artifacts/upstream-sync-v1-maintenance-2026-10-09/delivery/verification.json`。

- [x] 历史 Candidate 恢复 B 原始字节并匹配报告输入哈希；D 结构化结果移至
  [独立验证记录](../reports/upstream-sync-2026-10-validation.json)，原 D 报告不变。
- [x] Markdown/JSON 默认覆盖保护、显式 `--overwrite`、输入与路径冲突保护通过。
- [x] 原14项+新增7项 unittest 全部通过，exit0；B 完整输入/原工具复现 Markdown 字节一致，
  SHA256 `64074a979847a0db07395f93287bc8104ee96e88bfddccd1623eebd47f0ed817`。
  当前工具相同输入生成结果仅工具哈希不同；实际历史报告覆盖拒绝 exit2。
- [x] 冻结规则、复现方法、限制及下一任务已写入
  [长期流程](UPSTREAM_SYNC.md#42-历史报告复现与-v1-维护收口)与任务队列。

packages/版本锁/生产环境控制、原报告及 UBF 保持不变，两套缓存引用不变；
本轮未新增 HCU 测试或产品能力。候选仍 Candidate，整包升级继续 defer。
原始验证位于 `artifacts/upstream-sync-v1-maintenance-2026-10-09/`，含测试日志、
复现命令与 `reproduction-validation.json`；历史输入和原工具由 B 提交保存。
V1 工程化阶段到此收口，后续每两周手工观察，相关变化再定向审查。
该维护阶段之后，用户已确认 ASE-compatible BFGS M1–M4，并授权 M1 开发。

以下 M1/M2/E1/M3 记录各阶段当时的范围；当前完成状态及 solver 修订以 M4 节为准。

## ASE-compatible BFGS M1（2026-10-09，个人分支待审阅）

这是上游没有的本地新增功能，不是 NVIDIA 算子移植。分支
`wangleping/feat-ase-bfgs-m1` 从 develop `cfd22ad` 创建，当前未提交/合并/推送。
产品版本锁保持 Framework 0.2.0/Ops 0.4.1；架构参考冻结的 0.3.0-rc/0.5.0-rc
L-BFGS，数学独立对齐 ASE 3.29.0，来源与 SHA256 见
[合同](TORCH_ASE_BFGS_CONTRACT.md)。

- 已实现 M1 单体系固定晶胞 FP64 Ops：独立 H/历史/ID、prepare/step、版本化
  Restart；catalog/executor 接线；所有数值张量留在输入设备，无 ASE/Warp 运行依赖。
- CPU reference gate 退出0：493 passed/3 skipped，包含 BFGS 68 passed；跳过项为
  既有 Warp NHC 对照。compileall、Ruff 目标文件及 diff check 通过。
- HCU4/BW1000 64G 限时单卡退出0：3/9/36 自由度各8步、Restart、显式设备恢复、
  非默认 stream、device/dtype guard；profiler 记录 GPU 求解器活动。H 最大误差
  3.55e-14、坐标6.66e-16；Python 标量检查有同步，无性能提升声明。
- M1 当时尚无 BaseDynamics 优化器、真实 MLIP、Filter 和 Batch；后续 M2 见下节。
  不扩大为完整 BFGS/全 HCU/NVIDIA 支持。[报告](../reports/g2-ase-bfgs-m1.md)记录命令、
  退出码、误差与原始证据。完整 feature 保持 planned，M1 Ops 验证范围单列。

## ASE-compatible BFGS M2（2026-10-09，个人分支待审阅）

用户确认后在 `wangleping/feat-ase-bfgs-m2` 延续 M1 未提交工作；共享 develop、版本锁
和环境未更新。新增公共 `ASECompatibleBFGS`，复用 BaseDynamics 更新/Hooks/run，
初始模型求值和严格 `< fmax` 停止；单体系固定晶胞 FP64，完整历史冻结及本地 restart。
M1 数学只增加两处 profiler 标记，没有改 Hessian/方向/步长公式。

- CPU reference gate 退出0：532 passed/3既有 skipped（含 Framework BFGS39，
  Ops BFGS68）；最终 alias 入口补强后专项41 passed。具体先后/源码见
  [M2 报告](../reports/g2-ase-bfgs-m2.md)，不把专项新增两项算入前一 gate。
- CPU/HCU4 解析势逐步 ASE、三规模/四步位相等 restart、device guard 和屏蔽通过。
  真实 FP64 MACE-MPA-0 水分子两设备均11次更新/12次主轨迹求值收敛；CPU restart
  位相等，HCU MLIP 重算力有舍入差，续跑按量记录误差，不宣称位相等。
- eigh 独立评估 n=138/199/200/201/276/552，B=1/4/16，全部 solver info=0，
  残差/正交通过；仅是探针，不是异构 BFGS 支持。当前候选小矩阵 B=1 比 Torch 慢，
  B=16 收益约1.6–3.1倍；不采用外部旧报告的速度提升结论。
- **Torch 库内 host 数学已证实**：n=138 B=4 调用4次 LAPACKE_dstedc，HIP trace
  有三对角数据 D2H/特征向量 H2D。边界追踪 n=199 有主机调用，200/201 未观察到。
  隔离 native hybrid 探针未观察到受监测主机调用/矩阵回传，尚未注册产品后端。

未提交/合并/推送；完整 feature 仍 planned。UnitCellFilter、异构 Batch/inflight/CSP、
NVIDIA、compile、训练与生产性能未验收。建议 E1 适配 HCU eigensolver，保留
Torch reference/ASE 数学，待用户确认；之后仍交付 M3/M4。阶段顺序见
[已确认开发计划](ASE_BFGS_DEVELOPMENT_PLAN.md)。

## ASE-compatible BFGS E1（2026-10-09，个人分支待审阅）

用户确认后在 `wangleping/feat-ase-bfgs-e1` 保留未提交 M1/M2 工作。新增显式
`hip.bfgs-ase-rocsolver-v1` 与 BFGS 私有 native eigensolver；共用同一数学 body，
只替换分解；Torch reference 仍为默认，auto/None/warp 不准入。当前准入构建为
HCU Torch2.9.0+das.opt1.dtk2604、HIP6.3.26093、rocSOLVER3.20.0.80bb870c。

- CPU reference gate 退出0：550 passed/3既有skip（Ops240、fresh LJ1、Framework263、
  state22、ops24）；最终catalog import断言另跑registry18 passed。原生HCU4专项31
  passed，实际info=1非收敛、状态原子性、L/U、输入保护、非默认stream、schema/fake
  opcheck通过；不据此声明compile或Autograd。
- HIP Framework 解析势逐步ASE/四步位相等restart通过，最大q3.47e-16/H6.40e-14；
  MPA-0水分子11步/12次主轨迹求值收敛，q1.56e-15/H1.99e-12对ASE；实际模型
  restart仍按逐量误差验收。
- 产品eigh18组重构/正交/谱和info全通过。n138/199/200/201/552、B4原生trace无
  受监测LAPACK/矩阵回传，D2H仅1/4字节；Torch阳性对照有8次host调用。完整BFGS
  DOF9/138/201/552各3步同样无受监测host数学/矩阵回传。允许诊断标量及库索引
  元数据传输；证据只覆盖当前构建和已测规模。
- n<200 Jacobi、n>=200 DC；B1 Jacobi更慢，B16原生产品求解器收益1.94–3.00倍；
  不扩大为Batched BFGS或CSP性能。单水分子端到端主要仍由MLIP决定。

合同：[HIP-BFGS-EIGH](HIP_BFGS_EIGH_CONTRACT.md)，命令/退出码/误差/性能及来源：
[E1报告](../reports/g2-ase-bfgs-e1.md)。未提交/集成/推送，版本锁与系统环境不变；
完整feature仍planned。后续M3已确认并完成下述单体系验收，M4 Batch/refill/CSP仍必交付、待确认；
NVIDIA/其他构建、多卡/线程压力、compile/训练、生产性能未验收。

## ASE-compatible BFGS M3（2026-10-09，个人分支待审阅）

用户确认后在 `wangleping/feat-ase-bfgs-m3` 延续未提交 M1/M2/E1。
新增 `ASEUnitCellFilter` Ops 和公共 `ASECompatibleBFGSVariableCell`，完整
`3N+9` 广义坐标/力、ASE Filter 全部参数/默认值、联合严格收敛、配套晶胞重启；
同一个 BFGS 数学 body 和原生 eigensolver 未改变，不升级版本锁。

- 最终 CPU reference gate 退出0：621 passed/3既有skip（Ops268、fresh LJ1、
  Framework306、state22、ops24），包含 Filter28/变胞 Framework43；固定晶胞回归通过。
- CPU/HCU4 Torch/HIP 三条解析轨迹均与 ASE 在15/12/16步联合收敛；全部15个
  广义自由度含外压有限差分最大误差约2e-10；逐步方向/H/坐标/cell/力/应力及重启通过。
- HCU4 专项8 passed：CPU→HCU Filter/历史完整恢复、非默认stream及冻结；原生
  solver31项回归通过。CPU与HCU4显式HIP周期MPA-0 Al4小体系均5步/6次主轨迹
  模型求值收敛，配套restart通过；不扩大为分子晶体CSP或真实MLIP的全后端 gate。
- 15自由度完整变胞3步HIP trace无受监测host LAPACK/数值矩阵回传，D2H仅1/8字节
  检查/ID元数据。Torch同规模有主机LAPACK及112/120字节D2H，CPU LU阳性对照有效。
  现有Torch neighbor hook仍有host体系边界/范围元数据，不宣称整个MLIP工作流无host操作。
- constant_volume 按ASE去迹投影，不是每步严格体积约束；fmax/maxstep采用N+3广义行。
  M3为候选原子/晶胞/H/history原子提交增加O(d²) scratch复制；性能记录含探针/guard开销，
  无端到端加速声明。完整特性仍planned：M4异构Batch/refill/CSP为必交付、待用户确认。

合同：[UnitCellFilter](TORCH_ASE_UNITCELL_FILTER_CONTRACT.md)，证据与实际命令/退出码：
[M3报告](../reports/g2-ase-bfgs-m3.md)。未提交/集成/推送；其他Filter/constraints、
NVIDIA硬件、多卡/compile/训练及生产性能未验收。


## ASE-compatible BFGS M4（2026-10-09，个人分支待审阅）

2026-10-10 DTK26.10候选复验：独立HIP的真实285维单矩阵/B1/B4 DC与LOWER
均正确完成，旧错误谱/超时在已测范围未再出现；n199 CPU LAPACK仍存在，
285维DC还抓到私有CPU `steqr_host` 子块数值求解/矩阵往返，不满足设备内数学
合同。现有26.04构建Torch在隔离26.10库环境import失败（缺libhipblas.so.2），
完整26.10 Torch/BFGS未测。产品准入/默认DTK/全部Jacobi均未改，26.10不记为
支持版本。见[26.10报告](../reports/g2-ase-bfgs-dtk2610-eigh.md)；随后M5-1结果见下节。

2026-10-10 独立 eigh 调查补充：本机 Torch Default 已走 hipSOLVER，batched
仍逐矩阵调用；本机 DC 主机 LAPACK 边界 n<200。无 Torch 链接的独立 HIP
复现真实285-DOF的单矩阵/strided-batched UPPER/DC info=0错误谱；原样外部
扩展复现VMFault。同输入 Default/cuSOLVER 有界复验超时，MAGMA及Jacobi完成。
Jacobi库内仍有4字节收敛标志host读取，不能声称完全异步。详见
[调查报告](../reports/g2-ase-bfgs-eigh-investigation.md)；产品代码及默认选择未改，
历史通过结果不扩展为该输入在当前GPU Torch策略下已验证；随后M5-1结果见下节。

用户“继续”确认后在 `wangleping/feat-ase-bfgs-m4` 延续未提交 M1–M3。实现异构
固定晶胞/UnitCellFilter Batch，按稳定ID保留独立Hessian/历史/Filter/计数，按DOF
分组batched eigh，共享模型调用；严格逐体系冻结、重排、退出、原生补位与配对
Batch快照/旧单体系快照提升。没有更改通用BaseDynamics/FusedStage或FIRE/FIRE2。

- 最终CPU gate退出0：**650 passed/3既有skip**（Ops276、freshLJ1、Framework327、
  state22、dynamics Ops24）。HCU solver39、Batch/变胞14 passed，退出0。
- 真实46/92原子MPA-0 FIRE2→BFGS变胞两结构完整收敛；各ID前3次更新对ASE CPU
  坐标/晶胞误差约1e-14、H约2.55e-11。四次inflight弛豫全部收敛、退出与补位，
  约103.37秒/470主模型调用，其中模型21.32秒、BFGS更新60.49秒；无匹配加速对照。
  CPU真实固定晶胞通过配对checkpoint续算收敛，计时仅续算窗口；HCU固定晶胞也完整收敛。
- 真实285-DOF近alpha简并Hessian暴露本机构建strided-batched DC VMFault及
  info=0错误谱。私有HIP现直接统一使用Jacobi，添加设备内分解有效性守卫，
  保留ASE数学/FP64/current stream。旧E1/DC性能不代表当前代码。
- 完整147/285/147 DOF分组三步HIP trace无受监测host LAPACK，数值矩阵拷贝均
  在设备内，D2H仅1/4字节诊断；Torch阳性对照仍有host数学/数值往返。ID/边界
  等host调度允许同步；不扩大为整个MLIP工作流无host操作。
- 真实预热profile三步/六分解，eigh设备累计308ms，H更新7.8ms。M4当时建议可选M5-1
  同输入稳态性能试验/方案选择；后续完成结果见下节，不自动改变算法或启用已知故障DC。

合同：[Batch](TORCH_ASE_BFGS_BATCH_CONTRACT.md)，完整命令/实际退出码/误差/
失败记录/性能范围见[M4报告](../reports/g2-ase-bfgs-m4.md)。M1–M4功能完成上述窄
验收，全部未提交/集成/推送。仅当前HCU构建、小样本FP64 eager；NVIDIA硬件、
其他构建/Filters/constraints、DomainParallel、多卡、compile/训练、通用采样器
checkpoint与生产规模/吞吐未验收，不改上游版本锁。

## ASE-compatible BFGS M5-1（2026-10-10，性能定位完成）

用户确认后在 `wangleping/perf-ase-bfgs-m5-1` 完成性能合同、两项测量探针与记录。
11个BFGS/Filter产品源hash前后相同，准入/默认求解器/版本锁未改；未提交/集成/推送。

- 147/285维B1/B4，三个进程每项预热2次/7样本，全部数值门与实际退出码0。
  285/B1 raw约92.16ms、checked约92.64ms，独立有效性守卫约0.50ms。
  谱误差最大2.56e-11，方程/正交约1.14e-13；异构147/285/147核心坐标CPU门通过。
- 285/B1/B4的GPU kernel占hipprof窗口97.0%/97.6%，offd/rotate为主要热点。
  B4已通过strided-batched/grid.z批量执行；库内每轮4-byte收敛读取仍存在，API
  等待与GPU计算重叠，不能相加或全部解释为数据传输成本。
- 三次46/92原子 **MACE MPA-0 medium** FIRE2→UnitCellFilter BFGS均收敛，
  233阶段步/234次模型求值，各ID更新88/213次；工作流窗口56.55–56.95秒。
  checked eigh占更新约88.8%/整个窗口47.8%。禁用额外ASE模型对照；计时有同步、
  checkpoint与首模型冷初始化，不是生产吞吐或新后端加速。没有测试mace_off small。

证据：[M5-1报告](../reports/g2-ase-bfgs-m5-1.md)，
[合同](ASE_BFGS_M5_1_PROFILE_CONTRACT.md)。产品能力仍为M4已测范围，
M4全CPU/HCU gate本阶段未重跑。M5-1当时建议M5-2；后续已确认结果见下节。

## ASE-compatible BFGS M5-2（2026-10-10，隔离原型验收完成）

用户“继续”确认后，在 `wangleping/perf-ase-bfgs-m5-2` 实现自写HIP round-robin
Jacobi原型，仅probes、FP64 LOWER 147/285维、B=1..4；11个产品源hash及版本锁未变。
尚未注册产品后端，未提交/集成/推送。

- CPU oracle、HCU19例、13个非法输入拒绝、实际非收敛/非默认stream/不同矩阵
  Batch冻结通过。谱误差最大3.695e-13、相对方程4.935e-15、正交2.381e-13。
- 147/285/147分组10步ASE、restart位一致、单与Batch及失败原子性通过；坐标
  误差最大9.603e-15。自写旋转/排序/诊断在HCU，每sweep仍有显式标量host读取。
- 同M4两CIF **MACE MPA-0 medium** FIRE2→UnitCellFilter BFGS均收敛，233阶段步/
  234主轨迹求值；各ID前3次ASE对照坐标/cell最大1.421e-14、H2.553e-11。
- 三进程×7样本，相同输入checked solver相对rocSOLVER Jacobi耗时下降28.7%–56%；
  包含有效性守卫仍通过10%微基准门；直接分组数学136.715→68.556ms。
- 初次八卡作业占用HCU4期间的整段CSP计时仍排除，自身driver已停止（退出143）。
  后续用户确认新窗口三轮配对，六次全部退出0并收敛、均233步/234模型求值；
  174次资源观察无其他HCU4进程。完整同步工作流中位数57.308→46.726秒，
  耗时下降18.466%，约1.226倍。92原子最大配对坐标差异3.386e-4Å，
  同rocSOLVER重复最大1.472e-3Å；完整误差、样本与计时边界单独记录。

原型的排序launch约束、未中心化简并收敛与基线B4零矩阵失败均保留记录；
没有放宽数值门或改变ASE数学。产品能力仍为M4范围，未重跑M4全功能门。
见[M5-2报告](../reports/g2-ase-bfgs-m5-2.md)、[配对验收](../reports/g2-ase-bfgs-m5-2-e2e.md)、
[合同](HIP_BFGS_JACOBI_PROTOTYPE_CONTRACT.md)。M5-2仅隔离候选验收完成，未注册产品、
未完成M5整体或生产吞吐。用户明确要求取消147/285与B≤4限制；原受限接入方案
撤回；用户随后确认M5-3A动态扩展，结果见下节。产品接入另行确认。

## ASE-compatible BFGS M5-3A（2026-10-10，动态原型窄验收完成）

用户确认并提供含锂晶体与T2目录后，在 `wangleping/perf-ase-bfgs-m5-3a` 完成
隔离Jacobi原型动态n/B、64位偏移、完整配对launch和设备内稳定排序；取消147/285
与B≤4入口限制。可选chunk是临时工作区控制；逻辑Batch没有这个固定上限。
11个产品源和上游锁hash不变，未产品注册/提交/集成/推送；不同宽度组仍串行。

- CPU动态配对/oracle、旧HCU19例/13非法输入、动态32矩阵及B8/16/32四例通过；
  达到1113维和真实561维B8。分块与整批五项输出逐位一致，末块失败原子性/
  非默认stream通过。最大谱3.695e-13、方程4.935e-15、正交2.956e-13。
- 新11体系10步ASE数学通过，坐标最大2.159e-14、restart逐位一致；CPU专项
  Ops104/Framework105和HCU Framework14通过。没有重跑全项目650/53门。
- MPA-0 medium：8个独立含锂CIF（2–160原子）各3步独立CPU MLIP+ASE对照通过；
  8个独立T2 CIF经ASE解析各184原子，真实561维B8共享HCU力/应力的ASE CPU
  数学及独立HCU对照通过。两组均是短步验证，未完整收敛；T2独立CPU MLIP
  8体系三步对照超时退出124，不记为通过。所有容差沿用原值。
- 三进程×7交替微基准：旧尺寸checked耗时下降27.75%–57.19%；含锂/T2真实H
  分解窗口分别下降46.84%/47.07%。合成189 B16、513 B8、561 B1/B8退化，
  561 B8约慢44.80%，不登记普遍加速。chunk3也有额外耗时。
- 旧46/92原子完整变胞CSP新鲜配对各一次均收敛，233步/234模型求值，
  57.649/47.128秒，只作扩展回归。513维trace明确有HCU radix排序，API D2H
  均为诊断标量；主机标量同步仍存在。186次后段资源观察未发现HCU4竞争。

实现/命令/实际退出码/误差/退化及未覆盖范围见
[M5-3A报告](../reports/g2-ase-bfgs-m5-3a.md)、
[动态合同](HIP_BFGS_JACOBI_DYNAMIC_CONTRACT.md)。产品能力保持M4范围；
下一阶段M5-3B随后确认并完成隔离比选，见下节；M5整体未完成。

## ASE-compatible BFGS M5-3B（2026-10-10，异构并行隔离验收完成）

在 `wangleping/perf-ase-bfgs-m5-3b` 完成串行、2/4stream和原生异构Jacobi对照。
新增释放GIL的worker入口、producer/consumer事件和生命周期保护；原生异构用
实际n/int64偏移/tile列表合并launch，不做谱padding。probe数学adapter先准备全部H，
一起分解，全部通过后提交。11个产品源/原M5-3A源/版本锁不变，未产品注册或提交。

- CPU23维配对/双缓冲及12步adapter逐位门；Ops104/Framework105通过。
  四候选各32矩阵+9真实分组、11体系10步ASE、restart/晚组非收敛及坏谱原子性
  通过；32矩阵同输入五输出与串行逐位一致。最大谱1.052e-12、方程4.935e-15、正交
  2.956e-13；ASE数学坐标2.673e-14。无标量检查同步掩盖的stream生命周期门通过。
- Framework24组合通过：固定/变胞默认、pressure-scale、mask、CPU对照、共享
  模型、冻结/重排/退出/新ID、restart promotion和原生sampler/Fused refill。
  第一次refill夹具在CPU，显式HIP正确拒绝退出1；GPU夹具修正后退出0，日志保留。
- 三进程×7稳态样本：真实Li八组checked分解112.183→74.888ms（2stream，
  少33.24%），混合633.613→560.758ms（少11.50%）。T2同宽B8无明确收益；
  4stream无额外明显收益。原生异构数值通过但慢于2stream，暂留实验。
  profiler确认2stream实际不同队列kernel重叠43.910ms/81.493ms窗口。
- MPA-0 medium真实Li/T2各三轮serial/2stream fresh配对：每条轨迹1FIRE+10BFGS，
  主模型12次、各ID前3步ASE；前三次暖机后各候选21个完整更新样本。Li更新
  217.380→170.806ms（少21.43%），T2约0.88%差异不声明加速。均未完整收敛。
  同输入已保证分解逐位；真实10步末态位置差达1.719e-8 Å，未声称长期逐位确定。
- fresh2stream Li八体系独立CPU MACE/ASE前三步另行通过；T2为共享HCU力的
  ASE CPU数学门，M5-3A独立CPU MLIP超时门未重试，不记通过。目标卡资源采样
  未见第二个已分配进程。未重跑全项目门、未测生产端到端每成功结构吞吐。

建议多宽组采用2stream、单宽组保持batched，保留显式rocSOLVER；无固定n/B门。
详见[M5-3B报告](../reports/g2-ase-bfgs-m5-3b.md)和[合同](HIP_BFGS_HETEROGENEOUS_PROBE_CONTRACT.md)。
M5-3B结束时产品仍为M4实现；随后用户确认M5-4，当前产品状态见下节。

## ASE-compatible BFGS M5-4（2026-10-10，显式产品接入）

用户确认后新增 `hip.bfgs-ase-jacobi-streams-v1`。共享ASE数值核心、固定/变胞
Framework API及restart格式保持；单宽batched、多宽同进程双worker/双stream，
producer/completion event、record_stream与失败排空保护生命周期。没有经验n/B
白名单，实际受内存/整数表示/成本约束。`hip`仍选rocSOLVER Jacobi，默认仍Torch，
不进auto；不升级DTK/Torch/锁定源码，不改变FIRE/FIRE2。

- CPU总门退出0：668 passed/3既有Warp skipped；冻结旧共享数学四体系十二步
  全state/五项step字段逐位一致。既有rocSOLVER HCU39回归通过，其源hash未变。
- 最终Framework HCU24/native专项7通过：ASE固定/变胞default/pressure-scale/mask、
  restart/冻结/重排/退出/补位、真末组不收敛/错误原子性、stream分配器寿命、
  双调用者、partial submit排空、关闭/重建、schema/fake及极小谱/有符号零。
- 最终32冻结矩阵/9真实组/4批组合，达到1113维/B32子集和真实561 B8；
  谱/方程/正交最大误差1.052e-12/4.935e-15/2.956e-13。11体系十步ASE
  坐标2.673e-14、H7.816e-14；single/batch和restart逐位通过，原门未放宽。
- 零轮谱恢复修正：原减/加均值把diag(2e-18,1e-18,2)的小谱变成0；最终在设备
  直接排序原始对角值，极小谱和公共step与CPU逐位一致。迭代kernel保持，冻结
  原探针不修改；前后失败/通过记录均保留。

- 最终14个限时作业全部退出0；46/92原子MPA-0 medium完整变胞CSP再次全收敛，
  233阶段步/234主模型调用，广义fmax0.043552/0.048952，ASE H2.546e-11。
  Li/T2各八体系11阶段步/12主模型调用的短步门及Li独立CPU MLIP门通过；
  这16个新增CIF未完整弛豫，T2独立CPU MLIP仍未验收。
- 最终三进程×7同输入微基准：Li分解112.422→74.731ms、完整math194.772→149.094ms
  （对同检查自写单stream）。T2单宽基本持平；合成189 B32比旧rocSOLVER慢
  13.99%（分解）/8.87%（完整math），保留退化。profiler实测双queue重叠43.133ms。
  零轮修正前三轮完整CSP配对总耗时中位数94.997→81.947s，最终单次收敛79.450s；
  源版本和窗口分开记录，不混成最终源码三轮，不声称生产全目录吞吐。

真实MPA-0 medium（非OFF small）CSP、Li/T2短步、性能三轮配对及profiler最终
结果见[M5-4报告](../reports/g2-ase-bfgs-m5-4.md)与[合同](HIP_BFGS_JACOBI_STREAMS_CONTRACT.md)。
本轮未提交/集成/推送；仅已准入HCU gfx936 FP64/eager构建，保留主机标量等待。
其他硬件/构建、NVIDIA、compile/梯度、生产全目录和auto尚未验收。

## ASE-compatible BFGS 最终交付审阅（2026-10-10）

用户确认后完成源码/合同/来源/包装审阅。修正 Batch 步长范数溢出时静默零步
提交的问题，补齐与单体系相同的非有限步长报错；CPU、既有 HIP 与新 HIP 的
双宽组失败原子性回归通过，不改变正常有限轨迹的 ASE 公式。

CPU 总门退出0：669 passed / 5 skipped（3既有Warp、2在CPU环境不可用的HCU参数，
后二项已在HCU通过）。HCU Ops18、Framework24及11体系十步ASE门通过。
46/92原子MPA-0 medium完整变胞CSP再次2/2联合收敛，233阶段步/234主调用，
ASE H最大2.552e-11；该次总窗口79.014秒，仅正确性复验。
两个包wheel/sdist构建通过，独立解包CPU及包内HIP源码JIT smoke通过。

详见[审阅报告](../reports/g2-ase-bfgs-delivery-review.md)、
[使用指南](ASE_BFGS_USAGE.md)和[四组本地提交方案](ASE_BFGS_DELIVERY_PLAN.md)。
历史M5-4证据保持，不把原性能结果重标为审阅后代码的三轮性能门。
本轮仍未暂存/提交/合并/推送；提交阶段待用户确认。支持范围仍是已准入FP64/eager
HCU构建与明确小样本门，其他硬件/构建/生产吞吐未验收。

## ASE-compatible BFGS 本地提交收口（2026-10-11）

用户确认后按审阅清单完成四组带 DCO sign-off 的本地提交：Ops `d68ea01`、
Framework `bf52bca`、验证记录 `93955b0`，第四组为本节所在的工程文档提交。
共120个文件，产品/测试/probe与已通过的审阅哈希完全一致；第四组仅更新原清单
中的交付记录与状态。每组暂存清单、内容与diff检查通过，不重复计算未改变的
CPU/HCU数值门，不覆盖历史证据。完整SHA及文件记录见
[交付方案](ASE_BFGS_DELIVERY_PLAN.md)与 `artifacts/ase-bfgs-local-delivery-2026-10-11/`。

当前BFGS已在个人功能分支本地收口，尚未合入 `develop`、推送或发布。
FP64/eager、已准入构建及小样本验收边界保持；生产规模/其他硬件未验收。
本地 `develop` 集成是待确认的下一阶段。

## 当前能力摘要

| 功能 | 当前状态 | 已验证范围 | 明确限制 |
|---|---|---|---|
| Torch reference neighbor/cell-list | 已实现窄 reference | periodic/no-PBC、full/half、MATRIX/COO、mixed/triclinic Batch、image shift、容量、selective rebuild、连续 distance/vector 一二阶路径 | 不等于完整上游生产 cell-list；完整 target/pair/compile/opcheck/DomainParallel 仍未完成 |
| 显式 HIP neighbor/cell-list | 已实现并验证窄 capability | periodic、Batch、full-list、MATRIX/COO（Torch打包）、FP32/FP64、`skin=0`、自动 doubling capacity；允许调用间 full-PBC 晶胞改变，每次完整重建 | 不进入 `auto`；不支持 no-PBC、half、skin/selective rebuild、target/pair、旧 plan 变胞复用、native geometry backward、compile/opcheck、DomainParallel；MPA-0 HIP邻居单点已验收；HIP FIRE/FIRE2 短步/预算补位已验收，M5/M6两精度变胞性能及M7较大真实Batch验收见下文 |
| Torch geometry | 当前训练/梯度路径 | distance/vector 连续坐标的一阶/二阶路径 | native HIP geometry 仍是 forward-only 性能候选 |
| Langevin | 已完成窄 reference slice | 固定晶胞 BAOAB、float32/64、异构普通 Batch、空输入/错误、短统计 oracle、最小 integrator continuation state；当前 CPU 复核 `15 passed` | 不等于完整上游 Langevin/NVT；通用 checkpoint、`atom_ptr`/`_out`、inflight、分布式、完整行为套件、torch.compile、生产 Triton/HIP 未完成 |
| 固定晶胞 NHC | 已验证窄 Torch reference | FP32/64、C=1–8、Yoshida 3/5、异构 Batch、scalar refill、FusedStage/Hook、最小 continuation、CPU 谐势统计/扩展能量收敛、HCU4 parity 与短 MACE | Warp/NVIDIA 未执行；vector arrivals 映射、完整 checkpoint、DomainParallel、生产 HIP/Triton 未完成 |
| fixed-cell dynamics | reference slice | VV、FIRE、FIRE2、kinetics 及部分公共 wrapper | NPT/NPH、生产后端未完成；变胞证据另列 |
| stress/cell-force 基础 | 已验证窄 Torch reference | FP32/64、完整/下三角转换、CPU 一二阶/compile/opcheck、HCU4 一阶/stream、周期非正交真实 MACE 应力与 FP64 有限差分 | 基础转换本身不推进晶胞；Warp/NVIDIA、HCU 二阶/compile、完整 MLIP 训练、NPT/NPH 未验证；变胞应用另列 |
| MACE-MPA-0 单点 | 已验证窄 CPU/HCU4 | M1 CPU Batch/ASE及 FP64数值导数；M5 FP32/64 同几何 CPU/HCU完整 E/F/stress与邻居，48份导出核验；HIP邻居M2单点三快照 | 全目录/完整模型族未验证；单一精度须匹配默认 dtype；单点一致不代表弛豫轨迹等价 |
| MPA-0 原生变胞批量弛豫集成 | 已验证窄 CPU/HCU4 下游工作流 | toolkit 原生优化器与生命周期 API 由 `alchemi-app` 应用组合；两精度/两优化器单体系、补位和8结构验收；48份输出独立 CPU/HCU核验 | Li(CN)2 FIRE精度敏感、FIRE2应力未收敛；全目录/完整HCU gate、checkpoint/流式服务、混合变胞阶段、生产性能未验收 |

能力宽度以 [`BACKEND_CAPABILITY_MATRIX.md`](BACKEND_CAPABILITY_MATRIX.md) 和
[`FEATURE_COMPATIBILITY.yaml`](FEATURE_COMPATIBILITY.yaml) 为准；本表不把窄 slice 扩大为完整
上游支持。

## 阶段二收口证据

- Ops focused suite：`102 passed`。
- Framework 邻居/LJ focused suite：`19 passed, 1 warning`。
- HCU：gfx936/BW200，`HIP_VISIBLE_DEVICES=4`，46×32 compatibility gate 通过；FP32/FP64、
  empty、capacity overflow、unsupported request、重复 Hook 和自动容量增长均有证据。
- 自动容量 fixture：初始容量 `1` 增长到 `8`，最大实际邻居数 `6`，保守上界 `343`；active
  neighbor 集合与 Torch reference 一致。HIP padded capacity 形状不承诺与上游最小 16/16
  对齐策略一致。
- 当前阶段二 summary：[`reports/g2-framework-hip-neighbor-compatibility-gate.md`](../reports/g2-framework-hip-neighbor-compatibility-gate.md)。
  相关报告按主题见 [`reports/README.md`](../reports/README.md)。

## HIP 当前晶胞重建 M1（2026-10-05）

- 起点 `fb05b61`，分支 `codex/feat-hip-current-cell-neighbors`；已收口实现 `fecbc39`，已随 M1–M9 提交链合入 `develop` 并同步三个目标 remote。
- CPU：ops `87 passed, 6 skipped`；framework `33 passed, 4 skipped, 1 xfailed`。
  skip 为 HCU 专属用例；既有 xfail 为 `WARP-DEFAULT-001`，不是新增豁免。
- HCU4 BW1000 64G/gfx936：ops `22 passed`，framework `4 passed`；46×32 既有
  固定晶胞 gate 也通过。FP32/64、收缩/膨胀/剪切/wrap、多镜像、非默认 stream、
  输入与旧输出保护、显式溢出及 Hook 自动扩容均有窄范围证据。
- 新建每次调用的 cell metadata/plan 本来就是 stateless 行为；本轮明确准入并验证，
  修正容量错误为 Hook 可识别的 `NeighborOverflowError`，修复 registry 意外允许 auto 选 HIP。
- 新增证据仅 full-PBC/full/MATRIX/skin=0；旧 plan 的固定 metadata ABI 保持。
  M1 当时未验收 COO/HIP MACE；后续单点准入见 M2，短步变胞工作流见 M3。M4性能实验另列。
- [契约](HIP_CURRENT_CELL_NEIGHBOR_CONTRACT.md)、[报告](../reports/g2-hip-current-cell-neighbor-m1.md)。

## HIP COO 与 MPA-0 单点 M2（2026-10-05）

- 起点 `8120282`，分支 `codex/feat-hip-coo-mace-snapshot`；已收口实现 `3e394d7`，已随 M1–M9 提交链合入 `develop` 并同步三个目标 remote。
- COO 为 HIP MATRIX 核心加设备内 Torch 打包；直接 ops、framework、Hook 均准入。
  修正 Hook 对自动过量分配 MATRIX 的容量适配，只裁 padding、有效 counts 超容量则报错。
- CPU ops `88 passed, 10 skipped`；framework `38 passed, 12 skipped, 1 xfailed`。
  HCU4 ops `26 passed`、framework `17 passed`（其中5项CPU oracle）；46×32固定回归通过。
- 同 checkpoint/dtype/几何的真实 MPA-0 三个 Li/Na CIF（18原子）、两精度各 initial/shear/wrap，
  直接 compute 与 Hook 对照通过；边数610/602/610、atom/image 集合完全一致。
  最大 E/F/stress 差 FP64 `1.78e-15/9.81e-15/1.30e-16`，FP32 `3.81e-6/5.42e-6/3.54e-8`。
  FP64 force/strain 有限差分误差 `2.43e-10 eV/Å`、`1.68e-12 eV/Å³`，拓扑不变。
- 没有 FIRE/FIRE2、其他模型或性能准入结论；HIP 仍不进入 auto。
  [契约](HIP_COO_MACE_SNAPSHOT_CONTRACT.md)、[报告](../reports/g2-hip-coo-mace-snapshot-m2.md)。

## HIP 邻居变胞 FIRE/FIRE2 工作流 M3（2026-10-05）

- 起点 M2 `3e394d7`/`4136f0f`，分支 `codex/test-hip-variable-cell-workflow`；窄 gate 已通过，
  实现已收口 `6b1680a`，已随 M1–M9 提交链合入 `develop` 并同步三个目标 remote。无新增产品 kernel 或优化器修改。
- Torch reference 优化器/stress/align + 显式 HIP full-PBC/full/COO/skin=0；共享
  NeighborListHook 登记在 FusedStage BEFORE_COMPUTE，run 显式传入设备 Batch。
- CPU 新工作流16 passed/16 skipped、相关回归48 passed/12 skipped；HCU4 新工作流
  16 passed/16 deselected。解析势覆盖短步、inactive 精确冻结、survivor重排/arrival
  新状态、联合力/绝对应力退出、预算退出与补位 priming。
- 真实 MPA-0 三个 Li/Na CIF(18原子)，两优化器×两精度各五步，两邻居轨迹分支及
  当前 HIP 几何独立 reference 拓扑/E/F/stress 对照通过；同几何最大E/F/stress差
  FP64 `1.42e-14/1.50e-14/2.06e-16`，FP32 `3.81e-6/7.80e-6/8.29e-8`。
- 真实三结构容量2的三步预算补位通过，ptr `[0,4,14]→[0,10,14]`，ID完整、首kick
  前输出通过。预算退出不是物理收敛；不包含长弛豫、其他MLIP、训练、完整HCU或性能准入。
  HIP 仍不进入auto；query/workspace/skin优化留待下一里程碑确认。
  [契约](HIP_VARIABLE_CELL_WORKFLOW_CONTRACT.md)、[报告](../reports/g2-hip-variable-cell-workflow-m3.md)。

## HIP 邻居性能与 workspace M4（2026-10-06）

- 用户确认后，M3 已收口 `6b1680a`/`47cf259`；M4 分支
  `codex/perf-hip-neighbor-workspace`，已收口实现 `95f2ef6`，已随 M1–M9 提交链集成到 `develop`。
- 完成基线/hipprof和workspace候选；候选CPU规则+真实HIP回归30 passed。
  同进程AB/BA配对每项45样本，46×32 FP32邻居median `17.903→17.672ms`（下降1.29%），
  真实18原子MPA-0邻居+模型 `36.662→36.428ms`（下降0.64%）。未达预设5%门槛，
  撤回产品缓存改动，保留可重放实验；产品仍stateless、无新workspace/auto准入。
- 分项metadata约 `12.35/24.20ms`（32/64体系），远高于workspace查询/分配约0.3–0.4ms；
  trace支持存在逐体系小kernel与标量host/device往返；M5批量metadata已获确认，证据独立记录。
- 产品恢复后CPU ops27 passed/10 skipped、framework21 passed/24 skipped；HCU4 M3
  16 passed/16 deselected；没有长弛豫或完整CPU/HCU性能结论。
  [契约](HIP_NEIGHBOR_WORKSPACE_CONTRACT.md)、[报告](../reports/g2-hip-neighbor-workspace-m4.md)。

## HIP 邻居批量 metadata M5（2026-10-06）

- 分支 `codex/perf-hip-batch-metadata`；M4已收口 `95f2ef6`/`88b6b2f`。
  M5仅修改ops当前晶胞metadata批量推导，每调用仍完整重建；无新native query/COO。
  产品只有一个注册HIP后端，修改前源码仅作探针比较；实现已收口 `e2d569c`，已随 M1–M9 提交链集成到 `develop`。
- CPU ops45 passed/28 skipped、framework21 passed/28 skipped；HCU4 ops73和framework49 passed。
  真实MPA-0两精度/两优化器五步及FusedStage预算补位回归通过；不是长步物理收敛。
- 约20,000原子异构Batch的200/40/10结构：均FP32/64，包含压缩/膨胀/剪切、N/B/ptr补位及
  继续变胞。FP64 Torch reference→修改后HIP邻居median分别1432.148→11.011、
  323.633→10.988、117.454→10.894ms；FP32分别1450.214→10.617、
  317.542→10.539、111.241→10.472ms。每项六快照54样本，所测拓扑严格一致。
- 相对修改前HIP，目标200结构FP32下降87.65%，三轮一致，其他项无稳定退化；超过5%门槛。
  实际18原子MPA-0 FP64邻居+模型：FIRE 52.419→35.980ms、FIRE2 52.296→35.967ms
  （Torch reference→修改后HIP）；未包含优化器/hook/IO，不是20,000原子真实模型吞吐。
- 原小Batch seed20261006 FP32在6Å边界出现Torch多两条边，修改前后HIP一致；诊断边距
  6.000000088Å。失败保留，不改cutoff/容差；小Batch性能fixture换为明确seed20261007。
  不宣称任意FP32边界位级等价。`auto`、skin/selective rebuild、workspace、native COO、
  query并行、长收敛、其他MLIP、DD/compile/训练未新增。
- [契约](HIP_BATCH_METADATA_CONTRACT.md)、[报告](../reports/g2-hip-batch-metadata-m5.md)。
  M5已收口；后续M6查询协作及M7–M9验证已完成，本次按用户要求随提交链合入 `develop`。

## HIP 查询线程协作 M6（2026-10-06）

- 用户确认后M5实现 `e2d569c`、文档 `3073f63` 收口；M6分支
  `codex/perf-hip-cooperative-query`，实现收口 fe18147、文档 c2a8339；本次随提交链合入 `develop`。
- 仅改变native查询：每原子一64线程block分担候选cell，共享计数预留slot；
  native仍无序、公共排序保持；溢出/重叠明确失败、int64平展数量超限明确失败。
  正式产品一个HIP后端，接口/metadata/CSR/COO/梯度/默认选择保持。
- CPU ops51 passed/34 skipped、framework21 passed/28 skipped；HCU环境ops85与framework49 passed，
  含CPU oracle及真实HIP项，非全库gate。低层mixed-PBC/full/half、空输入、stream/容量及
  连续geometry梯度回归通过，不新增公共no-PBC/half准入。
- 两精度8配置、六变胞/补位快照三路径通过；FP64约20,000原子200/40/10结构
  Torch reference/M5 HIP/M6 HIP邻居median：1444.432/11.138/9.426、
  331.958/11.321/9.571、117.629/10.912/9.674ms。目标few FP64下降11.34%，
  三轮同向超过5%；其他配置无稳定退化，小Batch FP64下降37.76%。
- 实际18原子MPA-0两精度两优化器五步同快照拓扑/E/F/stress与预算补位回归通过。
  FP64 FIRE/FIRE2邻居+模型：55.141/38.605/36.527、54.872/38.331/36.218ms
  （Torch/M5/M6）；未含优化器/Hook/IO，不是较大真实模型或物理收敛验收。
- 独立few FP64查询分项3.550→2.343ms；排序约2.55ms、COO约0.71ms保持。
  trace候选kernel一次1.983ms；分项/trace不混入最终计时，不作为稳态全流程占比。
  原FP32 cutoff两边分歧保留，M5/M6集合完全相同；同名native模块别名的首轮计时作废并保留。
- 候选所测门槛通过并保留，auto/skin/workspace/native COO/长收敛/其他MLIP/DD/compile/训练未新增。
  [契约](HIP_COOPERATIVE_QUERY_CONTRACT.md)、[报告](../reports/g2-hip-cooperative-query-m6.md)。
  M7较大真实Batch综合验收已完成实验，FP64通过、FP32严格forces门未全通过，见下文。

## HIP 较大真实 Batch 综合验收 M7（2026-10-06）

- 用户确认后收口M6实现 `fe18147`、文档 `c2a8339`；分支
  `codex/test-hip-real-batch-validation`，实验与文档以签名提交 `00637c0` 收口，本次随提交链合入 `develop`；该阶段不修改产品代码。
- 真实1029原子/39结构、2030原子/92结构MPA-0，FP64优先、FP32补充；
  两路径都是显式cell-list（Torch reference/当前HIP），同模型/e3nn/dtype/参数。
- FP64两规模两优化器三轮三步及47/100真实CIF预算补位通过；同几何E/F/stress最大误差
  5.68e-13/3.41e-14/2.15e-15。1029 FIRE/FIRE2完整step median
  556.413→271.774 /554.916→272.487ms；2030 FIRE/FIRE2
  1157.782→492.640 /1165.159→492.503ms，相对Torch约2.04–2.37倍。
  完整step含Hook/模型/优化器，priming/校验/IO在计时外；模型耗时几乎相同。
- 测量点最大Torch allocated约10.87/21.03GiB，非全设备显存或完整进程峰值。
- FP32仅1029 FIRE完整组通过；1029 FIRE2、2030 FIRE/FIRE2在forces重复通用状态门失败，
  初始同几何拓扑/逐物理量输出均通过。纯Torch/Torch FIRE2也出现同样失败：3.03e-5。
  既有E/F/stress物理量容差通过，但forces又被更严格atol=2e-5状态门检查；
  未改容差，未把失败扩大归因为HIP kernel问题，也未记作完整组通过。
- CPU定向ops51/framework21、HCU环境ops85/framework49通过，非全库gate；
  M1–M6源/契约/报告一致性审阅完成。初次step2重编译样本与所有FP32失败保留。
- [契约](HIP_REAL_BATCH_VALIDATION_CONTRACT.md)、[报告](../reports/g2-hip-real-batch-validation-m7.md)。
  M7以 `00637c0` 收口；M8模型输出/优化器状态验收分层复验已确认并开始，详见下文。
  不新增长收敛、20k真实MLIP吞吐、其他模型/训练/compile/DD/skin/auto准入。

## MPA-0 FP32 状态验收分层 M8（已收口）

- 分支 `codex/fix-mpa0-fp32-state-validation`，M7签名收口 `00637c0`；M8签名收口 `04f80bd`。
- 用户限定只改 `probes/` 测试与文档；不改 framework/ops 产品源码、kernel、模型或全局容差。
- M7重复将模型缓存E/F/stress套用物理量门和通用mutable-state门。本阶段probe将前者仅按既有
  逐物理量容差比较；positions/cell/velocities及优化器私有状态仍按原状态容差，拓扑/整数仍精确。
- CPU单测3 passed；HCU真实Batch 8组合结果为6组完整通过。所有八组的邻居拓扑、E/F/stress
  物理量比较和轨迹几何/速度/优化器状态比较通过；两组FP32 FIRE2仅在最终额外的“每个晶胞均
  发生变化”断言失败，实际为46/47和99/100个晶胞变化，且保持原条件未放宽。
- Torch/Torch FP32控制3轮×3步没有几何/私有状态严格门失败。framework/ops产品源码、HIP kernel
  与数值阈值未改。
- [M8契约](HIP_MPA0_FP32_STATE_VALIDATION_M8.md)、[M8报告](../reports/g2-hip-mpa0-state-validation-m8.md)。

## FP32 FIRE2 per-system 晶胞移动断言 M9（已收口）

- M8两组FP32 FIRE2只因`moving == len(systems)`失败。定位到 source 36：
  `mp-1013795_Li3Ge3(MoO6)2.cif`。两组中 HIP 和 Torch reference 的终态晶胞均与初始值完全相同，
  且两路径晶胞差为0；终态应力约`7.68e-4 eV/Å³`。
- 仅将该批次的移动断言改为至少有一个晶胞变化；逐结构 backend parity、状态容差和物理量容差保持。
  未改变 neighbor contract 或 framework/ops/kernel。
- HCU4复跑受影响的1029/39与2030/92 FP32 FIRE2，均退出0；分别46/47和99/100结构变化，
  unchanged source 36详情保留在 probe summary。M8其余六组已在更严格条件下通过。
- [M9报告](../reports/g2-hip-fp32-fire2-cell-movement-m9.md)。结果已由签名提交 `fa1259b` 收口。

## Langevin 窄 slice 收口证据

- 公共固定晶胞 BAOAB reference、registry/executor 接线和 contract：
  [`g2-torch-nvt-langevin.md`](../reports/g2-torch-nvt-langevin.md)。
- 独立谐势短统计 oracle：[`g2-torch-nvt-langevin-stat.md`](../reports/g2-torch-nvt-langevin-stat.md)。
- 普通 Batch 的最小 integrator continuation state：
  [`g2-torch-nvt-langevin-restart.md`](../reports/g2-torch-nvt-langevin-restart.md)。
- 当前工作树 CPU 复核：三个兼容性文件合计 `15 passed`；历史 HCU 证据保留原报告中的真实卡号。

## NHC 固定晶胞 reference（2026-10-04）

- 实现与上游四项修正见 [`g2-torch-nvt-nhc.md`](../reports/g2-torch-nvt-nhc.md)；代码旁标记
  `NHC-YOSHIDA`、`NHC-REFILL`、`NHC-MASK`、`NHC-ETA`，未改变上游锁定版本。
- NHC focused ops `22 passed, 3 skipped`；三项跳过因 Warp 缺失。framework 生命周期
  与物理 oracle 合计 `21 passed`，已加入 CPU reference gate。
- 本次 CPU gate 退出码0：分进程 `67 passed, 3 skipped`、`1 passed`、`157 passed`、
  `22 passed`、`24 passed`，合计 `271 passed, 3 skipped`；compileall 与 diff检查通过。
- HCU4：FP32/64 × C=1/3/8 × Yoshida3/5 的12组 CPU/HCU 对照通过；真实 MACE
  水/甲烷异构 Batch 3步、2+1续算和第4步 mask 通过，退出码0、结束后设备正常。
- CPU FP64 谐势：dt减半，扩展能量峰峰误差降至约1/4；不是长 MACE NVT 统计或性能承诺。
- vector 参数支持普通 Batch，但 refill 需 scalar 参数；完整 workflow checkpoint、
  DomainParallel、Warp/NVIDIA、HCU compile/autograd 和生产后端未验收。

## Stress/cell-force 基础 reference（2026-10-04）

- 契约与证据：[TORCH_CELL_STRESS_FORCE_CONTRACT.md](TORCH_CELL_STRESS_FORCE_CONTRACT.md)、
  [g2-torch-cell-stress-force.md](../reports/g2-torch-cell-stress-force.md)。
- 保留旧四位置参数、`npt_nph` import 和 legacy `torch.ops`；新增 keyword-only
  backend/selection，reference 由 ops executor 执行，无 Warp/barostat 导入。
- 参考指定 fork 修正行晶格公式 `F_H=-V H^-T sigma`；源码标记 `CELL-STRESS-ROW`，
  [UBF-009](UPSTREAM_BUGFIXES.md#ubf-009行晶格的-stress-to-cell-force-乘法顺序错误) 记录来源、数值影响与 Warp 未执行边界。
- 新增 ops 15 项与 framework 11 项 focused 回归；现有 stress 工具选择集
  `12 passed, 1 skipped`，跳过项需要 CUDA；小模型实际 stress loss 反向到参数通过。
- 完整 CPU gate 退出0，分进程为 `82 passed, 3 skipped`、`1 passed`、`168 passed`、
  `22 passed`、`24 passed`，合计 `297 passed, 3 skipped`；compileall/diff检查通过。
  初次180秒限时在已有 Inductor 测试处结束；单编译线程复核后重跑整套通过，未改测试/容差。
- HCU4/BW1000/gfx936 FP32/64 转换、一阶反向、非默认 stream 及真实周期 MACE parity 通过，
  CPU/HCU probe 均退出0；FP64 MACE stress 有限差分最大绝对误差约 `1.93e-10 eV/Å³`。
- 真实 MACE FP32 仅 forward/parity；FP64 作能量有限差分 oracle，未训练真实 MACE。
  变胞 FIRE/FIRE2、晶胞对齐/打包/更新、NPT/NPH 和 production backend 不在本轮范围。

## MPA-0 弛豫里程碑1（2026-10-04）

- [契约](TORCH_MACE_MPA0_RELAXATION_CONTRACT.md) 与
  [CPU 报告](../reports/g2-mace-mpa0-m1-cpu-baseline.md)：模型/fork SHA 固定，NVIDIA 锁定不变。
- 基础 CPU 门再次为297 passed、3 skipped；应力管线另为12 passed、1 skipped。
  本阶段未修改产品运行时源码，新增 CPU probe，不移植晶胞/优化器。
- 真实 Li2MgGe/LiSnCl3/NaLiTl2 Batch 的610条周期边无跨体系边；CPU FP64 数值导数
  最大误差：force `3.67e-10 eV/Å`、stress `1.56e-10 eV/Å³`、cell-force `6.55e-10 eV/Å`。
- 初次 FP32 energy dtype 检查失败：ASE calculator 修改默认 dtype，MPA-0 ZBL 路径
  受该默认值影响。probe 显式配置每个精度并恢复，不 cast 输出/改模型/放宽容差；重跑退出0。
- M1 时点 HCU_MPA0=`not_run`；目标模型 HCU 证据由下文 M5 补充，OFF23 不代替 MPA-0。

## MPA-0 弛豫里程碑2（2026-10-04）

- [晶胞工具契约](TORCH_CELL_TOOLS_CONTRACT.md) / [报告](../reports/g2-cell-tools-m2.md)：
  九个原生工具移植为 ops-owned `cell_tools` reference，公共 align 解析/传递 selection，
  None/warp 保留，Torch 路径无 Warp 导入；不准入变胞优化器 capability。
- CPU 基础门335 passed、3 skipped（Warp）；晶胞工具28项、公共接线10项。
  首轮整体240秒超时，最后检查补跑通过；600秒时限完整重跑退出0。
- HCU4 FP64/32 六组 `[2,0,3]`、单体系零原子、空 Batch 精确 pack round-trip；
  CPU/HCU 最大绝对差 FP64 `8.88e-16`、FP32 `2.98e-8`；没有模型/优化步。
- UBF-010 更正左手晶胞 transform 和非连续 bridge 写回；Warp kernel 未实际执行。
  extend_batch_idx 的 append 布局与 batched pack 的 interleaved 布局分别保留，不能混用。
- M2 本身未包含 FIRE/FIRE2 更新；M3 接线和证据另列如下，目标 MPA-0 HCU 由 M5 另验收。

## MPA-0 弛豫里程碑3（2026-10-04）

- 原开发分支 `codex/feat-variable-cell-fire`，起点 `d5b5f3c`，实现/测试/契约 `29a70f3`；其提交已随本次集成进入 `develop`。
  [契约](TORCH_VARIABLE_CELL_FIRE_CONTRACT.md) / [报告](../reports/g2-variable-cell-fire-m3.md)。
- 原生六项 FIRE2 与指定 fork 修复后的完整3×3 FIRE reference 已接公共优化器；
  optimizer/stress/align selection 冻结，`None/warp` 保留；没有 ASE filter 新参数化。
  `AlignCellHook` 保护周期/active 子集和 cached F/v/stress 坐标系一致性，作为预处理。
- CPU 独立 NumPy 单步39项，公共接线18项；最终基础门392 passed、3 skipped、退出0。
  三个 skip 是原有 Warp NHC 对照；compileall、定向 Ruff 和 diff-check 通过。
  两种公共优化器解析矩阵势160步能量降至初值的千分之一以下，纯仿射分数坐标守恒。
- HCU4 FP32/64 合成异构 `[2,3]`：16组单步 + 4组公共解析势25步通过；单步最大
  CPU/HCU 差 FP64 `4.44e-16`、FP32 `1.46e-11`。不是目标模型 HCU 证据。
- MPA-0 三个真实 CIF `[4,10,4]` CPU FP32/64、两优化器各3步通过；r/H 都更新，最终
  独立重算 E/F/stress 一致，FIRE2 人工旋转后对齐协变通过。未判定完整弛豫收敛。
- UBF-011/012/013 摘取 fork 行晶格/完整晶胞 FIRE/联合控制修复；UBF-014 包含 fork
  掩码保护和本地 cached 字段旋转修正；源码与 [缺陷记录](UPSTREAM_BUGFIXES.md) 区分归因。
- M3 时点未验收 Warp/NVIDIA、MPA-0 HCU、变胞退出/补位/联合收敛、skin 缓存、checkpoint、
  NPT/NPH、DomainParallel、compile/优化轨迹梯度和生产 HIP/Triton。

## MPA-0 弛豫里程碑4（2026-10-05）

- 分支 `codex/feat-mpa0-relaxation-workflow`，起点 `317b988`，实现/测试/契约 `7a8af50`；
  [应用契约](TORCH_MACE_MPA0_RELAXATION_CONTRACT.md) / [M4 报告](../reports/g2-mpa0-relaxation-m4.md)。
- 复用原生 FusedStage/SizeAwareSampler/HostMemory/ConvergenceHook，完成退出/私有
  状态保护及补位 E/F/stress priming；UBF-015/016、LP-081/082 区分 fork 摘取与本地补充。
  M4 时的薄 CIF/JSONL 入口为本地 LP-083，现已移至独立下游工程 `alchemi-app`；没有新 filter/优化算法。
- 新增33项生命周期/应用回归，加原退出输出回归纳入 CPU gate；最终 CPU 门
  426 passed、3 skipped，compileall/Ruff/diff-check 通过。不是完整 HCU gate。
- HCU4 合成 FP64/FP32、两种优化器各4个异构体系，收敛/补位/ID/邻居/退出状态
  与 CPU 对照四组通过；FIRE2 181、FIRE 84 个 fused 步。M4 时点 HCU_MPA0=`not_run`，M5 另补充。
- 真实 CPU FP64 `[4,4,10]`、最大活跃2体系：FIRE 三个样本31/35/80步收敛；FIRE2
  NaLiTl2 53步收敛，另两项到达120秒墙钟预算。阈值0.01 eV/Å 与0.1 GPa。
  FP32 两种优化器三样本各2步 budget 路径通过，不能算 FP32 完整收敛。
- 实际导出 CIF 独立重读/model核验14份；NaLiTl2 两种优化器另用最终入口完整重跑。
  退出原因区分收敛/步数/墙钟/非法输入/计算异常/未开始/最终核验失败。
- scalar refill；普通向量 Batch 保留，向量 arrivals 来源映射明确拒绝。预读输入及
  HostMemory 占用随输入数增长，步间墙钟检查后还需完成最终导出核验。
  M4 时点 Warp/NVIDIA、目标 HCU、compile/训练梯度、checkpoint、混合变胞算法阶段、DD、
  NPT/NPH、生产后端与全目录均未验收。

## MPA-0 弛豫里程碑5（2026-10-05）

- 分支 `codex/test-mpa0-hcu-validation`，起点 `fa74ca1`；
  [M5 报告](../reports/g2-mpa0-relaxation-m5.md)。只新增验证探针与契约/证据，
  framework/ops/应用源码不变，无新增科学算法或上游 bug 修复。
- HCU4 BW1000 64G/gfx936，Torch2.9.0/HIP6.3.26093/DTK26.04；作业前后资源检查空闲。
  FP64 再 FP32，两种原生变胞优化器；目标锁定 MPA-0，eager Torch reference。
- CPU gate 426 passed/3既有 Warp skips；另33项工作流/应用 focused 回归通过。
  HCU 为真实单点两组、弛豫十二组定向验收，不是完整 HCU gate。
- 两精度单体系都收敛（FIRE31、FIRE2 53步）；3结构异构容量2的四组都收敛，
  实际补位后 source/ID唯一；FIRE31/35/80步，FIRE2 53/77/153步。
- 从用户目录选8个不同真实 CIF，4–16原子、共59原子、最大活跃4/max_atoms64；
  两精度 FIRE各8项收敛，FIRE2各7项收敛，Li(CN)2各500步预算退出（应力0.282/0.265 GPa）。
  力<=0.01 eV/Å 与绝对应力<=0.1 GPa 双阈值不变，budget不计为成功。
- 共48份输出（8种输入多配置重复），46份收敛/2份 max_steps；逐份 CPU/HCU 独立模型
  重算完整E/F/stress/邻居和导出哈希。最大差 FP64 E2.85e-14/F2.96e-14/stress3.22e-15，
  FP32 E1.53e-5/F1.93e-5/stress1.36e-6，单位依次eV/eV·Å⁻¹/eV·Å⁻³，均通过原定容差。
- 8结构应用及导出约26–42秒（含验证Hook，启动/独立核验另计），峰值 allocated
  FP64约626 MiB/FP32约347 MiB；不是整卡占用或生产吞吐。
- Li(CN)2 FIRE FP64/FP32均双阈值收敛，但88/364步、能量-72.656861/-74.576416 eV、
  体积133.015271/124.967972 Å³；没有判定原因或极小值性质，精度互换和重复稳定性未验收。
  当前范围仍为单优化器 scalar refill/单卡 eager；全目录、NVIDIA、训练/compile、
  checkpoint/DD/NPT-NPH/ASE filters/生产后端不在验收内。用户确认后续测量后，M5以 `4656cce` 收口提交。

## MPA-0 单个较大 Batch 耗时测量（2026-10-05）

- [报告](../reports/g2-mpa0-single-batch-timing.md)、`probes/mace_mpa0_batch_timing.py`；
  只新增隔离探针及证据，framework/ops/产品应用源码不变。只测FP64原生完整矩阵FIRE。
- 当前目录21755候选按文件名排序，Random(24)抽1600后只执行前64项；2287原子、
  每项3–132原子，HCU4容量2304/64结构、refill20、ordinary e3nn/Torch reference。
  初批全部装入，没有新到达结构；未测持续补位吞吐、16000原子容量或全部1600。
- 停止判据为原子物理力与原生`stress_to_cell_force`全部行范数最大值<=.05 eV/Å，
  无独立应力阈值；不同于M5双阈值及H100参考形变/cell_factor=N的UnitCellFilter判据。
  62项收敛、2项500步预算退出，0溢出/计算错误；实际更新总5191，中位数37.5步。
- 应用101.046秒（准备2.557+优化98.272+导出.218）；启动19.809、oracle3.564，
  探针总124.420秒。峰值allocated/reserved为23.486/35.096 GiB；进程allocator限80%。
  作业后无KFD PID；没有OOM不代表16000原子可运行。
- CPU/HCU各3个NumPy原生力案例通过；CPU核验64项ID/哈希/FP64原始数组/预算分类/
  CIF几何，最终原生力重算最大差3.47e-17 eV/Å；没有64份独立模型重算或完整HCU gate。
- 同等规模25批条件估算：共享启动约41–42分钟，每批进程重启约50分钟；不是全部收敛
  耗时或统计区间。H100输入、CuEq、容量和判据不同，不能作同工作量平台性能比较。
- 初轮dt_min=0被本地参数校验拒绝且未完成一个完整优化步，不计时外推；
  最终明确采用现有默认.002；本测量未修改算子。后续 LOCAL-004 已独立修正零下限，
  见下节，历史测量仍采用.002参数。

## MPA-0 8000原子门槛与分项计时诊断（2026-10-05）

- 用户追加授权8000门槛、仍只跑一批、估计模型与优化器等分项。
  [报告](../reports/g2-mpa0-batch8000-timing.md)；复用相同抽样前缀231结构/7937原子，
  每项3–176原子；下一项80原子会超容量，未纳入。FP64/e3nn/原生FIRE参数与上一测量相同。
- 在探针给邻居、compute、masked pre/post update、停止观测器加互不嵌套的同步墙钟，
  其余工作流记录残差。framework/ops/产品应用不变，没有新filter/算法/上游bug修复。
- HCU4同批初始求值80%、95%、默认100%allocator上限三次均在e3nn张量积前向OOM；
  模型求值及优化更新完成数0，没有弛豫输出。最后一次allocated峰值59.115 GiB、
  reserved62.604 GiB，额外3.95 GiB分配失败；没有改变allocator策略或测出理论最小内存。
- 默认上限：输入/模型准备4.986秒、初始冷邻居2.007秒、容量检查.000322秒、
  失败前部分模型.694秒、循环其余1.107秒，探针总31.939秒。
  未执行优化器，不能给完整E/F/stress求值/稳态分项占比、完整Batch耗时或8000容量1600项估算。
- 运行时确认e3nn TensorProduct/FullyConnectedTensorProduct与MACE SymmetricContraction，
  无CuEq/HCU等变加速库；OOM位置不是完整循环性能剖析或未来算子库收益证明。
- CPU oracle及计时返回/异常/Hook语义检查通过，Ruff/格式/编译/diff检查；
  HCU三次各CPU/HCU NumPy oracle通过，事后全部231输入哈希、容量、失败分类/分项总和核验。
  无完整产品gate、成功弛豫或完整模型parity新增证据；原2304容量结果保留。作业后设备空闲。

## MPA-0 4000原子门槛单Batch完整分项（2026-10-05）

- [报告](../reports/g2-mpa0-batch4000-timing.md)。复用既有探针，无源码/算法/上游bug修复。
  同一Random(24)前缀114结构/3959原子，每项3–176原子，下一项44原子超4000未纳入。
  FP64、e3nn、原生完整9项FIRE、raw力.05阈值、500实际更新预算、refill20不变。
- 邻居显式`torch_reference`，6Å/每次重建/COO/512上限检查；测量时优化HIP邻居仅覆盖
  固定晶胞/MATRIX等窄条件，变胞/COO未纳入当时支持范围。30.261秒属于Torch reference邻居。
- HCU4一次作业完成、allocator80%，111收敛/3 max_steps，无OOM/力溢出/计算错误。
  实际更新8679、中位数36步、P95为302.25；518次模型求值/500 fused步，不把reprime算作更新。
- 应用160.145秒=准备3.287+循环156.489+导出.369；启动19.261/oracle3.267另计，
  探针182.673秒。峰值allocated41.131/reserved49.938 GiB；只证明当前样本可容纳。
- 循环同步分项：模型118.819秒（75.93%）、邻居及容量检查30.261（19.34%）、
  原生优化器pre/post共4.404（2.81%）、停止观测器.730（.47%）、工作流残差2.276（1.45%）。
  模型含力/应力autograd，优化器含mask/私有状态保护；同步观测不是纯kernel或满载稳态时间。
- CPU/HCU各3个NumPy oracle通过；114项ID/抽样/哈希/lossless FP64状态/预算/CIF几何/
  计时分项核验通过，最终原生力差2.78e-17 eV/Å。无独立114项模型重算/完整CPU-HCU gate。
  作业后设备空闲、无KFD PID；没有新到达结构、持续refill吞吐或全目录成功率证据。
- 同等分布/参数按14.035批条件外推约37.8分钟，按15个完整批次安排约40.4分钟；
  不是1600实测、全部收敛耗时或统计区间，H100输入/CuEq/容量/判据仍不同。

## FIRE 零下限兼容修正（2026-10-05）

- [报告](../reports/g2-fire-zero-dt-min.md)。LOCAL-004 / LP-084：本地完整矩阵 FIRE
  reference 的 `dt_min<=0` 改为 `<0`，恢复锁定 NVIDIA/fork 已有的零下限语义。
  实际 `dt>0`、非法参数报错前状态保持、默认参数/API/公式/后端选择不变。
- 修复前四项零下限回归均复现拒绝；修复后 ops46/公共19项通过，完整 CPU gate
  **434 passed, 3 skipped**，compileall、Ruff、diff检查通过。
- HCU4仅测FP64：四类功率单步对NumPy误差0；三种非法边界无状态改写；
  公共解析势25步 CPU/HCU对照位置误差1.11e-16 Å；原抽样前三CIF/124原子
  MPA-0 + FIRE零下限十步正常，r/H/E/F/stress为有限设备内FP64且体积为正。
  不宣称十步达到收敛；作业后无KFD PID，不代表完整HCU gate。
- LOCAL-004已收口提交 `0e354f4`；历史4000原子完整Batch仍使用.002，未复测零下限完整Batch或性能。

## 应用工程分离（2026-10-05）

- MPA-0 CIF 弛豫入口和专属测试已迁至同级 `/home/wangleping/codes/alchemi-app`；本仓库继续维护
  toolkit 移植、上游兼容性及集成验证证据，不再承载材料/模型专用应用入口。
- 分离后验证：toolkit CPU gate **425 passed, 3 skipped, 1 deselected**；独立应用入口回归
  **9 passed**，CLI `--help` 和项目 TOML 检查通过。两处 `git diff --check` 通过；此项只移动
  应用代码并调整追踪文档，未改 framework/ops 算法，因此没有重跑 HCU。
- toolkit 分离提交 `730a848` 已并入本地 `develop`；应用工程提交 `9ae235c` 已建于其本地 `develop`，两者已出现在三个目标 remote 的 `develop` refs。现有 HCU 证据仍仅适用于各报告列明的窄验证范围。

## 运行时排障记录

此前“加载/设备同步在当前卡上异常变慢”已定位为：超时终止 JIT 编译后遗留 PyTorch C++
extension `FileBaton` stale lock，后续进程在加载阶段等待。确认没有属于本任务的
`ninja`/`hipcc`/应用进程后清理该缓存 lock，query extension 恢复正常加载，完整 46×32 HCU
gate 重跑通过。长时间验证和发布路径优先使用 AOT 或受控 JIT cache。

## Hygon distributed preflight（2026-09-21）

- 测试分支：`codex/test-distributed-md-preflight`；报告：
  [`reports/dcu-distributed-preflight-2026-09-21.md`](../reports/dcu-distributed-preflight-2026-09-21.md)。
- 初始只读环境：8×BW200/UBB BW1000、驱动 `6.3.30-V1.4.1a`、8 卡均约 2 MiB 显存占用、
  任意卡对拓扑为 HSW。
- 初次 2-rank `torchrun` 曾在 launcher 调用 `torch.cuda.is_available()` 时段错误，退出码 `139`；
  恢复后直接 Python 枚举、单进程 torchrun 和 2 子进程空 launcher 均通过，说明该段错误目前
  不是稳定复现的 launcher 必现故障。
- 历史重启前的真实 2-rank process group 曾初始化但在 1-element all-reduce 超时，随后设备运行时
  丢失；该段记录仅用于故障追溯，不参与当前 post-reboot preflight 决策。
- `hy-smi` 恢复后，纯 HIP runtime 在 HCU0/HCU1 通过；两个普通进程直接调用 RCCL 时，
  `ncclCommInitRank` 通过，但单元素 `ncclAllReduce` 在 P2P/IPC/XGMI 路径超时，退出码均为
  `124`。因此该故障已在 PyTorch 之外复现，但 P2P/RCCL/HIP runtime/驱动固件之间的根因
  尚未归因。SHM 对照在设备已异常后执行，标记为受污染，不作为 transport 结论。未扩大到
  4/8 卡，也不以环境变量 workaround 宣称通过。
- 换用 HCU6/HCU7 时，纯 HIP 仍通过；direct RCCL communicator 初始化仍通过，但 all-reduce
  同样停在 `before_ncclAllReduce`，随后 `hy-smi` 报 `Open mkfd failed`/设备不可用。两个
  不同双卡组合均复现，当前停止继续设备测试，等待管理员恢复并采集宿主机内核日志。
- 只读核验安装拓扑：`/var/log/rock-kernel/install-2.log` 记录 `selected 7+8->10`，
  `/etc/hfm/hys.cfg` 为 `OAM_7HSW_8HCU`。本节点应使用显式 `-t 10`；没有证据表明此前未
  显式指定 `-t` 导致当前故障，`-t 11` 不适用于该节点。
- 对照节点 dcu1 使用同一最小 direct RCCL probe 通过：`rank0_rc=0`、`rank1_rc=0`，两 rank
  均得到 `after_ncclAllReduce value=3`。这支持将当前 dcu2 故障优先定位为节点级驱动/固件、
  HFM/Hlink 配置或设备健康差异；dcu1 的 probe 输出未包含测试后 `hy-smi`，不扩展为完整节点
  健康结论。
- dcu2 重启后重新验证：直接 `torch.cuda.is_available()` 为 `True`、device count 为 2，
  两 rank `torchrun` 退出码为 `0`，all-reduce 和 P2P send/recv 均通过，测试后 `hy-smi`
  仍正常。用户同时确认“旧驱动、重启后、安装新驱动之前”测试已通过，因此不把新驱动升级
  视为恢复原因；当前状态记录为 post-reboot narrow PASS，重启前故障仍未归因。
- P1 单卡 device tensor smoke 已补齐：新增 `probes/distributed/probe_device.py`，对物理 HCU0--7
  逐卡隔离运行。8 张卡均通过 FP32/FP64/INT64 的 allocation、H2D/D2H、elementwise、同步和
  释放复用，并通过 FP32/FP64 matmul；allocator 快照在释放/复用后保持当前运行时基线。原先矩阵中
  的 `P1=NOT RUN` 已修正为当前 `P1=PASS`，不再把完整矩阵缺失误写成没有任何单卡证据。
- 新增 P4 集合通信探针 `probes/distributed/probe_collectives.py`。在 post-reboot dcu2 上以
  2 ranks、`fp32/fp64/int64`、4 KiB/1 MiB/64 MiB payload 运行，broadcast、all-reduce、
  all-gather、all-gather-into-tensor、等分及变长/零 split all-to-all、reduce-scatter 均
  精确校验通过，前后 `hy-smi` 正常。该结果只构成 P4/2-rank narrow PASS，不覆盖重启前故障，
  也不代表 4/8 rank、P2P 重复、async、DeviceMesh、长稳或 DomainParallel 已验证。
- 新增 P5 P2P 探针 `probes/distributed/probe_p2p.py`。post-reboot dcu2 首次 2-rank 有界运行
  在 180 秒内未完成，退出码 `124`；前后 `hy-smi` 均正常，未观察到设备丢失。日志显示进入
  unbatched P2P communicator 后无 rank summary；随后发现首次 harness 让两个 rank 以相同顺序
  提交对向 unbatched 操作，不作为 RCCL/硬件故障证据。按 RCCL operation contract 修正为两个
  方向有序验证后，2-rank、4 KiB、`fp32/fp64/int64` 的 pair `isend/irecv` 均通过，前后
  `hy-smi` 正常；随后 `pair_batch_isend_irecv`、有序 parity phases 的 ring、变长/非对称 ring、
  zero-length ring 以及 1000 次小消息重复均通过。随后以 4 ranks（HCU 0--3）和 8 ranks
  （HCU 0--7）运行完整 `--case all --payloads 4K,1M --repeats 1000` 矩阵，覆盖
  `fp32/fp64/int64`，所有 rank 均完成 `case_start -> case_pass`，`torchrun rc=0`，前后
  `hy-smi` 均正常。因此 P5/2/4/8-rank correctness matrix 已 PASS。该结论仍不包含性能、
  长时间稳定性、64 MiB P5 payload、多节点或 DomainParallel；探针保留 flushed case markers、
  显式 `device_id` 和 `--case` 单项选择。
- 新增 P6 async collective 探针 `probes/distributed/probe_async.py`。在 2/4/8 ranks、4 KiB/1 MiB、
  `fp32/fp64/int64` 上验证 async all-reduce、独立设备计算后的 wait、3 个 outstanding operation
  的正序/逆序 wait，以及 fp32 100 次重复；所有 rank 最终 JSON 为 `PASS`，`torchrun rc=0`，
  前后 `hy-smi` 正常。P6 correctness 已 PASS，但不代表 P11 通信计算 overlap 或 P12 长稳；
  P7 DeviceMesh 结果见下一条。
- 新增 P7 DeviceMesh/subgroup 探针 `probes/distributed/probe_devicemesh.py`。2/4/8 ranks 的 1D
  mesh、rank lookup、mesh group、1D submesh all-reduce 均通过；4-rank 2×2 和 8-rank 2×4
  2D mesh 的 row/col subgroup mapping 与精确 all-reduce 也通过，覆盖 `fp32/fp64/int64`、
  4 KiB/1 MiB，所有 rank 最终 JSON 为 `PASS`，前后 `hy-smi` 正常。该结果不等于 DomainParallel
  已支持；P11 overlap 另有独立 smoke 结果，生产通信计算 overlap 和 DomainParallel 仍未验证。
- 新增 P8 functional collectives 探针 `probes/distributed/probe_functional_collectives.py`。
  直接导入当前 DTK PyTorch 的 `torch.distributed._functional_collectives`，调用 functional
  `all_to_all_single` 与 `wait_tensor`，不以普通 `dist.all_to_all_single` 替代；在 2/4/8 ranks、
  4 KiB/1 MiB、`fp32/fp64/int64` 上验证显式 input/output split lists、zero/imbalanced splits、
  forward/reverse exchange，所有 rank 最终 JSON 为 `PASS`，`torchrun rc=0`，前后 `hy-smi` 正常。
  该结果仅覆盖 P8 functional collective correctness，不代表 P8-B PhysicsNeMo 兼容、通信计算
  overlap、性能、长稳或 DomainParallel 已验证。
- 新增 P9 autograd-sensitive communication smoke `probes/distributed/probe_autograd_collective.py`。
  在 2/4/8 ranks、4 KiB/1 MiB、`fp32/fp64` 上验证 `local requires_grad → all_gather → 可微 loss
  → backward`。direct `dist.all_gather` 的前向数据精确正确，但记录为
  `native_autograd_collective_supported=false`；显式
  `_functional_collectives.all_gather_tensor_autograd` 经 `wait_tensor` 后前向和反向梯度均
  精确通过，所有 rank 最终 JSON 为 `PASS`，前后 `hy-smi` 正常。该结果是 autograd runtime
  信息项，不等于 nvalchemi HALO/DomainParallel 已支持，也不覆盖 P11 通信计算 overlap、P12 长稳和 P13 显存行为。
- 新增 P10 通信带宽/latency baseline 探针 `probes/distributed/probe_performance.py`。在 2/4/8
  ranks 上测量 `all_reduce`、`all_gather`、`all_to_all_single` 和成对 P2P ping-pong，覆盖
  1 KiB/4 KiB/64 KiB/1 MiB/16 MiB/64 MiB；每项 10 次 warmup、20 次 steady-state sample，
  使用 `cuda_event`，每个 rank 保存 median/p90/min/max、样本、relative stdev 和按明确逻辑字节
  公式计算的 descriptive effective GiB/s。所有 correctness case 和性能运行均完成，
  `torchrun rc=0`，前后 `hy-smi` 正常；没有设置 transport/algorithm/channel 覆盖。该结果没有
  绝对 GB/s 门槛，不代表 P11 overlap、P12 长稳或 P13 显存行为。
- 新增 P11 通信计算 overlap 探针 `probes/distributed/probe_overlap.py`。在 2/4/8 ranks、1 MiB/16 MiB
  payload、1024×1024 独立矩阵计算上，分别测量同步 `all_reduce → matmul` 与
  `all_reduce(async_op=True) + 独立 CUDA stream matmul → wait`；两条路径各 5 次 warmup、10 次
  synchronized wall-clock samples，所有 rank 的通信和矩阵计算 correctness、async wait 与最终
  completion 均通过，`torchrun rc=0`，前后 `hy-smi` 正常。1 MiB 在各规模普遍是 overlap 更慢；
  16 MiB 仅部分 rank 有约 5--9% 的局部下降，按全 rank 完成时间未观察到稳定收益，且 8-rank
  1 MiB 有一个约 4 ms 离群样本。因此 P11 的 async/stream correctness smoke PASS，但不能据此
  宣称生产 DomainParallel 已实现通信计算 overlap，也没有进行性能优化或设定阈值。
- 新增 P12 长稳探针 `probes/distributed/probe_stability.py`。在 2/4/8 ranks 上均完成指南最低
  operation-count：small 10,000、medium 1,000、large 100；每个阶段按固定顺序混合
  `all_reduce`、`all_gather`、`all_to_all`、P2P ring 和 async all-reduce，使用 4 KiB/1 MiB/16 MiB
  FP32 device buffers、2 个 warmup cycles、每次操作完成同步和 exact correctness check。所有
  rank 最终 JSON 为 `PASS`，阶段进度完整，`torchrun rc=0`，前后 `hy-smi` 正常；每个阶段内部
  allocator allocated/reserved trace 保持稳定，阶段切换因首次分配更大 buffer 出现预期 reserved
  增长，释放后 allocated 回到约 512 B。该结果通过 P12 的 operation-count stability gate，
  但不是小时级 soak，也不替代 P13 的专门显存行为测试。
- 新增 P13 显存行为探针 `probes/distributed/probe_memory.py`。在 2/4/8 ranks、每 rank 64 MiB
  FP32 payload 上分别运行 `all_gather` 和 `all_to_all_single`，每项 2 个 warmup、5 次正式重复，
  记录 allocator 的 allocated/max/reserved/max，并在释放后及 `empty_cache()` 后复测。所有 rank
  最终 JSON 为 `PASS`，`torchrun rc=0`，前后 `hy-smi` 与 `hy-smi --showpids` 返回码均为 0；
  `all_gather` 的瞬时 allocated peak 增量随 2/4/8 ranks 为约 128/256/512 MiB，
  `all_to_all_single` 未观察到额外 peak，三轮释放后 allocated 均回到约 512 B，
  `empty_cache()` 后 reserved 均回到约 2 MiB。`--showpids` 采样已保存，但部分 PID 的 HCU 映射为空、
  VRAM 百分比为 `inf`，因此设备侧采样只作辅助证据，不替代 allocator 数字或可归因的进程峰值。
- P14 单节点拓扑已重新采集并完成记录。`hy-smi --showtoponuma --json` 与 `--showbus` 直接配对得到
  8 张 HCU 的 PCI/HCU/NUMA/OAM 映射；8×8 `showtopotype` 全为 `HSW`、`showtopoaccess` 全为
  `TRUE`、`showtopohops` 任意非自身卡对均为 1。PCIe 树确认每张 HCU 与对应 ConnectX-7
  `mlx5` NIC 为同一 PCIe 分支，8 个 IB port 均为 `ACTIVE/LINK_UP`。本次不启用自动绑核；collector
  JSON 的 PCI slot 排序与 hy-smi card 顺序不一致，最终映射以直接 `showbus`+`showtoponuma` 证据为准。

## 当前下一步

每个里程碑先说明目的与方案，用户确认后启动。当前计划见
[`PARALLEL_DEVELOPMENT_PLAN.md`](PARALLEL_DEVELOPMENT_PLAN.md)：

固定晶胞 NHC reference 已在本地 `develop`；stress/cell-force 基础已完成上述窄范围，
实现与测试已提交 `a76107e`。基础 slice 未移植晶胞及 FIRE/FIRE2。
MPA-0 + 原生 FIRE/FIRE2 弛豫里程碑1已完成基础收口、来源/契约和 CPU 单点验证。
通用晶胞工具 M2、原生变胞优化器 M3、应用生命周期 M4 已完成上述窄验证。
M5已收口 `4656cce`；2304容量FP64 Batch测量已收口 `7410445`；
8000门槛诊断已收口 `eff753d`；4000门槛完整Batch及分项已收口 `1dba058`。
零下限兼容修正 `0e354f4` 及收口提交已于 `620a485` 合并到 `develop`；remote 同步状态见上方当前基线。
后续建议队列：

1. 已完成 MPA-0 弛豫移植收口审查，见[报告](../reports/g2-relaxation-port-closure-review.md)。应用入口与专属测试现已移至独立下游工程 `alchemi-app`；toolkit 继续维护上游移植和兼容性跟踪。
2. 若要对齐 H100 运行控制或停止度量，在 `alchemi-app` 单独规划；H100 `UnitCellFilter(cell_factor=N)` 广义力判据与4000探针 native raw-force 判据不同。全目录长作业仍需单独确认。
3. ASE 3.29-compatible BFGS M1–M4及M5-4显式产品接入已完成本文明确目标数值门，含UnitCellFilter 3N+9、异构Batch/refill和真实小样本CSP；四组本地提交已收口，未合入develop或推送。新ID hip.bfgs-ase-jacobi-streams-v1取消经验n/B白名单，多宽双stream、单宽batched；hip仍rocSOLVER、默认Torch不变、不进auto。最终审阅CPU669/5skip，HCU环境Ops18/Framework24，M5-4原rocSOLVER39，1113/B32子集及真实561 B8和逐步ASE通过；性能输入相关。真实工作流、最终误差、profiler及范围见[M5-4报告](../reports/g2-ase-bfgs-m5-4.md)与[最终审阅](../reports/g2-ase-bfgs-delivery-review.md)。其他构建/NVIDIA/DomainParallel/全目录生产仍是独立后续范围。

用户已确认 HIP 邻居分阶段任务；M1 当前晶胞快照 MATRIX 已收口窄 gate，见
[报告](../reports/g2-hip-current-cell-neighbor-m1.md)。M2 MATRIX→COO 与 MACE 单点已收口实现 `3e394d7`；
M3 HIP邻居变胞 FIRE/FIRE2 已完成短步/预算补位窄 gate，实现 `6b1680a`，见[M3报告](../reports/g2-hip-variable-cell-workflow-m3.md)。
M4 已完成实测并拒绝未达门槛的workspace候选，产品保持stateless；M5批量metadata已收口；M6查询协作完成窄验收并收口，M7签名收口；M8已收口 `04f80bd`。M9已将过严的逐结构移动条件改为批次至少有一个晶胞移动，两组受影响FP32 FIRE2复跑通过，并由签名提交 `fa1259b` 收口。
M2 planner、NPT/NPH、DomainParallel、生产
Triton/HIP dynamics kernel 继续暂停。

## 验证和设备约定

- CPU gate：`scripts/check_cpu_reference.sh`。
- 本机所有后续单卡 HCU 验证统一使用 `HIP_VISIBLE_DEVICES=4`；历史报告中真实使用的 HCU 0
  结果保留，不改写。
- 受限沙箱中 `/dev/kfd`、`/dev/dri` 不可见只表示设备节点隔离，不能作为 HCU 能力结论。
- 实现、测试、数值结果、性能口径和未验证假设必须分别记录；没有 HCU 运行证据不能写成
  DCU verified。

## 文档权威关系

- 架构与稳定交接：[`PROJECT_HANDOFF.md`](PROJECT_HANDOFF.md)。
- 模块与应用总览：[`IMPLEMENTATION_STATUS_MATRIX.md`](IMPLEMENTATION_STATUS_MATRIX.md)。
- 当前任务队列：[`PARALLEL_DEVELOPMENT_PLAN.md`](PARALLEL_DEVELOPMENT_PLAN.md)。
- 后端能力：[`BACKEND_CAPABILITY_MATRIX.md`](BACKEND_CAPABILITY_MATRIX.md)。
- 逐特性契约：[`FEATURE_COMPATIBILITY.yaml`](FEATURE_COMPATIBILITY.yaml)。
- 上游来源/补丁：[`UPSTREAM.md`](UPSTREAM.md) 与 [`UPSTREAM_LOCK.yaml`](UPSTREAM_LOCK.yaml)。
- 历史日志：[`docs/history/STATUS.md`](history/STATUS.md)、
  [`docs/history/PROJECT_HANDOFF.md`](history/PROJECT_HANDOFF.md)、
  [`docs/history/PARALLEL_DEVELOPMENT_PLAN.md`](history/PARALLEL_DEVELOPMENT_PLAN.md)。
