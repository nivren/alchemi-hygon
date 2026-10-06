# 当前开发状态

更新时间：2026-10-05。本文只维护当前状态、已验证范围、下一步和限制；完整历史交接保留在
[`docs/history/STATUS.md`](history/STATUS.md)。

## 当前基线

- 当前交付基线：本地 `develop`，集成 merge commit `620a485`。它包含 NHC、stress/cell-force、MPA-0 M1–M5、Batch 测量、LOCAL-004 和本次应用分离收口；相应里程碑分支提交均已核实在 `develop` 历史中。
- toolkit 应用分离提交为 `730a848`，由 `codex/fix-fire-zero-dt-min` 合入；LOCAL-004 实现提交为 `0e354f4`。完整移植审查见[报告](../reports/g2-relaxation-port-closure-review.md)。
- 独立应用工程 `/home/wangleping/codes/alchemi-app` 已在本地 `develop` 创建首提交 `9ae235c`；此工程拥有 MPA-0 CIF 应用入口和专属测试。
- 首轮 remote refs 核验已通过：toolkit 三个 remote 的 `develop` 均包含集成状态提交 `9552588`，应用工程三个 remote 的 `develop` 均包含首提交 `9ae235c`；本条状态记录随其后的 toolkit commit 继续同步。
- 上游 framework/ops 锁定 SHA 见 [`UPSTREAM_LOCK.yaml`](UPSTREAM_LOCK.yaml)；产品源码位于
  `packages/framework` 和 `packages/ops`，`external/` 只作只读参考。

## 当前能力摘要

| 功能 | 当前状态 | 已验证范围 | 明确限制 |
|---|---|---|---|
| Torch reference neighbor/cell-list | 已实现窄 reference | periodic/no-PBC、full/half、MATRIX/COO、mixed/triclinic Batch、image shift、容量、selective rebuild、连续 distance/vector 一二阶路径 | 不等于完整上游生产 cell-list；完整 target/pair/compile/opcheck/DomainParallel 仍未完成 |
| 显式 HIP neighbor/cell-list | 已实现并验证窄 capability | periodic、Batch、full-list、MATRIX/COO（Torch打包）、FP32/FP64、`skin=0`、自动 doubling capacity；允许调用间 full-PBC 晶胞改变，每次完整重建 | 不进入 `auto`；不支持 no-PBC、half、skin/selective rebuild、target/pair、旧 plan 变胞复用、native geometry backward、compile/opcheck、DomainParallel；MPA-0 HIP邻居单点已验收；HIP FIRE/FIRE2 工作流待验收 |
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

- 起点 `fb05b61`，分支 `codex/feat-hip-current-cell-neighbors`；已收口实现 `fecbc39`，本地阶段分支；未合并或推送。
- CPU：ops `87 passed, 6 skipped`；framework `33 passed, 4 skipped, 1 xfailed`。
  skip 为 HCU 专属用例；既有 xfail 为 `WARP-DEFAULT-001`，不是新增豁免。
- HCU4 BW1000 64G/gfx936：ops `22 passed`，framework `4 passed`；46×32 既有
  固定晶胞 gate 也通过。FP32/64、收缩/膨胀/剪切/wrap、多镜像、非默认 stream、
  输入与旧输出保护、显式溢出及 Hook 自动扩容均有窄范围证据。
- 新建每次调用的 cell metadata/plan 本来就是 stateless 行为；本轮明确准入并验证，
  修正容量错误为 Hook 可识别的 `NeighborOverflowError`，修复 registry 意外允许 auto 选 HIP。
- 新增证据仅 full-PBC/full/MATRIX/skin=0；旧 plan 的固定 metadata ABI 保持。
  M1 当时未验收 COO/HIP MACE；后续单点准入见 M2，短步变胞工作流见 M3。性能优化待确认。
- [契约](HIP_CURRENT_CELL_NEIGHBOR_CONTRACT.md)、[报告](../reports/g2-hip-current-cell-neighbor-m1.md)。

## HIP COO 与 MPA-0 单点 M2（2026-10-05）

- 起点 `8120282`，分支 `codex/feat-hip-coo-mace-snapshot`；已收口实现 `3e394d7`，本地阶段分支，未合并或推送。
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
  实现已收口 `6b1680a`，未合并或推送；M4 已获确认。无新增产品 kernel 或优化器修改。
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
3. 固定晶胞 ASE 3.29-compatible BFGS、NPT/NPH、DomainParallel 与生产性能仍是独立后续范围。

用户已确认 HIP 邻居分阶段任务；M1 当前晶胞快照 MATRIX 已收口窄 gate，见
[报告](../reports/g2-hip-current-cell-neighbor-m1.md)。M2 MATRIX→COO 与 MACE 单点已收口实现 `3e394d7`；
M3 HIP邻居变胞 FIRE/FIRE2 已完成短步/预算补位窄 gate，实现 `6b1680a`，见[M3报告](../reports/g2-hip-variable-cell-workflow-m3.md)。
M4 实测与workspace优化已获确认。M2 planner、NPT/NPH、DomainParallel、生产
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
