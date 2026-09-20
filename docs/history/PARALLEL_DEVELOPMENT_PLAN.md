> 这是历史任务与 Point 日志，不是当前任务入口。当前队列请阅读 [`../PARALLEL_DEVELOPMENT_PLAN.md`](../PARALLEL_DEVELOPMENT_PLAN.md)。

# 并行开发工作计划

日期：2026-09-20

适用基线：当前产品 `develop`；B1 executor binding 已完成并已同步两个产品远端。

当前代码基线：`9e295ea`（阶段二收口及本计划文档修订已合入 `develop`，以远端最新
`develop` 为准）。

本文是团队进入具体开发时的操作入口，目的是让开发者从同一个 `develop` 创建个人分支，
在明确的文件边界内并行工作。它不替代 `AGENTS.md`、
`docs/BACKEND_PLATFORM_PIPELINE_PLAN.md`、`docs/PROJECT_HANDOFF.md` 和
`docs/FEATURE_COMPATIBILITY.yaml`；这些文件发生冲突时，以项目级指令和权威技术计划为准。

## 1. 总体判断

B1 已将实现绑定从 framework/ops dispatcher 中的 implementation-ID 分支收敛为 catalog
entrypoint + generic executor。新增 reference implementation 时，主要工作可以留在自己的
executor、测试、probe 和 report 中，不应再修改通用 executor 或大面积修改 framework。

因此现在适合并行开展固定晶胞 Torch reference 功能。并行的边界不是“每个人随意认领一个
目录”，而是每条线拥有一个 operation contract 和一组明确文件；共享的 catalog 汇总、CPU gate、
`STATUS.md` 与兼容矩阵由集成负责人最后串行收口。

## 2. 立即启动的任务

### 阶段二收口后的当前范围（2026-09-20）

`TORCH-NEIGHBOR-PBC-CELL` 的阶段一、阶段二已经按窄能力范围完成并合入 `develop`，不再
作为立即启动任务重复开发。后续的 `target_indices`、`pair_fn`/pair outputs、skin/rebuild、
变胞、native geometry backward、compile/opcheck 和 DomainParallel 都单独登记为后续
capability，不回填或扩大本次已收口的 HIP slice。

当前可以立即启动、且不改变已合并邻居实现的任务如下：

| ID | 当前切片 | 优先级 | framework 影响 | 交付边界 |
|---|---|---:|---:|---|
| `TORCH-NVT-LANGEVIN` | 补齐固定晶胞 reference 的行为/状态生命周期 | P0 | 低—中 | 上游行为套件、通用 checkpoint/restart、inflight 边界；不含生产 Triton/HIP |
| `TORCH-NVT-NHC` | 固定晶胞 Nose-Hoover chain reference | P1 | 中 | chain state、质量、Yoshida 更新、Batch/inflight 回归；不含 NPT/NPH |
| `TORCH-FIRE2-VARIABLE-CELL` | stress→cell-force 与 coupled FIRE2 | P1 | 低—中 | 先做 FP64 stress/cell-force oracle，再做原子/晶胞联合 step；不含普通 FIREVariableCell/NPT/NPH |
| `TORCH-BFGS-ASE-COMPAT` | 固定晶胞 ASE 3.29 BFGS 数值兼容 | P1 | 中 | 先做单体系 CPU FP64 Hessian/方向/restart；变胞另列 `TORCH-BFGS-ASE-UNITCELL` |

这些任务仍必须先完成 CPU contract，再申请 HCU 批验证；在用户确认具体 owner 和切片前，
这里只更新权威计划，不表示已自动创建分支或开始实现。生产 Triton/HIP、M2 planner、NPT/NPH
和 DomainParallel 继续暂停。

### T1 启动批次状态（历史记录，2026-09-19）

以下内容保留原始 T1 启动批次和阶段二执行证据；当前任务优先级以本节前面的“当前范围”、
`docs/STATUS.md` 当前摘要和 `docs/BACKEND_CAPABILITY_MATRIX.md` 为准。

- `TORCH-NVT-LANGEVIN` 已在 `develop` 合入 `1fafd7b`，完成固定晶胞 BAOAB Torch
  reference、registry/catalog、generic executor binding、公共 `NVTLangevin` 显式
  `torch_reference` 接线、CPU contract 和 HCU smoke；随后补充的短谐势统计 oracle 已在
  CPU/HCU 通过，随后新增并已合入普通 Batch 的最小 integrator continuation state。
  这些交付仍是可审查的窄 reference slice。
- 该状态不等于完整 Langevin/NVT 支持。上游完整统计/行为套件、checkpoint/restart、
  `atom_ptr`、`_out`、inflight refill、分布式 ownership、torch.compile 和 Triton/HIP
  生产实现仍未完成；本轮 restart 只保存积分器续跑元数据，完整 checkpoint/restart
  仍应另建任务，不回填本分支。
- 本里程碑满足第 6 节的“完成一个可审查里程碑后暂停”条件；后续开发前重新确认优先级，
  不自动启动 M2、NPT/NPH、DomainParallel 或生产优化。
- `TORCH-NEIGHBOR-PBC-CELL` 阶段一已在
  `codex/feature-torch-neighbor-pbc-cell-core` 完成候选实现和 CPU/HCU 验证：显式
  `backend="torch_reference", method="cell_list"` 覆盖 periodic/no-PBC、full/half、
  mixed/triclinic Batch、分层 build/query、容量错误、selective rebuild 和连续输出梯度。
  真实 46/92/4x92 HCU 基线显示该 reference 仍比 dense 慢 `2.38--2.84x`，不能进入 `auto`。
  阶段二首个隔离 HIP JIT build/binning probe 已在 BW200 编译运行并与 Torch FP32/FP64
  对齐；其子模块 device-loop median 约为 Torch 的 `1/11.5--1/11.5`，但尚未包含 sort/query/
  fill 或产品接线。证据见 `reports/g2-torch-reference-pbc-cell-list-core.md`。

### 低耦合扩展功能（不自动启动）

如果有额外开发者，可以同时开展下面三项。它们不改变 B1 的架构方向，但新增的
兼容条目应由集成负责人统一落盘。

| ID | 功能 | 优先级 | framework 影响 | 说明 |
|---|---|---:|---:|---|
| `TORCH-THERMOSTAT-UTILS` | Maxwell-Boltzmann 速度初始化、去 COM、velocity rescale | P1 | 低 | 补齐实际 NVT 初始化链路；需先冻结 seed、温度和 Batch 契约 |
| `TORCH-LJ-SWITCHING` | LJ cutoff switching 的能量、力和连续性 | P1 | 很低 | 主要是 ops reference 和 focused tests；framework 只取消明确拒绝 |

`LJ virial/stress` 暂不与 switching 分成两个同时修改同一实现文件的分支；它应在 switching
完成后单独排队，或由同一 owner 负责连续交付。

`TORCH-BFGS-ASE-COMPAT` 的“兼容”有严格含义：项目 `.venv` 当前锁定的 ASE 3.29
`ase.optimize.BFGS` 是数值 oracle，不是仅借用 BFGS 名称。首版不得把标准逆 Hessian
近似、L-BFGS、line search、曲率跳过/阻尼或不同的晶胞参数化伪称为该兼容路径。

## 3. 每条开发线的工作定义

### 3.1 `TORCH-NVT-LANGEVIN`

建议分支：`<developer>/feature-torch-nvt-langevin`

主要文件边界：

- 新增 `packages/framework/nvalchemi/_dynamics_reference/langevin.py`；
- 局部修改 `packages/framework/nvalchemi/dynamics/_ops/langevin.py` 和
  `dynamics/integrators/nvt_langevin.py`；
- 新增独立 compatibility test、HCU probe 和 report；
- 在 dynamics catalog 中登记 implementation，但不修改 generic executor。

最小验收：

- BAOAB `B-A-O-A-B` 顺序正确；`friction=0` 与 velocity Verlet 对照；
- 异构 Batch 的 `dt/kT/friction` 按 system 正确映射；
- 同一设备、输入和 seed 可复现，换 seed 通常产生不同轨迹；
- 普通 Batch 的 state/restart、空输入、非法参数和 mutation 语义明确；完整 checkpoint、
  inflight state 和分布式 ownership 不在首版范围；
- CPU 通过后再运行 HCU smoke；没有 HCU 证据只能标为 CPU verified。

不在本任务内：NPT/NPH、域分解、跨设备逐位随机一致、Triton/HIP 优化。

### 3.2 `TORCH-NEIGHBOR-PBC-CELL`

阶段一分支 `codex/feature-torch-neighbor-pbc-cell-core` 已合入 `develop`；阶段二分支为
`codex/feature-torch-neighbor-pbc-cell-stage2`。

主要文件边界：

- `packages/ops/nvalchemiops/torch_reference_cell_list.py` 或同一 operation 的 reference 模块；
- `packages/ops/test/torch/` 下的 cell-list contract tests；
- `packages/ops/nvalchemiops/_backend_catalog/neighbors.py`；
- 专用 PBC probe 和 report；尽量不修改 framework runtime。

阶段一（正确性 reference）验收：

- 覆盖正交/三斜晶胞、混合 PBC、image shift、Batch、空输入和 periodic full/half；
- 与 dense CPU FP64 对照 pair 集合、排序、距离、向量和有效邻居数；
- 分开检查 `num_neighbors`、active slice 和 capacity，不能用分配宽度代替有效邻居数；
- capacity overflow 必须明确失败或按契约扩容；不能截断、静默 fallback 或漏算；
- 新路径不得把 host `.cpu().tolist()` 或 Python 逐原子/逐 pair 循环当作默认设备内实现；
- build/query、预分配 scratch 和 Batch selective rebuild 使用同一语义契约；连续的 vector/distance
  路径验证一阶和二阶梯度。

阶段二（优化后端）在阶段一 oracle 和固定基准之后独立推进：

1. 先冻结一个共享基础 ABI：cell geometry/metadata、CSR cell storage、scratch/output buffer、
   capacity/overflow、stream 和 capability 描述。该层服务 HIP/Triton 模块，但不做运行时
   静默 fallback，也不把所有模块串成一个不可并行维护的巨型 kernel。
   当前小点 1 已先落地其中的 cell-key build 子 ABI：FP32/FP64 geometry 输入以及 `int32`
   shift/coordinate/key output buffer 由 Torch reference 定义，`torch_reference_cell_list`
   已复用；CPU contract 通过，但当前项目环境未暴露 HCU，尚无本次 HCU/performance 结论。
   记录见 `reports/g2-cell-key-build-abi.md`。
   小点 2 已将该子 ABI 映射为 lazy native HIP mutation-only custom-op，并以 gfx936 编译/加载
   验证；主机 BW200/gfx936 已验证 FP32/FP64 mixed-PBC/triclinic output parity、空输入、
   strided read-only input 和非默认 stream。wheel 只携带源码，正式 AOT 构建尚未接入；不登记
   `hip` capability，也不作性能结论。
   小点 3 已补正式 gfx936 AOT wheel 入口并在临时解包 wheel 上复验 HCU parity；该 wheel 仅针对
   当前 CPython/DTK/HIP PyTorch/架构，仍不登记 `hip` capability。下一小点先补 CSR count ABI
   和 Torch oracle，再选择 native atomic count 实现。
   小点 4 已完成 CSR count/start ABI 与 Torch oracle，PBC cell-list 已复用且主机 HCU probe
   通过；下一小点只评估 native atomic count，starts 暂保留 Torch scan，不进入 sort/fill/query。
   小点 5 已增加显式 native HIP atomic-count custom-op：它复用 count output ABI，以 current-stream
   zero + atomic add 写 caller-owned counts，并在 BW200/gfx936 对普通、空、strided key 和 stream
   输出与 Torch oracle 一致；reference/dispatcher/`auto`/AOT 均未改变，也没有性能结论。下一小点
   先对均匀与 dense-cell workload 做预热后的 isolated count timing，再决定是否推进接线或 scan。
   小点 6 已在 BW200/gfx936 以 32768 keys/4096 cells、20 warm-up、100 repeats/sample 和 20 HIP-event
   samples 测得 count-only Torch/HIP median：uniform `0.26104/0.20087 ms`，single-cell
   `0.26266/0.20278 ms`，局部 factor 为 `1.300x/1.295x`。这是保留 count candidate 的证据，不是
   complete neighbor speedup 或 `auto` 门槛通过；下一小点为独立 HIP exclusive scan contract/kernel。
   根据 ADR 0009，后续不再把 key、count、scan、fill 当作彼此无关的替换点：它们属于一个有 scan
   全局边界的 HIP build pipeline。允许融合 geometry/PBC/key 与 count；scan 保持独立并预分配
   rocPRIM workspace；fill 可在受控生命周期内复用 count buffer 为 cursor。内部 cell atom order
   只有在公共 neighbor order/集合契约仍成立时才可放宽。该决定不接入 runtime。
   小点 8 已将 scan 落为独立 native HIP rocPRIM boundary：显式 `(M,) int32` counts/starts、global
   offset 与 caller-owned `uint8` workspace，主机 BW200/gfx936 已通过 JIT output parity、empty、
   strided counts 与 non-default stream；CPU focused `29 passed`。它没有与 key/count 接线、没有
   scan 性能结论，也不改变 AOT/dispatcher/`auto`/`hip` capability。下一小点转向 batch-aware fused
   geometry/PBC/count 的 ABI/oracle 验证，而非提前 fill/query。
   小点 9 已完成该 batch-aware fusion candidate：按 `batch_idx` 读取 per-system metadata，将连续
   `cell_offsets` 加入 global key，在一个 native HIP kernel 内写 geometry/PBC/key/count。异构两体系
   FP32/FP64、empty B=1、strided inputs 和 stream 的 BW200 parity 已通过，CPU focused `33 passed`。
   下一小点只能以同输入、多样本 HIP events 比较 fusion 与独立 native key+count；未有性能结论，
   不改变 runtime 或进入 fill/query。
   小点 10 已完成该 comparison：BW200/gfx936 上相同 FP32 2x16384 Batch/4096-cell contract 的
   uniform/single-cell composed/fused median 分别为 `1.17562/0.79358`、`1.17380/0.79173 ms`
   （`1.481x/1.483x`，局部降低 `32.50%/32.55%`），样本方差小且 outputs 一致。它只保留 fusion
   candidate，不是 complete neighbor speedup 或 runtime 准入；下一小点为 CSR fill ABI/oracle。
   小点 11 已完成 CSR fill ABI/oracle：explicit cursor、stable atom order、global atom offset、
   capacity tail 和 public counts/starts 不变合同已由 Torch reference 与 27 项 focused tests 固化，
   HCU PBC reference probe 通过。下一小点才做 native HIP fill correctness candidate；公共顺序未
   解决前不接入 runtime。
   小点 12 已完成 native HIP atomic fill correctness candidate：current-stream cursor reset 与
   atomic slot 写入在 BW200/gfx936 上实际 JIT 编译/加载并完成 FP32/FP64、异构 triclinic Batch/
   mixed-PBC、strided inputs、empty B=1 和 stream 证据；每 cell atom set、cursor final state、CSR
   metadata 与 capacity tail 均一致。atomic 同 cell order 未指定，故仍不接 runtime，也没有性能
   结论。下一小点先在隔离 probe 串接 fused key/count、scan 和 fill，验证完整 Batch CSR build。
   小点 13 已完成该 isolated composition：三个已验证 HIP extension 顺序构成 Batch CSR build，
   workspace/output 仍 caller-owned。BW200/gfx936 的 FP32/FP64、异构 triclinic Batch/mixed-PBC、offset、
   strided input、empty 与 stream 中 keys/counts/starts 均逐元素一致，atom-list 按 cell set 一致；
   CPU focused `39 passed`。这不是 query/public order、完整 neighbor 或性能证据。下一小点将组合 CSR
   输出交给现有 Torch query，决定 atomic 内部顺序是否必须 canonicalize。
   小点 14 已完成该 public-order 判断：Torch direct query 的 pair sort 使 atomic 同 cell insertion
   order 不影响 public matrix/count/shift/distance/vector 的 row order。CPU 的 PBC/no-PBC、full/half
   permutation regression 和 HCU 的 mixed Batch FP32/FP64 full/half probe 均通过；不需为当前 direct
   query 新增 fill canonicalization。下一小点测完整 HIP build + Torch query，对 profile 决定 native
   irregular query 或 Triton materialization 的候选优先级；仍不接 runtime。
   小点 15 已完成完整 build/query 的窄 device-time benchmark 与受限 profile：native build 在
   uniform/clustered 上均比 Torch 慢约 `18.4%`，query 基本持平，full native 慢 `2.9%--3.4%`；
   query 占 Torch full 约 `82.5%--86.2%`。profile 含 JIT/module load，不能作稳态占比，但支持把
   后续重点转向 query/materialization。下一小点先冻结 query reference/candidate contract，再做
   最小 native HIP 或 Triton query correctness slice。
   小点 16 已按高 Batch 场景补测 `46/92 atoms × 32/64 systems` 的 uniform/clustered 矩阵：
   native build median 普遍慢 `31%--45%`，query 差异约 `-0.62%--+0.06%`，full 通常只慢
   `0.34%--0.63%`；query/materialization 的主导性在中小体系高 Batch 下更明显。另有 `2×16384`
   capacity 512 的显式 overflow，capacity 1024 重跑成功。详见
   `reports/g2-batch-scale-performance-matrix.md`。下一小点冻结 query/materialization contract，
   再做最小 query candidate correctness probe；不提前接 runtime。
   小点 17 已完成最小 native HIP query enumeration candidate：source-atom 邻近 cell 遍历、PBC
   image/shift、cutoff、self 排除和 full/half 过滤在 HCU 0 上通过 FP32/FP64、mixed PBC、非默认
   stream 及 capacity overflow probe；Torch reference 继续承担 stable materialization。详见
   `reports/g2-batch-cell-query-candidate-hip-boundary.md`。下一小点在高 Batch 矩阵中测 native
   enumeration + Torch materialization，之后再选择 HIP/Triton canonicalization/materialization。
   小点 18 已完成该高 Batch query 分解：`46/92 atoms × 32/64 systems` 的 8 个 workload 中，
   native enumeration + Torch matrix/count/shift materialization 相对完整 Torch query 的 median
   factor 为 `45.44x--101.16x`；但 optional distance/vector、autograd 和 API wall-clock 尚未覆盖。
   下一小点补齐这些 public continuous outputs 和梯度合同，再评估 HIP/Triton materialization。
   小点 19 已完成可复用 Torch geometry materialization：stable topology order、public
   matrix/count/shift、distance/vector 和 positions/cells 一二阶梯度合同均通过 CPU/HCU；相同
   高 Batch geometry 矩阵的 native enumeration + Torch materialization 相对完整 Torch geometry
   query 为 `35.20x--79.49x`。该 factor 是预热 HIP-event device timeline，不是 API wall-clock，
   也不代表完整 neighbor/端到端支持。下一小点分别评估 Triton 规则分块与 HIP 不规则子路径，
   仍不接 runtime。
   小点 20 已完成 materialization 阶段 breakdown：在固定 native candidate 上分别测 topology
   canonicalization/scatter、geometry distance/vector 和完整 helper；8 个高 Batch workload 均
   通过 parity，topology 占完整 helper `72%--75%`，geometry 占 `17%--19%`。因此下一小点
   优先做 HIP 不规则 topology compaction/scatter/sort correctness candidate，再以相同矩阵测量；
   Triton geometry 暂不作为第一候选，仍不接 runtime。
   小点 21 已完成 native HIP topology materialization correctness candidate：五字段稳定
   rocPRIM radix sort、counts scan 和 public scatter 对齐 Torch `_sort_pairs`，HCU 0/BW200/gfx936
   的 FP32/FP64、Batch mixed-PBC、full/half、empty、stream parity 通过。实现仍包含临时分配和
   host sync，只作为 correctness slice；下一小点在 `46/92 × 32/64` 高 Batch 矩阵上测其与 Torch
   topology stage 的稳态 device time，不接 runtime。
   小点 22 已完成该矩阵 benchmark：8 个 workload 均 parity 通过；HIP/Torch topology median
   factor 在 `46×32` 为 `0.830x--0.860x`，在 `46×64` 为 `1.175x--1.274x`，在 `92×32` 为
   `1.240x--1.365x`，在 `92×64` 为 `1.636x--1.696x`。当前 candidate 不具备稳定收益，下一小点
   先分析五次 radix sort、字段 gather、临时 workspace 和 stream/host sync 边界，再选择 single
   key/workspace reuse/专用 scatter 或 Triton 实验；不接 runtime。
   小点 23 已完成受限 `hipprof` kernel profile：在 `46×32` 与 `92×64` 的 uniform/clustered
   workload 上，四个 workload 的 public topology parity 均通过。按 candidate kernel 调用次数
   归并，sort 相关 duration 约为 `0.755/1.638 ms` 每次调用，明显高于 gather、prepare、order
   initialization 和 public scatter；高输入下 rocPRIM 从 merge-path 类 kernel 切换到 onesweep
   类 kernel。该结果仅定位热点，不是性能准入；exclusive scan、allocator 和 host sync 尚未从
   混合 trace 中单独量化。下一小点优先验证 single composite key 的顺序/溢出合同，再独立测
   workspace reuse 或 host-sync removal，不接 runtime。详见
   `reports/g2-batch-query-topology-materialization-profile.md`。
   小点 24 已完成 isolated single composite-key correctness/overflow candidate：以
   `row -> column -> shift_x -> shift_y -> shift_z` 编码到 signed `int64`，在 `5 atoms × 5`
   mixed-sign PBC fixture 上用一次 radix sort 对照当前五次稳定排序 candidate；默认 stream、
   non-default stream 和 public topology parity 均通过，`shift_bits` 范围溢出显式拒绝。CPU
   focused gate 为 `5 passed`，HCU 0/BW200/gfx936 probe 通过。`int64` 只是 packed key 的
   标量存储容器，不代表 int64 matrix-core throughput；未测性能、未改 runtime。下一小点在
   相同 `46/92 × 32/64` 高 Batch 矩阵上比较 single-key 与五次 int32 stable-sort 的 device
   time，再独立评估 workspace reuse/host-sync removal。详见
   `reports/g2-batch-query-topology-materialization-composite-key.md`。
   小点 25 已完成相同 benchmark contract 下的 composite-key device-time 对照：HCU 0/BW200/
   gfx936、FP32、`46/92 × 32/64`、uniform/clustered、3 warm-up/5 samples，8 个 workload
   的计时前后 parity 均通过。single-key 相对五次 int32 stable-sort 的 median factor 为
   `0.310x--0.406x`，相对 Torch topology 为 `0.289x--0.526x`；`46×32` 波动较大，最大
   relative population std 为约 `12.3%`。这是 topology-stage HIP-event evidence，不是 API
   wall-clock 或完整 neighbor 加速，不接 runtime。下一小点先把 composite candidate 扩展到
   现有 FP32/FP64、full/half、empty、mixed-PBC、capacity、stream correctness boundary，
   再拆分 workspace reuse/host-sync removal。详见
   `reports/g2-batch-query-topology-materialization-composite-key-benchmark.md`。
   小点 26 已复用现有完整 HCU query/materialization boundary，将 composite candidate 与
   Torch reference、五次 HIP stable-sort candidate 并列验证。HCU 0/BW200/gfx936 上 FP32/FP64、
   mixed-PBC、full/half、non-default stream、empty、native query capacity overflow 和
   composite public-capacity rejection 均通过；未测性能、未接 runtime。下一小点可独立拆出
   workspace lifecycle 或 host pair-count synchronization 实验，继续保留该 parity gate。详见
   `reports/g2-batch-query-topology-materialization-composite-key-boundary.md`。
   小点 27 已完成 composite-key caller-owned workspace reuse：key/order/row-start/sort/scan
   scratch 可跨调用复用，CPU `6 passed`，HCU 完整 boundary 通过。相同 `46/92 × 32/64`、
   uniform/clustered 矩阵的 8 个 workload parity 均通过；reused/direct 的 device-time factor
   为 `0.900x--1.026x`，显式同步 API wall-clock factor 为 `0.902x--1.129x`，没有稳定的
   全矩阵收益，因此不接 runtime。下一小点独立处理 `candidate_counts.sum().item()` host
   synchronization，不与 composite 算法或 workspace 生命周期混合。详见
   `reports/g2-batch-query-topology-materialization-workspace.md`。
   小点 28 已将五次 int32 与 single-int64 composite topology scatter 的 pair-count 判断留在
   device：按 candidate 总容量发射并在 kernel 内跳过 padding，移除了
   `candidate_counts.sum().item()` host sync。CPU `6 passed`，HCU 0/BW200/gfx936 correctness
   boundary 与 `46/92 × 32/64`、uniform/clustered 的 8 workload parity 均通过；point28 direct/
   point27 direct 的描述性 device median factor 为 `0.817x--0.914x`，API wall-clock 为
   `0.858x--1.042x`，不是交替配对样本，不能写成稳定加速。固定容量 launch 的稀疏场景代价
   尚未定量隔离，candidate 仍不接 runtime。下一步可比较 HIP build/query/topology + Torch
   geometry 的完整 hybrid pipeline 与 Torch reference。详见
   `reports/g2-batch-query-topology-materialization-host-sync.md`。
   小点 29 已把 native HIP query + composite-key topology + Torch geometry 串成 hybrid
   query/materialization path；CPU `11 passed`（含 geometry parity 与一/二阶梯度），HCU
   0/BW200/gfx936 的 FP32/FP64、mixed-PBC、full/half、empty、capacity、stream 和完整 public
   output parity 通过。固定 CSR metadata、不重复计 build 的 8 workload 中，hybrid/Torch
   device factor 为 `0.0120x--0.0257x`，API factor 为 `0.0124x--0.0256x`；这是窄 query/
   materialization scope，不是完整 neighbor/MD 加速结论。下一小点可独立实现并验证 forward-only
   HIP/Triton geometry candidate。详见
   `reports/g2-batch-query-hybrid-torch-geometry.md`。
   小点 30 已实现隔离 native HIP forward geometry candidate：在 public canonical topology 上
   直接计算 distance/vector，并加入 geometry-only 与 native query + composite topology 的
   hybrid 对照。CPU `7 passed`，HCU 0/BW200/gfx936 的 FP32/FP64、mixed-PBC、full/half、empty、
   capacity、zero-count、stream parity 通过；8 workload geometry-only factor 为 device
   `0.0784x--0.1220x`、API `0.1092x--0.1537x`，完整窄 hybrid factor 为 device
   `0.0069x--0.0146x`、API `0.0075x--0.0155x`。它仍是 forward-only isolated candidate，
   不接 dispatcher、AOT、`auto`、`hip` capability 或 autograd；没有覆盖 build-inclusive
   end-to-end。下一小点先决定 Torch geometry 梯度路径与 native autograd 的边界，再进行
   build-inclusive benchmark/收口。详见 `reports/g2-batch-query-native-hip-geometry.md`。
- 小点 31 已完成 build-inclusive isolated end-to-end benchmark：每次调用重新执行 cell-list
  build、query candidate、topology materialization 和 geometry，并比较 Torch reference、
  native discrete path + Torch geometry、native discrete path + native forward HIP geometry。
  HCU 0/BW200/gfx936 的 `46/92 × 32/64`、uniform/clustered 8 workload parity 全部通过；
  native build-only 相对 Torch 为 `1.311x--1.399x`，保留 Torch geometry 的 full factor 为
  device/API `0.0246x--0.0510x`/`0.0248x--0.0501x`，native geometry full factor 为
  `0.0193x--0.0384x`/`0.0195x--0.0381x`。这是预热后、固定 metadata/buffer/workspace、
  不含 JIT/首次分配的 isolated evidence，三条路径仍不接 runtime。下一步按 build 的
  key/count、scan、atomic fill 和 occupancy 拆分 profiling，再决定融合或 launch 优化；
  Torch geometry 梯度路径保持不变。详见
  `reports/g2-batch-cell-build-query-materialization-benchmark.md`。
- 小点 32 已先完成 native HIP build 的分阶段 profiling，而未猜测性修改 kernel。相同
  `46/92 × 32/64`、uniform/clustered 8 workload 全部 CSR parity 通过；74.5 万 global cells
  的 key/count/scan/fill 约 `0.80/0.70/0.81 ms`，149.1 万 cells 为约
  `0.80/0.98/1.04 ms`，native/Torch full build 为 `1.311x--1.423x`。受限 hipprof 的 API
  时间受 module load 污染，但 GPU OPS 和代码审查均显示 wrapper/custom-op 重复 ABI validation
  会发射 `any/reduce` device work，因此不应只调 atomic fill block size。下一小点先做
  private prevalidated direct-extension composition，公开 API 保留完整 validation；同一矩阵
  必须先通过 direct/public/Torch parity 和稳定收益再保留。详见
  `reports/g2-batch-cell-build-breakdown-profile.md`。
- 小点 33 已完成 private prevalidated direct-extension composition：公开
  `build_batch_cell_csr_hip_into` 仍执行完整 validation，trusted helper 仅对已验证的精确
  buffer 集合直接调用已有 key/count、rocPRIM scan 和 atomic fill extension。gfx936 上
  `46/92 × 32/64`、uniform/clustered 8 workload 全部 CSR parity 通过；trusted/public
  HIP-event factor 为 `0.0285x--0.0318x`，trusted/Torch 为 `0.0398x--0.0423x`，API wall
  也同方向。它仍是私有性能候选，不接 runtime。下一步先设计可安全复用的 prevalidated
  plan/lifecycle 并重测包含 query/materialization/geometry 的完整流程；若无法建立清晰
  的 validation ownership，就拒绝绕过路径并转向 clear/scan/fill 算法优化。详见
  `reports/g2-batch-cell-build-trusted-fastpath.md`。
- 小点 34 已增加 `_TrustedBatchCellBuildPlan`，将一次 checked public build 与固定
  tensor/workspace/storage 生命周期内的 trusted 重复调用绑定。gfx936 的 8 workload 全部
  plan/public/Torch CSR parity 通过；trusted/public HIP-event factor `0.0281x--0.0313x`，
  trusted/Torch `0.0394x--0.0413x`。plan 仍是 isolated build candidate，不接 runtime。下一步
  把它接入已有完整邻居 benchmark，比较 public build、trusted plan build 和 Torch reference
  的完整 build→query→topology→geometry；通过 full-pipeline parity 和性能门禁后再评估
  dispatcher/runtime 接线。详见 `reports/g2-batch-cell-build-trusted-plan.md`。
- 小点 35 已将 trusted plan 接入完整隔离 `build→native query→composite topology→Torch
  geometry` pipeline，并为 public/trusted 路径使用独立 geometry output buffer。gfx936 的 8
  workload 全部通过完整 parity；public native/Torch full factor `0.0249x--0.0501x`，trusted
  plan/Torch full factor `0.0121x--0.0257x`，trusted/public native `0.472x--0.544x`。这只是
  isolated forward evidence，不接 runtime。下一步设计 runtime-facing capability/错误合同，
  先完成 fixed-cell Torch reference 接线和回归，再决定显式 HIP 选择。详见
  `reports/g2-batch-cell-build-query-materialization-trusted-plan.md`。
- Point 36 的第一步已完成 fixed-cell Torch reference runtime boundary regression：periodic
  `NeighborListHook(backend="torch_reference", method="cell_list")` 到 Torch reference
  LJ energy/force 的纵向链通过；显式 `backend="hip"`、`method="cell_list"` 明确失败且不
  fallback。framework focused suite `33 passed, 1 warning`，ops registry/cell-list suite
  `31 passed`。这只增加 CPU wiring/error evidence，不登记 native HIP、不改变 `auto`/Warp。
  下一步将 capability/error contract 收敛为固定的选择记录和 native HIP wrapper 最小 ABI。
  详见 `reports/g2-fixed-cell-torch-reference-runtime-boundary.md`。
- Point 36.2 已完成共同 selection contract：`compute_neighbors` 和 `NeighborListHook` 共用
  `resolve_neighbor_list_backend` 构造 PBC/no-PBC、full/half、MATRIX/COO 与 method strategy，
  再生成中央 `BackendSelection`。framework focused suite `34 passed, 1 warning`；不登记
  native HIP，不改变 Warp/`auto`。下一步是 Point 36.3，冻结 native HIP wrapper 的最小 ABI
  和 admission gate。详见 `reports/g2-fixed-cell-neighbor-selection-contract.md`。
- Point 36.3 已完成 native HIP Batch neighbor runtime 的最小 ABI/admission gate：固定
  `checked build init -> trusted reusable CSR -> native unordered query -> composite-key
  canonical full topology -> Torch geometry`，并把 workspace/lifetime、full-list、fixed-cell、
  Batch、dtype 和 Torch gradient 边界显式化。CPU contract suite `15 passed`；不注册
  `hip`、不改变 Warp/`auto`，明确拒绝 unsupported request。下一步是建立不改变默认路径
  的 native wrapper 骨架/显式调用边界。详见 `reports/g2-native-hip-neighbor-runtime-abi.md`。
- Point 37 已完成显式 ops-side native HIP wrapper，并修复 composite workspace size-query
  的高 Batch int32/int64 storage 边界问题。wrapper 在 gfx936 HCU 0 上通过 `46×32`、`92×64`
  public output parity，以及 `46×32` Torch geometry 一阶/二阶位置梯度；CPU ops suite
  `109 passed, 1 warning`。它仍不注册 `hip`、不改变默认路径。下一步是 Point 38：接入
  framework 显式 HIP selection/executor 边界。详见
  `reports/g2-native-hip-neighbor-runtime-wrapper.md`。
- Point 38 已完成 framework 显式 HIP selection/executor 接线：新增
  `hip.neighbor.cell_list-v1` 和公开 ABI adapter，覆盖 periodic/fixed-cell/Batch/full/MATRIX
  的 native build/query/topology；ops `38 passed`、framework `11 passed, 1 warning`，并在
  `HIP_VISIBLE_DEVICES=4` 的 gfx936 上通过 `46×32`、`92×64` 对照。仍不覆盖 no-PBC、half、
  COO、skin/rebuild、变胞、target/pair、MACE/FIRE2 或性能准入。下一步是 Point39：第五张
  DCU 上的 build-inclusive API wall-clock 和适用 Hook 输出边界验证。详见
  `reports/g2-framework-hip-neighbor-executor.md`。
- Point 39 已完成 framework public HIP neighbor API 的 build-inclusive wall-clock 与 Hook
  边界验证，并修复了 Torch reference 大 Batch cell allocation/build 上限不一致、HIP
  executor workspace allocation 前 `candidate_counts` 未初始化两个可重复性问题。CPU
  cell-list focused suite `14 passed`；在 `HIP_VISIBLE_DEVICES=4` 的 gfx936/BW200 FP32
  fixed periodic/full/MATRIX、capacity 256 下，46×32 和 92×64 的 HIP/Torch warm public
  API median factor 分别为 `0.09398x`、`0.07839x`，public output 对照通过。cold 仅记录
  首次 runtime/extension load 与 allocation，warm 才作窄范围描述性比较；`skin=0` Hook
  通过，`skin>0` 明确拒绝。Point39 不改变 `auto`，也不覆盖 no-PBC、half、COO、
  target/pair、变胞、skin/rebuild、native geometry backward、compile/opcheck、
  DomainParallel 或 MACE/FIRE2。详见
  `reports/g2-framework-hip-neighbor-wallclock.md`。下一步为 Point40：补 explicit HIP
  executor 的 capacity/empty/dtype/format/unsupported-request 回归矩阵，并验证无 skin
  Hook 的重复调用契约。
- Point 40 已完成 explicit HIP framework neighbor contract matrix：CPU ops registry `7 passed`、
  framework neighbor `11 passed, 1 warning`；HCU `HIP_VISIBLE_DEVICES=4` 的 46×32 FP32/FP64
  parity、empty input、capacity overflow、no-PBC/half/COO rejection 和无 skin Hook 重复调用
  均通过。该点不改变 capability/`auto`，详见
  `reports/g2-framework-hip-neighbor-contract.md`。下一步为 Point41：纳入稳定 compatibility
  gate，并补 `max_neighbors=None` 自动容量增长/overflow 上界验证。
- Point 41 已完成阶段二最小收口：现有 contract probe 已作为稳定 compatibility gate，并补充
  `max_neighbors=None` 自动容量增长/有限上界验证。CPU ops `7 passed`、framework
  `11 passed, 1 warning`；gfx936/BW200、`HIP_VISIBLE_DEVICES=4` 的 46×32 HCU gate 通过，
  单原子周期 image fixture 的容量从 `1` 增长到 `8`，最大实际邻居数 `6`，保守上界 `343`，
  与 Torch reference parity 通过。阶段二在 periodic/fixed-cell/Batch/full-list/MATRIX/
  FP32/FP64/`skin=0` 窄范围内收口；不继续引入 lifecycle reuse 或扩大 capability。详见
  `reports/g2-framework-hip-neighbor-compatibility-gate.md`。当前 HIP 自动容量采用本地
  doubling policy，不承诺上游 `estimate_max_neighbors` 的最小 16/16 对齐 padded layout；
  这不改变 active neighbor 集合，但在扩大 public capability 前需单独统一容量形状。
阶段二当前状态（2026-09-20）：Point 1–41 已完成，阶段二已按上述窄能力范围收口；提交
`9f53f80` 已合并到 `develop`，并已推送到 `local-origin/develop` 与
`github-origin/develop`。收口证据包括 CPU ops/framework focused tests，以及
`HIP_VISIBLE_DEVICES=4`、gfx936/BW200 上的 46×32 HCU compatibility gate。当前交付仍只
覆盖 periodic/fixed-cell/Batch/full-list/MATRIX、FP32/FP64、`skin=0`；不把该结果扩大为完整
上游邻居后端支持。

此前出现的“加载/设备同步变慢”已定位为超时终止 JIT 编译后遗留 PyTorch C++ extension
`FileBaton` lock，后续进程因此在加载阶段等待；清理由本任务产生的 stale lock 后，query
extension 可正常快速加载，完整 46×32 HCU gate 重跑通过。该运行时排障记录已同步到
`docs/DEVELOPMENT_GUIDE.md`；后续长时间验证和发布路径优先使用 AOT 或受控 JIT cache。

2. 可并行的实现模块为 build/binning、query+count/fill、pair geometry/materialization 和
   Batch/rebuild orchestration。HIP 优先评估不规则 query、原子写入和显式容量控制；Triton
   只评估规则分块、compact/geometry 等实测合适的部分，不预设固定优先级。
3. 先做隔离 JIT/可行性 probe，再以 ops dispatcher/custom-op 和可独立构建 wheel 的正式
   AOT 边界接入；每个模块单独登记 capability、fake/meta/autograd/compile 状态。
4. 每个候选先通过相同 pair/shift/capacity/gradient oracle，再以固定设备、输入、warm-up 和
   重复样本比较。进入 `auto` 的门槛为代表性 neighbor workload 至少 2x，或目标
   MACE/FIRE2 端到端至少 20%，并且不能以改 dtype、顺序或能力宽度换取速度。

阶段一仍未覆盖、阶段二也不能默认宣称的功能：`target_indices`、`pair_fn`/pair outputs、
pair-centric/sorted query、完整 upstream compile/opcheck、DomainParallel ownership。它们应按
独立 capability 模块补齐，不能把 core 通过扩大为完整上游等价。

### 3.3 `TORCH-NVT-NHC`

建议分支：`<developer>/feature-torch-nvt-nhc`

主要文件边界：

- 新增 `packages/framework/nvalchemi/_dynamics_reference/nose_hoover.py`；
- 局部修改 NHC dispatcher 和 `nvt_nose_hoover.py`；
- 独立覆盖 chain state、Batch/inflight 和 extended-energy tests；
- 不接 NPT/NPH 的 barostat 代码。

最小验收：

- chain length、`Q`、`eta`、`eta_dot` 和 Yoshida order 的契约稳定；
- 异构 Batch 每个 system 独立使用温度、时间常数和自由度；
- state 初始化、补位、清理和 restart 不串状态；
- 固定晶胞短轨迹的温控统计、extended energy 和确定性回归可解释；
- 明确 NHC 与分布式全局 kinetic energy 的边界，不能提前宣称 DomainParallel 支持。

### 3.4 `TORCH-THERMOSTAT-UTILS`

建议分支：`<developer>/feature-torch-thermostat-utils`

先完成半天以内的契约审计，再实现：

- `initialize_velocities`：Maxwell-Boltzmann、seed、每 system 温度；
- `remove_com_motion`：质量加权的每 system 动量归零；
- `velocity_rescale`：按 system 目标温度缩放；
- 与现有 kinetics reference 统一单位，不重复实现另一套温度归约。

验收必须包括异构 Batch、不同质量、零/异常参数、COM 残差、温度统计和同设备 seed 行为。
它可以先作为独立 ops/reference operation 交付，公共 dynamics 初始化接线随后单独提交。

### 3.5 `TORCH-LJ-SWITCHING`

建议分支：`<developer>/feature-torch-lj-switching`

主要工作是把当前明确拒绝的非零 `switch_width` 变成可验证的 Torch reference：

- 从上游 switching 定义建立能量/力解析 oracle；
- 检查 switching 起点和 cutoff 处的连续性；
- 覆盖 full/half、PBC、Batch reduction、autograd force 和必要的混合二阶路径；
- 通过后才考虑 Triton/HIP，不凭 reference 结果写性能结论。

### 3.6 `TORCH-FIRE2-VARIABLE-CELL`

建议分支：`<developer>/feature-torch-fire2-variable-cell`

现有公共 `FIRE2VariableCell`、state、dispatcher ABI 和 `variable_cell` capability 解析已经存在；
本任务只补齐缺失的 Torch reference 与窄范围纵向验证。由一个 owner 按两个顺序里程碑交付：

1. `TORCH-CELL-STRESS-FORCE`：实现并验证
   `F_cell = -V * stress * inverse(cell).T`、`keep_aligned` 和 Batch 语义；
2. `TORCH-FIRE2-VARIABLE-CELL`：实现原子/晶胞 DOF 的共同归约、mix、clamp 和 affine update，
   再接现有公共 wrapper。

主要文件边界：

- `packages/framework/nvalchemi/_dynamics_reference/fire.py`；
- `packages/framework/nvalchemi/dynamics/_ops/fire.py`；
- `packages/framework/nvalchemi/dynamics/_ops/npt_nph.py` 中仅限 `stress_to_cell_force` 的
  reference binding，不迁移其他 NPT/NPH op；
- `packages/framework/nvalchemi/dynamics/optimizers/fire2.py` 的局部接线；
- 独立 compatibility test、variable-cell probe 和 report。

最小验收：

- 用 CPU FP64 有限应变或独立解析 oracle 验证 stress 符号、volume、cell inverse 和单位；
- 覆盖正交胞、三斜胞、`keep_aligned`、单体系、异构 Batch、空输入与奇异/非法 cell；
- 原子与晶胞状态的 `vf/vv/ff`、`maxstep`、`dt`、`alpha`、uphill/downhill 更新可独立对照；
- cell 改变后周期邻居重建不漏 pair；正确性阶段可使用现有 periodic dense reference，
  不依赖周期 cell-list 任务先完成；
- CPU contract 通过后，再用能够输出可信 stress 的解析模型或已验证模型运行 HCU smoke。

不在本任务内：普通 `FIREVariableCell`、NPT/NPH barostat、DomainParallel replicated cell state、
LJ virial/stress 的完整实现、生产 Triton/HIP 和性能结论。没有可信 stress/HCU 证据时只能登记
对应的 CPU reference slice。

### 3.7 `TORCH-BFGS-ASE-COMPAT`

建议分支：`<developer>/feature-torch-bfgs-ase-compat`

本任务为此前分子晶体弛豫工作提供可审计的 ASE 3.29 无 line-search BFGS 对齐路径。先冻结
固定晶胞的公开 API 和状态契约，再独立实现；变胞部分不与固定胞首版捆绑合入。

固定晶胞首版必须逐式对齐项目 `.venv` 的
`ase._4.optimize.bfgs.BFGSMethod` 与 `ase.optimize.BFGS`：

- 初始 Hessian 为 `alpha * I`（ASE 默认 `alpha=70`）；
- 以上一位置/梯度完成 Hessian BFGS 更新；
- 每步执行实对称 `eigh(H)`，以 `abs(eigenvalues)` 计算下降方向；
- 使用 ASE 的全局最大原子步长缩放，默认 `maxstep=0.2 Å`；
- 无 line search，且 ASE-compatible 模式不私自加入曲率拒绝、阻尼、reset 或 trust-region
  等改变轨迹的安全规则；
- checkpoint/restart 至少保存 Hessian、上一步位置、上一步力和 `maxstep` 的等价状态。

主要文件边界：

- 新增 `packages/framework/nvalchemi/_dynamics_reference/bfgs.py`；
- 局部新增 BFGS dispatcher/wrapper、catalog implementation 与独立 compatibility tests；
- 新增 ASE FP64 oracle test、HCU probe 和 report；
- 不修改 generic executor、公共 registry 或 M2 planner；不得把 BFGS 的内部线性代数需求提前
  扩张为公共通用 `eigh` operation。

固定晶胞最小验收：

- 用解析二次势和预设的 `position/gradient` 序列，逐步对照 ASE 3.29 的 Hessian、未裁剪方向、
  裁剪后位移与 restart；比较步向量而非符号不唯一的特征向量；
- CPU FP64 对照覆盖首次步、负特征值取绝对值、零位移 restart、`alpha`、`maxstep`、非法/非有限
  输入和收敛 Hook；明确首版只支持单个固定晶胞，异构 Batch、inflight 和 DomainParallel 必须
  显式拒绝，不能靠 padding 或共享 Hessian 静默伪支持；
- HCU 先以 `torch.linalg.eigh` 完成正确性 smoke 与基准；没有 HCU 证据只能登记 CPU verified。

变胞严格兼容是 `TORCH-BFGS-ASE-UNITCELL` 子里程碑，依赖固定胞验收和可信 stress：

- 采用 ASE `UnitCellFilter` 的原始胞 deformation-gradient 参数化，即 `3N+9` 自由度、
  `cell_factor`（默认原子数）、mask、hydrostatic/constant-volume/scalar-pressure 语义；
- 该路径与当前 `TORCH-FIRE2-VARIABLE-CELL`、`cell_filter` 的上三角 `3N+6` 原生表示不同。
  后者可继续发展，但不得标记为 ASE-compatible BFGS；
- 变胞 ASE 对齐不得在未确认此前工作实际使用的 filter（裸 `BFGS`、`UnitCellFilter`、
  `FrechetCellFilter` 或其他）前宣称轨迹一致。

`HIP-BFGS-EIGH` 不是首版承诺，而是有数据的条件任务：在目标晶体的 `D=3N` 与 `D=3N+9`、
FP64、预热后条件下分别记录 `eigh`、Hessian update、模型力计算与端到端每步时间。只有
`eigh` 确认为稳定瓶颈，才评估 HIP 小型实对称 eigensolver；先比较 Torch/hipSOLVER 路径，
不能仅因库名或设备名预设 Triton/HIP 更快。任何 HIP 路径仍必须以 ASE CPU FP64 步级 oracle
验证数学等价，且不能因性能回退到不同算法。

## 4. 共享文件与冲突控制

每条分支只直接拥有自己的 executor、测试、probe 和 report。以下文件是共享热点：

- `packages/ops/nvalchemiops/_backend_catalog/dynamics.py`；
- `packages/ops/nvalchemiops/_backend_catalog/__init__.py`；
- `docs/FEATURE_COMPATIBILITY.yaml`；
- `docs/STATUS.md`；
- `scripts/check_cpu_reference.sh`。

约定如下：

1. 不修改 `packages/ops/nvalchemiops/executor.py`、公共 registry 或 generic adapter；
2. catalog 只增加自己的 metadata，保持追加式修改，不重排既有登记顺序；
3. 每个开发者维护自己的 test/report；兼容矩阵和 STATUS 由集成负责人按实际证据串行更新；
4. CPU gate 脚本由集成负责人统一加入测试入口，避免多人同时改同一 shell 文件；
5. 任何跨任务 ABI、seed、state、planner 或 capability 变化，先写契约再改代码；
6. 不在个人分支合并其他人的分支，不改写共享历史。

## 5. 标准开工流程

从远端最新 `develop` 开始，不从旧的个人分支或候选指针开始：

```bash
git status --short --branch
git fetch --prune <product-remote> develop
git switch develop
git merge --ff-only <product-remote>/develop
git switch -c <developer>/<type>-<topic>
```

开工前在任务记录中写清：

- operation ID、上游 SHA/符号和本次不支持范围；
- 输入输出、shape/dtype/layout、单位、PBC/full-half、空输入和错误契约；
- mutation/alias、stream、确定性、梯度等级和 Batch/inflight 语义；
- 计划修改的文件、不会修改的共享文件和对应验收测试。

提交顺序保持可审查：

1. contract/test；
2. Torch reference；
3. dispatcher/catalog 接线；
4. HCU probe、report 和性能记录；
5. 兼容矩阵、STATUS 和交接文档。

每个提交使用 `git commit -s`；提交前至少运行：

```bash
git diff --check
scripts/check_cpu_reference.sh
```

有分配的 HCU 后，再运行对应的 HCU probe。未通过 HCU 时，报告必须写明 `HCU pending`，不能
把 CPU 或 import 成功写成 DCU verified。

## 6. 合入顺序与暂停点

建议由集成负责人按以下顺序审核：

1. `TORCH-NVT-LANGEVIN` 的剩余 reference 生命周期与上游行为 contract 可继续推进；
2. `TORCH-NVT-NHC` 可并行开始，但其 framework state 接线应独立审查；
3. `TORCH-FIRE2-VARIABLE-CELL` 可由独立 owner 并行开始，但必须先完成
   `TORCH-CELL-STRESS-FORCE`，不能借机迁移 NPT/NPH；
4. `TORCH-BFGS-ASE-COMPAT` 的固定胞 contract/reference 可与上述任务并行；其变胞
   `UnitCellFilter` 里程碑必须等固定胞验收与可信 stress，且不阻塞 FIRE2 的原生 `3N+6` 路径；
5. 低耦合的 thermostat utilities / LJ switching 在额外人力充足时并行；
6. 每个 operation 单独通过 CPU gate，再安排 HCU 批验证和 review；
7. 至少两个 operation 形成多个已验证实现、或确实出现可复现实验策略需求后，才重新评估 M2。

本批暂停条件：当前选定 operation 或其明确子里程碑完成一个可审查里程碑后，更新
`docs/STATUS.md`、对应 report 和兼容性条目，等待下一轮确认。期间不启动 M2、NPT/NPH、
DomainParallel 或生产 Triton/HIP kernel。

## 7. 这个协作方式为什么成立

它保留了 Git 最简单、最可恢复的协作模型：一个共享 `develop`、每人一个短分支、每项一个
operation、短提交、人工 review。文档只负责稳定边界，不承担 issue tracker、审批系统或实时
任务状态的职责。

如果团队规模很小，最简单的执行方式是只启动 Langevin 剩余 contract 和 NHC；若增加第三/四人，
材料结构弛豫优先时可选择 BFGS fixed-cell ASE compatibility 或 FIRE2 variable-cell；前者不等待
变胞 FIRE2，但不得提前宣称变胞 ASE 对齐。其余扩展项只有在
不争用共享文件且有人能完成完整 CPU/HCU 证据时才启动。
