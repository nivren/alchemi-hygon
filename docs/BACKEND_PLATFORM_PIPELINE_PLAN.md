# 后端能力、平台策略与 Pipeline 执行计划

状态：M1 已完成（2026-09-09）；M2 尚未开始。

本文是后续重构的权威计划。它解决三个不同问题：实现是否具备某项能力、某个平台/工作负载推荐什么实现、一次 pipeline 如何在不重复解析的情况下执行。三者不能再由一个后端字符串或若干局部 `if` 同时承担。

## 1. 目标与边界

目标是保留上游 API 和 `backend=None` 的默认语义，同时建立可审计、可版本化、可复现的后端选择体系：

```text
PlatformFingerprint -> ImplementationRegistry -> BackendProfile
                                           \-> PipelinePlanner -> Frozen BackendPlan
                                                                  \-> runtime dispatch
```

- `ImplementationRegistry` 只回答“这个实现能不能在给定契约下运行”。
- `BackendProfile` 只表达“在某个平台和工作负载下优先尝试什么”。
- `PipelinePlanner` 根据全 pipeline 的约束、代价和 profile 生成一次冻结计划。
- runtime 只执行 `BackendPlan`，不在每个算子入口重复解析，也不在生产路径临时 benchmark。

本计划不承诺立即实现 Triton/HIP，也不把当前 Torch reference 窄 slice 扩大成完整生产支持。当前已存在的 `torch_reference_cell_list` 是过渡实现；迁移完成后，cell-list 应成为 neighbor operation 的 strategy/method，而不是全局 backend 家族。

## 2. 核心语义

### 2.1 请求、实现和策略分离

删除固定的
`BackendName = Literal["auto", "torch_reference", "torch_reference_cell_list", "triton", "hip", "warp"]` 作为唯一类型。它把策略 token、实现家族、具体实现和未注册候选混在一起，扩展时会迫使每个调用点维护同一组字符串。

采用以下概念：

- `BackendRequest = str | None`：公共入口接受开放请求；未知请求由 registry 统一报错。
- `BackendFamily = str`：例如 `warp`、`torch_reference`、`triton`、`hip`，仅用于分组和报告。
- `ImplementationId = str`：registry 中的精确实现 ID，例如带版本/策略的内部 ID；它不直接暴露为每个 framework 函数的枚举。
- `method`/`strategy`：operation-specific 选择。neighbor 的 `dense`、`cell_list` 属于 neighbor strategy，不再伪装成全局 backend。

请求解释如下：

- `None`：未指定请求。若没有显式 policy，保持 `legacy-upstream-v1`，即上游 Warp 路径；不能因为设备名或 Warp 缺失而静默改变。
- `"auto"`：显式要求 planner/profile 选择；只有能力过滤和证据满足时才能选择优化实现。
- 家族名：请求某个实现家族，由 registry 选择该家族内满足契约的实现；没有合适实现时显式失败。
- 精确实现 ID：请求某个已登记实现，能力不满足时显式失败。

### 2.2 Registry 数据模型

每个登记项至少包含：

- `implementation_id`、`operation`、`family`、`strategy`；
- lazy executor/dispatcher 入口；
- 支持的 device、dtype、layout、PBC、full/half、gradient level、compile、distribution 和输入特征；
- mutation/alias、stream、确定性、容量和错误契约；
- evidence、证据版本和状态（实验、部分、已验证）；
- 可选的成本标签/benchmark profile，但 registry 不负责运行时测性能。

`BackendSelection` 至少记录 request、implementation ID、family、strategy、operation signature、profile ID、选择原因和证据版本。framework 先解析一次，再把 selection 传给底层 dispatcher；dispatcher 只校验 operation/契约，不二次解析同一请求。

### 2.3 Profile、Fingerprint 和 Frozen Plan

`PlatformFingerprint` 不能只看 `device.type` 或字符串中的 `cuda`。至少包含：

- PyTorch/DTK/driver/runtime 版本和 HIP/CUDA 构建信息；
- 实际设备型号、架构、显存和可见设备拓扑；
- Triton/HIP/通信库及相关 capability probe 版本；
- framework/ops 包版本和 registry schema 版本。

`BackendProfile` 是版本化配置，包含平台条件、工作负载条件、operation 级候选顺序、size buckets、特性限制、允许的显式 fallback 和 evidence gate。至少提供：

- `legacy-upstream-v1`：无显式 policy 时的上游兼容 profile；
- Hygon development/reference profile：在 HIP/Triton 完整证据之前只允许已验证的 Torch reference；
- NVIDIA/upstream profile：保留 Warp 语义，优化候选必须逐项登记。

`BackendPlan` 是 planner 的不可变输出，包含：

- profile/fingerprint、operation 选择、size buckets 和适用输入签名；
- 每个 operation 的 `BackendSelection`；
- fallback policy、warnings、约束冲突和 evidence；
- canonical serialization、plan fingerprint/hash。

同一 pipeline 内按 operation 选择是允许且通常必要的；不能假定一个全局 backend 覆盖 neighbors、模型、积分器、通信和存储。共享中间布局、转换成本、FusedStage/inflight 复用和 compile 边界必须进入 planner 约束。

## 3. 选择优先级、回退与 checkpoint

选择优先级固定为：

```text
component explicit override
    > pipeline policy override
    > loaded BackendProfile
    > legacy-upstream-v1 (only when request is None)
```

- capability 不满足时默认报错。
- 只有 profile 明确允许的 fallback 才能发生，并记录实际实现、原因、性能影响和 warning；不能把 fallback 当成 silent recovery。
- 离线 tuning 只生成候选 profile/artifact，由人工审阅后发布；生产运行不以即时 benchmark 改写计划。
- checkpoint 默认要求 plan hash 与当前环境匹配；跨平台或能力变化必须显式 `replan=True`，并记录旧/新 plan 与影响。
- 分布式运行由 rank 0 生成/加载计划并广播 canonical plan/hash；各 rank 不得独立选出不一致实现。

## 4. 分阶段实施

### M1：Registry 语义收敛

交付：

1. 将现有 `BackendCapability`/resolver 收敛为可注册的 `ImplementationRegistry`；移除固定 `Literal` 和分散的 known-backend 集合。
2. 把 `auto` 定义为 policy token，把 `None` 的 legacy 行为与实现 ID 分开记录。
3. 登记当前 Warp legacy、Torch dense reference 和 no-PBC Torch cell-list 过渡实现。
4. 将 neighbor 的 cell-list 改为 `backend="torch_reference", method="cell_list"`（或等价 operation strategy），逐步移除未发布的全局 `torch_reference_cell_list` 名称；`compute_neighbors` 与 `NeighborListHook` 仍保留兼容的公共 backend 参数。
5. 保留并扩大 pre-resolved `BackendSelection` 传递，确保 framework → dispatcher 单次解析。
6. 新增 ADR 0006；将当前 cell-list ADR 标记为过渡/被新语义取代，而不是删除历史。

门槛：registry 单测、dense/cell-list CPU/HCU contract、未知/未注册请求显式失败、legacy/default reverse guard、operation 文档审计通过。完成后停止并确认下一阶段。

#### M1 完成记录（2026-09-09）

- `ImplementationRegistry` 已替换固定 `BackendName`/known-backend 集合；选择记录包含 request、implementation ID、family、strategy 和 profile ID 占位。
- Warp legacy、Torch dense reference 与 no-PBC Torch cell-list 已登记。cell-list 现在通过 `backend="torch_reference", method="cell_list"` 选择；未发布的 `torch_reference_cell_list` 请求明确失败。
- framework `compute_neighbors` 与 `NeighborListHook` 将同一 selection 传入 Torch dispatcher；dispatcher 按 implementation ID 执行，不二次解析。
- CPU ops/framework 回归为 `25 passed`/`22 passed`；BW200/gfx936 HCU ops 为 `25 passed`，M1 新增 framework strategy smoke 为 `2 passed`。完整命令和限制见 `reports/g2-backend-registry-m1.md` 与 ADR 0006。

#### M1 follow-up：operation selection propagation（2026-09-09）

- FIRE/FIRE2 已拆为独立 operation 与 implementation ID：`fire` →
  `torch_reference.fire-v1`，`fire2` → `torch_reference.fire2-v1`。
- LJ、固定晶胞 VV/FIRE/FIRE2、periodic、kinetics、segmented reduction 和 observer
  路径已改为 framework 解析一次 `BackendSelection` 并传入 dispatcher；低层 dispatcher
  不再为同一调用二次解析 backend request。
- 变胞 FIRE/FIRE2 在 framework 初始化阶段按 `variable_cell` capability 解析；当前
  Torch reference 未登记该能力，因此显式请求会明确失败，默认仍保持 legacy Warp。
- 该 follow-up 的 CPU selection propagation 回归为 `3 passed`；配套 dynamics/reference
  slice 为 `103 passed, 1 deselected`（与 propagation 合并为 `106 passed, 1 deselected`），
  状态生命周期为 `22 passed`，导入边界与 reference wrapper 为 `15 passed`。报告见
  `reports/g2-backend-registry-m1-dispatch-propagation.md`。
- 以上不改变 M1 门槛，也不提前开始 M2；新增证据为 CPU 证据，不能替代 HCU、Triton/HIP
  或变胞 reference 验证。

### M2：PlatformFingerprint、Profile 与 Frozen Plan

交付：

1. 实现不依赖设备名猜测的 fingerprint，记录构建元数据、运行时能力和 probe/evidence 版本。
2. 实现 profile loader、版本校验、operation 规则、size buckets、capability filtering 和确定性 planner。
3. 生成 canonical plan/hash；提供 JSON/YAML 可审计输出和诊断报告。
4. 固化 `legacy-upstream-v1`；Hygon profile 在 HIP/Triton 证据不足前只允许 reference；不把运行时性能猜测写入默认 profile。
5. 为 PBC/full-half/dtype/gradient/compile/distribution/布局转换和输入规模定义可测试规则。

门槛：同一 fingerprint/profile 输入得到稳定 hash；能力缺失、规则冲突和非法 fallback 明确报错；直接调用可生成短生命周期 selection，但 pipeline 必须冻结 plan。完成后停止并确认下一阶段。

### M3：Framework/Pipeline 接线

交付：

1. 在 `BaseDynamics`/workflow 增加 policy/plan 注入，同时保留现有 component backend override。
2. 统一优先级：component > policy > profile > legacy；冲突不可静默覆盖。
3. 收集 neighbors、model/LJ、integrator、observer、storage、compile 和 distribution requirements，统一交给 planner。
4. 对 FusedStage 保证共享模型/邻居布局约束一致，stage update 可独立选择；对 inflight 使用有限 size buckets，禁止运行中产生不可追踪的实现漂移。
5. compile 在 planner 阶段预绑定；不支持 fullgraph 的实现登记 eager-only，不以 graph break 掩盖能力缺口。
6. 分布式由 rank 0 广播 plan/hash；checkpoint、trajectory 和 diagnostics 记录实际 plan。

门槛：固定晶胞 VV/FIRE/FIRE2、NeighborListHook、LJ、periodic、observer、FusedStage/inflight 的一次性选择与报告测试通过；缺能力/冲突/plan hash 不一致均可预测失败。完成后停止并确认下一阶段。

### M4：离线 tuning 与平台 profile 发布

交付：

1. 建立统一 tuning harness：先数值门禁，再按 cold/JIT、warmup、steady、端到端和显存/通信分项测量。
2. 复用 `perf_46/92/184/368`、no-PBC 与 periodic full-list，以及代表性的 `[46,92]` MACE/FIRE2 端到端工作负载。
3. 生成带输入签名、环境、证据和置信边界的候选 profile artifact；人工审阅后才进入发布 profile。
4. Hygon 平台按证据选择 Torch/Triton/HIP；不能预设 `hip > triton > torch`。NVIDIA profile 保留上游 Warp 优先语义。
5. 分离 workload profile（neighbor-heavy、MACE/FIRE2、训练、分布式），不要把某一规模的最佳结果推广为全局排序。

门槛：候选 profile 可重放、数值/契约回归通过、选择原因和性能影响可审计。只有达到对应 release gate 才允许 `auto` 选择优化实现。

## 5. 测试与文档门禁

- Registry：重复 ID、未知请求、家族/精确 ID、lazy import、strategy、能力过滤和显式 fallback。
- Planner：fingerprint/profile 解析、优先级、冲突、稳定 hash、size buckets、序列化、checkpoint strict/replan。
- Framework：compute_neighbors、Hook skin/rebuild、LJ、VV/FIRE/FIRE2、periodic、observer、FusedStage、inflight、默认 Warp 反向守护。
- 数值：dense/cell-list 同输入集合/位移/距离/力；CPU FP64 oracle；HCU smoke；必要时 gradcheck/gradgradcheck、compile 和分布式 plan hash。
- 性能：同一 harness 覆盖 no-PBC full/half、periodic full 和代表性 MACE/FIRE2 E2E；共享 HCU 数字只能标为相对证据，不能作为发布门槛。
- 每个里程碑同步 `docs/STATUS.md`、`docs/FEATURE_COMPATIBILITY.yaml`、必要的 ADR/UPSTREAM；报告实现、测试、数值结果和未验证假设四条轴。

## 6. 已锁定的设计假设

1. 没有显式 policy 时，核心 `backend=None` 永久保持上游 legacy 语义；Hygon 产品入口可以显式加载版本化 profile，但不能仅凭设备名偷偷改变默认。
2. registry 负责 capability，profile 负责推荐，planner 负责整个 pipeline，runtime 执行冻结 plan。
3. 默认 fallback 为显式错误；reference fallback 只有在 profile 允许时生效并记录影响。
4. 生产不运行时 benchmark 选后端；tuning 产物必须经过审阅和版本化。
5. checkpoint 默认严格复用 plan；跨环境必须显式 replan。
6. cell-list 过渡基线已封存在 `codex/feat-reference-cell-list`；M1 已完成名称/API 迁移，不额外承诺兼容未发布的过渡名称。

## 7. 后续 Codex session 恢复方式

下次工作从仓库恢复，不依赖模型的跨 session 记忆：

1. 先读根目录 `AGENTS.md`，再读 `docs/PROJECT_HANDOFF.md`、`docs/STATUS.md` 和本文。
2. 运行 `git status --short --branch`，保留当前用户改动；不要 reset、checkout 或覆盖式复制。
3. M1 已完成。下一阶段在用户确认后以 M2 为唯一实现范围；开始前报告将修改的文件、契约和测试，完成一个小里程碑后停下汇报并等待确认。
4. 如果工作树被清理或换了 clone，只有已提交并推送到共享分支的文档才能恢复；本次新增文档目前需要与现有改动一起由人类决定何时提交/同步。
