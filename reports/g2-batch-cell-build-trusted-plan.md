# G2 native HIP Batch cell-list prevalidated plan lifecycle

日期：2026-09-20。任务：`TORCH-NEIGHBOR-PBC-CELL` 阶段二，小点 34。

## 目标

Point 33 的 trusted helper 只是私有函数约定，缺少明确的 validation ownership 和 buffer
lifetime。Point 34 增加 `_TrustedBatchCellBuildPlan`，把一次完整 checked public build 与
后续 trusted 重复调用绑定起来：初始化阶段由公开 API 建立完整合同，后续调用复用精确的
tensor/workspace 引用，避免每次重复上层 validation。

plan 仅在以下条件保持不变时有效：tensor storage、shape、dtype、device、alias 关系、
cell metadata、capacity、workspace 和 global atom offset。positions 数值可以原地更新；
结构变化或重新分配 buffer 时必须重新 `initialize`。plan 不接 dispatcher、AOT、`auto`
或 `hip` capability。

## 验证合同

- 设备：HCU 0，BW200，`gfx936`，DTK 26.04，PyTorch HIP `6.3.26093`。
- FP32、mixed PBC、非正交 cell，`46/92 atoms × 32/64 systems`，uniform/clustered，8 个
  workload。
- 每组 3 warm-up、5 samples；固定 metadata、output buffer、workspace，JIT/module load
  不在稳态计时中。
- plan 初始化执行一次 checked public build；后续计时调用 `plan.run()`。每次计时前后比较
  shifts/mapping/keys/counts/starts、cell membership、cursor final state 和 capacity tail。

## 结果

全部 8 个 workload 通过 plan/public/Torch CSR parity。HIP-event median factor 为：

- trusted/public：`0.0281x--0.0313x`；
- trusted/Torch：`0.0394x--0.0413x`。

这与 Point 33 的 direct-extension 结果同量级，说明把 trusted 调用封装进 plan 没有引入
明显性能损失。显式同步 API wall-clock 的 median 方向也一致，约为 public 的
`0.0335x--0.0342x`；46×64 clustered 有一个 public API wall 样本约 `4.13 ms`，使该组
离散度升高，因此 API wall 仅作辅助证据，不能替代 HIP-event 主结论。

## 测试

- HCU smoke：46×32，1 warm-up/2 samples，退出码 0，CSR parity 通过。
- CPU focused：cell ABI、public build boundary、plan CPU rejection 共 `21 passed`。
- `py_compile` 和 `git diff --check` 通过。

## 结论与下一步

Point 34 完成了从“私有函数”到“显式生命周期对象”的最小安全边界，但 plan 仍是
isolated build candidate，不是生产 runtime API。下一步是把 plan 接入已有隔离的完整
邻居 benchmark，形成：

```text
plan.run() -> native HIP query -> composite-key topology -> Torch geometry
```

同时保留 Torch reference 作为 correctness/gradient oracle，并比较 public build、trusted
plan build 和 Torch build 的完整端到端成本。只有 full pipeline parity、异常边界和稳定
收益都通过后，才讨论 dispatcher/runtime 接线。

原始 samples：

- `artifacts/native-batch-cell-build-trusted-plan-benchmark/46x32/`；
- `artifacts/native-batch-cell-build-trusted-plan-benchmark/46x64/`；
- `artifacts/native-batch-cell-build-trusted-plan-benchmark/92x32/`；
- `artifacts/native-batch-cell-build-trusted-plan-benchmark/92x64/`。
