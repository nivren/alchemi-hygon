# G2 Point 37 native HIP Batch neighbor runtime wrapper

日期：2026-09-20

## 目标

把已验证的 native stages 组合成一个显式 ops-side wrapper，验证完整的

```text
trusted CSR build -> native candidate query -> composite-key topology
-> Torch distance/vector geometry
```

本点不注册 `backend="hip"`，不修改 dispatcher、`auto`、Warp 或 framework 默认路径。

## 实现

- `nvalchemiops.run_native_hip_batch_neighbor_into` 接受已完成 checked initialization
  的 `_TrustedBatchCellBuildPlan`、调用方持有的 candidate/public/geometry outputs 和
  composite topology workspace；每次调用复用同一 build plan/lifetime。
- wrapper 固定 full-list、fixed-cell、Batch 和 Torch geometry；native topology 只负责
  离散输出，Torch geometry 继续保留连续位置路径。
- admission gate 在任何 native buffer mutation 前执行；CPU、unsupported dtype/feature
  和无效 plan 明确失败，不 fallback。
- 修复 composite workspace size-query：原实现将 int32 candidate storage 强制解释为
  int64 sort key storage，高 Batch candidate 可能越过 tensor 边界。现在 size-query 使用
  正确 dtype/长度的临时 key/order/scan buffers。

## CPU 验证

- `packages/ops/test/torch/test_hip_batch_neighbor_runtime.py`：`16 passed`
- ops Torch suite：`109 passed, 1 warning`
- probe `py_compile`、`compileall`、`FEATURE_COMPATIBILITY.yaml` parse 和
  `git diff --check`：通过

## HCU 验证

环境：BW200，gfx936，HCU 0，Torch HIP `6.3.26093`，FP32，capacity 256。
独立 Torch CSR/query/geometry 作为 oracle；wrapper probe 不计性能。

| workload | cells | 结果 |
| --- | ---: | --- |
| 46 atoms/system × 32 systems | 745,472 | build、public matrix/shift/count、distance/vector parity 通过 |
| 92 atoms/system × 64 systems | 1,490,944 | build、public matrix/shift/count、distance/vector parity 通过 |
| 46 × 32 + position gradient | 745,472 | 一阶、二阶位置梯度通过 |

probe 输出均为 `status=passed`。这证明显式 wrapper 在代表性高 Batch workload 上可执行，
但不是性能结论。

## 当前边界

仍未完成 framework `backend="hip"` 注册、MACE/FIRE2 端到端调用、native geometry
backward、half-list、skin/rebuild、variable-cell、target/pair outputs、DomainParallel、
compile/opcheck 和 `auto` 性能准入。下一小点是 Point 38：在 framework 中接入显式 HIP
selection/executor 边界，同时保持默认 Torch/Warp 路径不变。
