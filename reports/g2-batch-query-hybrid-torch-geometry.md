# G2 Batch query hybrid HIP topology plus Torch geometry

日期：2026-09-20。任务：`TORCH-NEIGHBOR-PBC-CELL` 阶段二，小点 29。

## 目标与实现

本小点将已有 native HIP 离散路径与 Torch continuous geometry 组合：

```text
native HIP cell-list/query candidate
    -> native HIP composite-key topology
    -> Torch distance/vector geometry
```

新增 `materialize_batch_query_topology_geometry_into`，读取已经 canonicalized 的 public
matrix/shift/count，只用 positions/cells 计算 distance/vector。拓扑被视为离散输入，不参与
梯度；distance/vector 保持普通 Torch 表达式，因此 positions/cells 的一阶和二阶 geometry
gradient 仍可用。

该 helper 不接 dispatcher、AOT、`auto` 或 `hip` capability，也没有把 Torch geometry 宣称为
HIP/Triton geometry 实现。

## 正确性

CPU focused materialization suite 为 `11 passed`，包括 topology-to-geometry 与 Torch reference
parity、一阶/二阶 geometry gradient，以及既有 candidate/topology 边界。

HCU 0/BW200/gfx936 boundary 通过：

- FP32/FP64；
- mixed PBC；
- full/half；
- empty、capacity、non-default stream；
- native composite topology 后接 Torch geometry；
- public matrix/count/shift/distance/vector 与 Torch reference parity。

HCU boundary 的 native topology 仍是 forward-only；梯度合同由 CPU helper gradient test 覆盖，
没有宣称 native HIP topology 本身可微。

## 性能合同

设备为 HCU 0/BW200、`gfx936`、DTK 26.04、PyTorch HIP `6.3.26093`。FP32、capacity `256`、
固定 grid/CSR metadata，覆盖 `46/92 atoms × 32/64 systems` 的 uniform/clustered workload，
每个 workload 3 次 warm-up、5 次 samples。

比较范围是 query/materialization，不包含每次调用重新 build cell-list CSR metadata：

- hybrid：native HIP query + native composite topology（caller-owned workspace）+ Torch geometry；
- Torch：同一输入和已构建 CSR metadata 上的完整 Torch reference query/materialization；
- device time：HIP-event scope；
- API wall-clock：显式 `torch.cuda.synchronize()` 包围调用；
- 每个样本前后均检查 hybrid 与 Torch 的完整 public output parity。

## 结果

factor 定义为 `hybrid / Torch`，低于 `1.0x` 表示 hybrid 时间更低。

| atoms × systems | workload | device-time factor | API wall-clock factor |
|---|---|---:|---:|
| 46 × 32 | uniform | 0.0252x | 0.0252x |
| 46 × 32 | clustered | 0.0213x | 0.0211x |
| 46 × 64 | uniform | 0.0144x | 0.0142x |
| 46 × 64 | clustered | 0.0120x | 0.0124x |
| 92 × 32 | uniform | 0.0257x | 0.0256x |
| 92 × 32 | clustered | 0.0251x | 0.0243x |
| 92 × 64 | uniform | 0.0134x | 0.0132x |
| 92 × 64 | clustered | 0.0128x | 0.0127x |

在这个固定 scope 中，hybrid device factor 为 `0.0120x--0.0257x`，API factor 为
`0.0124x--0.0256x`，即约为 Torch 时间的 `1.2%--2.6%`。这主要反映 Torch reference 的
Python/逐阶段 query 成本与 native HIP 离散路径的差异，不能直接推广为完整 MD/MACE 或整个
neighbor API 的相同加速比。

## 结论与下一步

第29小点证明了：native HIP topology 可以直接接入 Torch geometry，完整 public neighbor
输出与 Torch reference 一致，同时 geometry gradient 路径可保留。这建立了可用的 hybrid
correctness/performance baseline。

当前仍有边界：cell-list build 不在本轮重复计时，Torch geometry helper 仍包含 validation 和
普通 Torch indexing，native topology 不接 autograd/dispatcher，未测试 FP64 performance、
compile/opcheck、rebuild、target/pair outputs 或端到端模型。

下一小点建议单独评估 geometry kernel：先做 forward-only HIP/Triton candidate，与当前 Torch
geometry helper 在相同 public topology 上比较数值和 device/API 时间；若要替换训练路径，另
行补齐 autograd/二阶梯度合同。

原始 samples 保存在：

- `artifacts/g2-point29-46x32/`；
- `artifacts/g2-point29-46x64/`；
- `artifacts/g2-point29-92x32/`；
- `artifacts/g2-point29-92x64/`。
