# G2 Batch query materialization 阶段 breakdown

日期：2026-09-20。任务：`TORCH-NEIGHBOR-PBC-CELL` 阶段二，小点 20。

## 目标

小点 19 已确认 native HIP enumeration + Torch geometry materialization 在高 Batch workload
上明显低于完整 Torch query，但还不能判断 Torch materialization helper 内部的主要热点。
本小点只做可复现的阶段分解，不实现新的 HIP/Triton kernel：

- `materialize_topology`：stable candidate canonicalization、matrix/count/shift scatter；
- `materialize_geometry`：在固定 canonical pairs 上计算并写回 distance/vector；
- `materialize_complete`：完整 `materialize_batch_query_candidate_into`，含输入检查、拓扑和 geometry。

`materialize_geometry` 使用计时前生成的 canonical pair，因而不把 topology sort/scatter 混入；
三个阶段都是固定 native candidate 上的预热 HIP-event device timeline，不是 Python API wall-clock。

## 实现与正确性

`packages/ops/nvalchemiops/_torch_batch_query_materialization.py` 现将 helper 内部拆成私有
canonicalize、topology scatter、geometry materialization 和 output reset 子阶段；公共 helper
接口和 reference ordering 不变。benchmark 通过 `--geometry --breakdown` 调用这些阶段，并在
计时前后比较：

- topology 阶段与完整 helper 的 matrix/count/shift；
- geometry 阶段与完整 helper 的 distance/vector；
- 完整 helper 与 Torch reference query 的全部 public outputs。

CPU focused：materialization、reference cell-list 和 HIP boundary 合计 `17 passed`；
compileall 与 `git diff --check` 通过。

HCU 0/BW200/gfx936 correctness boundary 重跑通过：FP32/FP64、mixed PBC、full/half、empty
input、non-default stream、capacity overflow、public parity 和 positions/cells 一阶、二阶
geometry gradient 均通过。HCU 环境为 DTK 26.04、PyTorch HIP `6.3.26093`。

## 测量合同

运行参数：HCU 0、FP32、cutoff `0.6`、capacity `256`、3 warm-up、5 samples、固定两套 mixed-PBC
cell geometry，覆盖 `46/92 atoms × 32/64 systems` 与 uniform/clustered positions。
HIP event scope 排除 JIT、分配、Python wall time 和 runtime selection；每个阶段计时前后都做
output parity。raw samples 位于：

- `artifacts/native-batch-cell-query-materialization-breakdown-46x32/`
- `artifacts/native-batch-cell-query-materialization-breakdown-46x64/`
- `artifacts/native-batch-cell-query-materialization-breakdown-92x32/`
- `artifacts/native-batch-cell-query-materialization-breakdown-92x64/`

下表为 median，单位 ms；百分比为对应阶段 median 除以完整 helper median。

| atoms × systems | workload | topology | geometry | complete | topology / complete | geometry / complete |
|---|---|---:|---:|---:|---:|---:|
| 46 × 32 | uniform | 1.17696 | 0.29136 | 1.63264 | 72.1% | 17.8% |
| 46 × 32 | clustered | 1.24832 | 0.29440 | 1.69232 | 73.8% | 17.4% |
| 46 × 64 | uniform | 1.21536 | 0.29712 | 1.67408 | 72.6% | 17.7% |
| 46 × 64 | clustered | 1.31696 | 0.30096 | 1.76192 | 74.7% | 17.1% |
| 92 × 32 | uniform | 1.21168 | 0.30016 | 1.65808 | 73.1% | 18.1% |
| 92 × 32 | clustered | 1.23824 | 0.30720 | 1.69952 | 72.9% | 18.1% |
| 92 × 64 | uniform | 1.26800 | 0.31248 | 1.71312 | 74.0% | 18.2% |
| 92 × 64 | clustered | 1.30032 | 0.32112 | 1.73536 | 74.9% | 18.5% |

`compare_timings.py` 以 complete 为 baseline 独立复核了 16 组 raw samples：topology 相对
complete 的 median ratio 为 `1.33--1.39`，geometry 相对 complete 的 ratio 为 `5.40--5.85`；
这里的 ratio 仅表示阶段时间关系，不是新 backend speedup。

## 结论和下一步

8 个 workload 一致显示，当前 Torch helper 的主要热点是 stable topology canonicalization/
scatter，约占完整 helper `72%--75%`；geometry 约占 `17%--19%`。因此下一小点的首个优化
hypothesis 应针对不规则 topology path：评估 HIP 的显式 compaction/scatter 或可用排序原语，
同时保留 Torch reference 作为 oracle。Triton geometry 仍可作为后续独立候选，但不是当前第一
优先级。

本小点没有接入 runtime、dispatcher、`auto` 或 `hip` capability，也没有宣称 HIP 必然优于
Triton。未覆盖 FP64 性能、API wall-clock、allocation/validation、rebuild、target/pair
outputs、pair-centric/sorted query、compile/opcheck、分布式和端到端 MACE/FIRE2。
