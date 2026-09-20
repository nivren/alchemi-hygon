# G2 Batch geometry/PBC/key/count fusion microbenchmark

日期：2026-09-19。任务：`TORCH-NEIGHBOR-PBC-CELL` 阶段二，小点 10。

## 测量合同

本次比较同一 native HIP build 前半段的两种显式 API 组合：

- **composed baseline**：对每个体系调用既有 native `cell_key`、native `cell_count`，再将 local
  keys 加上 global cell offset；两个体系的输出切片共同形成 global shifts/mapping/keys/counts。
- **fused candidate**：一次 native Batch geometry/PBC/key/count 调用；同一 per-atom kernel 写入
  相同的 global shifts/mapping/keys/counts，并对 counts 执行 atomic add。

计量是预热后的同一 Torch stream HIP-event device time，低者更好。包含每次 API 所发起的 GPU
memset、native kernel 和 composed key-offset add；不含 JIT/load、Python-side validation/loop、输入
生成、output comparison、host wall clock、scan、CSR fill、query、dispatcher 或完整 neighbor build。
两个实现以计时前后 `torch.equal` 比较所有四类输出。

固定环境与输入为 BW200/gfx936、DTK 26.04、PyTorch HIP `6.3.26093`、`HIP_VISIBLE_DEVICES=0`、
`OMP_NUM_THREADS=1`、FP32。Batch 为两个异构系统，每体系 16384 atoms，global cells 为 4096：
dimensions `[[16,16,8],[8,16,16]]`、offsets `[0,2048]`、PBC
`[[true,false,true],[false,true,false]]`。覆盖：

- `uniform`：fractional positions 在每体系内均匀采样；
- `single_cell`：所有 atoms 落在每体系 cell 0，构造 atomic contention。

每侧在计时前交替 20 次 warm-up；随后每个样本重复 API 100 次，交替收集 20 个样本。预注册的
保留阈值为：两个 workload 都有至少 5% 的融合中位数降低，且样本方差可解释。原始样本每行一个
正毫秒值，保留在不入库的 `artifacts/batch-cell-key-count-fusion-benchmark/`。

## 执行与结果

同一主机权限上下文中 `/dev/kfd`、`/dev/dri` 可见；运行前 `hy-smi --showuse` 显示全部 HCU 利用率
为 0%。以下限时命令退出码 0：

```bash
source scripts/activate_hygon_env.sh project
HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  PYTHONPATH=packages/framework:packages/ops timeout 120 \
  .venv/bin/python probes/batch_cell_key_count_fusion_benchmark.py \
  --output-dir artifacts/batch-cell-key-count-fusion-benchmark
```

| workload | composed median | fused median | composed / fused | fused 降低 | composed/fused 相对总体标准差 |
|---|---:|---:|---:|---:|---:|
| uniform | 1.17562 ms | 0.79358 ms | 1.481x | 32.50% | 0.82% / 1.57% |
| single_cell | 1.17380 ms | 0.79173 ms | 1.483x | 32.55% | 1.35% / 1.22% |

`hygon-dcu-performance/scripts/compare_timings.py` 从四个 raw sample 文件独立复核了中位数、
均值、min/max、总体标准差与 improvement；两种 workload 的计时前后 shifts/mapping/keys/counts
均逐元素一致。两侧的样本方差均小于 2%，并且两种 workload 都满足预先定义的 5% 门槛。

## 解释与边界

在这个固定的二体系 ordered-Batch FP32 workload 中，融合候选的 build-half device-time 中位数约为
独立 native composition 的 `1.48x`，局部降低约 `32.5%`。结果支持保留该融合 candidate：它避免了
每体系重复 key/count launch 和 key offset add，并同时保留相同的 global outputs。

这不是完整 cell-list 或邻居性能结论。比较没有包括 rocPRIM scan、CSR fill、排序、query、Python
runtime/dispatcher、不同 Batch sizes、不同 dtype、公开 neighbor order、梯度或完整 MACE/FIRE2；
也没有与 Torch reference API 或任何 Triton 实现比较。它不改变 `auto`、默认路径或 `hip`
capability，也不足以满足完整 neighbor 的 2x 或端到端 20% 准入门槛。

## 下一步

下一小点应先冻结 CSR fill 的共享 ABI 和 Torch oracle：明确 starts/counts/cursor 的生命周期、
cell atom list 的容量、full/half neighbor 公共 order 约束，以及 Batch global atom offsets。之后才
实现独立 native HIP fill candidate 并以 PBC/no-PBC、triclinic、Batch 对照验证；不提前接线 runtime。
