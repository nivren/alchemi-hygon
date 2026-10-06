# HIP 邻居 workspace 复用与性能契约（M4）

2026-10-06 用户确认；M3实现 `6b1680a`、文档 `47cf259`。
分支 `codex/perf-hip-neighbor-workspace`。先保留基线，再依据分项/trace证据修改。

## 不变量

沿用 M1–M3 full-PBC/full/MATRIX+COO/FP32/64/skin=0；显式HIP、auto关闭。
每次重新验证输入并重算 inverse/grid/radius/offset，新建 trusted plan，完整 CSR/query。
不复用 metadata、邻居或模型输出，不省略 checked 初始化，不改 dtype、cutoff、排序、
容量失败与梯度约定。public MATRIX/COO/geometry 输出仍独立所有，后续调用不覆盖旧结果。

实验候选仅复用内部 CSR、candidate 和 scan/sort scratch；单线程单设备/stream/dtype槽，
不同 key 替换，避免不同 stream 未排序的读写或线程竞争；超过128MiB不保留。
N/B/ptr/cell/capacity改变必须重建语义完整，scratch按容量/shape适配；输入异常不得污染
下次计算。scratch可能持有设备显存，生命周期与线程一致，有显式内部清理用于测试/测量。
不缓存输入tensor、plan或可微输出。此缓存只面向eager；compile/opcheck仍未准入。

## 测量和门槛

HCU4/gfx936，DTK26.04，已有native binaries，OMP_NUM_THREADS=1，资源检查后限时单卡。
固定种子20260920，46×32、92×64 full-PBC合成Batch，cutoff3Å/capacity128/COO，
FP32/64；真实MPA-0三CIF18原子cutoff6Å/capacity128，FP32。
同步wall API计时，5 warmup、每轮15 samples、三轮独立测量；保存冷首次API、全部原始
样本、median/mean/std/range。cold含runtime/extension加载，已有cache不代表cold JIT编译。
真实模型区分neighbor、model-only和neighbor+model，短端到端而非长弛豫吞吐。

分项instrumented wall timing含额外同步，仅定位host阶段；hipprof单独限时小trace定位
kernel/API，不与无profiler总耗时直接相比。先拓扑reference精确核对，再计时。
性能候选仅在三轮目标46×32 FP32 warm median改善至少5%、其余邻居项无超过5%稳定
退化、相关正确性gate通过后保留；噪声/未达到门槛则拒绝候选并报告。
不把分配次数下降或某次最佳值称为性能提升。

正确性要求候选容量/输出alias/异常后复用/stream/thread隔离测试和当前晶胞回归。
若保留产品改动，还须完成原M1/M2 ops与M3 framework/HCU真实模型回归；拒绝候选后
核对产品源码恢复并回归现有公共路径。CPU可验证cache规则但不得伪造HIP运行。
不新增query并行/native COO/skin/auto/模型或物理收敛准入。

证据目录 `artifacts/hip-neighbor-workspace-m4/`，报告
`reports/g2-hip-neighbor-workspace-m4.md`。实施结论：完成基线/trace/候选验证；
候选30项CPU/HIP测试通过，同进程配对45样本/项，目标下降1.29%、真实neighbor+model
下降0.64%，未达5%门槛，撤回产品缓存改动。归档实验可由benchmark显式重放；
产品仍stateless，不持有缓存显存。未新增workspace/query/auto能力；M4已收口
`95f2ef6`/`88b6b2f`，M5 metadata优化已获确认并独立验收。
