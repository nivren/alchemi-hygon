# HIP 真实 Batch 综合验收契约（M7）

用户已确认本阶段。起点为 M6 实现 `fe18147`、文档 `c2a8339`；仅新增验证探针和证据，
不调整 kernel、公共 API、默认后端、数值容差或模型实现。

## 输入与路径

- HCU4/gfx936 单卡，先只读检查共享资源；使用现有 DTK/PyTorch 环境。
- 已锁定 SHA256 的 MPA-0 medium checkpoint，普通 e3nn eager；关闭 cueq/compile，
  保留位置/应变自动微分，模型参数冻结。FP64 优先，FP32 补充。
- `/data/csp_data/catl_data/li_mp_data` 按文件名排序选取真实 CIF 前缀，
  达到约 1000/2000 原子，逐文件记录 SHA256、原子数；两个规模来自同一前缀清单。
- 两条路径为显式 `torch_reference` 与当前 `hip`，均 `method="cell_list"`、
  full-PBC/full-list、COO、skin=0、每次当前晶胞完整重建；动力学仍 Torch reference。
  COO 使用现有 Torch 打包，不宣称 native HIP COO。

## 验收与计时

- 每个规模/精度/FIRE 或 FIRE2 独立限时进程；先通过约1000原子显存检查再运行约2000。
- 同几何初始和三步后快照比较完整有重数的 `(i,j,sx,sy,sz)`，检查体系边界和 dtype/device，
  E/F/stress 使用既有 M3 各物理量容差。FP32 已知 cutoff 舍入分歧不可放宽或隐藏。
- 三轮独立初始条件下每路径三步，比较 mutable fields、优化器内部状态、步数和几何变化；
  整数/状态精确，浮点沿用 M3：FP64 rtol=2e-8/atol=2e-9，FP32 2e-4/2e-5。
- 同快照邻居、模型、邻居+模型：一次预热，三轮各两样本，轮换路径顺序。
  完整 step 包含 Hook、模型和优化器更新，priming/构建/模型加载/校验/IO 在计时外；
  同起点独立轨迹，额外预热轨迹不计样本。记录全部同步墙钟样本、median/mean/std/range。
- 记录 Torch allocated/reserved 峰值及增量；这不等于全部设备显存。
  显存不足/资源占用明确报告，不换精度、不截断邻居、不静默回退。
- 真实多结构 FusedStage 预算退出/到达补位分别运行两后端，检查 source/system IDs、ptr、
  到达 priming 和继续变胞、输出和计数；预算退出不代表物理收敛。生命周期计时不混入 step。
- 复核 M1–M6 源码/契约/报告/现状一致性，运行相关 CPU/HCU gate。

## 交付边界

无新性能门槛或优化承诺，报告实际 Torch/HIP 差异。不是长收敛、20k真实模型吞吐、
其他 MLIP、训练、compile、DomainParallel、no-PBC/half/skin/auto 验收。
完成后先汇报并提出收口方案，等待确认；不自动提交 M7、合并或推送。

## 本阶段结果

FP64四组通过；FP32一组通过、三组严格forces重复门失败，纯Torch/Torch控制也出现同一失败。
保持原容差与失败记录，不把物理量检查通过计作整个状态门通过。
CPU ops51/framework21，HCU ops85/framework49定向回归通过；M7未提交/合并/推送。
见 [M7报告](../reports/g2-hip-real-batch-validation-m7.md)；下一验收分层修订待确认。
