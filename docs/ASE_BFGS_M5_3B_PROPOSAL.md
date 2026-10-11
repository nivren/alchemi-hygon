# M5-3B 提案：异构 Hessian 的设备并行对照

2026-10-10，**用户已确认，隔离验收完成**。继 M5-3A 动态尺寸/Batch 扩展之后开展；仍在
probes 中开发，不改变产品 Registry、默认后端、版本锁或现有 State/Filter。

## 目标

在同一张 HCU 上比较三种异构分解调度：现有同宽组串行、有限多 stream、
原生异构 Jacobi。用相同真实 Hessian、相同数值门和完整分组 BFGS 窗口判断收益，
解决原子数都不同时等维组退化为 B1 的吞吐问题。用户仍提交一个异构 Batch。

## 方案

1. **串行基线**：复用 M5-3A 的含锂八个不同尺寸组、T2 的 561维 B8，以及混合
   组；保留 rocSOLVER 和动态自写 Jacobi 两个对照。模型共享 Batch 求值不变。
2. **多 stream 候选**：用 2/4 个受控 worker 发送不同尺寸组。先解决 pybind GIL
   和每 sweep `.item()` 阻塞导致的串行下发问题；只写 Python stream 循环不足以
   证明并行。每个 worker 有明确 device/current stream、独立临时张量，生产者/
   消费者 event 和张量生命周期保护。预加载扩展，避免线程竞争 JIT。不开多卡。
3. **原生异构候选**：接收临时 Tensor 列表与各体系实际维度/64位偏移/工作列表，
   合并旋转和矩阵元素任务，不对数学矩阵做谱 padding。每体系保留配对轮次、
   中心化、原始 norm、收敛诊断、冻结与双缓冲；先用 CPU 模拟验证不同奇偶尺寸
   的配对、空转轮次和缓冲选择，再实施 HIP。收敛检查可合并主机标量读取，
   不改阈值或允许未收敛成员通过。可暂复用设备排序逐组处理，单独量化该成本。
4. **完整数学对照**：现有 `_step_batch_with_eigh` 在组内更新 H 后立即同步分解，
   要实现跨组并行，隔离测试需要先准备全部候选 H，再批量提交分解，最后计算
   方向并在全组成功后统一提交。只在 probe adapter 中实施这个调度变化，保留
   原始数学表达式；不提前重构产品公共状态管理或引入长期 ragged storage。

TorchSim 本地参考 SHA `33cd5987c25c30af90c4003c49593ce7e3652da3`：
`torch_sim/optimizers/bfgs.py` 使用 padded State 和按实际尺寸的 Python 循环
`torch.linalg.eigh(H_group)`；这段路径没有提供跨尺寸分解并行。其 denominator
epsilon 和谱 clamp 不符合本项目严格 ASE 合同，不吸收。DTK 的 pointer-batched
接口也仍使用一个共同 n，不能直接把指针列表当作可变尺寸 solver。

## 验收

- 同输入谱/方程/正交、负谱/简并谱、输入保护、真实未收敛及晚组失败；保持
  M5-3A 和原 ASE 门，不以收敛末态代替逐步对照。
- 固定与变胞异构数学逐步 ASE、单体系、restart、冻结/重排/退出/补位，以及
  全组失败不提交；检查跨 stream 的 producer/consumer、输入提前释放与内存复用。
- 三进程、预热后七次交替配对；记录所有组规模、workspace、host 等待和实际
  kernel 时间重叠。不能只用 stream 创建次数或单次计时声称并发/加速。
- 用 M5-3A 真实输入验证共享模型调用和候选完整更新窗口；T2 的短步共享力
  ASE oracle 与独立 CPU MLIP 对照分开记录。完整收敛和端到端吞吐按实际完成
  范围报告。限时单卡，竞争窗口排除。

比较全部已完成的结果，包括退化、失败和额外显存；若多 stream 没有收益，
不保留无效复杂度。若原生异构方案未过数值门，保持已验收串行方案。完成后
给出选择依据及下一阶段产品接入目标，等待确认；不自动注册、提交、推送或合并。

结果见[M5-3B报告](../reports/g2-ase-bfgs-m5-3b.md)；建议2stream跨组调度，
单组batched不变。[M5-4产品接入](ASE_BFGS_M5_4_PROPOSAL.md)另待确认。
