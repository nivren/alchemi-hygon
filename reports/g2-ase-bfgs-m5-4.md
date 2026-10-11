# ASE-compatible BFGS M5-4：显式 HIP Jacobi 与异构双 stream 产品接入

2026-10-10，用户已确认的产品接入阶段，完成实现与下述限定范围的验收。
分支 `wangleping/feat-ase-bfgs-m5-4`，HEAD
`cfd22ad11fe0e1704dff48f53abee69135f8db99`；继承 M1–M5-3B 未提交工作。
这是 NVIDIA reviewed upstream 尚未提供的本地新增功能，非上游算子移植或 UBF。
未提交、合并、推送，未改变上游锁、默认后端或系统环境。

## 实现与接口

[用户确认的提案](../docs/ASE_BFGS_M5_4_PROPOSAL.md)与
[产品合同](../docs/HIP_BFGS_JACOBI_STREAMS_CONTRACT.md)定义本阶段边界。

- 注册 `hip.bfgs-ase-jacobi-streams-v1`，family=hip、strategy=jacobi_streams、
  default_strategy=False、auto_enabled=False。用户按完整 ID 显式选择；`hip`
  仍是既有 rocSOLVER Jacobi，BFGS 默认仍为 Torch reference。None/warp/auto
  的既有失败行为和 FIRE/FIRE2 默认保持。反转 catalog 顺序的测试保护这一选择。
- 新 executor 沿用 prepare_state/step/batch_step ABI，复用既有 ASE 数学。
  Batch 先准备全部候选 H，由私有联合分解入口调度，全部谱、方程、正交、
  方向和坐标检查通过后统一提交。单体系公式不变，不加阻尼、谱 clamp 或曲率跳过。
  CPU 四体系十二步，全部 state 与五项 step 字段和冻结旧数学逐位一致。
- 原生动态 FP64 Jacobi 来自本仓库原创 Apache-2.0 探针，使用 HIP 核函数及
  ATen 设备排序，不链接 rocSOLVER，不导入或复制外部 `torch_rocsolver_eigh`。
  元素偏移 int64，覆盖实际 n/B；没有 147/285 或 B≤4 门，也没有谱 padding。
  实际容量受内存、整数表示范围与计算成本约束。
- 单宽组直接使用 batched kernel；多宽组在同一进程、同一 HCU 内使用两个
  持久 worker 线程与 HIP stream，按元素量分配。模型依然整 Batch 求值。
  producer/completion event、record_stream、明确 device/current stream 保护
  跨 stream 依赖和分配器寿命。并发调用串行进入同一 scheduler；错误排空
  已接受工作，包括第二次 submit 失败的情况；支持关闭、拒绝旧对象和重新创建。
- Framework 使用既有显式 backend 传递，没有改 optimizer、BaseDynamics、
  FusedStage、Filter、ID 管理或 restart schema。私有 torch.library custom_op
  提供 schema/fake 元数据，拒绝 requires_grad；不据 fake 通过承诺 compile。

使用方式（`batch` 与 `neighbor_hooks` 按原有模型/邻居设置提供）：

```python
from nvalchemi.dynamics import ASECompatibleBFGSVariableCell

optimizer = ASECompatibleBFGSVariableCell(
    model=model, hooks=neighbor_hooks,
    backend="hip.bfgs-ase-jacobi-streams-v1",
)
relaxed = optimizer.run(batch, n_steps=500)
```

## 中断前发现并修正的数值边界

独立对角诊断 `diag(2e-18, 1e-18, 2)` 在原动态探针/接入版本的减均值、加均值
恢复中变为 `[0,0,2]`，退出1；这会把合法非零谱触发成 BFGS 精确零谱失败。
原始输入、输出和实际退出码保存在 `diagonal-before*`，没有通过放宽容差隐藏。

最终产品只修正零轮收敛分支：在设备上稳定排序原始对角值，恢复精确极小谱、
负零及稳定列顺序；迭代旋转与中心化谱恢复保持。修正后输出
`[1e-18,2e-18,2]`，对应 raw step `[0.1,0.1,0]` 与 CPU 逐位一致，退出0。
新 HCU 回归另含正零/负零两种排列、混合负谱及公共注册 BFGS step。

`kernel-provenance.json` 是修正前来源证据；最终来源差异应以
`final-kernel-provenance.json` 和 acceptance manifest 为准。冻结探针不修改，
非零轮矩阵五输出逐位对照；零轮谱以原始 CPU 对角排序为正确性基准。

## 验收与数值

最终队列：`artifacts/ase-bfgs-m5-4-2026-10-10/final-queue.json`。
每个作业的 `*-run.json` 保存完整 argv、限时、实际退出码及只读资源观察，
相邻 log/JSON 保存测试结果和误差；CPU 总门单独保留在 `cpu-final-gate.log`。
已通过的正确性门如下，其余真实工作流与性能复验在后节独立记录。

- CPU reference 总门退出0，668 passed/3既有 Warp skipped：Ops294、fresh LJ1、
  Framework327、state22、dynamics Ops24。该门覆盖最终 Python 数学/选择路径；
  随后零轮修正只改变原生谱输出，追加最终 HCU 门，不重复无变化的 CPU 全门。
- 最终 HCU Framework24 passed：固定晶胞与变胞 default/pressure-scale/mask
  逐步 CPU 对照、非默认 stream、bitwise restart、完整冻结、重排/退出/新ID、
  单体系快照提升及原生 sampler/FusedStage 补位。七步主模型求值8次。
- 最终原生 HCU7 passed：真未收敛的末组失败/全状态原子性及恢复、关闭/重建、
  双调用线程、producer/consumer提前释放及分配器复用、partial submit排空、
  schema/fake opcheck、输入三角/finite/梯度错误、极小谱/有符号零。
  既有 rocSOLVER HCU39 passed、对应源文件 hash 未变化。
- 最终32冻结矩阵、9真实组及4批处理组合通过：达到1113维；批处理覆盖
  147 B8、189 B16、199 B32、285 B8，真实561 B8。最大谱绝对误差1.052e-12、
  方程残差4.935e-15、正交残差2.956e-13；原门未放宽。
- 最终11体系十步 ASE 逐步门退出0，DOF为15/21/30/45/57/93/201/489/189/189/561。
  坐标、H、谱、raw、applied最大绝对差分别2.673e-14、7.816e-14、1.739e-11、
  3.517e-14、5.190e-15；restart逐位、single/batch对照和真零谱/未收敛/
  非法特征向量的全状态不提交通过。

## 性能窗口与真实工作流

相同输入下比较旧 rocSOLVER、自写单 stream、产品调度；三个独立进程，每次
预热两次、轮换顺序采样七次，共21个稳态样本。solver 使用冻结 H；完整数学
使用真实 H 加合成 q/f/history，包含更新、有效性检查、方向和原子提交，
属于算子微基准。rocSOLVER solver 控制有重复输入有限检查；完整数学控制
直接调用旧产品函数。并行收益主要以检查一致的自写单 stream 为对照。
保留冷启动、JIT 和 Torch allocator 峰值；后者不覆盖库的不可见分配。

真实工作流使用 **MACE MPA-0 medium** checkpoint，非 MACE-OFF small；
SHA256 `75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638`。
邻居为 Torch reference、skin=0。只读使用 Li/T2 CIF，Li 八体系原子数
2/4/7/12/16/28/64/160、变胞 DOF15/21/30/45/57/93/201/489；
T2 八个文件虽然目录含46，解析后每个184原子、DOF561。
短步 qualification（1 FIRE2 +10 BFGS、12主模型调用）与完整收敛分开。
各 ID 前三步为共享 HCU 力/应力的 ASE CPU 数学及独立 HCU 对照；
Li 另做独立 CPU MACECalculator/ASE 检查，T2 独立 CPU MLIP 历史超时不记通过。

完整46/92原子双体系 CSP 三组配对是在零轮修正前完成；最终产品亦已复验
完整收敛。三组性能与最终复验的源版本分别记录，不能混成同版重复样本。
接受探针总耗时含冷初始化和独立 oracle；同步 BFGS 更新窗口才是排除这些
成本的时间。无全目录成功率或每成功结构生产吞吐结论。

固定几何12次模型重复求值（交替 stream）实际能量、力、应力差分别
2.274e-13、1.191e-14、5.499e-16；这是诊断证据，不能证明长轨迹差全部源于模型。

## 支持边界

当前准入是 HCU4 BW1000/gfx936、DTK26.04、海光
Torch2.9.0+das.opt1.dtk2604/HIP6.3.26093、FP64、eager、forward-only。
错误构建/device/dtype/layout/UPPER/nonfinite/未收敛明确失败，无静默回退。
保留主机调度、每 sweep 标量检查和必要的诊断传输；不宣称零CPU、全异步、
全部片上、一次launch或多卡安全已全面验收。

本轮不包括其他构建/硬件、DTK26.10 Torch、NVIDIA、FP32、compile、优化器
梯度、DomainParallel、完整通用采样 checkpoint、全目录 CSP 压测或 auto 准入。
不保证所有谱/尺寸/Batch 都加速，稠密 H/scratch 保持 O(d²) 内存成本。

## 最终实测结果

零轮修正后的最终源码运行14个限时作业，全部退出0；每个作业的完整命令与
实际耗时保留在 `final-*-run.json`。三轮相同冻结输入，21个稳态样本的wall
中位数（ms）如下；CSP、MLIP与oracle均不在这些微基准窗口内。

| 输入与窗口 | 旧rocSOLVER | 自写单stream | 产品（多宽双stream） |
|---|---:|---:|---:|
| Li八宽，checked solver | 212.857 | 112.422 | 74.731 |
| T2 561 B8，checked solver | 985.284 | 521.120 | 521.247 |
| Li+T2，checked solver | 1197.958 | 633.427 | 559.939 |
| Li，完整math/guard/commit | 233.596 | 194.772 | 149.094 |
| T2，完整math/guard/commit | 904.511 | 505.516 | 505.624 |
| Li+T2，完整math/guard/commit | 1137.978 | 700.081 | 589.684 |
| 合成147 B1，checked solver | 36.709 | 20.767 | 20.887 |
| 合成189 B32，checked solver | 163.165 | 185.890 | 185.995 |
| 合成561 B8（rank18），checked solver | 958.938 | 665.496 | 665.531 |
| 合成147 B1，完整math | 40.563 | 21.085 | 21.190 |
| 合成189 B32，完整math | 163.936 | 178.431 | 178.473 |
| 合成561 B8（rank18），完整math | 1042.181 | 682.423 | 682.398 |

对检查相同的自写单stream，产品Li分解减少33.53%、完整math减少23.45%；
混合分解减少11.60%、math减少15.77%。单宽T2基本持平；单宽147略增加入口开销。
189 B32相对旧rocSOLVER分解慢13.99%、完整math慢8.87%，退化保留，不设静默
回退或经验白名单。三轮原始样本、范围和冷启动见 `final-benchmark-{1,2,3}.json`；
汇总 `final-results-summary.json` 保留，未覆盖修正前 `results-summary.json`。

首轮Torch allocator额外峰值：Li旧6.51/产品9.60 MiB，T2旧67.24/产品100.26 MiB，
混合旧69.50/产品102.54 MiB。性能收益伴随更多设备scratch，不承诺统一内存优势。

最终hipprof Li solver窗口83.971ms、13824 kernels，实际不同queue同时执行
最大值2、重叠43.133ms。窗口中D2H仅87次各1字节，未观察数值矩阵D2H/H2D；
D2D共103次/22903字节。该结论只适用于此已裁剪窗口，有profiler开销，不是
全部程序零传输/零主机证据，也没有在此新增LAPACK observer。

最终Li/T2各一个fresh rocSOLVER/产品配对，仍为短步qualification：各11阶段步、
12主模型调用，八个ID各十次BFGS，前三步ASE/独立HCU通过。Li最大ASE坐标
1.954e-14 Å、cell1.777e-14 Å、H9.309e-11；T2为2.132e-14/1.421e-14/8.854e-12。
独立CPU MLIP Li另一个fresh作业退出0：5阶段步/6主模型调用，各ID前三步门通过，
ASE坐标1.777e-14、cell1.421e-14、H6.754e-11。T2独立CPU MLIP仍未验收。

| 沿真实短轨迹的同步完整变胞更新 | 旧rocSOLVER中位数ms | 产品中位数ms | 样本 |
|---|---:|---:|---|
| Li八宽 | 306.337 | 160.950 | 每条轨迹预热3次后7个不同H |
| T2同宽561 B8 | 1064.569 | 758.805 | 每条轨迹预热3次后7个不同H |

这是一轮轨迹对照，窗口内含pack/unpack/状态提交，模型和独立oracle在外；
不等价于多轮相同H微基准或生产吞吐。T2收益来自求解器变化，不能归因双stream，
其同宽入口没有跨组并行。配对ID/status相同；Li十步末态位置差7.591e-9 Å、
cell3.635e-9 Å、能量5.352e-12 eV；T2位置4.619e-14 Å。末态差只作诊断，
不把模型重复求值的微小变化当作其全部根因，不新增或放宽ASE逐步门。

最终46/92原子双体系完整CSP退出0，233阶段步/234主模型调用，分别88/213次
BFGS更新，全收敛，广义fmax为0.043552/0.048952（阈值0.05）。前三步独立
CPU MACE/ASE坐标1.244e-14 Å、cell1.421e-14 Å、H2.546e-11；实际探针窗口
79.450s。该次未使用同步分项wrapper，不混入前面的三轮同步性能样本。

零轮修正前的三组fresh配对全部同233/234且完整收敛：接受探针总耗时中位数
旧94.997s/产品81.947s（约减少13.74%）；同步BFGS更新累计中位数
30.621s/18.628s（约减少39.17%）。首次产品冷编译总耗时115.987s另记，
没有混入warm比较。迭代kernel源码不变且最终微基准/完整CSP重验通过；
仍不把修正前CSP三轮声称为最终源码三轮，也不声称全目录生产吞吐。

## 可复验命令、记录与阶段收口

```bash
source scripts/activate_hygon_env.sh project
HIP_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  scripts/check_cpu_reference.sh
# 先只读确认单卡资源；每个子作业由队列中的限时包装器执行。
HIP_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MAX_JOBS=2 \
  NVALCHEMI_HIP_BFGS_JACOBI_BUILD_DIR="$PWD/artifacts/ase-bfgs-m5-4-2026-10-10/build" \
  .venv/bin/python probes/ase_bfgs_product_run_queue.py \
    --queue artifacts/ase-bfgs-m5-4-2026-10-10/final-queue.json
```

队列仅用于查看/复验；重新运行会写其输出路径，应先为新验收复制队列并更换
输出目录，避免覆盖历史证据。`acceptance-summary.json` 汇集完整命令/退出码、
最终source/artifact/input hashes、保护源与版本锁检查、诊断失败及剩余范围。
Ruff专项和diff whitespace检查通过；结束时无HCU作业在运行。

首次catalog inventory测试因未列新增ID而失败，更新该预期后全部通过；
零轮抵消诊断退出1的前版本记录保留。没有把任何未过门改成skip/放宽数值门。
现有12个保护源/锁中，仅Batch内部数学调度和catalog变化，既有数学表达式、
Framework optimizer/Filter、rocSOLVER和上游锁hash保持；M5-3B冻结源码/输入不变。
新增产品、测试、报告及契约更新均未提交/推送/合并。

下一阶段建议做 **M1–M5交付审阅与收口**：审阅完整未提交diff、来源/许可、
用户示例、契约与证据一致性，整理按范围拆分的本地提交清单；实施前等待用户
确认，不自动向develop合并或发布。生产全目录压测和auto选择仍是独立决策。
