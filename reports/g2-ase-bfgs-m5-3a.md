# ASE-compatible BFGS M5-3A：动态尺寸与 Batch 扩展

2026-10-10；分支 `wangleping/perf-ase-bfgs-m5-3a`，HEAD 仍为
`cfd22ad11fe0e1704dff48f53abee69135f8db99`。用户确认本阶段，并提供含锂晶体与
T2 数据目录。承接未提交的 M1–M4/M5-1/2；没有提交、推送或集成。

**动态原型实现与窄数值验收完成：取消 147/285 白名单和 B≤4 上限，支持实际
n/B；代表测试达到 n=1113、B=32，以及真实 561维 B8。性能随矩阵和规模变化，
部分合成案例退化，不能把扩展数值支持写成普遍加速。**
11个 BFGS/Filter 产品源与上游锁 hash 不变，当前产品 HIP 仍使用 rocSOLVER
Jacobi。本阶段是 NVIDIA 上游没有的本地新增求解器实验，未注册产品后端。

## 实现

[动态合同](../docs/HIP_BFGS_JACOBI_DYNAMIC_CONTRACT.md)先于实现建立。
[HIP 源码](../probes/native/ase_jacobi_candidate.cu)保留中心化、原始 norm 的
1e-15 收敛阈值、100 sweeps、双缓冲、逐矩阵冻结和原始旋转算术：

- 矩阵偏移/乘积改为 int64，配对及元素 launch 覆盖整个 Batch；按实际设备
  grid 上限使用 grid-stride。修复旧单256线程块对大维度配对覆盖不足的问题。
- 固定512槽排序替换为当前海光 Torch 的设备内 stable sort，int64列置换；
  同时重排 V。等值保留原始列顺序，不添加 epsilon。513维 trace 见后文。
- [Python入口](../probes/ase_bfgs_jacobi_candidate.py)接受动态连续 FP64 LOWER
  `[n,n]`/`[B,n,n]`，可选 `chunk_size` 控制临时工作区；它不是逻辑 B 上限。
  分块仍调用 batched kernel，设备内收集完整输出；OOM/不可表示大小明确失败。
- 保留输入、dtype、设备/current stream、无grad和 sweep 检查；矩阵不回 CPU。
  主机每 sweep 诊断标量同步仍存在。没有恢复故障 DC，也没有 clamp、阻尼、
  曲率跳过、线搜索或改变 ASE 的 abs 谱、maxstep、State/Filter。
- 等维组内批量执行；不同宽度组仍在同一 stream 串行。Framework 的整 Batch
  模型求值、独立 ID State、全组成功后提交保持原实现。

旧三份原型源码在 `artifacts/ase-bfgs-m5-3a-2026-10-10/original-m5-2/` 冻结。
147/285的 B1/B4 四组，w/V/info/residual/sweeps 五项输出与冻结原型逐位相同。
历史合同保留；当前限制以动态合同为准，不把验收尺寸列表写回 API 白名单。

环境为 DTK26.04、海光 Torch `2.9.0+das.opt1.dtk2604`、HIP6.3.26093、
rocSOLVER3.20.0.80bb870c、物理 HCU4 BW1000 64G/gfx936、O2无fast-math。
CPU/HCU资源、JIT冷加载与预热执行分开记录；没有升级系统或依赖。

## 测试与数值

原始目录 `artifacts/ase-bfgs-m5-3a-2026-10-10/`。

| 验收 | 实际结果 | 原始证据 |
|---|---|---|
| CPU 配对/排序/oracle | 退出0；18种维度全部无重复配对且完整覆盖，独立CPU FP64 32矩阵oracle | `oracles.json`、`cpu-prepare.log` |
| 旧HCU回归 | 退出0；19例及13非法输入拒绝，含真实285回归矩阵、简并/负/零谱、UPPER污染、B4冻结/stream | `old-correctness.json` |
| 动态HCU | 退出0；32矩阵 + 4个 B8/16/32 案例；chunk3 与整批五项输出逐位相同 | `dynamic-correctness.json` |
| stream/原子性 | 513维非默认stream；真实非收敛、非法chunk拒绝；B8末块失败后全部坐标/历史/H/计数不提交 | 同上 |
| 冻结旧原型对照 | 退出0；147/285 B1/B4 四组五项输出逐位相同 | `frozen-parity.json` |
| 十步ASE数学 | 退出0；旧147/285/147和新增189维B8 + 513/561/285混合；坐标/H/谱/raw/applied step均逐步对照 | `bfgs-old.json`、`bfgs-dynamic.json` |
| restart/单与Batch/异常 | 逐位restart、单与Batch一致，零谱/真非收敛/错误有限V拒绝且状态不提交 | 同上 |
| CPU专项回归 | 退出0；Ops104 + Framework105，无新增skip | `cpu-ops.log`、`cpu-framework.log` |
| Framework HCU专项 | 退出0；14 passed，其中8项显式HIP分支注入动态原型，另6项Torch对照；小体系freeze/stream/配对restore/旧单体系提升 | `hcu-framework.log` |

动态矩阵 n 为 1/2/3/7/138/147/189/199/200/201/285/511/512/513/552/561/
1104/1113。B子集包括旧B4、新147 B8、189 B16、285 B8、199 B32；另有真实561 B8。
不是全部 n×B 笛卡尔积；276及大尺寸B32未在本阶段单独验收。

旧+新增矩阵最大谱误差 **3.695e-13**、方程残差 **4.935e-15**、正交误差
**2.956e-13**。保持谱 atol2e-10/rtol2e-11、方程/正交1e-10的既有门。
新增11体系十步ASE最大误差：坐标 **2.159e-14**、H **5.684e-14**、谱
**1.683e-11**、raw step **3.572e-14**、applied step **6.307e-15**。
没有放宽 ASE 逐步容差。

CPU回归最初合并两个包运行，因同名 `test.conftest` 在收集阶段失败，实际退出4/1；
分进程后104/105通过。分析脚本也修正了本机profiler字符串ID解析和锁路径；失败
日志保留，最终 `analysis-final-with-api.log` 通过。这些不是 solver 数值失败。

## 用户数据与真实 MLIP

模型均为 **MACE MPA-0 medium**，不是 mace_off small；FP64、energy/forces/stress，
cueq/compile关闭，Torch reference neighbor、skin=0。checkpoint SHA256：
`75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638`。

目录包含21755/600个CIF，实际只盘点前160个含锂文件并选择以下样本，不声称
全数据集成功率。文件路径/hash/元素/晶胞/ASE实际原子数见 `real-inputs.json`：

| 数据 | 独立CIF | ASE实际原子数 | 变胞自由度/组 |
|---|---|---|---|
| `/data/csp_data/catl_data/li_mp_data` | 8 | 2/4/7/12/16/28/64/160 | 15/21/30/45/57/93/201/489，各B1 |
| `/data/csp_data/T2_46` | 8 | 各184 | 561维 B8 |

T2文件名带46，但 ASE 按CIF解析出的周期晶胞有184个原子，未手动复制扩大。
两组共16个独立输入；各加入标准差0.002Å的诊断位置扰动，种子7300+i，原文件
只读不修改。一次FIRE2步后四次BFGS更新，各ID前3步对照，共6次主工作流模型
求值；独立对照模型调用另列、不计入这6次。

- 含锂：独立 **CPU MACECalculator + ASE** 与独立HCU单体系均通过。
  ASE坐标/晶胞/H最大误差 **1.776e-14 / 1.421e-14 / 6.759e-11**；独立HCU
  坐标/H **4.441e-15 / 8.725e-12**。见 `li-csp.json`。
- T2：最初8体系独立CPU MLIP三步对照达到300秒限时，实际退出 **124**，
  `t2-csp.log` 保留，不登记该CPU MLIP门通过。后改为**同一HCU模型力/应力
  输入的 ASE CPU FP64数学对照**，并继续独立HCU模型/单体系对照，退出0。
  ASE坐标/晶胞/H **2.132e-14 / 1.421e-14 / 8.853e-12**；独立HCU
  坐标/H **7.105e-15 / 6.537e-13**。实际有四次 `[8,561,561]` 分解；
  最后八个真实Hessian不同。见 `t2-shared-csp.json`。

两组状态均为 `passed_step_qualification`，没有完整收敛，不用于端到端吞吐或
CPU/HCU模型加速结论。短步采用fmax=1e-6保证进入更新，沿用既有逐步误差门；
T2的共享力oracle不等价于CPU/HCU MLIP输出一致性。

## 性能：三进程 × 七次交替样本

扩展编译/加载与模型初始化不计入微基准；每项先记录首次调用并预热2次，再
交替执行已检查的相同输入。下面中位数均为21个稳态wall样本。原始cold、每进程
wall/stream时间、std/min/max和oracle见 `old-benchmark-{1,2,3}.json`、
`dynamic-benchmark-{1,2,3}.json`；不是所有输入的加速资格。

| 输入 | rocSOLVER Jacobi ms | 动态原型 ms | 耗时下降 |
|---|---:|---:|---:|
| 147 B1，旧真实捕获H | 40.604 | 17.383 | 57.19% |
| 147 B4，相同/不同矩阵 | 46.553 / 44.235 | 21.526 / 20.050 | 53.76% / 54.67% |
| 285 B1，旧真实捕获H | 92.775 | 46.362 | 50.03% |
| 285 B4，相同/不同矩阵 | 115.969 / 101.389 | 83.786 / 66.152 | 27.75% / 34.75% |
| 189 B16，合成同副本 | 43.112 | 59.965 | **-39.09%** |
| 199 B32，合成同副本 | 120.167 | 112.539 | 6.35% |
| 513 B1 / B8，合成同副本 | 133.561 / 401.101 | 109.012 / 499.306 | 18.38% / **-24.48%** |
| 561 B1 / B8，合成同副本 | 128.254 / 447.562 | 134.406 / 648.061 | **-4.80% / -44.80%** |
| 含锂八个真实H组，串行分解窗口 | 212.038 | 112.720 | 46.84% |
| T2八个不同真实H，561 B8分解窗口 | 985.281 | 521.498 | 47.07% |

旧组147/285/147完整直接数学窗口 **136.958→69.263ms**；旧微基准包含
有效性守卫时最小下降 **27.62%**，旧10%门保留。真实H分解窗口不包括MLIP、
Framework、H更新或完整弛豫；不同宽度仍串行，不能称为异构kernel并发收益。

513 B8分成chunk3后 **499.335→619.796ms**，约慢24.13%。分块是显式工作区
选项，不是普遍性能优化。合成513 B8的Torch allocator额外峰值：基线56.88MB、
原型84.40MB；561 B8分别70.51/105.06MB。这只统计Torch allocator，不能
据此比较底层库全部外部workspace或设备总峰值。真实测试巡检观察到含锂
5360MiB、T2 13986MiB进程显存读数，均不是完整峰值测量。

## 完整 CSP 回归与 trace

旧46/92原子输入、未扰动、FIRE2 20步→UnitCellFilter BFGS、fmax=.05，当前
rocSOLVER/动态原型新鲜配对各一次，全部退出0并收敛，均233阶段步/234模型求值，
ID更新88/213次。同步窗口 **57.649 / 47.128秒**；只做扩展回归，不用一对样本
替代M5-2的三轮端到端性能门。check-steps=0，误差字段0表示未执行该比较，
本次完整回归没有额外ASE检查；逐步门由前述测试承担。
末态配对坐标最大差异 **1.577e-4Å**、cell **4.654e-5Å**、能量
**3.916e-6eV**；一对运行不能估计扩展版的重复波动。

513 B1 checked solver的hipprof窗口 **112.767ms**，10315个kernel，kernel
union **109.838ms**。明确出现 `radixSortKVInPlace` 和 `sort_vectors`；稳定排序
在HCU执行。窗口内HIP API的D2H全部为1字节诊断标量，4104字节复制是D2D；
没有数值矩阵回传。LAPACKE_dstedc/dstedc_work observer无调用记录；此observer
只覆盖两个符号，不据此声称所有未受监测CPU私有函数均不存在。GPU矩阵数学与
显式主机诊断/下发分别记录，不声称单次launch、完全片上或全异步。

后半段任务186次资源观察，每次最多一个进程占用HCU4，未观察到竞争；前段
每次执行前及真实测试中也有巡检记录。这是采样观察，不是独占设备预约。
完整命令数组、每项timeout/实际退出码见 `remaining-commands.json`；监控为
`remaining-resources.jsonl`，最终统计为 `summary.json`，trace保留DB/pftrace/CSV。

## 复现与边界

从项目根激活环境。HCU设置 `HIP_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1 MAX_JOBS=2`；`ASE_JACOBI_PROBE_BUILD_DIR` 指向本阶段
`artifacts/.../build`，CPU oracle/回归使用 `HIP_VISIBLE_DEVICES=-1`。

```bash
source scripts/activate_hygon_env.sh project
.venv/bin/python probes/ase_bfgs_jacobi_dynamic_test.py --mode prepare --original artifacts/ase-bfgs-m5-2-2026-10-10/oracles.pt --output artifacts/ase-bfgs-m5-3a-2026-10-10/oracles.pt
.venv/bin/python probes/ase_bfgs_jacobi_dynamic_test.py --mode correctness --dataset artifacts/ase-bfgs-m5-3a-2026-10-10/oracles.pt --output artifacts/ase-bfgs-m5-3a-2026-10-10/dynamic-correctness.json
.venv/bin/python probes/ase_bfgs_jacobi_dynamic_csp.py --manifest artifacts/ase-bfgs-m5-3a-2026-10-10/real-inputs.json --input-set li --device cuda --backend hip --variable --qualification-only --perturb-positions .002 --fmax .000001 --fire-steps 1 --check-steps 3 --max-steps 5 --max-wall-seconds 180 --output artifacts/ase-bfgs-m5-3a-2026-10-10/li-csp.json
# T2共享力数学oracle额外使用 --input-set t2 --ase-shared-forces。
# 其独立CPU MLIP超时日志不覆盖；性能命令及各项界限见remaining-commands.json。
```

复现应使用新的artifact目录，保留历史结果。原始输入、checkpoint、最终源码和
产物hash/实际命令与退出码另存 `acceptance-manifest.json`。Git diff/新增文件空白、
probe语法和文档链接完成检查；未重跑全项目CPU650/HCU53，CPU209/HCU14属于本次
专项回归，不扩大为完整项目门。

仅当前匹配构建、FP64 LOWER、gfx936、forward-only eager；其他DTK/Torch、NVIDIA、
dtype、UPPER、grad/compile、多卡/多线程压力、全集CSP成功率和生产吞吐未验收。
任意n/B接口仍受整数表示与显存/计算资源约束。不同宽度组仍串行，原型未产品注册。
下一阶段为[M5-3B异构并行对照](../docs/ASE_BFGS_M5_3B_PROPOSAL.md)，等待用户确认。
