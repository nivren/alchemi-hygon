# ASE-compatible BFGS M5-2：三轮真实变胞 CSP 配对验收

2026-10-10；分支 `wangleping/perf-ase-bfgs-m5-2`，HEAD
`cfd22ad11fe0e1704dff48f53abee69135f8db99`，承接未提交的 M1–M4/M5-1/M5-2。
用户确认补齐 M5-2 的剩余端到端性能门。本阶段只运行已有隔离原型和分析，
产品与原型数学源码均不改，没有注册、提交、集成或推送。

**三轮配对全部通过：同步仪表化工作流中位数57.308→46.726秒，耗时下降18.466%，
约1.226倍加速。结合此前数值与微基准门，M5-2隔离原型验收完成，建议保留候选。**
这不等于产品已接入、M5整体完成或生产CSP吞吐验收。

## 输入、环境和测量边界

- 当前匹配环境：DTK26.04、HCU Torch `2.9.0+das.opt1.dtk2604`、
  HIP `6.3.26093`、rocSOLVER `3.20.0.80bb870c`，HCU4 BW1000 64G/gfx936。
  不使用DTK26.10，不安装或修改系统环境、时钟、功耗、亲和性。
- 模型为 **MACE MPA-0 medium**，FP64 energy/forces/stress，cueq/compile关闭。
  checkpoint为 `/home/wangleping/.cache/mace/mace-mpa-0-medium.model`，SHA256
  `75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638`。
- 同M4两CIF：46原子 `perf_46/formal_c1_1_100_z1_46.cif`、92原子
  `perf_v2_sorted/perf_v2_92/formal_c1_2_1000_z1_46.cif`；相对 `/data/csp_data/`。
  SHA256依次为
  `c4d09e2e187bc94eea6c11429a6c1de0b8766836acc6dc68d6f4c919cb1bc7b9`、
  `c463d1baad18cdef02ea998cf2b6ae7a0dc69807b12be1d524516e9331fb18ed`。
- 普通异构Batch、system_id/source_index=[0,1]，无inflight；Torch reference邻居、
  skin=0。固定晶胞FIRE2 20步→ASE UnitCellFilter 3N+9 BFGS，DOF147/285，
  alpha70、maxstep.2、strict广义fmax<.05、max_steps500。
- 轮次顺序rocSOLVER/原型、原型/rocSOLVER、rocSOLVER/原型。每个子进程限210秒，
  工作流限180秒；OMP/OPENBLAS各1线程，MAX_JOBS=2。
- 复用[CSP注入探针](../probes/ase_bfgs_jacobi_candidate_csp.py)及M5-1组件计时。
  原型只临时替换进程内私有native入口；沿用产品完整info/finite/残差/正交守卫。
  `actual_solver`记录真实实现，不能仅凭两边的 `backend="hip"` 元数据归因。
- `elapsed_s` 从模型构造/权重加载之后、stage.run之前开始，包含冷模型首forward、
  同步、Hooks和周期性**优化器 checkpoint 写入**；排除扩展JIT/load、模型构造/
  权重加载、计时结束后的最终文件导出。原始run-contract的“checkpoint included”
  指工作流中的状态写入；具体计时位置审计见 `environment-and-timing.json`。
  无额外ASE/独立模型求值、无profiler。这是同步仪表化工作流，不能称为普通
  生产吞吐或纯GPU kernel时间。

## 实际执行和证据

新原始目录：`artifacts/ase-bfgs-m5-2-e2e-2026-10-10/`；保留旧目录和被资源竞争
污染的序列，旧计时仍排除，不拼接到新配对。

~~~bash
source scripts/activate_hygon_env.sh project
export HIP_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MAX_JOBS=2 PYTORCH_ROCM_ARCH=gfx936
export ASE_JACOBI_PROBE_BUILD_DIR="$PWD/artifacts/ase-bfgs-m5-2-e2e-2026-10-10/build"
ulimit -c 0
python artifacts/ase-bfgs-m5-2-e2e-2026-10-10/run_pairs.py
~~~

driver在每次启动前、运行中每3秒及结束后读取 `hy-smi --showpids`，仅豁免自己
隔离子进程组。发现其他HCU4作业则停止自己的组、排除计时，不停止他人任务。
06:49:58–06:58:24 UTC共174次观察，未观察到其他HCU4进程；这是采样证据，
不保证两个观察之间绝对独占。初始检查曾有其他卡作业，不影响HCU4准入。

| 项目 | 实际退出码/结果 | 原始证据 |
|---|---|---|
| 初始资源输出解析 | 1，未启动GPU工作；补充DTK空进程输出识别后运行 | `parser-rejected.txt`、`exits.tsv` |
| 六次CSP子进程、最终driver | 全部0，没有排除的配对 | `commands.txt`、`exits.tsv`、`driver-status.json`、`driver.log` |
| CPU末态/合同/资源/hash分析 | 0，接受3配对 | `analyze_pairs.py`、`analysis.log`、`acceptance-summary.json` |
| timing统计脚本 | 0，提示每实现仅3样本 | `compare-timings.txt` |
| 产品/原型/锁定源检查 | 11产品源、6原型源、版本锁hash与开测前相同 | `run-contract.json`、`source-check.json` |

每次完整命令、日志、JSON、components原始样本、最终Batch和restart快照均保存；
环境、构建产物、资源观察及文件hash见同目录。CPU分析设置 `HIP_VISIBLE_DEVICES=''`，
加载最终Tensor到CPU，不另占用HCU。最终文件清单见 `manifest.json`。
初始解析失败是本轮driver对空KFD进程输出识别不完整，非求解器或DTK数值故障。

## 性能结果

| 配对 | 顺序 | rocSOLVER秒 | 自写HIP秒 | 耗时下降 |
|---|---|---:|---:|---:|
| 1 | rocSOLVER→原型 | 57.308456 | 46.719778 | 18.477% |
| 2 | 原型→rocSOLVER | 56.074667 | 47.167339 | 15.885% |
| 3 | rocSOLVER→原型 | 57.336177 | 46.726088 | 18.505% |
| 中位数 | 每实现3次 | **57.308456** | **46.726088** | **18.466%** |

基线范围56.075–57.336秒、population σ=.588秒；原型46.720–47.167秒、σ=.210秒。
每轮均有收益，下降大于本窗口波动；三样本符合已确认配对范围，但不能据此推断
任意CSP成功率/生产分布或统计置信区间。

同窗口组件累计时间的跨三次中位数：

| 组件 | rocSOLVER秒 | 自写HIP秒 |
|---|---:|---:|
| 模型计算 | 14.074855 | 14.024698 |
| BFGS更新 | 30.626752 | 20.558607 |
| checked eigh | 27.248225 | 17.202561 |
| eigensystem guard | .140792 | .139633 |

组件有嵌套关系，不能相加成完整工作流；native_cpu_return只到CPU返回，不等于
GPU完成时间。本轮差异主要出现在eigh/优化更新窗口，与此前微基准方向一致，
没有用更少模型求值、移除守卫或改变收敛阈值获取收益。

## 收敛、数值和末态差异

六次均退出0、两个结构status=[2,2]，共12次成功弛豫。每次233阶段步、234主轨迹
模型求值，两个ID的BFGS更新88/213次；实际分解是不同宽度的两个B1组，
不是B4端到端吞吐实验。所有301次分解的info均为0，六次共1806次，并保留
既有有限性/方程残差/正交守卫。

46原子广义fmax约.043551845；92原子基线范围.04905818–.04916030，
原型.04907746–.04916849，均严格小于.05。UnitCellFilter按广义力收敛；
物理原子fmax约.0528–.0541可以大于.05，不将其描述成原子/应力分别通过.05。

配对末态最大绝对差异：

| 体系 | 坐标Å | 晶胞Å | 总能量eV |
|---|---:|---:|---:|
| 46原子，三配对最大 | 3.972e-12 | 1.640e-12 | 3.411e-13 |
| 92原子，第1配对 | 2.376e-4 | 4.455e-5 | 3.783e-6 |
| 92原子，第2配对 | 1.407e-4 | 1.094e-5 | 3.576e-7 |
| 92原子，第3配对 | 3.386e-4 | 4.171e-5 | 8.996e-7 |

92原子同rocSOLVER三次间最大坐标/晶胞/能量差异为1.472e-3Å/1.240e-4Å/
4.958e-6eV；同原型为9.928e-4Å/8.228e-5Å/9.640e-6eV。配对差异低于这些
同后端最大重复波动，但不能从这六个末态单独确定漂移原因或保证全长轨迹一致。
每对的force/stress/volume和RMS坐标差异亦保存在analysis JSON。

本轮 `--check-steps 0`，JSON中的ASE错误零值表示未执行该对照，不能写成零误差。
严格数值资格来自此前同一源码的HCU19例、10步ASE及真实CSP每ID前3次ASE对照，
详见[M5-2数值报告](g2-ase-bfgs-m5-2.md)。本轮仅补全真实性能/成功率/计数与
描述性末态比较，不新增宽松轨迹容差；没有重跑M4 CPU650/3既有skip、HCU53全门。

## 结论和下一阶段

M5-2的数值、≥10%微基准和真实工作流可重复收益门均已具备，隔离候选保留。
产品显式HIP仍为rocSOLVER全Jacobi，默认Torch reference，能力仍为M4窄范围；
原型仅FP64 LOWER n147/285、B≤4、当前匹配Torch/DTK26.04/gfx936 eager，
每sweep主机标量同步仍存在。无NVIDIA/其他规模/其他构建/多卡/生产资格。

原先建议受限产品接入；用户随后明确要求取消147/285维与B≤4限制，下一阶段
调整为[M5-3动态尺寸与Batch扩展](../docs/ASE_BFGS_M5_3_PROPOSAL.md)：先修正
固定排序容量、配对launch覆盖与索引/批量边界，复用异构分组和全组失败不提交，
完成更广尺寸/大组数值与性能验收。扩展通过后再确认产品接入；本轮未修改
求解器代码或开始产品注册，M5-2历史验收范围不因新计划扩大。
