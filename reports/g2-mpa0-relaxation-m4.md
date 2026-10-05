# M4：MPA-0 原生变胞批量弛豫工作流

日期：2026-10-05。分支 `codex/feat-mpa0-relaxation-workflow`，起点 `317b988`，实现/测试/契约 `7a8af50`。
范围为普通单优化器 FusedStage 的变胞邻居、联合收敛、退出/scalar 补位与薄 CIF 应用。
目标模型 HCU 留待 M5。本分支未合并、未推送。

## 实现与来源

- `BaseDynamics` 的四种掩码入口保护全部私有系统状态：cell velocity、dt/alpha、
  功率计数及 scratch。Batch 字段按 storage level 选择掩码，保护 N=B 单原子情况；
  在 observer Hooks 前恢复退出字段，FusedStage 共享模型评估也恢复已退出 E/F/stress。
- `FusedStage.refill_check` 改变组成后调用原 compute Hooks、重算 E/F/stress，
  不计优化步；priming 前更新 active_batch，计算异常时可导出真实到达体系。
  原 index_select/append 状态同步、sampler ID 和 sink API 保留。
- `examples/relax_mace_mpa0.py` 读取/校验 CIF，配置原生 sampler、优化器、Hooks、
  FusedStage 和 HostMemory。source_index 与 admission-order system_id 分开；
  每体系计步，收敛与步数/墙钟预算分别记录。最终 CIF 重读后独立建邻居和模型评估，
  记录实际导出坐标系的 E、体积、最大力和最大绝对应力、输入/输出/模型哈希与参数。
- 复用原生多条件 ConvergenceHook：fmax<=0.01 eV/Å **且**
  max(abs(stress))<=0.1 GPa（0.0006241509 eV/Å³）。可通过应用参数配置。
  压缩应力同样用绝对值检查。部分/混合占位、非法/超容量输入明确记录。
- 退出类别为 converged、max_steps、wall_timeout、invalid_input、calculation_error、
  not_started、final_check_failed；budget 不标成功。最后核验失败也不保留成功状态。

NVIDIA framework `4dfe372` 与 ops `26dbceb` 的锁定不变。
[UBF-015/016](../docs/UPSTREAM_BUGFIXES.md) / [LP-081–083](../docs/UPSTREAM.md) 区分：
fork framework `b12b54c` 的私有状态保护/admission priming、旧 schema 下本地窄适配及
observer 修正、本地应用脚本。没有新增晶胞参数化、ASE filter 或优化算法。
没有同步 fork 的新版 admission/schema/compile/训练依赖。

实施中应用回归发现预算 Hook 放在自动迁移 Hook 前会丢失退出原因，最终核验也必须
直接消费 model outputs（compute 不创建缺失的 Batch 输出字段）；两处为本地接线问题，
已修正并由应用/异常回归保护，不登记为新的上游缺陷。

## CPU 门与邻居/工作流证据

`HIP_VISIBLE_DEVICES='' bash scripts/check_cpu_reference.sh` 退出0：
ops149 + import1 + framework230 + state22 + dynamics-ops24 = **426 passed、3 skipped**。
三项 skip 为已有 Warp NHC 对照，不是 M4 新增 skip；framework/ops 分进程。
compileall、定向 Ruff、git diff --check 通过。

新增33项：四种 mask 入口、非零晶胞运动后的私有状态/输出冻结、observer、N=B、
scalar refill survivor/fresh-state、向量 arrivals 拒绝、首步 E/stress priming、来源 ID
唯一性、解析势联合收敛、真实 CIF I/O/JSONL/结果指标、步数/墙钟/计算与 priming
异常、部分占位。原退出输出回归也加入正式 gate；已有 NHC 生命周期门继续通过。

邻居独立 image oracle 用小晶胞枚举 [-4,4]^3，覆盖3Å cutoff 两侧3.01/2.99Å、
压缩/剪切、多周期镜像、capacity 增长和 skin=0/0.4。skin>0 原有晶胞变化缓存
失效逻辑通过；邻居源码没有新增替代实现。实际应用默认 skin=0、full periodic COO。

原始证据：`artifacts/m4-workflow/cpu-gate-final.log`。

## HCU4 合成验证

先只读 hy-smi 确认空闲，`HIP_VISIBLE_DEVICES=4`、OMP=1、timeout150秒；退出0。
BW1000 64G / gfx936 / 项目 Torch2.9.0 / HIP6.3.26093 / DTK26.04。
`probes/variable_cell_workflow_reference.py --device cuda` 对照 CPU，4个 `[2,3,3,2]`
异构体系从最大活跃2体系中补位，联合收敛后 sink 保存每个来源/ID 恰好一次。
检查当前 E/stress priming、无跨体系邻居、晶胞实际在HCU、非零私有状态后的退出冻结。

| 优化器 | dtype | fused 总步数 | 最大绝对应力 eV/Å³ |
|---|---|---:|---:|
| FIRE2 | FP64 | 181 | 9.99424e-5 |
| FIRE | FP64 | 84 | 9.85580e-5 |
| FIRE2 | FP32 | 181 | 9.99333e-5 |
| FIRE | FP32 | 84 | 9.85509e-5 |

CPU/HCU 步数相同，positions/cell/energy/stress parity 按 FP64 rtol/atol=2e-11、
FP32=3e-5 检查。没有放宽既有单步门。这是合成工作流四组，**HCU_MPA0=not_run**。
现有 NeighborListHook 的 Dynamo wrapper 出现 recompilation-limit 告警；本轮未验收
compile 或性能，实际 operation family 仍为显式 torch_reference。
原始证据：`artifacts/m4-workflow/hcu4-synthetic.log`。

## 实际 MPA-0 CPU

Checkpoint 为可信本地 `mace-mpa-0-medium.model`，SHA256
`75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638`。
MACE0.3.15/e3nn0.4.4、ordinary eager，cueq/compile_model=False。
每个作业匹配进程默认 dtype/权重/输入，并在结束后恢复默认 dtype。

三个 M1 真实 CIF：NaLiTl2(4)、Li2MgGe(4)、LiSnCl3(10)，目录排序 `[4,4,10]`，
sampler 首批 `[4,10]`，退出后补入剩下4原子体系。max_atoms=20、max_batch_size=2、
dt=0.02、max_steps=300/体系、wall=120秒；两算法保留各自上游默认控制参数。
下表应力/力来自重新读取实际导出 CIF 的独立模型核验。

| 优化器 | 结构 | 状态 | 步数 | fmax eV/Å | max(abs(stress)) GPa |
|---|---|---|---:|---:|---:|
| FIRE | NaLiTl2 | converged | 31 | 0.00287861 | 0.0412495 |
| FIRE | Li2MgGe | converged | 35 | 0.00149852 | 0.0928081 |
| FIRE | LiSnCl3 | converged | 80 | 0.00817191 | 0.0291459 |
| FIRE2 | NaLiTl2 | converged | 53 | 0.00902073 | 0.0570618 |
| FIRE2 | Li2MgGe | wall_timeout | 65 | 0.000596887 | 0.256783 |
| FIRE2 | LiSnCl3 | wall_timeout | 118 | 0.00298434 | 0.171711 |

两个 FIRE2 budget 结果应力仍超阈值，保存结果并保持 wall_timeout。
FIRE2 初轮 Batch 输出指标来自写出前坐标；独立 export probe 核验实际 CIF 坐标系。
NaLiTl2 用最终入口分别重跑：FIRE31步、FIRE2 53步，重新读写后核验成功。
FIRE 三样本也用最终入口完整复跑，31/35/80步全部收敛，JSONL 使用导出坐标系指标。
FP32 两种优化器另做三样本各2步，全部 max_steps；该证据只证明短步/补位/导出，
没有验收 FP32 完整收敛。两算法可能给出不同轨迹/局部极小值，不要求它们能量相同。

`probes/mace_mpa0_export_verify.py` 单独加载模型/读取结果文件，重建当前邻居、
检查实际输出 CIF 的14次单点，确认所有 converged 仍满足双阈值。FP64 最大能量差
8.89e-16 eV、fmax差1.23e-14 eV/Å；FP32 两个标量差均为0（同设备/同精度重评）。
这些差值是模型输出一致性证据，没有材料预测准确率或跨设备逐位一致声明。

原始目录：`artifacts/m4-workflow/mpa0-{fire,fire2}-fp64*/`、
`mpa0-{fire,fire2}-fp32/`；每个目录含 CIF、results.jsonl，JSONL 记录来源与哈希。
独立核验：`artifacts/m4-workflow/export-verify-final.log`，FIRE Batch 使用
`mpa0-fire-fp64-batch-final/results.jsonl`，两份独立单体系使用 `mpa0-*-fp64-final/`。

## 使用与下一步

```bash
source scripts/activate_hygon_env.sh project
HIP_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 .venv/bin/python examples/relax_mace_mpa0.py \
  --input-dir /data/csp_data/catl_data/li_mp_data --limit 3 \
  --output-dir artifacts/my-mpa0-smoke \
  --checkpoint /home/wangleping/.cache/mace/mace-mpa-0-medium.model \
  --optimizer fire2 --device cpu --dtype float64 \
  --max-batch-size 2 --max-atoms 256 --max-steps 300 --max-wall-seconds 120
```

output-dir 必须是新目录；选择小样本和预算后启动。应用预读输入、HostMemory 暂存全部
毕业结果，内存随输入数增长；墙钟在步间检查，最终导出/核验可以略超预算。
scalar refill；向量 arrivals 来源映射明确拒绝，普通向量 Batch 保留。
未验收：目标 MPA-0 HCU、FP32完整收敛、全目录成功率/吞吐、通用流式服务或重启、
多种变胞优化器混合阶段、NPT/NPH、ASE filters、完整checkpoint、训练/compile、
Warp/NVIDIA、DomainParallel、生产HIP/Triton。M4 没有扩大 native HIP neighbor 能力。

M5 待用户确认：用同一入口在 HCU4 做真实目标单体系→异构小 Batch→可控扩大批量，
先 FP64 再 FP32；按每体系双阈值、来源/ID、补位、实际导出单点及 CPU/HCU 物理量
对照验收，记录真实耗时/显存及未收敛原因。全目录长作业另行授权。
