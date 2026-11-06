# 项目示例

示例使用工程公共接口。请先在项目根目录准备并激活
[Hygon 项目环境](../docs/DEVELOPMENT_ENVIRONMENT.md)：

```bash
source scripts/activate_hygon_env.sh project
```

## 01：构造 Batch 并计算周期邻居

[`01_batch_neighbors.py`](01_batch_neighbors.py) 展示 ASE 结构转换、组合多个体系和
计算邻居。接口来自上游 `AtomicData.from_atoms`、`Batch.from_data_list` 与
`nvalchemi.neighbors.compute_neighbors`；参考上游
[`01_data_structures.py`](../packages/framework/examples/basic/01_data_structures.py) 和
[`03_ase_integration.py`](../packages/framework/examples/basic/03_ase_integration.py)。
上述上游示例仅供阅读参考；本脚本运行时调用本工程的 `nvalchemi` 接口，
不访问或导入 `external/` 上游仓库。

无需输入文件，运行两个完整周期的小体系（CPU / Torch reference）：

```bash
python examples/01_batch_neighbors.py
```

默认截断半径是 0.5 Å，仅用于看清周期边界的演示，不是材料建模建议。预期有两个体系、
四个原子，`batch_ptr=[0, 2, 4]`，每个体系两条有向边；跨周期边的 shift 分别为
`[-1, 0, 0]` 和 `[1, 0, 0]`。边次序不作为契约。

也可传入一个或多个自己的周期 CIF，并按模型要求设置截断半径。下面是 dcu2 上两类
结构文件的路径示例，其他机器请替换为自己的路径：

```bash
python examples/01_batch_neighbors.py --cutoff 3.0 --cif \
  /data/csp_data/perf_46/formal_c1_1_100_z1_46.cif \
  /data/csp_data/catl_data/li_mp_data/mp-1222651_Li2MgGe.cif
```

输出包含设备、实际请求的 backend、截断半径、`batch_ptr`、各体系有向边数、前八条
`[中心原子, 邻居原子]` 和与之对应的整数周期位移 `shift`。COO 中原子索引属于整个
Batch；脚本检查每条边两端属于同一个体系。完整周期结构以外的输入会明确报错。
邻居列表只是几何结果，本示例不加载模型，也不计算能量或力。

在已分配或确认空闲的 DCU 上，可运行 Torch reference 或显式 HIP 邻居：

```bash
HIP_VISIBLE_DEVICES=4 python examples/01_batch_neighbors.py --device cuda
HIP_VISIBLE_DEVICES=4 python examples/01_batch_neighbors.py --device cuda --backend hip
```

这里的 `4` 是 dcu2 的设备示例；在其他机器上选择自己的设备。HIP 仅支持已准入的
full-PBC/full-list、FP32/FP64、`skin=0` 等范围，不进入 `auto`；这里固定使用 FP64、
COO 和 `cell_list`，COO 由设备内 Torch 打包。具体能力见
[后端能力矩阵](../docs/BACKEND_CAPABILITY_MATRIX.md)。

本示例只覆盖一次邻居计算；不涉及动力学中的缓存、重建生命周期或性能比较。

## 本例验证记录

2026-11-06 在 dcu2、DTK 26.04 / 海光 Torch 2.9.0+das.opt1.dtk2604、FP64 环境验证：

- CPU Torch reference：默认两个小体系，以及上文 46 原子分子晶体 + 4 原子锂化合物 CIF；
  全部有向边与周期位移分别与 ASE `neighbor_list` 对照一致，且没有跨体系边。
- HCU4（BW1000 64G）：Torch reference 和显式 HIP 均对同样两组输入与 ASE 对照一致。
  两个真实 CIF 在 cutoff=3.0 Å 时分别产生 416 和 32 条有向边。
- 非完整周期输入明确报错。本例脚本 SHA-256：`f4cd2235d1bd773c74f26b3198dc9458547024ac699763eeceefe6852af44ab9`。

首次 HIP 执行可能需要在用户级 Torch 扩展缓存中 JIT 编译，时间计入首轮运行；
本轮首次运行在编译中达到 240 秒超时，检查并清理本次留下的锁后，复用已完成
的编译部分重新运行通过。该记录不代表其他环境的编译时间或性能。
原始日志位于 dcu2 项目的 `artifacts/example-batch-neighbors-2026-11-06/`。

## 02：MACE + FIRE2 变胞结构弛豫

[`02_mace_variable_cell.py`](02_mace_variable_cell.py) 读取一个周期 CIF，使用本项目的
`MACEWrapper`、`FIRE2VariableCell`、邻居和晶胞对齐接口，优化原子坐标与晶胞并保存 CIF。
示例参考上游的 [`02_geometry_optimization.py`](../packages/framework/examples/basic/02_geometry_optimization.py)
及项目已有变胞验证；运行时不访问 `external/`。它使用 FP64、零外压以及显式
`torch_reference` 优化器、晶胞对齐和邻居后端。

需自行准备可信的 MACE checkpoint 与模型支持元素的完整三维周期结构；脚本会检查元素
覆盖，不支持时明确报错。下例中的路径是
已在 dcu2 上核对的 MACE-MPA-0 和锂化合物文件，在其他机器上请替换。模型和数据本身不随
工程分发，使用前需确认各自的来源和使用条件。

```bash
OMP_NUM_THREADS=1 python examples/02_mace_variable_cell.py \
  --checkpoint /home/wangleping/.cache/mace/mace-mpa-0-medium.model \
  --cif /data/csp_data/catl_data/li_mp_data/mp-1222651_Li2MgGe.cif \
  --output relaxed.cif --steps 100
```

在已分配或确认空闲的 DCU 上，将命令前缀替换为 `HIP_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1`，
并增加 `--device cuda`；设备编号按机器实际情况选择。也可以传入分子晶体结构文件，例如
`/data/csp_data/perf_46/formal_c1_1_100_z1_46.cif`。

默认最多运行 100 步，`--steps` 可调整。输出包括实际步数、能量、最大原子力和最大
绝对应力；两者分别不超过 **0.01 eV/Å** 和 **0.1 GPa** 才显示 `converged: True`。
步数耗尽也会保存当前结构，但若未达到双阈值，结果明确显示 `converged: False`；这时
该结构只是中间结果。输出路径由用户指定，已有同名文件可能被覆盖，应选择新路径。
阈值写在脚本中，仅供示例；具体任务需根据模型与研究目的调整。本例不提供重启、轨迹或
批量调度。晶胞对齐可能同时旋转原子坐标和晶胞，比较输入与输出坐标时应考虑坐标系变化。

2026-11-06 在 dcu2、MACE-MPA-0 medium（SHA-256
`75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638`）、FP64 下验证：

- CPU：LiSnCl3 输入 5 步、46 原子分子晶体输入 3 步均完成，均未达到双阈值，属于短流程验证。
- HCU4（BW1000）：Li2MgGe 输入在 88 步达到双阈值，最大力 0.00026039 eV/Å、
  最大绝对应力 0.09042778 GPa。
- 对导出 CIF 重新读取，并用 MACE 的 ASE calculator 在 CPU 上重算能量、力和应力，与脚本
  报告对照；以上只说明这些输入及给定条件，不代表所有材料都能收敛，也不构成性能结论。

日志、输入哈希与导出核验位于 dcu2 项目的 `artifacts/example-mace-fire-2026-11-06/`；
更完整的历史多结构变胞验收见[验证报告](../reports/g2-hip-fire2-convergence-2026-10-09.md)。

## 03：MACE 能量、力和应力单点计算

[`03_mace_single_point.py`](03_mace_single_point.py) 读取一个或多个周期结构，使用
`AtomicData`、`Batch`、`compute_neighbors` 和 `MACEWrapper` 计算能量、原子力和应力。
它参考上游 [`01_data_structures.py`](../packages/framework/examples/basic/01_data_structures.py)
的结构与邻居接口、[`04_mace_nvt.py`](../packages/framework/examples/advanced/04_mace_nvt.py)
的模型加载，以及项目已有单点验证；运行时不访问 `external/`。

准备可信的 MACE checkpoint 和完整三维周期 CIF。以 dcu2 上已核对的模型及两类结构为例：

```bash
OMP_NUM_THREADS=1 python examples/03_mace_single_point.py \
  --checkpoint /home/wangleping/.cache/mace/mace-mpa-0-medium.model \
  --cif /data/csp_data/catl_data/li_mp_data/mp-998591_LiSnCl3.cif \
        /data/csp_data/perf_46/formal_c1_1_100_z1_46.cif
```

在其他机器上替换路径；在已分配或确认空闲的 DCU 上，命令前缀可改为
`HIP_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1` 并追加 `--device cuda`。脚本使用 FP64 和
显式 Torch reference 邻居。每个结构输出一行 JSON，包括输入路径、总能量（eV）、按输入
原子顺序排列的力（eV/Å），以及 3×3 应力张量（eV/Å³，沿用 MACE/ASE 符号约定）。
可以将标准输出重定向到文件。单点计算不更新结构，也不表示结构已经弛豫。

脚本检查输入元素是否被 checkpoint 支持，不支持时明确报错；这只能验证元素覆盖，
不能保证模型适合具体材料。模型与结构需要自行提供，使用前请确认来源和使用条件。

2026-11-06 在 dcu2、MACE-MPA-0 medium（SHA-256
`75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638`）验证：
CPU 和 HCU4（BW1000）分别计算上述 LiSnCl3 与 46 原子分子晶体混合 Batch，能量、全部
原子力和全部应力分量与 CPU 上 MACE 的 ASE calculator 对照通过。该结果限于指定模型、
输入、FP64 和 Torch reference 邻居；不代表所有材料或性能已验收。日志与哈希位于
`artifacts/example-mace-single-point-2026-11-06/`。

## 04：固定晶胞 MACE NVE 分子动力学

[`04_mace_nve.py`](04_mace_nve.py) 参考上游
[`04_nve_energy_conservation.py`](../packages/framework/examples/basic/04_nve_energy_conservation.py)，
用本项目 `NVE`、`MACEWrapper` 和邻居接口运行单个周期结构的短轨迹。使用 FP64、
Torch reference 积分和邻居后端，每次按当前坐标重建邻居；初始力在第一次积分前求值。
`external/` 只作为阅读参考，不是运行依赖。

先激活项目环境，再传入可信模型及其支持元素的周期 CIF：

```bash
OMP_NUM_THREADS=1 python examples/04_mace_nve.py \
  --checkpoint /home/wangleping/.cache/mace/mace-mpa-0-medium.model \
  --cif /data/csp_data/catl_data/li_mp_data/mp-998591_LiSnCl3.cif \
  --output nve.traj --steps 10 --dt 0.5 --temperature 300 > nve.csv
```

其他机器请替换路径；在已分配或确认空闲的 DCU 上，可添加前缀 `HIP_VISIBLE_DEVICES=4`
和选项 `--device cuda`。也可使用分子晶体输入，如
`/data/csp_data/perf_46/formal_c1_1_100_z1_46.cif`。模型和结构文件需自行准备。

`--dt` 单位是 fs。`--temperature` 仅用于初始速度的 Maxwell–Boltzmann 随机采样，
固定随机种子为 42，随后移除质心平动，不强制缩放到精确指定温度；运行过程中没有温控。
因此它展示 NVE，温度会随动能与势能交换而变化。输入最好先弛豫，时间步应根据模型和
材料另外验证，不能把默认参数直接视为生产设置。

输出 ASE `.traj` 文件含初始帧与每步的坐标、速度、势能和力；CSV 标准输出含步数、
时间、势能、动能与总能量。可用 `ase.io.read("nve.traj", index=":")` 读取。
选择新的输出路径，已有同名文件会被覆盖。晶胞全程固定，坐标未主动折回主胞；邻居计算
仍处理周期边界。轨迹用于分析，本例没有实现从轨迹恢复积分器状态的重启接口。
短轨迹中总能变化很小不等于长时间能量守恒已通过。

2026-11-06 在 dcu2、MACE-MPA-0 medium（哈希同示例 03）、FP64、dt=0.5 fs 下验证：

- LiSnCl3：CPU 与 HCU4 分别运行 5 步（2.5 fs），轨迹各含 6 帧。
- 46 原子分子晶体：CPU 运行 3 步（1.5 fs），轨迹含 4 帧。
- 三条轨迹逐帧与 CPU 上 MACE ASE calculator + ASE Velocity Verlet 独立积分对照通过，
  并核对晶胞不变。坐标最大绝对差不超过 1.87e-10 Å，力不超过 4.86e-9 eV/Å。
- 相对初始帧的最大总能量变化：LiSnCl3 为 7.55e-7 eV，分子晶体为 7.82e-4 eV。
  这是指定输入的极短流程验证，不代表热平衡、长期能量守恒、长时性质或生产性能。

日志、输入哈希、轨迹和逐帧核验保存在 dcu2 项目
`artifacts/example-mace-nve-2026-11-06/`。本环境的 ASE 对
`MaxwellBoltzmannDistribution` 发出弃用提示，运行与上述数值核验均完成。
