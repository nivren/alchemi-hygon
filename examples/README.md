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
