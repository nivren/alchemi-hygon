# Stress/cell-force 基础 reference 契约

任务：`TORCH-CELL-STRESS-FORCE`；起点 `e6dd99c`；开发分支
`codex/feat-cell-stress-force`。本契约只涉及应力生成工具的验证、应力到晶胞力的转换与
framework 接线，不推进晶胞，不新增 FIRE/FIRE2、NPT/NPH 或 ASE filter 支持。

## 来源与数学语义

- 锁定 framework `4dfe3723def34df3fadb245981081ccf8c94c257`：
  `nvalchemi/models/_utils.py` 的 `prepare_strain/apply_strain/autograd_stresses`，
  `nvalchemi/dynamics/_ops/npt_nph.py::stress_to_cell_force`。
- 锁定 ops `26dbceb61e30cca80e1a5805eebeb51d7dc68fd1`：
  `nvalchemiops/dynamics/utils/cell_filter.py::stress_to_cell_force`。
- 公式修正参考 fork ops `c4c3bb56b6c6ae0d5e85b4e1ed14bd4bd4748a2a` 与
  framework `4bf8bc8084a2f588c332e0bf090978ba4ba76c4b`；只摘取转换公式和相关说明。
  不同步 fork develop 或修改上游锁定。

晶格向量是 `H` 的行，`r = s @ H`。公开 Cauchy stress 拉伸为正，
`sigma = -W/V = (dE/depsilon)/V`，`V = abs(det(H))`。
原子位置与晶胞通过同一个对称应变变形；不能只改变晶胞而把 Cartesian 位置固定，
再把该偏导当作完整 stress。

固定分数坐标时，`F_H = -dE/dH = -V * inverse(H).T @ sigma`。
单位：H/positions 为 Å，V 为 Å³，energy/virial 为 eV，stress 为 eV/Å³，F_H 为 eV/Å。
锁定上游反向的乘法顺序是 `UBF-009 / CELL-STRESS-ROW`；保留公共接口，但更正 Torch
和 legacy Warp 转换公式。legacy 数值轨迹可能变化，Warp 运行证据必须单列。

`keep_aligned=True` 保留 `[0,0],[1,0],[1,1],[2,0],[2,1],[2,2]` 六项，
其余置零，实际是下三角投影。该参数默认值保留；它不自动旋转输入晶胞。
`False` 返回完整 3×3 晶胞力。应力通常对称，转换不自行对称化、不做 atom-count
归一化、不加外压。调用方可在明确的应用契约下传入 `sigma + p_ext * I`。

## 数据、执行与梯度

| 项目 | 要求 |
|---|---|
| 输入 | stress/cell `[B,3,3]`，volume `[B]`；同一设备和 dtype |
| dtype/device | float32/float64，CPU 或 HIP Torch 的 cuda 设备；不降精度、不 CPU 回退 |
| layout | 支持 strided 非连续张量；返回新建的 contiguous `[B,3,3]`，不 alias 输入 |
| volume | 有限、严格正；调用方负责给出物理体积，算子不静默重算；autograd 将其作为独立输入 |
| cell | 有限、可逆；左手晶胞可接受，物理体积取绝对行列式；不承诺病态矩阵精度 |
| stress | 有限；不猜测 Voigt、单位、符号或自动对称化 |
| 空输入 | B=0 返回同设备/dtype 的空张量，保留可微路径 |
| 错误 | shape/非有限/非正体积/奇异矩阵 ValueError；dtype TypeError；设备不一致 ValueError |
| mutation/alias | 不改写任何输入；legacy 输出仍为独立 buffer |
| Batch | 按体系独立运算；无原子数量、拓扑、容量或跨体系归约 |
| PBC/邻居 | 转换不接受邻居/PBC；应力源必须正确处理其 PBC image shifts 与连续 geometry |
| stream/确定性 | 当前 PyTorch stream；参数合法性检查可同步 host；无随机数；不承诺跨设备逐位一致 |
| 梯度 | reference 的 stress/cell/volume 一阶、二阶；CPU gradcheck/gradgradcheck 验收 |
| custom_op | reference functional custom op，mutates_args=()，fake/meta 与 differentiable backward |
| backend | None/warp 保留 legacy 选择；reference 显式登记；未知/不支持请求明确失败 |

framework 保留原四个位置参数和旧 `torch.ops.nvalchemi.stress_to_cell_force`。
公共 Python 入口新增 keyword-only backend/selection；预解析 selection 不二次解析。
reference 接线无需导入 Warp；NPT/NPH 函数仅将依赖改为按需导入，不改变积分算法。

## 验收与证据边界

- CPU：解析周期谐键、FP64 strain/cell 中心差分，非正交/旋转/剪切/拉伸/压缩，
  full 与六项投影、异构 Batch 对逐体系、空输入、非法参数、alias、strided layout。
- 连续转换：CPU gradcheck/gradgradcheck；schema/fake/autograd/AOT opcheck 与
  CPU compile smoke 独立记录，不能泛化到完整 dynamics compile。
- 现有应力工具：小模型实际 stress loss 混合二阶反向到模型参数，不能用 detach/零梯度替代。
- MACE：缓存真实 checkpoint，周期非正交异构 Batch；FP64 stress 与 affine strain energy
  中心差分对照，并实际调用 reference 转换；FP32 验证 forward 与 CPU/HCU parity。
  不推进晶胞、不训练真实 MACE。
- HCU：只读确认资源后，仅 HCU4 限时 smoke，FP32/64 CPU/HCU parity、短真实 MACE
  与非默认 stream；无资源写 pending，不等待。不因 CPU 通过标记 HCU verified。
- 数值容差在测试/探针中按物理量/dtype 定义；finite difference 必须避开 cutoff 拓扑跃迁。

本契约不证明 LJ stress、完整 MLIP 训练、变胞算法/收敛、晶胞对齐/打包、neighbor
变胞生命周期、完整 checkpoint、DomainParallel、Warp/NVIDIA、生产 HIP/Triton 已通过。
