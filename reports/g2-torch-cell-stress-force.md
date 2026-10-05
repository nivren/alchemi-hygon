# Stress/cell-force 基础 Torch reference 验证

日期：2026-10-04。分支 `codex/feat-cell-stress-force`，起点 `e6dd99c`；
实现与测试已提交为 `a76107e`；文档独立提交。未合并到 develop、未推送。

本轮完成给定晶胞上的应力生成工具验证、应力到晶胞力的 reference 与 framework 接线。
用户明确排除晶胞及 FIRE/FIRE2 移植；本结果不构成变胞 MD 或变胞弛豫支持。
契约：[TORCH_CELL_STRESS_FORCE_CONTRACT.md](../docs/TORCH_CELL_STRESS_FORCE_CONTRACT.md)；
决策：[ADR 0011](../adr/0011-stress-cell-force-reference.md)。

## 实现与来源

- framework 锁定 `4dfe3723def34df3fadb245981081ccf8c94c257`；ops 锁定
  `26dbceb61e30cca80e1a5805eebeb51d7dc68fd1`。锁定与 `external/` 均未改变。
- 新 ops [stress_reference.py](../packages/ops/nvalchemiops/stress_reference.py)：
  functional custom op、fake/meta、一阶与可再次求导的 backward；输入同设备同 dtype，
  FP32/64，支持 strided/empty，输出不 alias，不改写输入，不静默 CPU 回退。
- 新 framework [_ops/stress.py](../packages/framework/nvalchemi/dynamics/_ops/stress.py)：
  按 operation 单次解析 selection，经 catalog 声明的 ops-owned executor 执行。
  旧 `npt_nph` import、四位置参数、默认 keep_aligned 和 legacy `torch.ops` 名称保留；
  新增 keyword-only backend/selection。None/warp 仍选择 legacy，未知/不支持请求明确失败。
- 原 `npt_nph.py` 的 Warp/barostat 导入下沉到调用内，reference 无需安装 Warp。
  相对起点的 AST 比较确认：忽略 import 后，其余 24 个函数完全一致。
  没有移植或修改 NPT/NPH 积分算法；legacy 导入隔离的数值执行未验证。
- 保留已有 `prepare_strain/apply_strain/autograd_stresses`，不重写模型应力管线。

### 行晶格公式修正：UBF-009

行晶格约定 `r=s@H` 下，固定分数坐标晶胞力应为 `F_H=-V H^-T sigma`。
锁定 ops 原为 `-V sigma H^-T`，非正交晶胞上不等价。
reference 与本地 legacy Warp kernel 均修正，代码标记 `CELL-STRESS-ROW / UBF-009`；
同时澄清 tensile-positive stress 与实际下三角六项投影。
完整记录见 [UPSTREAM_BUGFIXES.md](../docs/UPSTREAM_BUGFIXES.md#ubf-009行晶格的-stress-to-cell-force-乘法顺序错误)，
补丁索引为 LP-072/073。默认后端选择保留，legacy 晶胞力及后续轨迹可能改变。

参考用户指定 fork 的局部修复：

| fork | 本轮读取的 develop SHA | 采用的相关修复 |
|---|---|---|
| nvalchemi-toolkit-ops | `eadc3eeebaeceb832f7341eb59303dc588f857bd` | [c4c3bb56b6c6](http://10.208.16.64:3000/wangleping/nvalchemi-toolkit-ops/commit/c4c3bb56b6c6ae0d5e85b4e1ed14bd4bd4748a2a) 的转换顺序与说明 |
| nvalchemi-toolkit | `b12b54c48fc8001adb84654c3b42f2bef21db710` | [4bf8bc8084a2](http://10.208.16.64:3000/wangleping/nvalchemi-toolkit/commit/4bf8bc8084a2f588c332e0bf090978ba4ba76c4b) 的 wrapper 说明 |

只采用本任务相关的数学修正，未整体同步 fork develop，也未采用其 FIRE/FIRE2、晶胞更新、
仿射映射或优化器归一化改动。fork 修复状态不代表 NVIDIA 官方当前分支状态。

## CPU 测试

新 focused 回归共 26 项：

- ops [test_torch_stress.py](../packages/ops/test/torch/test_torch_stress.py) 15 项：
  非正交周期谐键独立解析值、FP64 九项晶胞力中心差分、完整/投影、FP32/64、strided、
  Batch 对逐体系、alias、empty/zero、非法输入；两种投影的 gradcheck/gradgradcheck；
  schema/fake/autograd/AOT opcheck 和 CPU `aot_eager/fullgraph` compile。
  旧乘法顺序对 oracle 的最大偏差超过 `0.1 eV/Å`，未通过放宽容差遮盖。
- framework [test_cell_stress_force_reference.py](../packages/framework/test/compatibility/test_cell_stress_force_reference.py)
  11 项：旧 import/torch.ops、无 Warp 导入、现有 strain/stress 工具的独立谐势验证；
  实际 `StressMSELoss` 通过混合二阶反向到小模型参数，梯度与解析值一致且非零；
  selection 只解析一次、显式错误、legacy 选择，以及仅登记 metadata 的第二 executor
  可经公共 API 执行。

现有 upstream stress 工具选择集：`12 passed, 1 skipped, 66 deselected`；
跳过的 halo stress 管线测试需要 CUDA，本轮 CPU 选择集不宣称覆盖该项。

完整 `scripts/check_cpu_reference.sh` 退出0，分进程结果：

| 进程 | 结果 |
|---|---|
| ops registry/reference/cell-list/stress/NHC | 82 passed, 3 skipped, 1 warning |
| 独立 framework lazy import | 1 passed |
| framework compatibility、Hooks、LJ | 168 passed, 1 deselected |
| state management NVE/FIRE | 22 passed, 34 deselected |
| VV/FIRE/FIRE2 ops | 24 passed, 73 deselected |

合计 **297 passed, 3 skipped**；三项跳过因 Warp 未安装。warning 来自既有测试路径。
compileall 与 `git diff --check` 通过；修改的 Python 文件通过 Ruff check，新文件及
`npt_nph.py` 通过 Ruff format check。Feature Contract 可解析且 stable ID 不重复。
补充 CPU 手工 smoke：公共 framework 入口在显式 backend 和预解析 selection 两种调用下，
`aot_eager/fullgraph` 的 forward/backward 均通过，日志为 `public-compile.log`；
该两项不计入 pytest 的297项。

2026-10-04 为 MPA-0 里程碑1收口基础时，完整 CPU 门再次退出0：
`82 passed, 3 skipped`、`1 passed`、`168 passed, 1 deselected`、`22 passed`、
`24 passed`，合计297项通过；应力管线另为 `12 passed, 1 skipped`。
AST 复核再次确认24个非 stress 函数除 import 外不变。原始日志见
`artifacts/mace-mpa0-m1/cpu-gate.log` 与 `stress-pipeline.log`。
本次未重跑 HCU，以下 HCU4 证据仍属于原基础 slice，不是 MPA-0 验证。

初次完整门采用180秒限时，在已有 Inductor scatter 编译测试处超时，退出124，无测试失败结论。
随后固定单编译线程，该测试单独通过（4.61秒）；以相同测试和数值容差重跑完整门通过。
没有更改编译测试或将其 skip。记录的是有效运行配置，不将初次超时归因为产品 bug。

## 单卡 HCU 与真实周期 MACE

环境：project `.venv`、DTK26.04、PyTorch2.9.0/HIP6.3.26093；
`HIP_VISIBLE_DEVICES=4`，BW1000 64G，`gfx936:sramecc+:xnack-`。
运行前 `hy-smi --showpids --showmemuse` 正常，各卡显示0%显存；有一个未关联卡索引的
KFD PID、0 MiB VRAM，不据此推断节点完全独占。HCU4 可用，仅暴露该卡运行120秒限时探针。
探针退出0，结束后各卡仍0%显存、无 KFD PID，管理接口正常。

[cell_stress_force_reference.py](../probes/cell_stress_force_reference.py) 的 CPU/HCU 两次运行均退出0：

- FP32/64 × keep_aligned 两种模式 × 五种应力（零、拉伸、压缩、剪切、混合），
  共4组各5体系的转换和 stress/cell/volume 一阶反向 CPU/HCU parity 通过；
  转换前向最大误差均为0。非默认 stream 与设备不一致显式拒绝通过。
- 使用真实 `MACE-OFF23_small.model` 缓存 checkpoint，SHA256：
  `165cce4cfec5a34b9c64d4ebf95de15d71106bb584b7291c8470f0749977c46f`。
  H2O/CH4 异构 Batch `batch_ptr=[0,3,8]`，周期非正交晶胞；108条邻居边、
  84个非零 image-shift 分量，使用显式 Torch reference 邻居与连续 geometry。
- FP64 对六个独立对称应变分量作能量中心差分（步长 `1e-5`）；另对两个非对角
  晶胞分量作固定分数坐标中心差分，实际调用 public stress 转换。
  每次差分核验 canonical atom/image edge 集合不变；输入 Batch 的晶胞/坐标未被修改。
- FP32 MACE 验证 forward 和 CPU/HCU parity。其大原子参考能量导致临近 FP32 总能量
  相减噪声较大，未作为 stress 有限差分 oracle，也未放宽容差；数值导数由 FP64 验收。

最大绝对误差：

| 物理量 | CPU | HCU4 |
|---|---|---|
| FP64 MACE stress 对有限差分，eV/Å³ | `1.92824e-10` | `1.92824e-10` |
| FP64 cell-force `[1,0]` 对有限差分，eV/Å | `3.24484e-9` | `3.24484e-9` |
| FP64 cell-force `[2,1]` 对有限差分，eV/Å | `8.79918e-9` | `8.79918e-9` |

| MACE CPU/HCU parity | stress，eV/Å³ | cell-force，eV/Å |
|---|---|---|
| FP32 | `8.19564e-8` | `1.66893e-6` |
| FP64 | `9.71445e-17` | `2.04697e-15` |

有限差分门：stress `rtol=3e-4, atol=3e-7 eV/Å³`，cell-force 使用同 rtol、
`atol=3e-7 * max(volume) eV/Å`。parity 门：FP32 `rtol=2e-4, atol=2e-5`，
FP64 `rtol=2e-9, atol=2e-10`，按对应物理单位应用。未测性能，无 warm/JIT 加速比承诺。

## 复现与原始证据

```bash
# 仓库根目录，CPU 与 ops/framework 分进程；单编译线程避免无界并发编译。
HIP_VISIBLE_DEVICES='' NVALCHEMI_TEST_DEVICE=cpu TORCHINDUCTOR_COMPILE_THREADS=1 \
  timeout 480 bash scripts/check_cpu_reference.sh

source scripts/activate_hygon_env.sh project
export PYTHONPATH="$PWD/packages/framework:$PWD/packages/ops"
OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q --tb=short \
  packages/framework/test/models/test_pipeline.py \
  -k 'PrepareStrain or stress or stresses'

# checkpoint 需是现有本地文件；不自动下载或更换模型。
OMP_NUM_THREADS=1 timeout 120 .venv/bin/python probes/cell_stress_force_reference.py \
  --device cpu --checkpoint "$HOME/.cache/mace/MACE-OFF23_small.model" \
  --output artifacts/cell-stress-force/cpu.json

# 先只读确认 HCU4 资源，再单卡运行。
hy-smi --showpids --showmemuse
HIP_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 timeout 120 \
  .venv/bin/python probes/cell_stress_force_reference.py \
  --device cuda --checkpoint "$HOME/.cache/mace/MACE-OFF23_small.model" \
  --cpu-reference artifacts/cell-stress-force/cpu.json \
  --output artifacts/cell-stress-force/hcu.json
hy-smi --showpids --showmemuse
```

原始目录 `artifacts/cell-stress-force/` 不入库：`cpu-gate.log`、
`cpu-gate-timeout.log`、`compile-diagnostic.log`、`upstream-stress-tools.log`、
`public-compile.log`、`cpu.log/cpu.json`、`hcu.log/hcu.json`、`hcu-preflight.log/hcu-after.log`。
fork 内容和局部差异只作为只读参考，临时文件位于 `/tmp`，不是产品源码或发布依据。

## 未验证及不在本轮范围

- 晶胞对齐/打包/仿射推进、变胞 neighbor 生命周期、FIRE/FIRE2 耦合、NPT/NPH、ASE filter、
  DomainParallel、完整 checkpoint 未移植或验收；转换完成不能推定这些应用可用。
- stress/cell-force 的 CPU 一二阶与 HCU 一阶证据分开；HCU 二阶、compile/opcheck，
  以及 CPU Inductor 对这个新 custom op 的独立编译均未验收，CPU smoke 使用 aot_eager。
- 小模型 stress loss 参数反向不代表真实 MACE/全 MLIP 训练通过；真实模型仅 eval 应力及
  能量有限差分。其他模型族、LJ virial/stress 仍遵循原有未支持边界。
- legacy Warp kernel 公式已修改，但本机 Warp 缺失，无 Warp/NVIDIA 数值 parity；
  不声称新后端与未执行的 legacy 路径已等价。
- 无生产 HIP/Triton 转换、病态晶胞精度、长物理轨迹、收敛或性能验证。
