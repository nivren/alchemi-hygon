# MPA-0 弛豫 M2：原生晶胞基础工具

日期：2026-10-04。分支 `codex/feat-cell-tools-reference`，起点 `f802d49`，实现/测试 `036827e`。
未合并、未推送。NVIDIA 锁定不变；指定 fork ops `eadc3ee` 的 cell_filter 已只读核对。
契约：[TORCH_CELL_TOOLS_CONTRACT.md](../docs/TORCH_CELL_TOOLS_CONTRACT.md)；
决策：[ADR 0012](../adr/0012-native-cell-tools-reference.md)。

## 实现

移植九个已有 entrypoint：align、positions/velocities pack 与 unpack、forces/masses
pack、extend_batch_idx/extend_atom_ptr。Torch reference 属 ops，保持原函数参数顺序、
预分配输出、原地写入/返回对象、单体系和 batched 分支；未重写优化器或新增 filters。
operation `cell_tools` 仅登记 row_cell/six_cell_dofs 和 gradient_order=0。
framework 公共 align 保持原 import 与位置参数，增加 keyword-only backend/selection；
None/warp、原 torch.ops 名称和 fake 注册保留。第二 executor、单次 selection 和
无 Warp 导入均实际调用验收。catalog 没有导入 executor 或新增 central dispatch 分支。

上游实际六项 layout 是下三角 `[00,10,20]` 和 `[11,21,22]`；unpack 上三角归零。
batched pack 按体系交错；extend_batch_idx 则保留所有原子后追加晶胞索引的语义。
二者不能配对，下一阶段按上游 torch/fire2 的 ext_ptr 块长构造交错索引。

UBF-010：左手 H 取负后用于 transform 导致分数坐标翻转，改为原始 H；bridge 对
positions.contiguous() 未写回的问题同步修正。代码注释、LP-074/075 与
[缺陷记录](../docs/UPSTREAM_BUGFIXES.md#ubf-010晶胞对齐的左手坐标与非连续写回错误)
已登记。非法输入检查属于 reference 契约加强，不当作上游 bug。

## CPU 验证

- 工具28项：FP32/64、左右手非正交晶胞、ASE lower standard_form、metric、分数坐标、
  周期位移/解析径向能量不变；旧负体积 transform 复现；非连续写回；异构 `[2,0,3]`、
  单体系零原子、空 Batch；pack 精确 round-trip、六项投影、质量复制、两种索引布局；
  shape/dtype/device/idx/ptr/capacity/alias/非有限/梯度/整数溢出失败。
- framework10项：原 import/custom-op、无 Warp、single selection、legacy 选择、错误
  capability、第二 executor、全部 entrypoint ABI；legacy bridge 使用 stub 验证写回。
- 基础门：335 passed、3 skipped。三项 skip 是原有 NHC Warp 对照，环境没有 Warp。
  首次 `timeout 240 scripts/check_cpu_reference.sh` 的五个 pytest 进程均通过，但整体
  退出124；随后 compileall、git diff --check 单独退出0，并以600秒时限重跑完整门禁。
  最终完整门禁退出0，记录见 `artifacts/cell-tools-m2/cpu-gate-final.log`。
- CPU probe 六组退出0；原始 `cpu.json` / `cpu.log`。首次 probe 报告序列化误用了
  to_dict，已改为 BackendSelection.as_dict，重新运行通过；无产品数值修正/容差放宽。

CPU 命令（project .venv；framework/ops pytest 由脚本分进程）：

```bash
OMP_NUM_THREADS=1 timeout 600 scripts/check_cpu_reference.sh
OMP_NUM_THREADS=1 PYTHONPATH=packages/framework:packages/ops \
  .venv/bin/python probes/cell_tools_reference.py --device cpu \
  --output artifacts/cell-tools-m2/cpu.json
```

## HCU4 窄验证

按 dcu-hy-smi 技能先只读检查显存、利用率和进程：HCU4 显存/利用率0%，无进程。
仅使用物理卡4，`HIP_VISIBLE_DEVICES=4` 映射为 `cuda:0`，没有多卡或全目录作业。
DTK26.04、Torch2.9.0、HIP6.3.26093，设备 BW1000 64G。

```bash
source scripts/activate_hygon_env.sh project
HIP_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 PYTHONPATH=packages/framework:packages/ops \
  timeout 120 .venv/bin/python probes/cell_tools_reference.py --device cuda:0 \
  --output artifacts/cell-tools-m2/hcu4.json
```

整体退出0，六组 FP64/32 × `[2,0,3]`、`[0]`、`[]`。每组在目标设备实际调用全部九个
entrypoint 和 framework align；检查几何、精确 pack/unpack、索引归属、质量、非法奇异
cell。CPU/HCU 比较只在收集证据时转 CPU，算子内部没有 CPU 路径/降精度。

| 量 | FP64 最大绝对误差 | FP32 最大绝对误差 |
|---|---:|---:|
| 对齐分数坐标 | 1.11e-16 | 8.94e-8 |
| CPU/HCU 全浮点输出 parity | 8.88e-16 | 2.98e-8 |
| 同设备 pack/unpack | 精确 | 精确 |

几何/parity 门 FP64 rtol/atol=2e-12，FP32=2e-5；整数与 pack round-trip 精确。
原始输出 `artifacts/cell-tools-m2/hcu4.json` 和 `hcu4-rerun.log`，不入库。

## 未验证与下一步

这是晶胞 primitive 完成，未执行 FIRE/FIRE2 变胞优化步、仿射 motion 或 MPA-0 HCU
推理/弛豫。能量不变证据为解析径向能量；目标模型旋转后的 E/F/stress 尚未验收。
AlignCellHook 的 optimizer 生命周期与后端传递待 M3，不宣称 Hook 可用。
Warp kernel/完整原测试/NVIDIA parity、compile/autograd、非默认 stream 专项、病态
晶胞精度、生产 HIP/Triton 和性能均未验证。

下一里程碑 M3：先移植原生六项 FIRE2，再移植指定 fork 修复的全3×3 FIRE；独立验证
单步与功率分支、联合 kick/mixing/reset、仿射更新、真实 Cartesian maxstep 和公共 API。
开始前仍需用户确认；邻居生命周期/退出补位/应用入口留在 M4，真实目标 HCU 留在 M5。
