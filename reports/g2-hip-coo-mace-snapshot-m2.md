# HIP 邻居 M2：COO 适配与 MPA-0 单点

日期：2026-10-05。用户确认后先收口 M1：实现 `fecbc39`、文档 `8120282`。
本阶段分支 `codex/feat-hip-coo-mace-snapshot`，起点 `8120282`。
已完成下列实现和窄验证；用户确认下一里程碑后，收口实现 `3e394d7`。
仅提交到本地阶段分支，未合并或推送。

## 实现

显式 `hip.neighbor.cell_list-v1` 新增 periodic/full/COO 准入；HIP 不进入 auto。
native build/query/materialization 继续输出 MATRIX，ops 直接 COO 请求复用
`torch_reference_cell_list._result`；framework 与 Hook 复用已有设备内 Torch
`_neighbor_matrix_to_coo`，无需新写 native COO kernel 或模型专用转换。
详细 shape/dtype/PBC、容量、stream、梯度和来源见[M2 契约](../docs/HIP_COO_MACE_SNAPSHOT_CONTRACT.md)。

真实模型测试发现既有 Hook 的容量适配假定 source MATRIX 宽度不超过 staging 容量：
HIP 自动返回宽80，但按实际最大邻居数计算的 staging 为64，复制失败。
修正 `_pad_reference_result`：先检查所有有效 counts 能放入 staging，再仅复制
`min(source_width,capacity)` 列；若有效邻居不能放入则抛 `NeighborOverflowError`。
裁去的仅是 padding；CPU periodic/no-PBC 回归分别复现修正前失败并验证修正后行为。

原有 CPU “HIP COO 拒绝”测试实际输入是 CPU，改名明确仍验证禁止 CPU 回退。
原固定晶胞探针将已准入的 COO 从拒绝项替换为仍未支持的 skin；half/no-PBC 拒绝保持。
所有既有 MATRIX、默认 None/Warp、reference 和整数 shift 语义保持，未修改 external。

## 验证

环境：DTK26.04、项目 `.venv`、Torch2.9.0/HIP6.3.26093，HCU4 BW1000 64G/gfx936，
MACE0.3.15、e3nn0.4.4，单卡限时 gate、`OMP_NUM_THREADS=1`。
开始检查 HCU4 显存/利用率0%、Normal；KFD 中存在本任务 CPU pytest 的零显存进程。
结束 HCU4 空闲、Normal、无 KFD PIDs，快照保存于 `resources-after.log`。
未占用其他卡、未更新软件或下载模型。

| Gate | 结果（退出0） | 原始日志：`artifacts/hip-coo-mace-snapshot-m2/` |
|---|---|---|
| CPU ops 聚焦回归 | 88 passed、10 skipped、1 warning | `cpu-ops.log` |
| CPU framework 独立进程 | 38 passed、12 skipped、1 xfailed、1 warning | `cpu-framework.log` |
| HCU ops current-cell MATRIX/COO | 26 passed | `hcu-ops.log` |
| HCU framework MATRIX/COO/Hook | 17 passed，含5项CPU adapter/Hook oracle | `hcu-framework.log` |
| HCU 原46×32固定晶胞回归 | passed | `hcu-fixed-regression.log` |
| HCU MPA-0 单点/Hook/有限差分 | passed | `mace-hcu.log`、`mace-hcu.json` |

CPU skip 为显式 HCU 专属测试；xfail 是既有 `WARP-DEFAULT-001`，未添加豁免。
最终 compileall、diff空白检查及特性契约 JSON/唯一 ID/证据路径检查通过；没有运行全量套件。
ops gate 新增4组 dtype×stream 的直接 dispatcher COO：当前晶胞7种快照及空原子，
与 reference 精确比较 atom/image 多重集合、row_ptr、shape/dtype/device，显式溢出仍失败。
framework 新增8组 dtype×入口×stream：收缩、多镜像、完整三斜剪切、wrap、零边、恢复，
比较边集合和体系分段，验证无跨体系边、输入与先前输出保护及 Hook 容量增长。

两个新增 registry 测试在实现前失败，见 `before-implementation-ops.log`。
两个 padding 回归在修正前失败，见 `before-hook-fix.log`；首次模型测试的失败日志
保留为 `mace-before-hook-fix.log`。最终 gate 是修正后的独立重跑，不将失败结果记作通过。

## 真实模型数值结果

可信本地 checkpoint `mace-mpa-0-medium.model`，SHA256
`75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638`。
输入复用 MPA-0 CPU 基线三个 CIF：Li2MgGe(4)、LiSnCl3(10)、NaLiTl2(4)，
`batch_ptr=[0,4,14,18]`；文件哈希记录于 JSON，输入文件不修改。
模型与进程默认 dtype 同精度，普通 e3nn eager、冻结权重，保持位置/应变 autograd。
两精度各 initial、affine_shear、wrap 快照，直接 `compute_neighbors` 与复用同一 Hook
均对照同设备 Torch reference；reference 直接调用框架既有 dense 路径，不借用 HIP 拓扑。

各精度总边数都是 `610/602/610`，canonical `(i,j,shift)` 精确一致，0条跨体系边。
下表是三个快照中 HIP 与同设备 Torch reference 的最大绝对差：

| 精度 | 能量 eV | 力 eV/Å | 应力 eV/Å³ |
|---|---|---|---|
| FP64 | 1.78e-15 | 9.81e-15 | 1.30e-16 |
| FP32 | 3.81e-6 | 5.42e-6 | 3.54e-8 |

容差沿用 MPA-0 基线，未放宽：FP64 rtol2e-8、E/F atol2e-7、stress2e-9；
FP32 rtol2e-4、E atol2e-3、F2e-4、stress2e-6。Hook 结果同样通过。
FP64 initial 使用 eps1e-5 的中央有限差分：最大力原子(全局7)的 x 分量误差
`2.43e-10 eV/Å`；对称 xy 应变应力投影最大误差 `1.68e-12 eV/Å³`，差分中拓扑不变。
这检验连续几何的力/应力，未验证训练力损失、模型二阶梯度或优化轨迹梯度。

## 复现与限制

激活项目环境后，CPU ops 文件集合与 M1 六文件相同；framework 在 M1 六文件基础上
加入 `packages/framework/test/compatibility/test_hip_coo_neighbors.py`，分进程运行，
`HIP_VISIBLE_DEVICES=''`。HCU topology 的两个 pytest 进程各限时180秒：

```bash
source scripts/activate_hygon_env.sh project
source artifacts/hip-current-cell-neighbor-m1/native_env.sh
export HIP_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1
timeout 180 .venv/bin/python -m pytest -q packages/ops/test/torch/test_hip_current_cell_neighbors.py
timeout 180 .venv/bin/python -m pytest -q packages/framework/test/compatibility/test_hip_coo_neighbors.py packages/framework/test/compatibility/test_hip_current_cell_neighbors.py
timeout 180 .venv/bin/python probes/framework_neighbor_hip_contract.py
MPLCONFIGDIR=/tmp/hip-m2-mpl timeout 180 .venv/bin/python -u probes/hip_coo_mace_snapshot.py \
  --checkpoint /home/wangleping/mace-mpa-0-medium.model \
  --data-root /data/csp_data/catl_data/li_mp_data \
  --output artifacts/hip-coo-mace-snapshot-m2/mace-hcu.json
```

M1 的 `native_env.sh` 配置 gfx936/单编译进程及任务独立缓存；native C++ 本轮未修改。
无这些已有产物时，应先独立完成冷编译，数值 gate 的180秒不含首次编译预算。
probe elapsed13.42秒是此次整个探针墙钟，未做预热/重复性能比较，不是优化或吞吐证据。

M2 新增变胞证据只限 full-PBC/full/MATRIX+COO/skin=0。模型验收只限上述 MPA-0，
不能扩展为 AIMNet2/UMA 或全 MLIP。no-PBC/half、skin/selective rebuild、旧 plan 变胞复用、
target/pair、native geometry backward、compile/opcheck、DD 未准入；不进入 auto。
未运行 FIRE/FIRE2、未验收退出/补位或长弛豫，没有性能优化结论。

下一阶段已获用户确认：分别用已有 Torch reference 变胞 FIRE/FIRE2 与显式 HIP COO Hook
组合，验证少量 Li/Na 结构的短步轨迹、当前晶胞邻居、E/F/stress、优化器状态及
退出/补位边界；性能测量与 workspace/query 优化继续作为后续独立阶段。
