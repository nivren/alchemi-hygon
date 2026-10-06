# HIP 当前晶胞批量 metadata 契约（M5）

2026-10-06 用户确认（含后续补充的变胞/20,000原子/三后端比较）。
起点M4实现 `95f2ef6`、文档 `88b6b2f`，分支 `codex/perf-hip-batch-metadata`。

## 优化与语义

沿用framework锁定4dfe372、ops锁定26dbceb的API及当前移植契约。
只替换 `_hip_neighbor_executor._current_cell_metadata` 的逐体系推导，使用设备内
批量inverse/norm/floor/ceil；每次输入当前cell，完整重算grid/radius/offset/inverse。
MAX_NBINS8192；dims下限1；超限时每体系最大轴（首个tie）逐次ceil-halving，
与 `_grid_spec` 原规则一致；额外搜索cell+1、int32范围检查和奇异cell失败保留。
检查大dims乘积避免int64溢出，减半避免加1溢出。

不缓存metadata/plan/workspace、不新增query并行或native COO，不改full-PBC/full-list/
FP32/64/COO/skin=0、容量、输出所有权、梯度或默认None/Warp/auto策略。
整数metadata及邻居集合要求精确匹配；inverse按FP64 2e-12、FP32 2e-6比较；
真实模型和优化状态沿用M3逐物理量容差。stream行为沿用Torch当前stream和原native接口。

## 正确性与性能

CPU/reference先验：正交/倾斜/旋转/压缩/剪切/大网格缩减/空B/奇异cell，dims/radius/
offset精确对照旧逐体系，inverse对照。HCU4实测真实HIP集合，保留M1/M2/M3 gate。

性能三路径：设备内Torch reference cell-list、M4原HIP、M5HIP；相同checkpoint/dtype/
位置/cell/ptr、容量256、full-PBC/full/COO；无silent CPU或精度回退。
合成密度0.04 atom/Å³、cutoff6Å、seed20261006，异构size交替0.8/1.2×平均：
200结构均值100、40结构均值500、10结构均值2000，共20,000原子；另保留小Batch。
小Batch使用明确seed20261007：原seed的FP32 cutoff机器精度邻域有两条边与
Torch reference舍入分歧，原HIP与优化HIP相同。失败保留，不放宽集合检查，
不宣称任意FP32 cutoff边界位级等价；此限制见M5报告。
连续initial/压缩/膨胀/剪切快照，退出→arrival后新N/B/ptr，再继续剪切/压缩；
记录source IDs，重建必须在新布局/当前晶胞上正确。固定重复仅辅助基线。
每快照先三路径拓扑对照，再同步wall计时：1 warmup，3轮×3样本/路径，轮转顺序，
保存每样本、median/mean/std/range、有效邻居数、每路径峰值显存及首次API。
记录逐快照计时；几何变换/fixture生成/正确性核对不计入邻居API耗时。

真实MPA-0 Li/Na三CIF18原子两精度，两优化器各5步；先优化HIP正常驱动得到公共
快照序列，三路径在同快照重算并测neighbor/model-only/neighbor+model，避免分叉。
另执行原M3私有状态、预算退出/补位回归。真实短轨迹不代表20,000原子模型计算，
预算退出不是物理收敛。每项单HCU4限时，空闲资源检查，不调整系统/时钟/亲和性。

候选准入：变胞200结构FP32相对原HIP邻居median改善>=5%，三个轮次一致；其余
HIP配置无>5%稳定退化，拓扑/模型/生命周期通过。Torch reference比值独立报告，
不以HIP设备执行替代HIP邻居选择。分项/hipprof单独采样，性能最终无profiler。
无NVIDIA实际基线、全模型族、长收敛、DD、compile/opcheck/训练准入。

证据 `artifacts/hip-batch-metadata-m5/`；报告 `reports/g2-hip-batch-metadata-m5.md`。
当前：实现完成，CPU ops45/framework21及HCU ops73/framework49通过；
FP32/64八配置变胞三路径比较完成，所测性能门槛通过。具体数值和剩余边界以报告为准。
产品只有一个注册HIP后端，修改前代码仅作probe基线；尚未提交/合并/推送。
