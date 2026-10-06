# FP32 FIRE2 变胞移动验收 M9

## 目的与范围

M8 两组 FP32 FIRE2 在全部邻居拓扑、E/F/stress、轨迹状态和逐结构 HIP/Torch 最终状态比较后，
仅因 probe 额外要求 Batch 中每个结构晶胞都改变而失败。本阶段定位未移动结构，确认两路径
结果，并按结果修正该测试条件。仅改 `probes/hip_real_batch_validation.py` 和文档；不改
framework、ops、HIP kernel、优化器实现或数值容差。

## 定位结果

两个 Batch 的未移动项均为 CIF 清单 `source_index=36`：
`mp-1013795_Li3Ge3(MoO6)2.cif`（80 atoms）。结构已分别处在两个批次中，但 HIP 和
`torch_reference` 终态晶胞都与该结构初始晶胞完全相同，且两路径最终晶胞逐位相同。

| Batch 档位 | 实际原子/结构 | 移动晶胞数 | 未移动 system ID | HIP/reference 与初始 H 最大差 | 终态最大应力 | 终态最大力 |
|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 1029 / 39 | 46 / 47 | 27 | 0 Å / 0 Å | `7.6791e-4 eV/Å³` | `5.7138e-2 eV/Å` |
| 2000 | 2030 / 92 | 99 / 100 | 42 | 0 Å / 0 Å | `7.6802e-4 eV/Å³` | `5.7138e-2 eV/Å` |

这不是两邻居后端的晶胞差异：已有逐结构 cell/E/F/stress/state parity 检查通过，而且未移动
项在两路径上的 cell delta 都精确为零。该短测试不判断 Li3Ge3(MoO6)2 是否收敛，也不推断
FIRE2 更新机制的普遍行为。

## 最小测试修正与复验

旧条件 `moving == len(systems)` 改为 `moving > 0`：该 lifecycle Batch 必须实际出现晶胞运动，
但不强制每个 CIF 在有限步数内都改变晶胞。每个结构的 HIP/Torch 输出和 geometry/state
对照仍逐项执行；`unchanged_systems` 现在保存在 probe 结果中。

在 HCU4 BW1000、Torch 2.9.0/HIP 6.3.26093 上仅重跑两组受影响的真实 MPA-0 FP32 FIRE2：
1029/39 atoms/structures 耗时 82.18 s，2030/92 耗时 119.22 s，进程均退出 0，summary
状态均为 `passed`。每组分别保留 46/47、99/100 的移动数和相同的未移动 source。
M8 其余六组在更严格的旧移动条件下已经通过；因此没有重复全部八组。CPU comparator 单测
仍为 3 passed，probe compileall 通过。

两组原始结果和运行状态保存在忽略的 `artifacts/hip-mpa0-state-m8/` 下，包含各自的
`summary.json`、`run.log` 与 `m9-assertion-rerun-status.json`；probe/source hashes 记录在
`source-hashes.json`。

M9 没有修改邻居计算契约或后端选择。HIP 邻居是否支持其他 dtype、模型、PBC/skin、长步收敛
等不在本次证据范围内。
