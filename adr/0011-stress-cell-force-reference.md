# ADR 0011：应力到晶胞力的独立 reference 边界

日期：2026-10-04；状态：采用；任务：`TORCH-CELL-STRESS-FORCE`。

## 决策

保留 framework strain/autograd stress 工具。ops 新增 functional、设备内的
`stress_to_cell_force` reference，通过 metadata 声明 executor。
framework 独立 `dynamics/_ops/stress.py` 解析一次 selection 后调用 generic executor。
原 `npt_nph` import 路径 re-export 同一入口；其原有 Warp 函数只改为按需导入依赖。

保留四个原位置参数，新增 keyword-only backend/selection。None/warp 仍调用原
`torch.ops.nvalchemi.stress_to_cell_force`。reference 使用独立的
`nvalchemiops::stress_to_cell_force` custom op，登记 fake 和可再次求导的 backward。
reference 的一二阶梯度支持不扩大 legacy autograd。

采用 fork ops `c4c3bb56b6c6` 的行晶格公式 `-V H^-T sigma`，同时更正本地 legacy
kernel；源码标记 `CELL-STRESS-ROW`，登记 UBF-009。默认后端选择不变，legacy 变胞
调用的数值轨迹可能改变。六自由度投影仍为下三角，不更换参数化。

## 边界

- ops 转换不依赖 framework/Warp；catalog 不导入 executor。
- 不迁移晶胞对齐/打包/仿射更新、FIRE/FIRE2、NPT/NPH 算法，不准入 variable_cell。
- 不新增通用变胞优化器或通用模型 stress 适配层，也不实现 ASE filter。
- volume 是独立可微输入；物理调用由调用方传 `abs(det(H))`。
- CPU/HCU、转换梯度、模型混合二阶、compile 和 legacy 运行证据分别记录。

验收与证据见 `docs/TORCH_CELL_STRESS_FORCE_CONTRACT.md` 和
`reports/g2-torch-cell-stress-force.md`。
