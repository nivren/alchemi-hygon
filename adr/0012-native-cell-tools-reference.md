# ADR 0012：原生晶胞工具的独立 Torch reference

日期：2026-10-04；状态：采用；任务：MPA-0 弛豫 M2。

保留锁定上游六自由度晶胞布局、单体系/batched 分支、预分配输出与 mutation。
新增 ops-owned `cell_reference` executor，通过声明式 `cell_tools` operation 绑定九个
原名 entrypoint；不新增通用晶胞优化器、ASE filter 或参数化。
framework `align_cell` 保留位置参数和 None 返回值，新增 keyword-only backend/selection；
None/warp 仍调用原 custom op，Warp/bridge 按需导入。Torch reference 为 eager、
forward-only 的设备内路径，不准入 `variable_cell` 或梯度等级 1。

`extend_batch_idx` 上游为 append 布局，而 batched pack 是 interleaved 布局。分别保留并
明确不能配对；后续 FIRE2 adapter 按上游 torch/fire2 的 ext_ptr 块长构造交错索引。
不把这个不同用途的工具改为另一个索引语义。

确认 UBF-010 的负体积 transform 与非连续 bridge 写回缺陷，分别更正本地 Warp kernel
和 framework bridge，并在 reference 中保护。下三角文案沿用指定 fork 修订。
默认后端不变，受缺陷影响的 legacy 输入结果可能改变；Warp 数值执行尚未验证。
具体错误/别名边界、CPU/HCU 门和限制见 `docs/TORCH_CELL_TOOLS_CONTRACT.md`。
