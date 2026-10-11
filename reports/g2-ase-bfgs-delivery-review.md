# ASE-compatible BFGS 最终交付审阅

日期：2026-10-10。用户确认 M1–M5 最终交付审阅与收口；本阶段没有暂存、提交、
合并或推送。分支 `wangleping/feat-ase-bfgs-m5-4`，HEAD
`cfd22ad11fe0e1704dff48f53abee69135f8db99`。原始记录在
`artifacts/ase-bfgs-delivery-review-2026-10-10/`，历史 M1–M5 产物保持。

## 实现与审阅修正

M1–M4 的 Ops、固定/UnitCellFilter 变胞 Framework、异构 Batch、独立 ID 状态、
冻结/重排/退出/补位和配对重启均已完成明确范围的验收。M5-4 的显式
`hip.bfgs-ase-jacobi-streams-v1` 已接入产品，单宽 batched、多宽同进程双 stream；
没有经验性的 147/285 维或 B≤4 限制。默认仍 Torch，`hip` 仍 rocSOLVER Jacobi，
新实现不进 auto。未改 FIRE/FIRE2、通用 BaseDynamics、版本锁或系统环境。

此次完整接口/源码审阅发现一个 Batch 数值失败边界遗漏：有限 raw direction 的
范数可溢出为 Inf；单体系明确拒绝，而 Batch 将 scale 算成 0 后提交计数与历史。
CPU 复现为 alpha=1e-300、force=1e8，前后结果分别保留在
`step-length-before.json` 和 pytest 日志。共享 Batch 数学增加与单体系相同的
`_finite("step length", length)` 检查，不修改 ASE 公式、谱、dtype 或正常有限步。
两个不同宽度组的回归确认晚组失败时所有坐标、力和完整状态不变；CPU、原 HIP
和新 HIP 实现均通过。没有用 CPU 回退、阻尼或放宽容差修正。

另补齐两个既有 executor 的 Batch 类型/NumPy docstring、变胞构造参数类型；
整理一个新增 HCU 测试文件的局部格式。合同明确空 Batch 拒绝及步长非有限报错，
API 页面补充 BFGS eager 范围和 ASE 正拉应力约定。

## 验收结果

| 检查 | 实际结果 | 范围 |
|---|---|---|
| `scripts/check_cpu_reference.sh` | 退出 0；669 passed / 5 skipped | Ops295、fresh LJ1、Framework327、state22、dynamics Ops24 |
| HCU 环境 Batch + native Ops | 退出 0；18 passed | 9 个 CPU Batch 案例、2 个 HCU 溢出案例与 7 个 native HCU stream/allocator/关闭重建/失败排空案例 |
| Framework HCU | 退出 0；24 passed | 固定/变胞、Filter 参数、冻结/重排/退出/补位、重启与 current stream |
| 公共异构 BFGS 十步 ASE | 退出 0；11 体系 | 15/21/30/45/57/93/201/489/189/189/561 DOF；single/batch、逐位相等重启及真实失败门 |
| MPA-0 medium 完整变胞 CSP | 退出 0；2/2 联合收敛 | 46/92 原子，233 阶段步、234 主模型求值，前 3 步独立 CPU MLIP/ASE 对照 |
| Ops / Framework wheel 与 sdist | 两组构建均退出 0 | 0.4.1 / 0.2.0；所有 BFGS 模块与四个产品 C++/HIP 源文件在包中 |
| wheel 解包 CPU smoke | 退出 0 | /tmp 启动，所有包模块来自解包目录；异构 fixed/variable Framework、Ops 配对 restart 与 UCF roundtrip |
| wheel 解包 HIP smoke | 退出 0 | 独立缓存由包内源码实际 JIT 构建，显式后端执行 3/6 DOF 异构 Batch |
| 使用指南 CPU 示例 | 退出 0 | 第一段代码从独立 wheel 目录实际运行，异构 Batch 与配对重启 |
| 局部 lint / format / 来源 | 退出 0 | Ops lint18/format17 文件；Framework11 文件；71 个 BFGS 源码/测试/probe 的 SPDX 与 Python 语法 |

CPU 跳过中 3 项是既有缺 Warp 的 NHC 对照，2 项是新 HCU 参数在 CPU 隔离环境
不可用；这两项在 HCU 门已实际通过。没有把跳过记为通过。既有 registry 测试
`test_backend_registry.py` 有三处基线格式差异，HEAD 原文件也被同版本 Ruff
判为需格式化；此次只检查其 lint 并保留无关格式，不宣称全仓 format 通过。

公共十步 ASE 最大绝对差：坐标 2.673e-14、Hessian 7.816e-14、谱 1.739e-11、
raw step 3.517e-14、applied step 5.190e-15。重启逐位一致；精确零谱、真实不收敛
和错误特征向量均在提交前失败。

CSP 广义 fmax 为 0.04355185 / 0.04930759，门限 0.05；物理原子 fmax 为
0.04439251 / 0.05315668，后者体现 UnitCellFilter 广义力判据，不等价于物理
原子力阈值。前 3 步最大 ASE 坐标/晶胞/Hessian 差为
1.243e-14 / 1.421e-14 / 2.552e-11。整个 probe 窗口 79.014 秒，含初始化与独立
对照；仅一次正确性复验，不增加三轮配对或生产加速结论。邻居后端为 Torch reference。

具体 HCU 命令、限时、实际退出码和资源观察见 `hcu-review-queue.json`、各
`*-run.json`；CPU/构建/Ruff 实际输出见各 `.log`。本次未重新运行未改变的 native
kernel 大矩阵和三轮性能套件；对应历史证据与适用版本见
[M5-4 报告](g2-ase-bfgs-m5-4.md)。步长非有限检查是新的标量同步边界，本次不把
历史微基准重新标为审阅后版本的性能 gate。

## 来源、许可与包装

保持 NVIDIA 原包 LICENSE 和已有来源署名。新数值实现为本地独立编写的
Apache-2.0 代码，ASE 3.29.0 是测试 Oracle，未复制其完整实现文件，也不是产品
运行依赖。当前环境四个 ASE 源文件 SHA256 与
[BFGS 合同](../docs/TORCH_ASE_BFGS_CONTRACT.md)冻结值完全一致。

新 HIP Jacobi 来自本仓库原创 M5-2/3A/3B 原型，M5-4 kernel 来源记录保持；
`agy-eigh-test` 为只读调查资料，产品无导入或链接到该工程的依赖。源码/测试/
probes 的 SPDX 核对通过；原 license checker 强制 NVIDIA 全文模板，不能将其
视为本地原创署名的通过门，也没有改署名或全局重构 checker 来凑结果。

两个 wheel 各有 LICENSE；无 `.so`、模型 `.pt` 或 CIF。Ops 原生源码随 sdist/wheel
交付，在准入 DTK/Torch 环境 JIT 编译；wheel smoke 在项目之外执行并断言实际
import 路径，因此没有用 editable 的产品源码替代包内实现。

独立 wheel smoke 的测试脚本曾修正 unpack 参数、Batch 索引字段、stress 字段及
初始 stress 配置；均为审阅夹具问题，未修改产品实现来适应错误用法。最终运行
记录为通过，使用指南采用实际接口。

## 交付与未覆盖范围

[使用指南](../docs/ASE_BFGS_USAGE.md)提供可运行 CPU Ops 异构/restart 示例及已有
MLIP Framework 接入要求。[分批交付方案](../docs/ASE_BFGS_DELIVERY_PLAN.md)与
[精确文件清单](../docs/ASE_BFGS_DELIVERY_FILES.json)把全部最终改动分为 Ops、
Framework、验证记录、工程文档四组；不暂存 raw artifacts、外部资料或编译产物。

当前仅明确的 FP64/eager、gfx936/DTK26.04 海光 Torch 构建及小样本门通过。T2
独立 CPU MLIP 完整门仍未验收；Li/T2 新增 16 CIF 短步不代表全部完整收敛。
其他硬件/构建、NVIDIA、compile、优化器梯度、其他 Filter/constraints、多卡、
全目录成功率和生产吞吐未验证。原性能退化案例及标量 host 同步边界保留。

清单共 120 个文件，四组分别为 27 / 13 / 56 / 24 个。
下一阶段待确认：按清单完成四组带 DCO sign-off 的本地提交并记录 SHA；不自动合并或推送。
