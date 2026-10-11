# ASE-compatible BFGS 分批本地交付方案

2026-10-10 审阅时：用户确认的是最终交付审阅，尚未执行暂存或提交。
当前分支 `wangleping/feat-ase-bfgs-m5-4`，基线
`cfd22ad11fe0e1704dff48f53abee69135f8db99`。M1–M5 已完成各自明确范围的开发与
验收；审阅时全部工作仍在同一未提交工作树。此次审阅结果见
[交付审阅报告](../reports/g2-ase-bfgs-delivery-review.md)。

按最终代码组织四个提交，保留 M1–M5 阶段报告，不重造已经不存在的历史源码快照。
每个提交使用 DCO sign-off，按清单中的具体文件暂存；不使用 `git add -A`。
各组精确路径见 [文件清单](ASE_BFGS_DELIVERY_FILES.json)，未经另行授权不合并、推送。

| 顺序 | 建议提交消息 | 范围与检查 |
|---|---|---|
| 1 | `feat(ops): add ASE-compatible batched BFGS backends` | Ops 状态/dispatcher/reference/UnitCellFilter、两个 HIP solver、registry/loader、Ops 测试与当前数值/后端合同；CPU Ops gate、已完成 HCU 数值与寿命门 |
| 2 | `feat(framework): integrate ASE-compatible BFGS workflows` | 固定/变胞优化器、按 ID 的 Batch 生命周期、lazy exports、Framework 测试、API 文档和 CHANGELOG；CPU Framework gate、HCU 24 项与真实 CSP 门 |
| 3 | `test(bfgs): retain eigensolver investigations and validation records` | 调查/数值/性能/CSP probes、原创 native 原型、M1–M5 脱敏阶段报告；源码语法、SPDX、阶段证据对应关系 |
| 4 | `docs(bfgs): record delivery scope and acceptance evidence` | ADR、feature/status/handoff/matrices/queue、分阶段提案、使用指南、交付审阅报告/清单与 CPU 总门入口；文件清单覆盖检查、文档链接、完整 CPU 总门和 `git diff --check` |

第 1、2 组是完整的最终功能 slice，不把当前共同文件拆成假想的阶段版本。
完整文档链接与统一验收入口在第 4 组完成后闭合；各历史阶段报告记录当时的
源码哈希、命令和范围，不用此次最终结果覆盖历史记录。

提交前再次确认分支/HEAD/工作树和文件内容与审阅清单一致。若有并行新改动，先
隔离并核对归属，再更新清单。已经通过的数值门不因单纯 Git 分组而重复执行；
新的源码改变或失败才触发相关重验。完成后记录四个 SHA 与各组实际文件列表。

`artifacts/` 原始数据、轨迹/CIF/checkpoint/模型、build cache、wheel/sdist、`.so` 与
外部工程不进入提交。两个包版本仍为 Framework 0.2.0/Ops 0.4.1，本轮不发布发行包。
当前 wheel 已在独立解包目录完成 CPU 使用与 HIP 源码 JIT smoke；这不是修改环境
安装依赖或验证其他机器的安装。

上游 `make license` 检查器绑定 NVIDIA 全文版权模板，不适用于原创本地文件的
SPDX 署名；本轮保留来源真实署名及原包 LICENSE，用局部 SPDX/来源核对验收。
已有 registry 测试文件的格式债保持，新增 BFGS 源码/测试与修改的产品文件通过
Ruff lint/format 检查；具体范围在审阅报告记录，不声明全仓 pre-commit 已通过。

## 本地执行记录（2026-10-11）

用户确认后按上述四组执行，合计 120 个文件。提交前完整文件集合、分支/HEAD 与
审阅哈希逐项一致；前三组暂存 blob 与审阅内容逐字节一致。第四组只在原清单内
更新本地交付状态、提交记录及相应矩阵，没有改变产品、测试或历史阶段报告。

| 顺序 | 本地提交 | 文件数 |
|---|---|---|
| 1：Ops | `d68ea01` | 27 |
| 2：Framework | `bf52bca` | 13 |
| 3：验证记录 | `93955b0` | 56 |
| 4：工程文档 | 本节所在的 `docs(bfgs): record delivery scope and acceptance evidence` 提交 | 24 |

第四组的完整 SHA 可用 `git log -1 --format=%H -- docs/ASE_BFGS_DELIVERY_PLAN.md`
查询；其 SHA 不写回自身内容。四个完整 SHA、父提交、DCO sign-off 及实际文件列表
另存 `artifacts/ase-bfgs-local-delivery-2026-10-11/`。

每组经过暂存清单/内容审阅与 `git diff --cached --check`；没有绕过 Git hooks。
数值源码、测试和探针未改变，沿用最终审阅已经通过的 CPU/HCU、ASE、CSP 与
包装门，没有把 Git 分组操作记为新硬件测试。历史原始验收记录不覆盖。

本提交完成第四组本地收口。四组均有 DCO sign-off，尚未合入 `develop`、推送或
发布。下一阶段可准备本地 `develop` 集成；实施前仍需给出目标并等待用户确认。
