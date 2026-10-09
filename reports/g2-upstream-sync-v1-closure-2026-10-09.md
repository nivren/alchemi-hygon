# G2 / Upstream Sync V1 — A–D 与 FIRE2 补测集成收口

日期：2026-10-09。用户已确认签名收口 D 和补测，并整理可审阅的独立集成 PR。
本地成果整理完成；在线 PR 尚未建立，源分支推送与后续集成等待明确授权。

## 1. 分支与提交身份

| 对象 | 精确提交 / 分支 |
|---|---|
| 稳定 develop / PR base | `2e211ab93a4f1ad0c8c720eecb22129d846c7792` |
| A：指定集成基线 gate / 流程 | `132ff32d90e4cf56b422a34674a5a93c596490af` |
| B：只读 Delta 工具 | `84b4ba95ac008d4022acfcff6ea076a5d7793a55` |
| C：16 项 UBF 审查 | `ad189987b94b8b97b3332ee8982c960a3ce6a5e6` |
| D：隔离升级评估 | `a0a37635c8cb35814099cace8b2fc6cdb94b4e27` |
| 稳定基线 FIRE2 完整弛豫补测 | `8306b7f569f5993bb5e73a18a0cf574373615a7d` |
| 两成果支线汇合 | `86235170cde14d07739d807ce52166fe1c687aab` |
| 独立 PR head 分支 | `wangleping/integration-g2-upstream-sync-v1` |
| 本地 PR worktree | `artifacts/g2-upstream-sync-pr-2026-10-09/worktree` |

A–C 提交链原样保留；D 与补测分别从 C 开展，汇合使用合并提交并保留双方历史。
冲突仅在 STATUS、PARALLEL_DEVELOPMENT_PLAN 和 reports/README 三个文档中；
逐处保留 D 与补测记录，没有产品源码冲突。提交均带 Signed-off-by，不改写共享历史。
本报告的文档提交位于汇合提交之后；最终 head 和 diff 清单保存于原始收口证据。

## 2. PR 交付范围

- A：已集成 HIP Neighbor M1–M9 的指定 CPU/HCU4 gate、状态修正和长期流程入口。
  此前已授权的 M1–M9 fast-forward 保留，不补造历史 PR。
- B：可重复运行的只读 `scripts/upstream_delta.py`、14 项 Git/AST 夹具、冻结候选记录和首轮报告。
- C：UBF-001～016 的字段与官方证据；核心 equivalent 3、partial 5、different 1、unknown 7，
  全部 retain；科学案例回归和未覆盖边界分别记录。
- D：隔离模块探针、依赖/API/文本冲突评估、候选状态和暂缓整体升级的决策。
- 补测：FIRE2 + HIP 邻居固定三样本完整联合收敛探针、CPU 夹具、契约及报告；
  能力矩阵仅补充这一小范围的真实证据，仍保留 implemented/partial 状态。
- 工程交接：本收口报告、STATUS、PROJECT_HANDOFF、任务队列和报告索引。

PR diff 不包含 `packages/`、产品版本锁、生产环境配置、上游源码导出、独立 venv、
checkpoint、CIF 数据或原始轨迹。Framework `4dfe3723def34df3fadb245981081ccf8c94c257`
和 Ops `26dbceb61e30cca80e1a5805eebeb51d7dc68fd1` 继续为 Baseline。
候选 Framework `f59e28dc6701583a13bb294e536076c2a7379e98` / Ops
`5fffc0726b1e71bb8b68e10b5f87942064759f5b` 保持 Candidate。

## 3. 集成复验

集成 worktree 的产品文件与稳定 develop 完全相同；复验使用项目环境 Python3.12.13、
海光 Torch2.9.0+das.opt1.dtk2604、DTK26.04，CPU 显式 `HIP_VISIBLE_DEVICES=''`、OMP1。
候选三原样模块使用 D 的独立环境和显式文件导入；没有启动新 HCU 作业。

| 本轮复验 | 结果 | 退出码 |
|---|---|---:|
| 变胞 reference / 生命周期 / HIP 工作流 CPU | 59 passed、16 既有设备 skip | 0 |
| Delta Git/AST unittest 夹具 | 14 passed | 0 |
| PR194 改编的本地科学 pytest 案例 | 6 passed | 0 |
| 完整弛豫探针 CPU 分类夹具 | 4 passed | 0 |
| 候选三独立原样模块 CPU probe | 11 checks passed | 0 |

科学案例第一次误用了 unittest，退出5并显示0 tests；该调用未计为通过。
保留原始日志和命令后改用 pytest，6项实际执行通过；无需修改被测代码或容差。

原始收口证据位于 `artifacts/g2-upstream-sync-pr-2026-10-09/`：
`cpu-commands-and-exits.json` 保存实际参数、PYTHONPATH、时间和退出码；
`remote-develop-before.json` 保存三远端只读 SHA 核验；
`closure-validation.json` 保存提交/文件/证据完整性和最终 head。
三个远端 develop 在 2026-10-09 07:44 UTC 均为上表稳定基线。

## 4. 复用的硬件证据及限制

本轮保留此前实际测试的源码身份、probe 哈希和环境；不把文档合并重新记作硬件测试。
D 与补测探针/CPU 夹具的内容与已收口提交及原测试哈希一致；产品子树与原验收源码一致。

| 既有证据 | 实际范围 | 不能推断 |
|---|---|---|
| [A 基线验收](g2-integration-a-2026-10-09.md) | CPU425/3skip；HCU ops101、framework77/1xfail；真实MPA-0两精度/两优化器/两规模共8/8通过 | 完整上游/Warp或新的性能准入 |
| [D 隔离评估](upstream-sync-2026-10.md) | 三候选原样模块CPU/HCU4各11项；stock Batch/L-BFGS导入缺Warp退出1；24个文件出现三方文本冲突 | 候选整包、真实Batch/grouped引擎、全量subtree语义合并已验证 |
| [FIRE2 补测](g2-hip-fire2-convergence-2026-10-09.md) | HCU4三Li/Na CIF、FP32/64×HIP/reference四作业共12个结构运行全部联合收敛；真实容量2补位、终态与导出CPU/HCU重算通过 | 任意材料/大Batch收敛率、性能、restart、compile或DomainParallel |

补测采用 fmax<=0.01 eV/Å 与 max(abs(stress))<=0.1 GPa 的联合物理判据，
三结构更新77/153/53次；其中HIP两精度共6个结构运行。
优化器、晶胞工具与应力仍为 Torch reference；HIP只用于当前晶胞邻居完整重建。

对照原冻结快照核验912份源码/夹具、3份环境控制文件、两套缓存引用不变；
候选导出的1380个原文件哈希不变。没有检查整个已安装依赖目录的所有文件，
环境核验范围为控制文件与运行元数据。
C 记录的原 MACE 模块收集失败继续保留；本轮定向通过不覆盖该模块或 Warp/NVIDIA。

## 5. 决策与下一阶段确认点

本轮选择暂不整体升级：候选 Ops 的 Torch>=2.10 声明与当前海光Torch2.9不符，
其他声明依赖、API适配闭包和历史分叉仍需独立处理。16项UBF全部保留，不升级Torch/DTK。
日常功能移植可在稳定版本继续，上游观察按 [长期流程](../docs/UPSTREAM_SYNC.md) 独立开展。

下一步是明确授权将本 PR 源分支推送到 github-origin 并建立以 develop 为 base 的 draft PR。
理由来自用户确认的规则6及 UPSTREAM_SYNC 第6节：阶段确认不自动授权推送/合并。
在线PR经审阅后再确认合并与指定远端同步，逐处核验同一 develop SHA。
正常移植任务（固定晶胞BFGS、L-BFGS评估、HIP邻居扩展/优化或DomainParallel）随后单独选择，
不因上游新增特性自动改变优先级。
