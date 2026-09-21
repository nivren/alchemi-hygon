# DCU Toolkit — 共享脚本索引（本目录为占位）

## 重要：本目录是占位索引，不是脚本存放处

`dcu-toolkit-skills/scripts/`（即本目录）这一级**没有、也不应直接放可执行脚本**。
它只是一个包级说明文件，用来告诉读者"真正的脚本在哪里"。

**可执行的 `.sh` 脚本全部在各子 skill 自己的 `scripts/` 目录下**，例如：

- `dcu-rccl-test/scripts/rccl_test_single_node.sh` — 单节点 RCCL 通测一键流程
- `dcu-env/scripts/check_iommu_acs.sh` — BIOS ACS/IOMMU 检测（**主要针对 PCIe 形态卡：K100-AI/BW10/BW100/BW150 等；OAM 产品 BW1000/BW1100 基本无影响，无需为此改 BIOS/内核**）
- `dcu-cpu-affinity/scripts/gen_affinity.sh` — NUMA 绑核方案生成
- `dcu-driver/scripts/check_driver.sh` — 驱动加载验证
- `dcu-hy-smi/scripts/hy_smi_query.sh` — hy-smi 查询封装
- `dcu-hy-smi/scripts/examples.sh` — 原仓库示例脚本（参考）
- `dcu-topology/scripts/collect_topology.sh` — 拓扑采集
- `dcu-env/scripts/vendor_dcuprofiletools/...` — 海光官方环境检测工具（第三方打包，含 `system_check.sh` 等）

> 引用约定：文档中凡提及脚本，**一律写完整相对路径**（如 `dcu-topology/scripts/collect_topology.sh`），
> 不写裸名（如 `collect_topology.sh`）以免机检误报"文件缺失"。shell 命令示例里"已进入脚本目录后直接 `bash xxx.sh`"
> 属正常操作，不在该约定约束内。

## 历史坑（避免重复排查）

早期版本的本 README 曾罗列 `env_detect.sh` / `collect_metrics.sh` / `parse_hy_smi.py` /
`gen_report.py` 等脚本，并标注"\[待创建\]"——但这些脚本**从未创建**，也从未被任何子 skill
引用，属于"文档许诺但未实现"的悬空描述，会让人误以为这里有现成脚本。
现已删除该误述，本目录明确为占位索引。若日后确有"包级共享脚本"需求，再在此处创建，
并保持本说明的占位性质不变。

> 引用约定：文档中凡提及脚本，**一律写完整相对路径**（如 `dcu-topology/scripts/collect_topology.sh`），
> 不写裸名（如 `collect_topology.sh`）以免机检误报"文件缺失"。shell 命令示例里"已进入脚本目录后直接 `bash xxx.sh`"
> 属正常操作，不在该约定约束内。

判断标准：文档叙述里出现 `<某>.sh` 时，先按"完整相对路径 → 子 skill scripts/ → 外部仓库/示例占位"顺序核对，
确认都不是才视为真缺失。
