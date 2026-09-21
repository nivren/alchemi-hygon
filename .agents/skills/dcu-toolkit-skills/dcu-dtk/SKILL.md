---
name: dcu-dtk
description: 海光DCU DTK(开发工具包)下载、安装与激活。覆盖从分发源下载最新DTK(按系统/内核选包)、解压到/opt、建dtk软链、source激活环境变量。仅物理机需要装/激活；docker容器内自带DTK不需配置。DCU上架部署、环境搭建、框架编译(pytorch/vllm等)前置必读。
license: MIT
metadata:
  version: 1.0.0
  author: DCU Engineer
  hermes:
    related_skills: [dcu-driver, dcu-docker, dcu-env, dcu-hy-smi]
---

# dcu-dtk — 海光 DCU DTK 开发工具包安装与激活

DTK（DCU Toolkit）是海光 DCU 的开发工具包，提供 HIP 兼容编译器、运行时、数学库与
框架支持（PyTorch DCU 版 / vLLM / SGLang 等编译运行依赖）。**物理机部署 CUDA 系框架
前必先装并激活 DTK**；docker 容器内镜像已自带 DTK，无需配置。

## 1. 分发源

> ⚠️ 明确指定保留以下原文 URL（含内网分发域），agent 直接可用。

- 最新 DTK 下载根：`https://download.sourcefind.cn:65024/1/main/latest`
  - 打开后**按操作系统选择**对应 DTK 包；**若无完全对应系统，选内核版本最接近**的那个
- DTK 文档：`https://download.sourcefind.cn:65024/1/main/latest/Document`
- 版本配套关系表：`https://download.sourcefind.cn:65024/file/1/DTK与驱动版本配套关系表.md`
  - 该表列出 DTK 版本 ↔ 驱动版本 ↔ 系统/内核 的配套要求，**每次使用前必须联网复核**（见 Pitfall #1）

## 2. 安装流程（仅物理机）

```bash
# 2.1 从 latest 选系统对应的 DTK 包下载（无对应系统则选内核最接近）
#     以下为示意，实际文件名/路径以官网 latest 目录实时为准
cd /tmp
curl -L -O "https://download.sourcefind.cn:65024/1/main/latest/<系统对应DTK包>.tar.gz"

# 2.2 解压到 /opt
tar -xzf <系统对应DTK包>.tar.gz -C /opt
#     解压后 /opt 下出现如 dtk-<version> 目录（具体目录名以包内为准）

# 2.3 建软链 /opt/dtk -> 解压目录（统一入口，框架/编译按 /opt/dtk 找）
ln -sfn /opt/dtk-<version> /opt/dtk
```

## 3. 激活（仅物理机）

```bash
source /opt/dtk/env.sh
# 验证：
which hipcc          # DTK 提供的 HIP 编译器应在 PATH
echo $DTK_PATH       # 应输出 /opt/dtk
hy-smi -a            # 驱动/设备可见（依赖 dcu-driver 已装）
```

> 激活即把 `/opt/dtk/bin` 等加入 `PATH`、设 `DTK_PATH` 等环境变量；
> **每次新 shell 跑框架/编译前都要 `source /opt/dtk/env.sh`**（或写入 `~/.bashrc`）。

## 4. 适用范围（重要）

| 环境 | 是否需要装/激活 DTK | 说明 |
|---|---|---|
| 物理机（裸金属） | **必须** | 下载解压 + 软链 + source 激活，框架编译运行前置 |
| docker 容器 | **不需要** | 基础镜像已自带 DTK，直接 `source /opt/dtk/env.sh` 或已在镜像 ENV 内，**不要重复下载安装** |

- 容器内若要确认 DTK 是否已激活：`which hipcc` / `echo $DTK_PATH`；为空则镜像未带，需换带 DTK 的镜像（见 dcu-docker）。
- 物理机激活后若进容器，容器内是独立 ENV，仍需在容器内单独 `source /opt/dtk/env.sh`（前提是镜像内 `/opt/dtk` 已存在——通常基础镜像有）。

## 5. 版本配套关系（基线快照 + 联网复核）

> ⚠️ **本段为基线快照（取自官方配套关系表），非实时数据。每次使用前必须联网打开配套关系表核查最新版本**：
> `https://download.sourcefind.cn:65024/file/1/DTK与驱动版本配套关系表.md`
> 以联网查到的最新 DTK / 驱动配套为准。
> 注：表中 `K100_AI` 即本包其他处写的 `K100-AI`。

| DTK 版本 | 配套发布的驱动版本 | 驱动版本要求 | 硬件类型 |
|---|---|---|---|
| DTK-26.04 | rock-6.3.30-V1.4.1a.run | >=6.3.30-V1.4.1a | BW, K100_AI, K100, Z100L |
| DTK-25.04.2 | rock-6.3.16-V1.1.0a.run | >=6.3.16-V1.1.0a | BW, K100_AI, K100, Z100L |
| DTK-25.04.1 | rock-6.3.6-V1.9.0c.run | >=6.3.6-V1.9.0c | BW, K100_AI, K100, Z100L |
| DTK-25.04 | rock-6.3.3-V1.8.0.run | >=5.7.1-V1.7.0 | BW, K100_AI, K100, Z100L |
| DTK-24.04.3 | rock-5.7.1-6.2.26-V1.5.aio.run | >=5.7.1-V1.0 且 <=5.7.1-V1.6.3 | K100_AI, K100, Z100L |
| DTK-24.04.2 | rock-5.7.1-6.2.18-V1.1.2.aio.run | >=5.7.1-V1.0 且 <=5.7.1-V1.6.3 | K100_AI, K100, Z100L |
| DTK-24.04.1 | rock-5.7.1-6.2.17-V1.1.1.aio.run | >=5.7.1-V1.0 且 <=5.7.1-V1.6.3 | K100_AI, K100, Z100L |
| DTK-24.04 | rock-5.7.1-6.2.13-V1.0.1a.aio.run | >=5.7.1-V1.0 且 <=5.7.1-V1.6.3 | K100_AI, K100, Z100L |
| DTK-23.10.1 | rock-5.2.0-5.16.18-V01.9.2.run | >=5.2.0-V10.9.2 / >=5.7.1-V1.0 且 <=5.7.1-V1.6.3 | K100, Z100L |
| DTK-23.10 | rock-5.2.0-5.16.18-V01.9.2.run | >=5.2.0-5.16.18 / >=5.7.1-V1.0 且 <=5.7.1-V1.6.3 | K100, Z100L |
| DTK-23.04.1 | rock-5.2.0-5.16.18-V01.9.2.run | >=5.2.0-5.16.18 / >=5.7.1-6.2.13a 且 <=5.7.1-6.2.18 | K100 |
| rock-4.5.2-5.11.40-V01.8.1.run | >=4.5.2-5.11.38 / >=5.7.1-6.2.13a 且 <=5.7.1-6.2.18 | Z100L |
| DTK-23.04 | rock-4.5.2-5.11.39-V01.5.run | >=4.5.2-5.11.38 / >=5.7.1-6.2.13a 且 <=5.7.1-6.2.18 | Z100L |
| DTK-22.10.1 | rock-4.5.2-5.11.38-V01.4.2.1a.run | >=4.5.2-5.11.38 / >=5.7.1-6.2.13a 且 <=5.7.1-6.2.18 | Z100L |
| DTK-22.10 | rock-4.5.2-0-5.11.36-1-V01.3.2.run | >=4.5.2-5.11.36 / >=5.7.1-6.2.13a 且 <=5.7.1-6.2.18 | Z100L |
| DTK-22.04.2 | rock-kernel-refactory-rock-4.5.2-0-5.11.34-V01.3.run | 仅支持当前一个驱动版本 | Z100L |
| DTK-22.04.1 | rock-4.5.2-0-5.11.32-Linux.rpm | 仅支持当前一个驱动版本 | Z100L |
| DTK-22.04 | rock-4.5.2-0-5.11.32-Linux.rpm | 仅支持当前一个驱动版本 | Z100L |
| DTK-21.10.1 | rock-4.0-23-5.6.25-3.10.0-957.el7.x86_64.rpm | 仅支持当前一个驱动版本 | Z100L |

- 选 DTK 时必须先确认**已装驱动版本**满足配套要求（见 dcu-driver），版本不匹配会导致
  hipcc 编译出的程序运行异常或设备不可用。
- 驱动版本查法：`hy-smi -a` 或 `cat /opt/hyhal/version`（若驱动已装）。

## 6. Pitfalls（实测/官方确认）

1. **每次使用前联网核查配套关系**：DTK 与驱动强绑定，官网配套表会更新。skill 内版本号为
   基线快照，**不构成承诺**。agent 执行安装前**必须用 curl 拉取最新配套表**，解析后与
   本 skill 第 5 节快照比对，以联网拉到的为准：
   ```bash
   # 该直链可直接 curl 获取（无需浏览器渲染），返回 UTF-8 表格文本
   curl -sL "https://download.sourcefind.cn:65024/file/1/DTK与驱动版本配套关系表.md"
   # 解析：提取 DTK 版本列与驱动版本要求列，与本 skill 第5节比对；
   # 若远程出现本快照没有的新 DTK 版本或驱动要求变更，以远程为准并告知用户。
   ```
   - 若运行环境无法访问该域名（纯内网隔离/无外网），则以本 skill 第 5 节快照为基线，
     并在输出中明确提示「配套表未联网复核，版本以快照为准，请人工确认」。
2. **容器不重复装 DTK**：docker 基础镜像已带 DTK，物理机下载解压那套**不要进容器**；
   容器内只需 `source /opt/dtk/env.sh`（若镜像 ENV 未自动加载）。误在容器内跑 tar 解压
   到 /opt 会污染镜像且与环境预期冲突。
3. **软链统一入口**：务必 `ln -sfn /opt/dtk-<version> /opt/dtk`，框架/编译脚本按
   `/opt/dtk` 定位；直接写死版本目录会导致升级时多处改路径。
4. **激活非持久**：`source /opt/dtk/env.sh` 只对当前 shell 生效。编译/跑框架的每条命令
   前都要 source，或写入 `~/.bashrc` / 构建脚本开头。CI/远程执行（ssh 非交互 shell）尤其
   容易漏 source 导致 `hipcc: command not found`。
5. **选包按内核而非仅看系统名**：同一发行版不同内核，DTK 包可能按内核区分。latest 目录
   无完全对应系统时，**选内核版本号最接近**的包，避免 glibc/内核符号不匹配。

## 7. 与包内其他 skill 的衔接

- **dcu-driver**：DTK 依赖驱动，先按 dcu-driver 装好并验证（`hy-smi -a` 可见），再装 DTK。
- **dcu-docker**：容器内 DTK 由镜像提供；选带 DTK 的基础镜像，不要容器内重装。
- **dcu-env**：安装前用 dcu-env 检测 OS / kernel / 已装驱动版本，确定该下哪个 DTK 包。
- **dcu-hy-smi**：装完 DTK 后 `source /opt/dtk/env.sh` + `hy-smi -a` 联合验证环境就绪。

## References

- DTK 下载：`https://download.sourcefind.cn:65024/1/main/latest`
- DTK 文档：`https://download.sourcefind.cn:65024/1/main/latest/Document`
- 配套关系表：`https://download.sourcefind.cn:65024/file/1/DTK与驱动版本配套关系表.md`
- 关联：dcu-driver（驱动）、dcu-docker（容器DTK）、dcu-env（环境检测）、dcu-hy-smi（验证）
