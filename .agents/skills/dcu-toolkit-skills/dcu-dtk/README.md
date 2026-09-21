# dcu-dtk — 海光 DCU DTK 开发工具包安装与激活

DTK（DCU Toolkit）是海光 DCU 的开发工具包（HIP 编译器、运行时、数学库、框架支持）。

## 适用范围

- **物理机**：需下载安装并激活（`source /opt/dtk/env.sh`），框架编译运行前置。
- **docker 容器**：镜像已自带 DTK，**不需下载安装**，只需激活或镜像已激活。

## 关键流程（详见 `SKILL.md`）

1. 从 `https://download.sourcefind.cn:65024/1/main/latest` 按系统选 DTK 包（无对应系统选内核最接近）
2. 解压到 `/opt`，建软链 `ln -sfn /opt/dtk-<version> /opt/dtk`
3. 激活：`source /opt/dtk/env.sh`
4. 验证：`which hipcc` / `echo $DTK_PATH` / `hy-smi -a`

## ⚠️ 版本配套（必须联网复核）

DTK 与驱动强绑定。每次使用前核查配套关系表最新版本（agent 用 curl 拉取直链，无需浏览器）：
`https://download.sourcefind.cn:65024/file/1/DTK与驱动版本配套关系表.md`
以联网查到的最新 DTK / 驱动配套为准（skill 内版本号为基线快照，非实时承诺）。
