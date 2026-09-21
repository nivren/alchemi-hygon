---
name: dcu-driver
description: 海光DCU驱动安装、版本兼容、驱动加载验证与排障。覆盖 rock 系列驱动与硬件(Z100/Z100L/K100/K100-AI/BW100/BW150/BW1000/BW1100)及DTK版本兼容矩阵、CentOS/Ubuntu依赖安装、驱动run包安装流程、lsmod验证驱动加载、容器内调试挂载。DCU上架部署、驱动升级、环境搭建必读。
license: MIT
metadata:
  version: 1.0.0
  author: DCU Engineer
  hermes:
    tags: [dcu, hygon, driver, rock, dtk, installation, deploy, hymgr]
    related_skills: [dcu-env, dcu-docker, dcu-topology, dcu-hy-smi, dcu-ontrack-experience]
---

# dcu-driver — 海光 DCU 驱动安装与验证

> 📌 **遇到具体报错先查排查经验库**：驱动加载失败、容器看不到卡、vbios/内核冲突等具体现象的处理经验（含 env/修复命令），统一收在 **`dcu-ontrack-experience`** 这个 skill 里。本 skill 只讲安装流程与兼容矩阵，具体操作手法请去那里按"现象关键词"反查，避免重复维护。

## 1. 驱动兼容列表（硬件 × 驱动 × DTK）

| 驱动版本 | 支持硬件 | 推荐 DTK 版本 |
|---|---|---|
| rock-4.0-23-5.6.25 | Z100 / Z100L | 21.04 / 21.10 / 22.04 |
| rock-4.5.2-5.11.40 | Z100 / Z100L | 22.04 / 22.10 / 23.04 |
| rock-5.2.0-5.16.18 | Z100 / Z100L | 22.10 / 23.04 |
| rock-5.2.0-5.16.29 | Z100 / Z100L / K100 | 23.04 / 23.10 |
| rock-5.7.1-6.2.26+ | Z100 / Z100L / K100 / K100-AI | 24.04 / 25.04 |
| rock-6.3.8+ | BW1000 / K100-AI / K100 / Z100L / Z100 | 25.04* |
| rock-6.3.30+ | BW1100 / BW1000 / BW150 / BW100 / BW10 / K100-AI / K100 / Z100L / Z100 | 26.04 |

> 选型原则：先按**硬件型号**锁定驱动大版本，再按目标 **DTK/OS** 版本选具体驱动。新硬件（BW 系列、K100-AI）只能用 6.3.x 以上驱动。

## 2. 驱动安装流程

### 2.1 安装依赖包

**CentOS：**
```bash
yum install -y rpm-build gcc-c++ cmake automake elfutils-libelf-devel libdrm libdrm-devel pciutils
yum install -y kernel-devel-`uname -r` kernel-modules-extra
```

**Ubuntu：**
```bash
apt install -y gcc g++ cmake automake libelf-dev libdrm-amdgpu1 libtinfo5 pciutils libdrm-dev
apt install -y linux-headers-`uname -r` linux-modules-extra
```

> ⚠️ **kernel 版本必须与 kernel-devel / linux-headers 版本一致**（`uname -r` 动态取值已处理）。
> ⚠️ **新版本驱动依赖 cmake > 3.2、gcc > 7.3**。安装前确认：`cmake --version`、`gcc --version`。

### 2.2 安装驱动

1. **下载驱动**：<internal_driver_mirror>/6/main/latest驱动
2. **赋权**：`chmod +x rock*.run`
3. **安装**：
   ```bash
   ./rock*.run          # 非虚拟场景推荐添加 -A 参数
   ```
4. **重启驱动服务**：
   ```bash
   systemctl restart hymgr
   ```

**安装注意事项：**
- 安装参数可用 `./rock*.run -h` 查看
- **安装新驱动无需先卸载旧驱动**：安装过程会自动卸载老驱动
- 安装中会遇 `Do you want to remove legacy hymgr config file? [Y/n]:`
  - **Y**：重新生成 `/etc/hymgr.cfg`
  - **N**：保留该文件（已有调优配置时选 N）
- 非 Z100L 环境会提示 `Update vbios? [y/N]:`
  - **首次安装建议 Y**（更新 vbios 后重启物理机）
  - 后续升级驱动可选 N
  - ⚠️ **云计算场景**：物理节点重启受严格管控，可不更新 vbios（Y 需重启物理机，N 不重启）

### 2.3 验证驱动加载

**6.2.x 及以下的驱动（模块名 `hydcu`）：**
```bash
lsmod | grep -E "hydcu|hycu"
# 预期：hydcu / hydcu_sched / hyttm / hykcl / hy_extra / amd_iommu_v2 / drm_kms_helper / drm
```

**6.3.x 及以上的驱动（模块名 `hycu`）：**
```bash
lsmod | grep -E "hydcu|hycu"
# 预期：hycu / hycu_sched / hyttm / hykcl / hydrm_buddy / hy_extra / iommu_v2 / drm_kms_helper / drm
```

> 🔑 **关键区分**：6.2.* 以前驱动模块名是 `hydcu`；**6.3.* 以后改为 `hycu`**。
> 验证脚本必须同时匹配两者（`grep -E "hydcu|hycu"`），否则 6.3 机器会误判"驱动未加载"。

## 3. Pitfalls（实测/官方确认）

1. **驱动模块名 6.2/6.3 差异**：`hydcu`（≤6.2）vs `hycu`（≥6.3）。验证务必 `lsmod | grep -E "hydcu|hycu"`，单匹配一种会漏。本包验证机型均为 6.3.x（`hycu`）。
2. **容器内调试挂载**：**6.3.x 驱动下，docker 调用 DCU 时推荐挂载 `/sys/kernel/debug`**，否则容器内无法获取 DCU 进程信息（hy-smi `--showpids` 等失效）。docker run 加 `-v /sys/kernel/debug:/sys/kernel/debug` 或 compose 挂载。
3. **cmake/gcc 版本门槛**：新驱动要求 cmake > 3.2、gcc > 7.3。老旧 OS（如 CentOS 7 默认 cmake 2.8）需先升级，否则驱动编译/安装失败。
4. **kernel-devel 版本必须一致**：`uname -r` 与 `kernel-devel-$(uname -r)` / `linux-headers-$(uname -r)` 必须同版本，否则驱动模块编译失败。升级 kernel 后必须重装对应 devel 包再装驱动。
5. **hymgr 服务**：安装后 `systemctl restart hymgr` 是必须步骤（不重启服务，hy-smi / 设备节点可能不可用）。
6. **vbios 更新与重启**：非 Z100L 首次安装选 Y 更新 vbios 后**必须重启物理机**才生效；云环境因重启管控可选 N（牺牲 vbios 更新换稳定性）。
7. **`-A` 参数**：非虚拟（裸金属）场景推荐 `./rock*.run -A`（具体含义见 `./rock*.run -h`，通常与虚拟化/直通相关）；虚拟化/云场景按 -h 说明决定是否加。
8. **驱动自带的 hy-smi**：装驱动后 `hy-smi` 位于 `/opt/hyhal/bin/hy-smi`（宿主机与容器内均有，不一定在 PATH）——见 dcu-hy-smi skill。
9. **重装驱动后必须重启物理机**：驱动升级（尤其带 vbios 更新，见注意事项第 4 条）后，必须重启设备驱动才完全生效；不重启直接跑 rccl-test 会因驱动状态不一致导致带宽异常或设备不可用。来自内部测试手册"基本环境准备"。
10. **BIOS 关闭 IOMMU/ACS（rccl-test 前置，仅 PCIe 卡适用）**：关闭 IOMMU/ACS 的要求**主要针对 PCIe 形态卡**（K100-AI、BW10、BW100、BW150 等）——这类卡走 PCIe P2P 拓扑，BIOS 须关闭 ACS 和 IOMMU（或内核 `iommu=off amd_iommu=off`），否则通信带宽不达标。**OAM 形态产品（BW1000、BW1100）基本无影响，无需为此改 BIOS/内核**。检测见 dcu-env `scripts/check_iommu_acs.sh`。
11. **重启后驱动可能未自动加载**：部分设备装驱动并重启物理机后，驱动未自动加载，`hy-smi` 报设备不可用；需手动执行 `hy-smi --loaddriver` 重新加载驱动，再 `hy-smi -a` 验证设备可见。

## 4. 与包内其他 skill 的衔接

- **dcu-env**：安装前用 dcu-env 检测 OS / kernel / 已装驱动版本，确定该装哪个驱动
- **dcu-hy-smi**：装完驱动后第一件事——`hy-smi -a` 验证设备可见；`/opt/hyhal/bin/hy-smi --showtoponuma` 验拓扑
- **dcu-topology**：驱动加载正常后，`dcu-topology/scripts/collect_topology.sh` 才能拿到真实卡↔NUMA（依赖 hy-smi）
- **dcu-docker**：容器内用 DCU 必须挂 `/dev/kfd` `/dev/dri` 且 6.3.x 额外挂 `/sys/kernel/debug`

## References

- 驱动下载：<internal_driver_mirror>/6/main/latest驱动
- 关联：dcu-env（环境检测）、dcu-hy-smi（驱动自带SMI）、dcu-topology（拓扑采集）、dcu-docker（容器挂载）
