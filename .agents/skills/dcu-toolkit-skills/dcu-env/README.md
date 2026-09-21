# dcu-env — 海光 DCU 远程环境检测

远程环境检测 skill。检测目标机的 OS、内核、CPU 型号、DCU 驱动加载状态、DTK 版本、
PCIe 拓扑等，为驱动安装（dcu-driver）和拓扑采集（dcu-topology）提供前置信息。

## 核心内容

- OS / kernel / gcc / cmake 版本检测（驱动安装前置门槛）
- DCU 驱动加载验证（`lsmod | grep -E "hydcu|hycu"`）
- 设备节点 `/dev/kfd` `/dev/dri` 检查
- DCU 型号识别（参考 `../references/dcu-models.md`）

## 目录

```
dcu-env/
├── SKILL.md                      环境检测流程与判定
├── references/
│   └── bw100-pcie-real.md       实测 PCIe 拓扑记录（BW100）
└── scripts/
    └── vendor_dcuprofiletools/  第三方官方工具（dcuprofiletools，含 env_check）
```

> `scripts/vendor_dcuprofiletools/` 是海光官方性能/环境检测工具，作为 vendor 依赖收纳，
> **已被 `.gitignore` 排除**，不进 git 仓库（需要时从 sourcefind 官方仓库拉取）。

## 快速用法

```bash
# 检测单台
bash <(ssh root@host "cat env_check.sh")   # 或直接在目标机跑 vendor 工具

# 驱动加载验证（6.2 hydcu / 6.3 hycu 双匹配）
lsmod | grep -E "hydcu|hycu"
```

## 衔接

- 装驱动前：用 dcu-env 确认 kernel-devel 版本、gcc/cmake 是否满足（见 dcu-driver 门槛）
- 装驱动后：用 dcu-driver/check_driver.sh 验证，再用 dcu-topology 采拓扑

详见 `SKILL.md`。
