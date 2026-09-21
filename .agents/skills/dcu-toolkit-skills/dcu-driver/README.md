# dcu-driver — 海光 DCU 驱动安装与验证

DCU 驱动安装、版本兼容、加载验证与排障 skill。覆盖 rock 系列驱动与硬件（Z100/Z100L/
K100/K100-AI/BW100/BW150/BW1000/BW1100）及 DTK 版本兼容矩阵、CentOS/Ubuntu 依赖安装、
驱动 run 包安装流程、lsmod 验证驱动加载、容器内调试挂载。

## 核心内容

- **驱动兼容矩阵**：7 行硬件×驱动×DTK（Z100→BW1100 全系列），见 `SKILL.md` 第 1 节
- **安装流程**：依赖包（CentOS/Ubuntu 双分支）→ 驱动 run 包（`-A` 参数）→ `systemctl restart hymgr`
- **加载验证**：`scripts/check_driver.sh`（兼容 6.2 `hydcu` / 6.3 `hycu` 模块名）
- **Pitfalls**：模块名差异、容器内 `/sys/kernel/debug`、cmake/gcc 门槛、kernel-devel 一致、vbios 重启

## 目录

```
dcu-driver/
├── SKILL.md                 兼容矩阵 + 安装流程 + Pitfalls
└── scripts/
    └── check_driver.sh      驱动加载验证（模块/hymgr/设备节点/hy-smi/debug挂载）
```

## 快速用法

```bash
# 安装前检测（dcu-env）
# 安装后验证
bash scripts/check_driver.sh
# 预期：hycu 模块(6.3.x)加载、hymgr active、/dev/kfd /dev/dri 存在、hy-smi 见8卡
```

详见 `SKILL.md`。
