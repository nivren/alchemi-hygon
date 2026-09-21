# BW 系列 DCU 真实 PCIe 硬件 ID（实测）

> 本文件记录 dcu-env 在真实设备上运行 dcuprofiletools 时发现的关键硬件事实。
> 优先于 SKILL.md / dcu-models.md 中的推断值。

## 实测环境

- 设备：BW100 机型（8 卡）
- 系统：Kylin V10 (kernel 4.19.90-89.11.v2401.ky10)

## 真实 lspci 输出（节选，每张卡一行，共 8 行）

```
07:00.0 Co-processor [0b40]: Chengdu C-3000 IC Design Co., Ltd. BW [1d94:6370] (rev 01)
0a:00.0 Co-processor [0b40]: Chengdu C-3000 IC Design Co., Ltd. BW [1d94:6370] (rev 01)
0f:00.0 Co-processor [0b40]: Chengdu C-3000 IC Design Co., Ltd. BW [1d94:6370] (rev 01)
16:00.0 Co-processor [0b40]: Chengdu C-3000 IC Design Co., Ltd. BW [1d94:6370] (rev 01)
...（共8张）
```

- 厂商 ID：`1d94` = Chengdu C-3000 IC Design Co., Ltd.（海光 DCU）
- 设备 ID：**`6370`**
- 设备类：`Co-processor [0b40]`

**查询命令（供 dcu-topology 复用）：**
```bash
lspci -nn | grep -iE '1d94:|co-processor'
```

## 与 dcuprofiletools 工具的冲突（重要陷阱）

工具的 `dcu_performance_analyzer.py` 内置 `dcu_id_map` 仅含：
`54b7→Z100, 55b7→Z100L, 62b7→K100, 6210→K100-AI, 6211→K100-AI-ECO, 6320→BW1000`

**`6370` 不在其映射表中** → 工具在 BW100 上运行 `check_pcie_devices` 时报
**“检测到 0 张 DCU”**（实测有 8 张）。

**结论**：处理 BW 系列时，**不要信任该工具的 DCU 数量/型号识别**。
dcu-topology 必须自行解析 `lspci -nn`（关键字 `1d94:` 或 `Co-processor`）来统计卡数。

## 型号 ↔ PCIe ID 对照（修正后）

| 厂商:设备ID | 型号 | 来源 |
|---|---|---|
| 1d94:6370 | **BW100（真机实测 8卡）** | 真机 lspci |
| 1d94:6320 | BW1000（工具源码标注，待实测） | dcuprofiletools |
| 54b7 | Z100 | dcuprofiletools |
| 55b7 | Z100L | dcuprofiletools |
| 62b7 | K100 | dcuprofiletools |
| 6210 | K100-AI | dcuprofiletools |
| 6211 | K100-AI-ECO | dcuprofiletools |

> ⚠️ 用户曾备注 BW100/BW150/BW1100/BW10 共用某内部代号（记为 6320），
> 但**真机实测 BW100 = 1d94:6370**。以实测为准。
> BW150/BW1100/BW10 的实测 ID 待后续设备采集补充（第二台 BW150×8 机型因密码问题本次未连上）。
