# dcu-ontrack-experience

DCU 故障排查经验速查手册（源自 Hygon ontrack 工单沉淀）。

按"**现象关键词 → 设备型号 → 根因 → 规避/修复手段**"反查历史踩坑经验，覆盖驱动/容器、vLLM/SGLang 推理、RCCL/NCCL 通信、库缺失/编译/框架版本、硬件 RMA 提示五大类。

## 怎么用
遇到报错先在 `SKILL.md` 里按关键词 grep，核对**设备型号 + 驱动/DTK 版本 + 报错原文**是否匹配再动手。

## 重要提醒
- 本手册**仅供参考，不保证完全修复**，实际环境千差万别，勿无脑套用。
- 涉及**重启物理机 / 重装驱动 / 刷 vbios / 改 BIOS / 拔电源**等高危操作务必谨慎：确认环境（产线/客户）、确认有带外恢复通道、优先在测试机验证。详见 `SKILL.md` 的「使用须知」。

## 入口
完整经验表与免责声明见同级 `SKILL.md`。

## 速查参考（references/）
子 skill 主文只做索引，详细手册在 `references/` 下按需查阅：

| 文件 | 内容 |
|---|---|
| `references/xid_sxid_errors.md` | **XID / SXID 错误码速查**（来源：HYGON DCU XID/SXID 错误手册 v2.2.0）。XID 驱动内核日志错误、SXID HySwitch 互联错误；含查看方法、错误码含义、处置建议 |
| `references/pack-enrich-maintenance.md` | 存量 enrich 维护规范（不新建子 skill 时如何并入 references、脱敏自检） |
| `references/ontrack-csv-extraction.md` | ontrack 工单 CSV 提取方法（主键切分法，避免 csv.reader 拆错工单边界） |

> XID/SXID 是海光 DCU 报错定位的关键入口——遇到驱动/硬件类报错，先 `dmesg` 抓 XID 再在本表反查。
