# ontrack 工单 CSV 提取方法（references/ontrack-csv-extraction.md）

从 Hygon ontrack system 导出的 CSV 工单里提取可复用经验时，用下面这套方法，避免 csv.reader 把多行线索拆错、漏读。

## 为什么不用 csv.reader
ontrack 导出的 CSV 单元格内大量包含换行、竖线、图片占位（`!image-xxx.png!`）、管道符，csv.reader 按 RFC 解析会把一个工单拆成多"行"，导致：
- 主键（CSD 编号）和线索分离
- wc -l 统计的"工单数"虚高（实测 7955 物理行 ≠ 635 真实工单）

## 主键切分法（推荐）
真实工单主键格式为 `|CSD-数字|`，用正则切分即可准确拿到每个工单边界：

```python
import re
text = open(csv_path, encoding='utf-8').read()
# 每个工单以 "=== CSD-xxxxx | 型号=[...] ===" 起头
tickets = re.split(r'(?=== CSD-\d+ \|)', text)
# tickets[1:] 即各工单块；型号在标题行，线索在"线索:"之后
```

## 覆盖校验（必须做，用户要求全量覆盖）
提取后核对"没漏读"，用主键计数 + 字符覆盖率双指标：
```python
ids = re.findall(r'CSD-(\d+)', text)
print("去重工单数:", len(set(ids)))
# 字符覆盖率：提取稿字符数 / 原 CSV 字符数（去空白），>99.8% 视为基本全量
```

## 去重与并集
两份 CSV 各自提取后，按 CSD 编号求并集：
```python
s1, s2 = set(ids1), set(ids2)
print("交集:", len(s1 & s2), "并集:", len(s1 | s2))
# 实测：文档1=635、文档2=685、交集=1、并集=1319
```

## 二次筛选信号词（落地前用 grep 量化）
剔除无复用价值的噪声，保留带硬核修复信号的：
- 保留信号：`export \w+` / `unset \w+` / `VLLM_\w+` / `NCCL_\w+` / `LD_LIBRARY_PATH` / `source /opt` / `vbios` / `cmake` / `gcc` / `GRUB` / `BIOS` / `iommu` / `acs` / `exclude=kernel` / `max_map_count` / `HYHAL` / `seccomp`
- 剔除信号：`pan.baidu.com` / `密码` / `提取码` / 真实 IP / "已解决可关闭" / "待反馈" / 纯"更换...镜像...解决"无根因

## 脱敏自检（落地前必跑）
在 skill 目录下 grep 确认无内网域/真实 IP/凭证/用户名残留：
```
grep -rnoE "harbor\.sourcefind\.cn|image\.sourcefind\.cn|密码|提取码|<真实IP>" .
```
内网域统一记 `<internal_registry>`。
