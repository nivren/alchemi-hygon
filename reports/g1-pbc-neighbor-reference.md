# G1 PBC neighbor reference slice

日期：2026-09-05  
上游锁定：framework `4dfe3723def34df3fadb245981081ccf8c94c257`，ops `26dbceb61e30cca80e1a5805eebeb51d7dc68fd1`

本步只验证 Torch reference 的周期邻居拓扑和整数 image shift，不宣称 LJ PBC 力、skin/rebuild、半邻居、应力或 NVE 已支持。shift 约定沿用上游矩阵契约：

```text
r_ij = r_i - r_j - shift @ cell
```

实现支持批量 cell、部分 PBC 和一般可逆三斜胞的全邻居列表；小输入使用 eager Torch/Python 枚举，后续再按实测瓶颈替换为 Triton/HIP。PBC `half_fill=True` 明确报 `NotImplementedError`，避免猜测周期 image 的唯一归属规则。

CPU 验证：

```bash
PYTHONPATH=packages/ops \
  /home/wangleping/codes/nvalchemi-toolkit/.venv/bin/python -m pytest -q \
  packages/ops/test/torch/test_torch_reference_backend.py

PYTHONPATH=packages/framework:packages/ops \
  /home/wangleping/codes/nvalchemi-toolkit/.venv/bin/python \
  probes/pbc_neighbor_reference.py --device cpu
```

结果：ops `8 passed`；framework `compute_neighbors` 与 `NeighborListHook` 的 PBC 测试纳入此前 framework 邻居测试并通过。正交胞边界案例使用 cell `(2, 10, 10)`、cutoff `0.5`、坐标 `0.1/1.9`，两行邻居为 `[[1], [0]]`，shift 为 `[[[-1, 0, 0]], [[1, 0, 0]]]`。非正交 cell 也通过同一 shift 契约测试。

HCU 重跑命令：

```bash
source /opt/dtk-26.04/env.sh
PYTHONPATH=packages/framework:packages/ops \
  HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 timeout 60 \
  /home/wangleping/codes/nvalchemi-toolkit/.venv/bin/python \
  probes/pbc_neighbor_reference.py --device cuda
```

退出码 `0`；设备为 `BW200, UBB BW1000`，backend 为 `torch_reference`，邻居矩阵、shift 和计数与 CPU/解析案例一致。原始摘要：`artifacts/g1/pbc_neighbor_reference_hcu0.json`。

限制：当前框架 LJ wrapper 仍明确拒绝带 PBC shift 的输入；下一步应先把 shift 纳入力/能量公式并用解析 FP64 对照，再接入短 NVE。周期 half-list、动态 skin/rebuild、邻居容量自动扩展和生产 Warp 默认路径均未改变。
