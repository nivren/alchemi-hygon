# G2 fixed-cell neighbor runtime selection contract

日期：2026-09-20。任务：`TORCH-NEIGHBOR-PBC-CELL` 阶段二，Point 36.2。

## 目标

将 framework 的两个邻居入口：

- `nvalchemi.neighbors.compute_neighbors`；
- `nvalchemi.hooks.NeighborListHook`；

统一到同一个 runtime topology selection contract，避免未来 native HIP wrapper
在两个入口分别实现 capability 判断。

## 实现

新增 `nvalchemi._backend.resolve_neighbor_list_backend`，统一构造：

- `periodic` / `no_pbc`；
- `full` / `half`；
- `matrix` / `coo`；
- 显式 neighbor `method` strategy；
- device、dtype 和中央 registry 的 `BackendSelection`。

该 helper 只描述邻居拓扑能力，不把 topology selection 推断成 geometry、force
gradient 或完整 native pipeline capability。`backend=None`/`warp` 仍不接收 registry
strategy；显式 unsupported backend 仍由中央 registry 明确失败，不执行 silent fallback。

## CPU 验证

```bash
PYTHONPATH=packages/framework:packages/ops .venv/bin/python -m pytest -q \
  packages/framework/test/models/test_neighbors_torch_reference.py \
  packages/framework/test/hooks/test_neighbor_list_torch_reference.py \
  packages/framework/test/models/test_lj_torch_reference.py
```

结果：`34 passed, 1 warning`，退出码 `0`。

`py_compile` 和 `git diff --check` 通过。

## 证据边界

本点只改变 framework 内部 selection construction，不登记 native HIP，不改变
Torch reference、Warp、`auto` 或 registry capability 宽度。未新增 HCU kernel 或
性能证据；Point 35 的 gfx936 isolated full-pipeline 结果仍不能直接进入 runtime。

## 下一步

Point 36.3 定义 native HIP runtime wrapper 的最小 ABI 和 admission gate：明确 build
workspace/lifetime、query candidate、public topology、Torch geometry 及 unsupported
gradient/rebuild/variable-cell 请求；在该 gate 通过前继续保持 `backend="hip"` 未注册。
