# G1 数据导入链审计（2026-09-05）

## 结论

锁定源码中的 `AtomicData`/`Batch` 公共数据 API 目前被 Warp 运行时依赖阻断，不能在没有 Warp 的基础 Torch 环境中导入。该问题发生在导入阶段，不是某个具体邻居或动力学算子调用阶段。

## 复现环境与命令

使用探索环境的 Python 3.12 和项目源码路径：

```sh
PYTHONPATH=packages/framework:packages/ops \
  /home/wangleping/codes/nvalchemi-toolkit/.venv/bin/python -c 'import nvalchemi.data.atomic_data'
```

退出码为 1，错误为 `ModuleNotFoundError: No module named 'warp'`。以下模块在独立进程中均得到相同结果：`nvalchemi.data.atomic_data`、`nvalchemi.data.data`、`nvalchemi.data.batch`、`nvalchemiops`、`nvalchemiops.torch`。`import nvalchemi` 与 `import nvalchemi._optional` 可通过，因为它们不触发数据存储或 ops 初始化。

## 静态依赖链

```text
nvalchemi.data.__init__
  -> transforms / Compose
  -> Batch
     -> level_storage
        -> buffer_kernels
           -> import warp; wp.init()
        -> import warp; wp.init()
  -> datapipes
     -> zarr / Dataset / DataLoader / Batch
```

`AtomicData` 本身在 `atomic_data.py` 顶层只直接依赖 NumPy、periodictable、Torch、Pydantic 和 `OptionalDependency`；ASE 与 pymatgen 使用 `OptionalDependency.*.require` 装饰器延迟检查。但 Python 导入 `nvalchemi.data.atomic_data` 前会先执行 `nvalchemi.data.__init__`，因此仍会经过 `Batch` 的 Warp 链。

`Batch` 的存储实现依赖 `level_storage.py`。该文件顶层导入 Warp 并调用 `wp.init()`；`buffer_kernels.py` 也顶层导入 Warp 并调用 `wp.init()`。`nvalchemiops/__init__.py` 同样顶层导入 Warp 并初始化 Warp。当前环境未安装 Warp；PhysicsNeMo 可被发现，但不在这条数据导入失败链上。

## 影响与边界

- 这是 `data.atomic`、`data.batch` 和基础 `ops` 命名空间的 import-time 耦合，属于 G1 前置阻断，不应通过安装 NVIDIA Warp 或静默 CPU 回退掩盖。
- 上游 API、字段语义、Batch 索引偏移和分层存储逻辑尚未修改；本报告只记录审计证据。
- 后续隔离必须保持 `from nvalchemi.data import AtomicData, Batch` 的公共导入路径，并把 Warp 相关实现变成明确的可选后端或在调用点 fail-fast；不能删除高性能路径或改变 Batch 语义。
- 当前未运行上游测试，也未进行 DCU 设备导入；本结果不等同于 Warp 在 gfx936 上不可用。

## 下一小步

先为数据 API 设计最小导入边界：让仅使用 `AtomicData` 的代码不触发 Warp 存储初始化，同时保留 `Batch` 的显式后端错误信息。实现前补充独立 import smoke 测试和依赖矩阵，再决定是延迟导入还是拆分 Warp-backed storage 模块。
