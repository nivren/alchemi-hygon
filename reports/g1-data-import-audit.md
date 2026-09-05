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

## 第一阶段实现结果

`packages/framework/nvalchemi/data/storage_backend.py` 现已加入无 Warp 的 `TorchStorageBackend`，覆盖协议中的 uniform/segmented fit mask、masked put、defrag 和 segment expansion。`probes/storage_backend_probe.py` 在项目 `.venv` 中退出 0，验证了：

- uniform 行按源顺序写入首个空槽，复制掩码和目标掩码原位更新，defrag 保留顺序并清零尾部；
- segmented 长度 `[2, 3, 1]` 的选择、目标 batch pointer 追加和容量边界；
- 包含零长度 segment 的 defrag、pointer 尾部填充和 element index expansion；
- 全部计算仍在 Torch 当前 device 上，模块加载未导入 Warp。

该实现目前尚未接入 `LevelStorage`；`from nvalchemi.data import AtomicData, Batch` 仍会触发原有 Warp import-time 链。下一步只处理后端注入和延迟导入，不同时实现 Triton/HIP。

## 第二阶段实现结果

`LevelStorage` 已通过 `StorageBackend` 组合调用 Torch reference；uniform/segmented 的 fit mask、put、defrag 和 segment expansion 不再从 `level_storage.py` 顶层导入 `buffer_kernels`。无 Warp 环境中，`from nvalchemi.data import AtomicData, Batch` 已可成功导入，且构造的 storage 默认报告 `backend.name == "torch"`。

验证命令：

```sh
PYTHONPATH=packages/framework:packages/ops \
  /home/wangleping/codes/nvalchemi-toolkit/.venv/bin/python -c \
  'from nvalchemi.data import AtomicData, Batch; print(AtomicData.__name__, Batch.__name__)'
```

退出码为 0。`PYTHONPATH=packages/framework:packages/ops /home/wangleping/codes/nvalchemi-toolkit/.venv/bin/python -m pytest -q packages/framework/test/data/test_batch.py` 得到 90 passed、4 skipped（无 CUDA 设备）、1 failed；唯一失败为 `test_pin_memory`，错误是当前进程没有 HIP GPU（`RuntimeError: No HIP GPUs are available`），不涉及 storage backend。`probes/storage_backend_probe.py` 仍退出 0。

当前仍保留 `level_storage.py` 中的 Warp helper 声明作为 NVIDIA 参考，但它不在导入阶段初始化；后续应把该遗留声明移入专用 Warp backend，避免假对象兼容层长期存在。
