# G1 存储后端契约与多属性复制

日期：2026-09-05

## 变更

`UniformLevelStorage.put` 和 `SegmentedLevelStorage.put` 对多个共同属性复用第一次属性确定的复制集合。uniform 存储复用同一目标槽位；segmented 存储复用同一 `batch_ptr` 起点和 segment offset。这样 `copied_mask`、属性数据和 segment 边界保持一致。

## 可重跑验证

```bash
PYTHONPATH=packages/framework:packages/ops \
  /home/wangleping/codes/nvalchemi-toolkit/.venv/bin/python \
  -m pytest -q packages/framework/test/data/test_storage_backend.py

PYTHONPATH=packages/framework:packages/ops \
  /home/wangleping/codes/nvalchemi-toolkit/.venv/bin/python \
  -m pytest -q packages/framework/test/data/test_level_storage.py
```

结果：

```text
test_storage_backend.py: 3 passed
test_level_storage.py: 96 passed
```

新增 smoke 子进程继承 DTK/动态库环境，只覆盖 `PYTHONPATH`，验证 `from nvalchemi.data import AtomicData, Batch` 成功且 `warp` 不在 `sys.modules`。新增的两个数据测试使用 CPU Torch reference，覆盖双属性 uniform 和 segmented put 的同槽位/同 offset 语义。
