# 项目环境 pytest 基线

日期：2026-09-06 UTC  
环境：项目 `.venv`、CPython 3.12.13、DTK 26.04、海光 Torch `2.9.0+das.opt1.dtk2604`。

项目环境原先没有 pytest。本轮按北外镜像安装并固定：

```text
pytest==8.4.2
pytest-asyncio==1.4.0
```

可重跑方式：

```bash
source scripts/activate_hygon_env.sh project
pytest -q packages/ops/test/torch/test_torch_reference_backend.py
pytest -q packages/framework/test/hooks/test_optional_imports.py \
  packages/framework/test/hooks/test_neighbor_list_torch_reference.py
```

结果：

- ops reference：`10 passed`，退出码 `0`；
- framework optional-import/neighbor Hook：`10 passed`，退出码 `0`；
- pytest 插件加载 `pytest-asyncio==1.4.0`，不再出现 `asyncio_mode` 配置警告。

两个包的上游测试目录都使用顶层 `test` 包名。把两包测试目录放在同一个 pytest 命令中会触发 `ImportPathMismatchError`，因此当前基线按包分开执行；这属于测试入口组织问题，不是算子失败。

可重建规格：

- 输入规格：`configs/hygon-reference.in`；
- 当前主机精确冻结：`configs/hygon-reference-lock.txt`；
- 生成脚本：`scripts/freeze_hygon_env.sh`；
- 历史/本轮冻结摘要：`reports/probe-environment-freeze.txt`。
