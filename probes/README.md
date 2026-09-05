# 隔离探针

从项目根目录运行，详见 docs/PROBE_PLAN.md。这些脚本不属于产品 API，也不安装或修改 external。

```sh
UV_CACHE_DIR=/tmp/alchemi-uv-cache uv venv --python 3.12 .venv
UV_CACHE_DIR=/data/envs/uv-cache uv pip install --python .venv/bin/python --index-url https://mirrors.hust.edu.cn/pypi/web/simple /data/envs/torch-2.9.0+das.opt1.dtk2604-cp312-cp312-manylinux_2_28_x86_64.whl /data/envs/triton-3.3.0+das.opt1.dtk2604.torch290-cp312-cp312-manylinux_2_28_x86_64.whl -c configs/probe-constraints.txt numpy packaging pyyaml
```

已有 .venv 时不要重建；安装增量依赖使用 configs/probe-constraints.txt。不运行上游 CUDA extras 的 uv sync。
