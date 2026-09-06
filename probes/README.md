# 隔离探针

从项目根目录运行，详见 docs/PROBE_PLAN.md。这些脚本不属于产品 API，也不安装或修改 external。

常用运行环境可以从项目根目录加载：

```sh
source scripts/activate_hygon_env.sh project
```

`project` 使用本项目 `.venv`；当前项目环境仍可能缺少部分 framework 依赖。已验证的 framework/HCU 探针环境可显式使用：

```sh
source scripts/activate_hygon_env.sh exploration
```

脚本会加载 DTK 26.04、设置项目 `PYTHONPATH`、HUST PyPI 镜像和 `/data/envs/uv-cache`。`HIP_VISIBLE_DEVICES`、`OMP_NUM_THREADS`、超时和探针参数仍由每条命令显式设置。

```sh
UV_CACHE_DIR=/tmp/alchemi-uv-cache uv venv --python 3.12 .venv
UV_CACHE_DIR=/data/envs/uv-cache uv pip install --python .venv/bin/python --index-url https://mirrors.hust.edu.cn/pypi/web/simple /data/envs/torch-2.9.0+das.opt1.dtk2604-cp312-cp312-manylinux_2_28_x86_64.whl /data/envs/triton-3.3.0+das.opt1.dtk2604.torch290-cp312-cp312-manylinux_2_28_x86_64.whl -c configs/probe-constraints.txt numpy packaging pyyaml
```

已有 .venv 时不要重建；安装增量依赖使用 configs/probe-constraints.txt。不运行上游 CUDA extras 的 uv sync。
