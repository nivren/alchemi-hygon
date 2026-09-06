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

`exploration` 只指向前期探索仓库的只读环境，用于复现已有依赖和对照结果；正式安装目标仍是 `project`。当前缓存 MACE 的最小 wrapper/batching smoke 可这样运行：

```sh
source scripts/activate_hygon_env.sh exploration
python probes/mace_wrapper_reference.py \
  --device cpu \
  --checkpoint /home/wangleping/.cache/mace/MACE-OFF23_small.model \
  --cif /data/csp_data/perf_46/formal_c1_1_10_z1_46.cif \
        /data/csp_data/perf_46/formal_c1_1_11_z1_46.cif
```

HCU 运行必须在 DTK 已加载且设备节点可见的主机环境中显式设置 `HIP_VISIBLE_DEVICES`；结果和超时退出码写入 `artifacts/g1`，详见 `docs/PROBE_PLAN.md` 与 `reports/g1-mace-wrapper-batch.md`。

脚本会加载 DTK 26.04、设置项目 `PYTHONPATH`、HUST PyPI 镜像和 `/data/envs/uv-cache`。`HIP_VISIBLE_DEVICES`、`OMP_NUM_THREADS`、超时和探针参数仍由每条命令显式设置。

```sh
UV_CACHE_DIR=/tmp/alchemi-uv-cache uv venv --python 3.12 .venv
UV_CACHE_DIR=/data/envs/uv-cache uv pip install --python .venv/bin/python --index-url https://mirrors.hust.edu.cn/pypi/web/simple /data/envs/torch-2.9.0+das.opt1.dtk2604-cp312-cp312-manylinux_2_28_x86_64.whl /data/envs/triton-3.3.0+das.opt1.dtk2604.torch290-cp312-cp312-manylinux_2_28_x86_64.whl -c configs/probe-constraints.txt numpy packaging pyyaml
```

已有 .venv 时不要重建；安装增量依赖使用 configs/probe-constraints.txt。不运行上游 CUDA extras 的 uv sync。
