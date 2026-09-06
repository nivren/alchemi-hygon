# G1 MACE 依赖审计

日期：2026-09-06  
范围：确认真实 MACE 探针的依赖边界和可运行前置条件，并记录用户认可的本地 checkpoint；不修改项目环境。

## 审计命令

```bash
source scripts/activate_hygon_env.sh exploration
python - <<'PY'
import importlib.metadata as md
import importlib.util
import torch

print("python", __import__("sys").executable)
print("torch", torch.__version__, torch.version.hip)
for name in ("mace", "e3nn", "ase", "torch_scatter", "torch_geometric"):
    print(name, importlib.util.find_spec(name), end="")
    try:
        print("", md.version(name))
    except md.PackageNotFoundError:
        print("")
PY
```

实际审计使用探索环境 `/home/wangleping/codes/nvalchemi-toolkit/.venv`，并另外检查项目环境 `.venv` 和 `UV_CACHE_DIR=/data/envs/uv-cache` 的缓存元数据。

## 结果

- 探索环境可以导入 `mace`、`e3nn` 和 `ase`，版本分别为 `mace-torch 0.3.16`、`e3nn 0.4.4`、`ase 3.29.0`。其 Torch 为海光构建 `2.9.0+das.opt1.dtk2604`。
- MACE 的声明依赖还包括 `torch-ema`、`prettytable`、`matscipy`、`h5py`、`torchmetrics`、`python-hostlist`、`configargparse`、`GitPython`、`PyYAML`、`tqdm`、`lmdb`、`orjson`、`matplotlib`、`pandas` 等；这些依赖是否需要安装要以选定 checkpoint 和实际导入路径为准。
- 项目 `.venv` 目前没有 MACE/e3nn/ASE 运行时，仍只保持已冻结的海光 Torch/Triton、NumPy 等探针依赖。此次没有执行安装，也没有替换海光 Torch/Triton。
- `/data/envs/uv-cache` 中能找到 `mace-torch 0.3.15` 和 e3nn 的缓存/索引痕迹，因此后续可以优先做离线解析或从缓存安装；缓存存在不等于 wheel 与海光 Torch、gfx936 运行时兼容。
- 用户确认 `~/.cache/mace` 中的模型权重可作为可信来源。`MACE-OFF23_small.model` 可加载为 `ScaleShiftMACE`，覆盖 H/O/C/N 等元素，`r_max=4.5`；SHA256 为 `165cce4cfec5a34b9c64d4ebf95de15d71106bb584b7291c8470f0749977c46f`。
- `probes/mace_probe.py` 已在探索环境 CPU 和 source DTK 26.04 的 BW200/gfx936 HCU 上退出 `0`，检查能量、力、力损失到模型参数的非零梯度；详细结果见 `reports/g1-mace-wrapper-batch.md`。这只是原始 MACE 模型证据，不等同于 framework wrapper 或 dynamics 支持。
- `probes/mace_wrapper_reference.py` 已加入正式 wrapper + Torch reference neighbor + Batch smoke；H₂O 单/双体系、单个 `perf_46` 和两个 `perf_46` CIF 的 CPU/HCU batching 均通过。向量化周期 Torch reference 邻居后，双体系 HCU 结果为 `batch_ptr=[0,46,92]`、1754 条边、无跨体系边、逐体系总力最大分量约 `9.6e-7`；向量化前的 180 秒超时（退出 `124`）仍作为优化前证据保留，详见 wrapper 报告。

## 解释和边界

审计命令在受限探针沙箱中观察到 `torch.cuda.is_available()=False`；该沙箱不暴露 `/dev/kfd`、`/dev/dri`，不能据此判断 HCU 不可用。此前 source DTK 26.04、可见 BW200/gfx936 环境中的 Torch/HCU 证据仍以对应 reports 为准。

P06 目前为 `partial`：探索环境已有真实 checkpoint 与依赖，项目 `.venv` 尚未安装 MACE/e3nn/ASE；当前证据覆盖的是探索环境中的 wrapper/reference batching，不等于项目环境依赖已封装，也不等于完整 dynamics、训练或生产 cell-list/Triton/HIP 支持。下一步先审核项目 `.venv` 的最小依赖解析，再决定是否从缓存安装；不把探索环境证据写成项目环境支持。
