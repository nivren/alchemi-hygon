# G1 MACE 依赖审计

日期：2026-09-06  
范围：确认真实 MACE 探针的依赖边界和可运行前置条件，并记录用户认可的本地 checkpoint；随后将经解析的最小依赖安装到项目环境并复核 Torch/HCU。

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
- 项目 `.venv` 现有 MACE/e3nn/ASE 运行时为 `mace-torch 0.3.15`、`e3nn 0.4.4`、`ase 3.29.0`、`matscipy 1.1.1`；海光 Torch/Triton、NumPy 约束保持不变。额外补齐了 framework 导入所需的 `jaxtyping`、`periodictable`、`tensordict`、`pydantic`、`dm-tree`、`zarr`、`plotext`、`loguru` 等。
- `/data/envs/uv-cache` 中能找到 `mace-torch 0.3.15`、e3nn 和 matscipy 的缓存/索引痕迹；北外镜像可以解析并安装与 `numpy==1.26.4` 匹配的 `matscipy 1.1.1`。缓存/安装成功仍不等于所有模型和 gfx936 性能路径已兼容。
- 用户确认 `~/.cache/mace` 中的模型权重可作为可信来源。`MACE-OFF23_small.model` 可加载为 `ScaleShiftMACE`，覆盖 H/O/C/N 等元素，`r_max=4.5`；SHA256 为 `165cce4cfec5a34b9c64d4ebf95de15d71106bb584b7291c8470f0749977c46f`。
- `probes/mace_probe.py` 已在探索环境 CPU 和 source DTK 26.04 的 BW200/gfx936 HCU 上退出 `0`，检查能量、力、力损失到模型参数的非零梯度；详细结果见 `reports/g1-mace-wrapper-batch.md`。这只是原始 MACE 模型证据，不等同于 framework wrapper 或 dynamics 支持。
- `probes/mace_wrapper_reference.py` 已加入正式 wrapper + Torch reference neighbor + Batch smoke；H₂O 单/双体系、单个 `perf_46` 和两个 `perf_46` CIF 的 CPU/HCU batching 均通过。向量化周期 Torch reference 邻居后，双体系 HCU 结果为 `batch_ptr=[0,46,92]`、1754 条边、无跨体系边、逐体系总力最大分量约 `9.6e-7`；向量化前的 180 秒超时（退出 `124`）仍作为优化前证据保留，详见 wrapper 报告。

## 项目环境依赖 dry-run（2026-09-06）

项目 `.venv` 沿用 `configs/probe-constraints.txt` 的海光 Torch/Triton 与 `numpy==1.26.4` 约束，执行：

```bash
UV_CACHE_DIR=/data/envs/uv-cache uv pip install --dry-run \
  --python .venv/bin/python \
  --index-url https://mirrors.bfsu.edu.cn/pypi/web/simple \
  -c configs/probe-constraints.txt 'mace-torch==0.3.15'
```

历史上使用 HUST 镜像的在线解析因 TLS 响应提前关闭退出 `1`；同一缓存的离线解析也因候选不完整退出 `1`。切换北外镜像后，在线 dry-run 退出 `0`，解析 45 个包，计划安装 31 个，其中包含 `matscipy==1.1.1`，与项目 `numpy==1.26.4` 约束相容。

随后按同一 constraints 安装到项目 `.venv`，并核对 `torch==2.9.0+das.opt1.dtk2604`、`triton==3.3.0+das.opt1.dtk2604.torch290`、`numpy==1.26.4` 未变化。PhysicsNeMo 没有安装：它绑定 NVIDIA/Warp，不是单进程 Torch/HCU MACE wrapper 的必需依赖；profiling 和域并行能力仍保留为显式可选边界。

## 项目环境规格（2026-09-06）

本报告早期的 `configs/probe-constraints.txt` 只承担 Torch/Triton/NumPy ABI 约束，不能作为完整环境冻结。当前可重建入口已独立登记：

- 直接输入：`configs/hygon-reference.in`；
- 精确包冻结：`configs/hygon-reference-lock.txt`；
- 海光 wheel 名称/SHA256：`configs/hygon-wheel-manifest.txt`；
- 生成脚本：`scripts/freeze_hygon_env.sh`；
- pytest：`8.4.2`，pytest-asyncio：`1.4.0`；
- 当前冻结摘要：`reports/probe-environment-freeze.txt`。

冻结文件包含 MACE/e3nn/ASE/matscipy 运行栈和 framework 基础依赖；它不包含源包本身（运行时由项目 `PYTHONPATH` 暴露），也不包含 PhysicsNeMo。

## 解释和边界

审计命令在受限探针沙箱中观察到 `torch.cuda.is_available()=False`；该沙箱不暴露 `/dev/kfd`、`/dev/dri`，不能据此判断 HCU 不可用。此前 source DTK 26.04、可见 BW200/gfx936 环境中的 Torch/HCU 证据仍以对应 reports 为准。

P06 目前为 `partial`：探索环境与项目 `.venv` 均有真实 checkpoint 和 MACE 依赖；当前证据覆盖 wrapper/reference batching，不等于完整 dynamics、训练、cuEquivariance、生产 cell-list/Triton/HIP 或 PhysicsNeMo 域并行支持。项目环境的两个 `perf_46` HCU Batch 证据见 wrapper 报告及 `artifacts/g1/mace_wrapper_project_perf46_batch2_hcu0.*`。
