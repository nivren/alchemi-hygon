# G1 Torch reference backend slice

日期：2026-09-05
分支：`codex/g0-initialization`
上游锁定：framework `4dfe3723def34df3fadb245981081ccf8c94c257`，ops `26dbceb61e30cca80e1a5805eebeb51d7dc68fd1`

## 实现范围

`packages/ops/nvalchemiops/torch_reference.py` 提供不导入 Warp 的 Torch reference：

- no-PBC、连续 batch 的 dense neighbor matrix；支持 full/half、COO、距离和位移输出；
- 邻居容量不足时抛出 `NeighborOverflowError`，不截断相互作用；PBC、target rows 和未实现参数显式 `NotImplementedError`；
- LJ 能量、每原子能量和负梯度力；full/half 采用相同 pair 能量约定，并保留 `create_graph=True` 的二阶梯度路径；
- 输入 tensor 的 device 和 dtype 保持在 Torch 路径中，当前实现没有 CPU 搬运回退。

`nvalchemiops` 根包现在只在调用 `initialize_warp()` 或导入 Warp-facing 子包时初始化 Warp。`nvalchemiops.torch_reference` 可单独导入；已有 neighbors、interactions、dynamics、math、jax 和 Torch Warp adapters 在各自包边界显式初始化 Warp，以保留原有 Warp 路径。

`packages/ops/pyproject.toml` 将 Warp 从基础依赖移到 `warp`/Warp 适配器 extras，并增加 `torch-reference` extra；framework 的 uv dependency metadata 已同步。使用 PyPI 构建 wheel 成功，且 wheel 的基础 `Requires-Dist` 只有 `numpy`，不会在 HCU reference 环境强制安装 Warp。原镜像返回 403 的构建尝试未改变项目环境。

## 可重跑验证

CPU/无 Warp 导入与五项 reference 测试：

```bash
PYTHONPATH=packages/ops \
  /home/wangleping/codes/nvalchemi-toolkit/.venv/bin/python -m pytest -q \
  packages/ops/test/torch/test_torch_reference_backend.py
```

结果：`5 passed`，退出码 `0`。测试覆盖 Warp 未加载、异构 batch、full/half/COO、LJ 能量/力、二阶梯度、容量溢出、显式 PBC 失败和单原子零容量。

HCU 单卡验证（先加载 `/opt/dtk-26.04/env.sh`，使用 `HIP_VISIBLE_DEVICES=0`）：

```bash
bash -lc 'source /opt/dtk-26.04/env.sh && \
  PYTHONPATH=packages/ops HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
  .venv/bin/python -c "import torch; from nvalchemiops.torch_reference import neighbor_list, lj_energy_forces; assert torch.cuda.is_available(); p=torch.tensor([[0.,0.,0.],[1.1,0.,0.]],device=\"cuda\",dtype=torch.float64); m,n=neighbor_list(p,2.0); q=p.detach().requires_grad_(); e,f=lj_energy_forces(q,m,n,epsilon=1.,sigma=1.,cutoff=2.); print(torch.cuda.get_device_name(), m.tolist(), n.tolist(), e.sum().item(), torch.linalg.vector_norm(f).item())"'
```

结果：设备 `BW200, UBB BW1000`，邻居 `[[1], [0]]`、计数 `[1, 1]`，总能量 `-0.9833724493736826`，力范数 `2.245906038631372`，退出码 `0`。

wheel 元数据检查：

```bash
UV_CACHE_DIR=/tmp/alchemi-uv-cache uv build --wheel \
  --default-index https://pypi.org/simple \
  --directory packages/ops --out-dir /tmp/alchemi-ops-dist
unzip -p /tmp/alchemi-ops-dist/*.whl '*/METADATA' | sed -n '1,45p'
```

构建退出码为 `0`；基础依赖为 `numpy`，`warp-lang` 只出现在显式 Warp extras，`torch-reference` 只声明 Torch。构建时绕过了本机返回 403 的镜像，未安装或替换项目环境中的包。

## 限制

本报告只证明 Torch reference 的 no-PBC 小规模路径在 CPU/HCU 可运行，不代表上游 `nvalchemiops.torch.neighbors` 或 `nvalchemiops.interactions.lj` 公共 Warp API 已替换。PBC/cell-list/skin、switching/virial、NVE、自动 dispatcher、Triton/HIP kernel、双卡 ownership 和性能结论仍未实现或验证。下一步是把 reference 接到受控的公共 dispatcher，再按 gfx936 的实测瓶颈逐算子评估 Triton 与 HIP。
