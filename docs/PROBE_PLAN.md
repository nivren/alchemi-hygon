# G0/G1 可重跑探针

先运行 `hy-smi` 确认资源并取得作业分配。以下 GPU 命令只在有设备访问权限的终端、已分配卡上运行；把 `<id>` / `<id0>,<id1>` 换为分配卡号，不默认使用全部 8 卡。G0 探针通常超时 60 秒，双卡 90 秒；P06 direct/wrapper 使用表中单独列出的 120/180 秒。沙箱隐藏 `/dev/dri`，不得把沙箱内不可见报告为服务器无卡。

| 探针 | 命令 | 本轮状态与范围 |
|---|---|---|
| P00 | `hy-smi --showproductname --showdriverversion --showmeminfo vram --showtopotype` | 已只读盘点，详见 ENVIRONMENT |
| 源码 | `.venv/bin/python probes/audit_upstream.py` | 退出 0，AST 与清单；不导入/执行上游 |
| P01/P04/segment CPU | `OMP_NUM_THREADS=1 .venv/bin/python probes/torch_probe.py --device cpu` | 结果见 reports/g0-validation.md |
| P01/P04/segment DCU | `HIP_VISIBLE_DEVICES=<id> OMP_NUM_THREADS=1 timeout 60 .venv/bin/python probes/torch_probe.py --device cuda` | 资源阻塞；张量、同步、FP64 一二阶、segment、FFT；不是 custom-op opcheck |
| P02 编译 | `hipcc --offload-arch=gfx936 probes/hip_probe.cpp -o artifacts/g0/hip_probe` | 编译退出 0；gfx936 为用户明确指定适配目标 |
| P02 运行 | `HIP_VISIBLE_DEVICES=<id> timeout 60 artifacts/g0/hip_probe` | 资源阻塞；仅 32 元素，不覆盖 gather/scatter/atomic |
| P03 | `HIP_VISIBLE_DEVICES=<id> TRITON_CACHE_DIR=$PWD/artifacts/triton-cache timeout 60 .venv/bin/python probes/triton_probe.py` | 资源阻塞；1025 FP32 元素、tail mask、5 次预热/20 次调用，冷启动与 host-loop 稳态分开 |
| P05-ref | `PYTHONPATH=packages/ops OMP_NUM_THREADS=1 .venv/bin/python probes/neighbor_lj_reference.py --device cpu`；HCU 版先 `source /opt/dtk-26.04/env.sh` 并设置 `PYTHONPATH=packages/ops HIP_VISIBLE_DEVICES=<id>` | Torch reference dispatcher：两体系、无 PBC、full/half neighbor matrix、LJ 能量/负梯度力，并记录实际 backend；CPU 与单卡 HCU 已通过，不代表生产 nvalchemiops API 已移植 |
| P05-framework | `PYTHONPATH=packages/framework:packages/ops /home/wangleping/codes/nvalchemi-toolkit/.venv/bin/python -m pytest -q packages/framework/test/models/test_neighbors_torch_reference.py`；HCU MATRIX/PBC 版先加载 `/opt/dtk-26.04/env.sh` 并设置 `HIP_VISIBLE_DEVICES=<id>` | `compute_neighbors` 的显式 `torch_reference`/`auto` 入口、异构 batch、COO 转换和 PBC shifts；探索环境测试及 BW200 HCU 路径已通过；项目 `.venv` 仍在补 framework 依赖 |
| P05-PBC | `source /opt/dtk-26.04/env.sh && PYTHONPATH=packages/framework:packages/ops HIP_VISIBLE_DEVICES=<id> timeout 60 /home/wangleping/codes/nvalchemi-toolkit/.venv/bin/python probes/pbc_neighbor_reference.py --device cuda` | full periodic neighbor topology、正交/三斜胞 shift；CPU 与单卡 HCU 已通过；不覆盖 half-list、skin/rebuild 或性能路径 |
| P05-PBC-LJ-NVE | `PYTHONPATH=packages/framework:packages/ops /home/wangleping/codes/nvalchemi-toolkit/.venv/bin/python probes/pbc_lj_nve_reference.py --device cpu`；HCU 版先加载 `/opt/dtk-26.04/env.sh` 并设置 `HIP_VISIBLE_DEVICES=<id>` | full-list 周期 LJ 的独立 FP64 对照与无 skin/no-switching 的短 velocity-Verlet 轨迹；CPU 与 BW200 HCU 已通过；独立探针不等同于完整 Warp-backed `nvalchemi.dynamics.NVE` |
| P05-skin | 同上，增加 `--skin 0.5` | Torch reference Hook 的小输入 cached skin、raw displacement 触发重建、周期 wrap 与 LJ cutoff 过滤；CPU 与 BW200 HCU 已通过；不是生产 per-system GPU rebuild 性能证据 |
| P06-direct | `source scripts/activate_hygon_env.sh exploration` 后执行 `python probes/mace_probe.py --device cpu --checkpoint /home/wangleping/.cache/mace/MACE-OFF23_small.model`；HCU 版增加 `HIP_VISIBLE_DEVICES=<id> OMP_NUM_THREADS=1 timeout 120` | 探索环境的真实 MACE direct model；CPU 与 BW200/gfx936 HCU 已通过，26 个非零参数梯度；不证明 toolkit wrapper 或 dynamics |
| P06-wrapper | `source scripts/activate_hygon_env.sh exploration` 后执行 `python probes/mace_wrapper_reference.py --device cpu --checkpoint /home/wangleping/.cache/mace/MACE-OFF23_small.model [--cif <file1> <file2> ...]`；HCU 版增加 `HIP_VISIBLE_DEVICES=<id> OMP_NUM_THREADS=1 timeout 180` | H₂O 单/双体系和单个 `/data/csp_data/perf_46` 结构的 wrapper CPU/HCU 已通过；两个 `perf_46` CIF 的 CPU batching 已通过，`batch_ptr`/逐体系 energy/无跨体系边/逐体系总力均检查；同一批次 HCU 超时退出 `124`，阶段探针定位到 `compute_neighbors` 的 Torch reference 周期邻居构造，仍不算通过 |
| P07 | `HIP_VISIBLE_DEVICES=<id0>,<id1> OMP_NUM_THREADS=1 timeout 90 .venv/bin/torchrun --standalone --nproc-per-node=2 probes/distributed_probe.py` | 资源阻塞；all-reduce 与双向 P2P，非域分解/LJ 验收 |

MACE 审计和 checkpoint 证据见 `reports/g1-mace-dependency-audit.md`、`reports/g1-mace-wrapper-batch.md`。当前只在探索环境运行；项目 `.venv` 仍未安装 MACE/e3nn/ASE。后续如需安装，先用 `uv pip install --dry-run --python .venv/bin/python -c configs/probe-constraints.txt 'mace-torch==0.3.15'` 审核解析，再决定是否从缓存安装；禁止替换海光 Torch/Triton。探针输出 checkpoint/CIF SHA256、逐体系 batch 边界、energy/force 和 direct model 参数梯度；未知 checkpoint 不下载，不以 mock 代替真实 MLIP。

P04 自定义算子的 fake/meta、opcheck、gradcheck/gradgradcheck，P07 两卡 LJ ownership/pair 守恒尚待 G1 后端实现；P05-ref/P05-PBC/P05-PBC-LJ-NVE/P05-skin 和 P06-wrapper 仍是 Torch/reference 小输入，不掩盖 half-list、完整 dynamics API、per-system rebuild 优化、训练 wrapper 混合二阶梯度和性能路径缺口。P01/P02/P03/P07 本轮仍未运行；P06 仅有上述 partial 证据。无 NVIDIA 对拍 fixture。

运行时保存 stdout/stderr 和单独退出码至 artifacts/g0，并将脱敏数值摘要写 reports；记录代码提交、设备、dtype、种子。不得将未运行、编译成功或 CPU 成功写成 DCU verified。
