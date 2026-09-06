# G0 可重跑探针

先运行 `hy-smi` 确认资源并取得作业分配。以下 GPU 命令只在有设备访问权限的终端、已分配卡上运行；把 `<id>` / `<id0>,<id1>` 换为分配卡号，不默认使用全部 8 卡。进程超时 60 秒，双卡 90 秒。沙箱隐藏 /dev/dri，不得把沙箱内不可见报告为服务器无卡。

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
| P06 | `HIP_VISIBLE_DEVICES=<id> OMP_NUM_THREADS=1 timeout 60 .venv/bin/python probes/mace_probe.py --device cuda --checkpoint /path/to/trusted.model` | 缺指定可信 checkpoint/MACE 依赖及空闲卡；真实 MACE 直接模型，非 mock；不证明 toolkit wrapper 可用 |
| P07 | `HIP_VISIBLE_DEVICES=<id0>,<id1> OMP_NUM_THREADS=1 timeout 90 .venv/bin/torchrun --standalone --nproc-per-node=2 probes/distributed_probe.py` | 资源阻塞；all-reduce 与双向 P2P，非域分解/LJ 验收 |

MACE 前先用 `uv pip install --python .venv/bin/python -c configs/probe-constraints.txt 'mace-torch==0.3.15'` 审核解析，禁止换掉海光 Torch/Triton。checkpoint 须来自可信来源且支持 H/O；脚本输出 SHA256、能量/力 loss 和非零参数梯度数量。未指定 checkpoint 时不下载不可信 pickle，也不以 mock 代替真实 MLIP。

P04 自定义算子的 fake/meta、opcheck、gradcheck/gradgradcheck，P07 两卡 LJ ownership/pair 守恒尚待 G1 后端实现；P05-ref/P05-PBC/P05-PBC-LJ-NVE 仅是 Torch reference 小输入，不掩盖 half-list、skin/rebuild、完整 dynamics API 和性能路径缺口。P01/P02/P03/P06/P07 本轮仍未运行。无 NVIDIA 对拍 fixture。

运行时保存 stdout/stderr 和单独退出码至 artifacts/g0，并将脱敏数值摘要写 reports；记录代码提交、设备、dtype、种子。不得将未运行、编译成功或 CPU 成功写成 DCU verified。
