# G1 MACE wrapper 与 Batch 证据

日期：2026-09-06  
上游锁：framework `4dfe3723def34df3fadb245981081ccf8c94c257`，ops `26dbceb61e30cca80e1a5805eebeb51d7dc68fd1`。

本报告验证缓存的用户认可 checkpoint `/home/wangleping/.cache/mace/MACE-OFF23_small.model` 经过正式 `MACEWrapper` 和 `compute_neighbors(backend="torch_reference")` 的路径。运行环境是已有探索环境 `/home/wangleping/codes/nvalchemi-toolkit/.venv`，不是项目 `.venv`；项目环境尚未安装 MACE/e3nn/ASE。

## 代码修复

`MACEWrapper.from_checkpoint` 的签名允许本地 `Path | str`，但此前无条件调用 MACE 下载器。MACE 下载器把本地路径当 URL，导致 `Path` 触发 `decode` 错误，字符串路径触发 `unknown url type`。现在已有文件先在本地加载，命名 foundation checkpoint 仍走原下载器。

回归命令：

```bash
source scripts/activate_hygon_env.sh exploration
python -m pytest -q packages/framework/test/models/test_mace.py \
  -k 'from_checkpoint_uses_existing_local_path or from_checkpoint_normalizes_load_device'
```

结果：`3 passed, 89 deselected`，退出码 `0`。

## 直接 MACE checkpoint

```bash
source scripts/activate_hygon_env.sh exploration
python probes/mace_probe.py --device cpu \
  --checkpoint /home/wangleping/.cache/mace/MACE-OFF23_small.model
source scripts/activate_hygon_env.sh exploration
HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 timeout 120 \
  python probes/mace_probe.py --device cuda \
  --checkpoint /home/wangleping/.cache/mace/MACE-OFF23_small.model
```

CPU 和 HCU 均退出 `0`。checkpoint SHA256 为
`165cce4cfec5a34b9c64d4ebf95de15d71106bb584b7291c8470f0749977c46f`；两次均得到能量 `-2077.7669553149117`、force loss `0.4526716740143`、26 个非零参数梯度。HCU 设备报告为 `BW200, UBB BW1000`。原始输出：

- `artifacts/g1/mace_off23_small_cpu.json`
- `artifacts/g1/mace_off23_small_hcu0.json`

这证明 MACE 原始模型的前向、力和参数梯度路径，不证明 framework wrapper 或 dynamics。

## Framework wrapper 与 batching

最小 H₂O 命令：

```bash
source scripts/activate_hygon_env.sh exploration
python probes/mace_wrapper_reference.py --device cpu \
  --checkpoint /home/wangleping/.cache/mace/MACE-OFF23_small.model
source scripts/activate_hygon_env.sh exploration
HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 timeout 120 \
  python probes/mace_wrapper_reference.py --device cuda \
  --checkpoint /home/wangleping/.cache/mace/MACE-OFF23_small.model
```

CPU/HCU 均退出 `0`，能量为 `-2078.119873046875`，force norm 为 `4.194570064544678`，逐体系总力最大分量约 `1.2e-7`。HCU 设备为 `BW200, UBB BW1000`。

真实结构 batching 命令使用用户确认元素覆盖范围的 `perf_46` CIF：

```bash
source scripts/activate_hygon_env.sh exploration
python probes/mace_wrapper_reference.py --device cpu \
  --checkpoint /home/wangleping/.cache/mace/MACE-OFF23_small.model \
  --cif /data/csp_data/perf_46/formal_c1_1_10_z1_46.cif \
        /data/csp_data/perf_46/formal_c1_1_11_z1_46.cif
source scripts/activate_hygon_env.sh exploration
HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 timeout 180 \
  python probes/mace_wrapper_reference.py --device cuda \
  --checkpoint /home/wangleping/.cache/mace/MACE-OFF23_small.model \
  --cif /data/csp_data/perf_46/formal_c1_1_10_z1_46.cif \
        /data/csp_data/perf_46/formal_c1_1_11_z1_46.cif
```

CPU batching 退出 `0`，结果为：

- `num_systems=2`，`natoms_per_system=[46,46]`，`batch_ptr=[0,46,92]`；
- `neighbor_edges=1754`，`cross_system_edges=0`；
- 两个 energy 为 `-39190.83203125`、`-39190.81640625`；
- 两个体系的总力分量均在约 `9e-7` 内。

HCU batching 退出 `124`（180 秒超时），没有 JSON 结果，不能记为 HCU batching 通过。原始文件：

- CPU：`artifacts/g1/mace_wrapper_perf46_batch2_cpu.json`、`.stderr`、`.exit`；
- HCU：`artifacts/g1/mace_wrapper_perf46_batch2_hcu0.json`、`.stderr`、`.exit`。

当前探针已断言 `batch_idx/batch_ptr`、逐体系 energy、无跨体系邻居边和逐体系总力；尚未覆盖完整 dynamics、训练 wrapper 的混合二阶梯度、skin 重建和多卡域分解。HCU 超时原因尚未定位，不得改写为设备不支持。

同一个 `perf_46` CIF 的单体系 HCU 命令退出 `0`：

```bash
source scripts/activate_hygon_env.sh exploration
HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 timeout 180 \
  python probes/mace_wrapper_reference.py --device cuda \
  --checkpoint /home/wangleping/.cache/mace/MACE-OFF23_small.model \
  --cif /data/csp_data/perf_46/formal_c1_1_10_z1_46.cif
```

设备为 `BW200, UBB BW1000`，46 原子、858 条边，energy 为 `-39190.8359375`，逐体系总力最大分量约 `3.6e-7`。原始输出：`artifacts/g1/mace_wrapper_perf46_10_hcu0.json`、`.stderr`、`.exit`。因此当前证据把问题收窄为两个 46 原子体系同时进入 HCU Batch 时的超时；单体系、H₂O 双体系和 `perf_46` CPU 双体系均可运行。

将两个 `perf_46` 结构的 HCU 命令在同一张卡上再次重跑，仍在 180 秒退出 `124`，没有 JSON 输出。证据为 `artifacts/g1/mace_wrapper_perf46_batch2_hcu0_retry.json`、`.stderr`、`.exit`。这不是一次性冷启动/JIT 超时；当前不再通过增加 timeout 来掩盖问题。

## HCU 超时定位的最小诊断

用两个相同的 H₂O 组成 Batch，在同一 source DTK 26.04、`HIP_VISIBLE_DEVICES=0` 环境运行同一 `MACEWrapper + compute_neighbors(torch_reference)` 路径：

- 退出码 `0`，设备 `BW200, UBB BW1000`；
- `batch_ptr=[0,3,6]`、12 条邻居边、跨体系边 `0`；
- 两个 energy 均为 `-2078.119873046875`；
- 两个体系逐体系总力最大分量约 `2.4e-7`。

证据：`artifacts/g1/mace_wrapper_h2o_batch2_hcu0.json`、`.stderr`、`.exit`。这说明小图的 HCU batching 基础路径可运行；`perf_46` 两体系超时仍需区分图规模、PBC/JIT、显存和共享资源，不能归因于 Batch 语义或直接判定 HCU 不支持。
