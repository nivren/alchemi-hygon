# 当前开发状态

更新时间：2026-09-22。本文只维护当前状态、已验证范围、下一步和限制；完整历史交接保留在
[`docs/history/STATUS.md`](history/STATUS.md)。

## 当前基线

- 当前分支：`develop`。
- 阶段二实现提交：`9f53f80`；上一轮文档收口提交：`9e295ea`；当前文档重构提交：`97325a4`。
- 两个产品远端 `local-origin/develop`、`github-origin/develop` 均已同步到 `97325a4`。
- 上游 framework/ops 锁定 SHA 见 [`UPSTREAM_LOCK.yaml`](UPSTREAM_LOCK.yaml)；产品源码位于
  `packages/framework` 和 `packages/ops`，`external/` 只作只读参考。

## 当前能力摘要

| 功能 | 当前状态 | 已验证范围 | 明确限制 |
|---|---|---|---|
| Torch reference neighbor/cell-list | 已实现窄 reference | periodic/no-PBC、full/half、MATRIX/COO、mixed/triclinic Batch、image shift、容量、selective rebuild、连续 distance/vector 一二阶路径 | 不等于完整上游生产 cell-list；完整 target/pair/compile/opcheck/DomainParallel 仍未完成 |
| 显式 HIP neighbor/cell-list | 已实现并验证窄 capability | periodic、fixed-cell、Batch、full-list、MATRIX、FP32/FP64、`skin=0`、`max_neighbors=None` 本地 doubling capacity | 不进入 `auto`；不支持 no-PBC、half、COO、skin/rebuild、target/pair、变胞、native geometry backward、compile/opcheck、DomainParallel |
| Torch geometry | 当前训练/梯度路径 | distance/vector 连续坐标的一阶/二阶路径 | native HIP geometry 仍是 forward-only 性能候选 |
| Langevin | 已完成窄 reference slice | 固定晶胞 BAOAB、float32/64、异构普通 Batch、空输入/错误、短统计 oracle、最小 integrator continuation state；当前 CPU 复核 `15 passed` | 不等于完整上游 Langevin/NVT；通用 checkpoint、`atom_ptr`/`_out`、inflight、分布式、完整行为套件、torch.compile、生产 Triton/HIP 未完成 |
| fixed-cell dynamics | reference slice | VV、FIRE、FIRE2、kinetics 及部分公共 wrapper | NHC、NPT/NPH、变胞 stress/FIRE2、生产后端未完成 |

能力宽度以 [`BACKEND_CAPABILITY_MATRIX.md`](BACKEND_CAPABILITY_MATRIX.md) 和
[`FEATURE_COMPATIBILITY.yaml`](FEATURE_COMPATIBILITY.yaml) 为准；本表不把窄 slice 扩大为完整
上游支持。

## 阶段二收口证据

- Ops focused suite：`102 passed`。
- Framework 邻居/LJ focused suite：`19 passed, 1 warning`。
- HCU：gfx936/BW200，`HIP_VISIBLE_DEVICES=4`，46×32 compatibility gate 通过；FP32/FP64、
  empty、capacity overflow、unsupported request、重复 Hook 和自动容量增长均有证据。
- 自动容量 fixture：初始容量 `1` 增长到 `8`，最大实际邻居数 `6`，保守上界 `343`；active
  neighbor 集合与 Torch reference 一致。HIP padded capacity 形状不承诺与上游最小 16/16
  对齐策略一致。
- 当前阶段二 summary：[`reports/g2-framework-hip-neighbor-compatibility-gate.md`](../reports/g2-framework-hip-neighbor-compatibility-gate.md)。
  相关报告按主题见 [`reports/README.md`](../reports/README.md)。

## Langevin 窄 slice 收口证据

- 公共固定晶胞 BAOAB reference、registry/executor 接线和 contract：
  [`g2-torch-nvt-langevin.md`](../reports/g2-torch-nvt-langevin.md)。
- 独立谐势短统计 oracle：[`g2-torch-nvt-langevin-stat.md`](../reports/g2-torch-nvt-langevin-stat.md)。
- 普通 Batch 的最小 integrator continuation state：
  [`g2-torch-nvt-langevin-restart.md`](../reports/g2-torch-nvt-langevin-restart.md)。
- 当前工作树 CPU 复核：三个兼容性文件合计 `15 passed`；历史 HCU 证据保留原报告中的真实卡号。

## 运行时排障记录

此前“加载/设备同步在当前卡上异常变慢”已定位为：超时终止 JIT 编译后遗留 PyTorch C++
extension `FileBaton` stale lock，后续进程在加载阶段等待。确认没有属于本任务的
`ninja`/`hipcc`/应用进程后清理该缓存 lock，query extension 恢复正常加载，完整 46×32 HCU
gate 重跑通过。长时间验证和发布路径优先使用 AOT 或受控 JIT cache。

## Hygon distributed preflight（2026-09-21）

- 测试分支：`codex/test-distributed-md-preflight`；报告：
  [`reports/dcu-distributed-preflight-2026-09-21.md`](../reports/dcu-distributed-preflight-2026-09-21.md)。
- 初始只读环境：8×BW200/UBB BW1000、驱动 `6.3.30-V1.4.1a`、8 卡均约 2 MiB 显存占用、
  任意卡对拓扑为 HSW。
- 初次 2-rank `torchrun` 曾在 launcher 调用 `torch.cuda.is_available()` 时段错误，退出码 `139`；
  恢复后直接 Python 枚举、单进程 torchrun 和 2 子进程空 launcher 均通过，说明该段错误目前
  不是稳定复现的 launcher 必现故障。
- 历史重启前的真实 2-rank process group 曾初始化但在 1-element all-reduce 超时，随后设备运行时
  丢失；该段记录仅用于故障追溯，不参与当前 post-reboot preflight 决策。
- `hy-smi` 恢复后，纯 HIP runtime 在 HCU0/HCU1 通过；两个普通进程直接调用 RCCL 时，
  `ncclCommInitRank` 通过，但单元素 `ncclAllReduce` 在 P2P/IPC/XGMI 路径超时，退出码均为
  `124`。因此该故障已在 PyTorch 之外复现，但 P2P/RCCL/HIP runtime/驱动固件之间的根因
  尚未归因。SHM 对照在设备已异常后执行，标记为受污染，不作为 transport 结论。未扩大到
  4/8 卡，也不以环境变量 workaround 宣称通过。
- 换用 HCU6/HCU7 时，纯 HIP 仍通过；direct RCCL communicator 初始化仍通过，但 all-reduce
  同样停在 `before_ncclAllReduce`，随后 `hy-smi` 报 `Open mkfd failed`/设备不可用。两个
  不同双卡组合均复现，当前停止继续设备测试，等待管理员恢复并采集宿主机内核日志。
- 只读核验安装拓扑：`/var/log/rock-kernel/install-2.log` 记录 `selected 7+8->10`，
  `/etc/hfm/hys.cfg` 为 `OAM_7HSW_8HCU`。本节点应使用显式 `-t 10`；没有证据表明此前未
  显式指定 `-t` 导致当前故障，`-t 11` 不适用于该节点。
- 对照节点 dcu1 使用同一最小 direct RCCL probe 通过：`rank0_rc=0`、`rank1_rc=0`，两 rank
  均得到 `after_ncclAllReduce value=3`。这支持将当前 dcu2 故障优先定位为节点级驱动/固件、
  HFM/Hlink 配置或设备健康差异；dcu1 的 probe 输出未包含测试后 `hy-smi`，不扩展为完整节点
  健康结论。
- dcu2 重启后重新验证：直接 `torch.cuda.is_available()` 为 `True`、device count 为 2，
  两 rank `torchrun` 退出码为 `0`，all-reduce 和 P2P send/recv 均通过，测试后 `hy-smi`
  仍正常。用户同时确认“旧驱动、重启后、安装新驱动之前”测试已通过，因此不把新驱动升级
  视为恢复原因；当前状态记录为 post-reboot narrow PASS，重启前故障仍未归因。
- P1 单卡 device tensor smoke 已补齐：新增 `probes/distributed/probe_device.py`，对物理 HCU0--7
  逐卡隔离运行。8 张卡均通过 FP32/FP64/INT64 的 allocation、H2D/D2H、elementwise、同步和
  释放复用，并通过 FP32/FP64 matmul；allocator 快照在释放/复用后保持当前运行时基线。原先矩阵中
  的 `P1=NOT RUN` 已修正为当前 `P1=PASS`，不再把完整矩阵缺失误写成没有任何单卡证据。
- 新增 P4 集合通信探针 `probes/distributed/probe_collectives.py`。在 post-reboot dcu2 上以
  2 ranks、`fp32/fp64/int64`、4 KiB/1 MiB/64 MiB payload 运行，broadcast、all-reduce、
  all-gather、all-gather-into-tensor、等分及变长/零 split all-to-all、reduce-scatter 均
  精确校验通过，前后 `hy-smi` 正常。该结果只构成 P4/2-rank narrow PASS，不覆盖重启前故障，
  也不代表 4/8 rank、P2P 重复、async、DeviceMesh、长稳或 DomainParallel 已验证。
- 新增 P5 P2P 探针 `probes/distributed/probe_p2p.py`。post-reboot dcu2 首次 2-rank 有界运行
  在 180 秒内未完成，退出码 `124`；前后 `hy-smi` 均正常，未观察到设备丢失。日志显示进入
  unbatched P2P communicator 后无 rank summary；随后发现首次 harness 让两个 rank 以相同顺序
  提交对向 unbatched 操作，不作为 RCCL/硬件故障证据。按 RCCL operation contract 修正为两个
  方向有序验证后，2-rank、4 KiB、`fp32/fp64/int64` 的 pair `isend/irecv` 均通过，前后
  `hy-smi` 正常；随后 `pair_batch_isend_irecv`、有序 parity phases 的 ring、变长/非对称 ring、
  zero-length ring 以及 1000 次小消息重复均通过。随后以 4 ranks（HCU 0--3）和 8 ranks
  （HCU 0--7）运行完整 `--case all --payloads 4K,1M --repeats 1000` 矩阵，覆盖
  `fp32/fp64/int64`，所有 rank 均完成 `case_start -> case_pass`，`torchrun rc=0`，前后
  `hy-smi` 均正常。因此 P5/2/4/8-rank correctness matrix 已 PASS。该结论仍不包含性能、
  长时间稳定性、64 MiB P5 payload、多节点或 DomainParallel；探针保留 flushed case markers、
  显式 `device_id` 和 `--case` 单项选择。
- 新增 P6 async collective 探针 `probes/distributed/probe_async.py`。在 2/4/8 ranks、4 KiB/1 MiB、
  `fp32/fp64/int64` 上验证 async all-reduce、独立设备计算后的 wait、3 个 outstanding operation
  的正序/逆序 wait，以及 fp32 100 次重复；所有 rank 最终 JSON 为 `PASS`，`torchrun rc=0`，
  前后 `hy-smi` 正常。P6 correctness 已 PASS，但不代表 P11 通信计算 overlap 或 P12 长稳；
  P7 DeviceMesh 结果见下一条。
- 新增 P7 DeviceMesh/subgroup 探针 `probes/distributed/probe_devicemesh.py`。2/4/8 ranks 的 1D
  mesh、rank lookup、mesh group、1D submesh all-reduce 均通过；4-rank 2×2 和 8-rank 2×4
  2D mesh 的 row/col subgroup mapping 与精确 all-reduce 也通过，覆盖 `fp32/fp64/int64`、
  4 KiB/1 MiB，所有 rank 最终 JSON 为 `PASS`，前后 `hy-smi` 正常。该结果不等于 DomainParallel
  已支持；P11 overlap 另有独立 smoke 结果，生产通信计算 overlap 和 DomainParallel 仍未验证。
- 新增 P8 functional collectives 探针 `probes/distributed/probe_functional_collectives.py`。
  直接导入当前 DTK PyTorch 的 `torch.distributed._functional_collectives`，调用 functional
  `all_to_all_single` 与 `wait_tensor`，不以普通 `dist.all_to_all_single` 替代；在 2/4/8 ranks、
  4 KiB/1 MiB、`fp32/fp64/int64` 上验证显式 input/output split lists、zero/imbalanced splits、
  forward/reverse exchange，所有 rank 最终 JSON 为 `PASS`，`torchrun rc=0`，前后 `hy-smi` 正常。
  该结果仅覆盖 P8 functional collective correctness，不代表 P8-B PhysicsNeMo 兼容、通信计算
  overlap、性能、长稳或 DomainParallel 已验证。
- 新增 P9 autograd-sensitive communication smoke `probes/distributed/probe_autograd_collective.py`。
  在 2/4/8 ranks、4 KiB/1 MiB、`fp32/fp64` 上验证 `local requires_grad → all_gather → 可微 loss
  → backward`。direct `dist.all_gather` 的前向数据精确正确，但记录为
  `native_autograd_collective_supported=false`；显式
  `_functional_collectives.all_gather_tensor_autograd` 经 `wait_tensor` 后前向和反向梯度均
  精确通过，所有 rank 最终 JSON 为 `PASS`，前后 `hy-smi` 正常。该结果是 autograd runtime
  信息项，不等于 nvalchemi HALO/DomainParallel 已支持，也不覆盖 P11 通信计算 overlap、P12 长稳和 P13 显存行为。
- 新增 P10 通信带宽/latency baseline 探针 `probes/distributed/probe_performance.py`。在 2/4/8
  ranks 上测量 `all_reduce`、`all_gather`、`all_to_all_single` 和成对 P2P ping-pong，覆盖
  1 KiB/4 KiB/64 KiB/1 MiB/16 MiB/64 MiB；每项 10 次 warmup、20 次 steady-state sample，
  使用 `cuda_event`，每个 rank 保存 median/p90/min/max、样本、relative stdev 和按明确逻辑字节
  公式计算的 descriptive effective GiB/s。所有 correctness case 和性能运行均完成，
  `torchrun rc=0`，前后 `hy-smi` 正常；没有设置 transport/algorithm/channel 覆盖。该结果没有
  绝对 GB/s 门槛，不代表 P11 overlap、P12 长稳或 P13 显存行为。
- 新增 P11 通信计算 overlap 探针 `probes/distributed/probe_overlap.py`。在 2/4/8 ranks、1 MiB/16 MiB
  payload、1024×1024 独立矩阵计算上，分别测量同步 `all_reduce → matmul` 与
  `all_reduce(async_op=True) + 独立 CUDA stream matmul → wait`；两条路径各 5 次 warmup、10 次
  synchronized wall-clock samples，所有 rank 的通信和矩阵计算 correctness、async wait 与最终
  completion 均通过，`torchrun rc=0`，前后 `hy-smi` 正常。1 MiB 在各规模普遍是 overlap 更慢；
  16 MiB 仅部分 rank 有约 5--9% 的局部下降，按全 rank 完成时间未观察到稳定收益，且 8-rank
  1 MiB 有一个约 4 ms 离群样本。因此 P11 的 async/stream correctness smoke PASS，但不能据此
  宣称生产 DomainParallel 已实现通信计算 overlap，也没有进行性能优化或设定阈值。
- 新增 P12 长稳探针 `probes/distributed/probe_stability.py`。在 2/4/8 ranks 上均完成指南最低
  operation-count：small 10,000、medium 1,000、large 100；每个阶段按固定顺序混合
  `all_reduce`、`all_gather`、`all_to_all`、P2P ring 和 async all-reduce，使用 4 KiB/1 MiB/16 MiB
  FP32 device buffers、2 个 warmup cycles、每次操作完成同步和 exact correctness check。所有
  rank 最终 JSON 为 `PASS`，阶段进度完整，`torchrun rc=0`，前后 `hy-smi` 正常；每个阶段内部
  allocator allocated/reserved trace 保持稳定，阶段切换因首次分配更大 buffer 出现预期 reserved
  增长，释放后 allocated 回到约 512 B。该结果通过 P12 的 operation-count stability gate，
  但不是小时级 soak，也不替代 P13 的专门显存行为测试。
- 新增 P13 显存行为探针 `probes/distributed/probe_memory.py`。在 2/4/8 ranks、每 rank 64 MiB
  FP32 payload 上分别运行 `all_gather` 和 `all_to_all_single`，每项 2 个 warmup、5 次正式重复，
  记录 allocator 的 allocated/max/reserved/max，并在释放后及 `empty_cache()` 后复测。所有 rank
  最终 JSON 为 `PASS`，`torchrun rc=0`，前后 `hy-smi` 与 `hy-smi --showpids` 返回码均为 0；
  `all_gather` 的瞬时 allocated peak 增量随 2/4/8 ranks 为约 128/256/512 MiB，
  `all_to_all_single` 未观察到额外 peak，三轮释放后 allocated 均回到约 512 B，
  `empty_cache()` 后 reserved 均回到约 2 MiB。`--showpids` 采样已保存，但部分 PID 的 HCU 映射为空、
  VRAM 百分比为 `inf`，因此设备侧采样只作辅助证据，不替代 allocator 数字或可归因的进程峰值。
- P14 单节点拓扑已重新采集并完成记录。`hy-smi --showtoponuma --json` 与 `--showbus` 直接配对得到
  8 张 HCU 的 PCI/HCU/NUMA/OAM 映射；8×8 `showtopotype` 全为 `HSW`、`showtopoaccess` 全为
  `TRUE`、`showtopohops` 任意非自身卡对均为 1。PCIe 树确认每张 HCU 与对应 ConnectX-7
  `mlx5` NIC 为同一 PCIe 分支，8 个 IB port 均为 `ACTIVE/LINK_UP`。本次不启用自动绑核；collector
  JSON 的 PCI slot 排序与 hy-smi card 顺序不一致，最终映射以直接 `showbus`+`showtoponuma` 证据为准。

## 当前下一步

具体 owner 和分支需用户确认后再启动。当前计划见
[`PARALLEL_DEVELOPMENT_PLAN.md`](PARALLEL_DEVELOPMENT_PLAN.md)：

1. 固定晶胞 Nose-Hoover chain reference；
2. `TORCH-CELL-STRESS-FORCE` 与 coupled FIRE2 variable-cell；
3. 固定晶胞 ASE 3.29-compatible BFGS，变胞 ASE 语义另列子里程碑。

暂不启动 M2 planner、NPT/NPH、DomainParallel、生产 Triton/HIP kernel 或扩大当前 HIP
neighbor capability。

## 验证和设备约定

- CPU gate：`scripts/check_cpu_reference.sh`。
- 本机所有后续单卡 HCU 验证统一使用 `HIP_VISIBLE_DEVICES=4`；历史报告中真实使用的 HCU 0
  结果保留，不改写。
- 受限沙箱中 `/dev/kfd`、`/dev/dri` 不可见只表示设备节点隔离，不能作为 HCU 能力结论。
- 实现、测试、数值结果、性能口径和未验证假设必须分别记录；没有 HCU 运行证据不能写成
  DCU verified。

## 文档权威关系

- 架构与稳定交接：[`PROJECT_HANDOFF.md`](PROJECT_HANDOFF.md)。
- 模块与应用总览：[`IMPLEMENTATION_STATUS_MATRIX.md`](IMPLEMENTATION_STATUS_MATRIX.md)。
- 当前任务队列：[`PARALLEL_DEVELOPMENT_PLAN.md`](PARALLEL_DEVELOPMENT_PLAN.md)。
- 后端能力：[`BACKEND_CAPABILITY_MATRIX.md`](BACKEND_CAPABILITY_MATRIX.md)。
- 逐特性契约：[`FEATURE_COMPATIBILITY.yaml`](FEATURE_COMPATIBILITY.yaml)。
- 上游来源/补丁：[`UPSTREAM.md`](UPSTREAM.md) 与 [`UPSTREAM_LOCK.yaml`](UPSTREAM_LOCK.yaml)。
- 历史日志：[`docs/history/STATUS.md`](history/STATUS.md)、
  [`docs/history/PROJECT_HANDOFF.md`](history/PROJECT_HANDOFF.md)、
  [`docs/history/PARALLEL_DEVELOPMENT_PLAN.md`](history/PARALLEL_DEVELOPMENT_PLAN.md)。
