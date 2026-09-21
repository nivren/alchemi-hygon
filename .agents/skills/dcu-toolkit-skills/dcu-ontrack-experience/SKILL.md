---
name: dcu-ontrack-experience
description: DCU 故障排查经验速查手册。遇到驱动/容器/vLLM/SGLang/RCCL/编译类报错时，按"现象关键词"反查根因与规避手段。数据源自 Hygon ontrack 工单沉淀，仅保留技术可规避的经验，已脱敏（行业/客户/项目名、内网域、IP、凭证均已去除）。
metadata:
  version: 1.0.0
---

# DCU 故障排查经验速查（dcu-ontrack-experience）

## 触发条件
遇到以下任一类问题、且不确定根因时，先在本手册按关键词 grep：
- 容器里 `hy-smi` 看不到卡 / DCU 不可见 / 驱动加载失败
- vLLM / SGLang 启动崩溃、精度乱码、toolcall 失效、长上下文报错
- RCCL/NCCL 多机多卡通信卡死、带宽低、allreduce 异常
- 推理/训练报 `libxxx.so` 缺失、hip/nvcc 编译失败、numpy/transformers 版本冲突
- 掉卡、VMFault、内核切换后驱动 down
- **`dmesg` 出现 XID 错误码（如 XID 190/301）** → 查 `references/xid_sxid_errors.md`（正向错误码速查：错误码→含义/原因/处理）
- **HySwitch 互联报 SXID 错误码（如 SXID 207）** → 查 `references/xid_sxid_errors.md`（SXID 列表 + HySwitch HW/HFM 处理策略）

## 纪律（必读）
- 本手册**只沉淀踩坑经验与可复用规避手段**，不写性能数值、不写逐行日志、不写客户名/行业/项目名。
- 内网镜像仓库域统一记为 `<internal_registry>`（落地时替换为实际 harbor/image 域名）。
- 每条格式：**现象关键词 → 设备型号 → 根因 → 规避/修复手段**。
- 硬件损坏类（UE 错误、ubb 未供电、PCIe 卡槽故障、锁频 RMA）不在本手册——那类走换卡/报修，非技术可避。

## ⚠️ 使用须知（务必先读）
1. **本手册仅供参考，不保证完全修复**。所列经验来自历史工单沉淀，实际环境（驱动版本、DTK、机型批次、固件）千差万别，同样报错可能根因不同。使用前先核对**设备型号 + 驱动/DTK 版本 + 报错原文**是否匹配，不要无脑套 env。
2. **涉及高危操作务必谨慎**：
   - 重启物理机 / 重装驱动 / 刷 vbios / 改 BIOS（关 IOMMU、ACS、NUMA）/ 拔插电源线 / 改内核启动参数 等操作，可能影响整机业务或导致设备离线。
   - 高危操作前：**先确认是否在产线/客户环境**（避免在业务运行中机器直接操作）、**确认有带外管理（BMC/IPMI）可恢复通道**、**必要时应先与硬件/产线团队确认**。
   - 本手册中带"重启物理机""重装驱动""刷 vbios""改 BIOS""拔电源"字样的条目，执行前多想一步，优先在测试机/空闲节点验证。
3. **先观测后动手**：能用 `hy-smi` / `dmesg` / `lspci` / 日志定位的，先定位再改，不要靠试错堆 env。
4. 若按手册操作后仍异常，说明非该经验覆盖场景，回到正常排障流程（查日志、提工单、问研发），不要反复堆砌环境变量。

---

## A. 容器 / 驱动 / 可见性

| 现象 | 设备 | 根因 | 规避/修复 |
|---|---|---|---|
| 容器内 hy-smi 看不到卡、DEVCOUNT=0 | 全系 | 未挂 hyhal | 创建容器必挂 `-v /opt/hyhal:/opt/hyhal:ro`；已挂仍异常→容器内重解压 hyhal 包 |
| 容器内起服务报 amdsmi/lib 缺失 | 全系 | 未 source DTK 环境 | 服务启动前 `source /opt/dtk/env.sh`（含 cuda/env.sh） |
| 装驱动后 hy-smi 仍报错 / 驱动 down | K100-AI/BW | 驱动与 vbios 版本不匹配 | 装驱动时各选项选 Y 自动刷 vbios；多卡不一致跑 `vbios_upgrade.sh` |
| 增减桥接卡/换卡后 topo 异常 | K100-AI | 未重刷固件 | 换卡后必须重装驱动重刷 vbios |
| 虚拟机直通 DCU 驱动加载失败 | Z100L/BW | 内核/直通延迟 | 物理机先装驱动刷 vbios，再进虚拟机装（不刷 vbios）；`modprobe hygon` 等映射延迟 |
| 自升内核后驱动编译失败 | 全系 | kernel-devel 不匹配 | 装系统后 `/etc/yum.conf` 加 `exclude=kernel*`；已踩坑则回退内核或装对应 kernel-devel |
| 定制 OS 装驱动失败（凝思 6.0.80 等） | BW | 无适配记录 | 换社区版 openEuler 22.03 SP3 / 凝思 6.0.99 |
| 多卡通信掉卡 / 带宽低 | K100-AI/BW | 多为 PCIe 卡且 BIOS 未关 IOMMU+ACS（**OAM 产品 BW1000/BW1100 基本不受影响，先确认卡形态再处理**） | PCIe 卡：BIOS 关 IOMMU、关 ACS（关 ACS 后 NUMA 需 BIOS 团队确认同步）；OAM 机型排查其他根因，勿盲目改 BIOS |
| `linux-modules-extra` 缺失（IOMMU v2 报错） | 全系 | 未装额外内核模块 | `apt install linux-modules-extra-$(uname -r)` |
| hyqual 容器报 libkmod 缺失 | BW150 | 容器未映射内核模块 | 创建容器加 `-v /lib/modules/$(uname -r):/lib/modules/$(uname -r)` |
| 新支点 OS 脚本无法执行 / tmp 无权限 | 全系 | OS 限制 | `bash xxx.sh` + 改用用户目录 tmp |
| `max_map_count` 不足导致推理失败 | BW | 内存映射限制 | `echo 262144 > /proc/sys/vm/max_map_count`（重启失效） |
| 安全套件与驱动冲突 | 全系 | 北信源等拦截 | 卸载冲突安全套件 |

---

## B. vLLM / SGLang 推理部署（最大头）

### B1. 启动崩溃 / 精度乱码
| 现象 | 设备 | 根因 | 规避/修复 |
|---|---|---|---|
| 多模态/长上下文精度乱码 | K100-AI/BW150 | 默认注意力后端不兼容 | `--mm-encoder-attn-backend TRITON_ATTN --attention-backend TRITON_ATTN` |
| W8A8 路径异常 | K100-AI | 量化实现 | `W8A8_SUPPORT_METHODS=3` + `ROCBLAS_INT8_ENABLE=0` |
| FP8 启动报错 | K100-AI | 融合算子 | `VLLM_USE_FUSED_QA_KVA_GEMM=0` + `VLLM_USE_FUSED_DTBMM=0` |
| PP 分片报错 `len(partitions)!=pp_size` | K100-AI | 残留 PP 环境变量 | `unset VLLM_PP_LAYER_PARTITION` |
| Bert/bge-m3/XLMRoberta 启动 NotImplementedError | 全系 | V1 引擎不支持这些架构 | `VLLM_USE_V1=0` |
| vLLM 0.9.2 v1 精度异常 | K100-AI | V1 注意力 | `VLLM_USE_FLASH_ATTN_PA=0` |
| Gemma3 输出空 | K100-AI | dtype | 指定 `--dtype bf16` |
| 服务仅监听 127.0.0.1 | 全系 | 默认 host | `--host 0.0.0.0` |
| vLLM 启动崩溃，traceback 断在 `await run_server_worker(...)` 且无后续 `Error` 行 | 全系（海光 vllm_hcu 0.18.x + GLM-5.x） | 启动脚本 `--speculative_config` 写了已废弃的 `"method":"deepseek_mtp"`（0.18.x 已改名 `mtp`，海光插件下该旧名初始化即崩） | 改 `"method":"mtp"` + `"num_speculative_tokens":1`；token 数 2→1 无效（已验证），根因是方法名非层数。仍崩则整段删 `--speculative_config` 重启隔离 MTP vs 权重/fp8。详见 `glm-tool-call-debug` skill 的 MTP startup CRASH 段 |
| 容器 hy-smi `failed to check docker container` | 全系 | seccomp | 创建容器加 `--security-opt seccomp=unconfined` |
| 长上下文 VMFault（特定镜像） | K100-AI | 分段实现 | `VLLM_USE_PIECEWISE=1`（仅特定镜像） |

### B2. 高并发 / 超时 / toolcall
| 现象 | 设备 | 根因 | 规避/修复 |
|---|---|---|---|
| 高并发崩溃 / TimeoutError | K100-AI/BW150 | RPC/迭代超时过短 | `VLLM_RPC_TIMEOUT=100000` `VLLM_ENGINE_ITERATION_TIMEOUT_S=600` `--speculative-disable-by-batch-size 33` |
| toolcall 失效 | K100-AI | 解析器/模板 | 启动加 `--tool-call-parser deepseekv32` 并去 jinja；推理 toolcall 失败加 `--chat-template-content-format=string` |
| 思考模式关不掉 | K100-AI | 缺 reasoning parser | `--reasoning-parser qwen3` 或改 chat_template.jinja |
| Ray DAG 超时（>300s） | K100-AI | ray timeout 太小 | `RAY_CGRAPH_get_timeout=3600` 等 |

### B3. 文生图 / 多模态大模型
| 现象 | 设备 | 根因 | 规避/修复 |
|---|---|---|---|
| GLM-Image / Qwen-Image 推理异常 | BW150 | AR 阶段占比大、aiter 接口不稳 | 建议走 vLLM-Omni；sglang 的 aiter flash_attn 需 patch shim（`patch_aiter_flash_shim.py`） |
| sglang 文生图无法走 aiter 接口 | BW150 | aiter 未导出兼容接口 | Patch `aiter.__init__` 导出 `flash_attn_func` 兼容 SGLang |

---

## C. RCCL / NCCL 集合通信

| 现象 | 设备 | 根因 | 规避/修复 |
|---|---|---|---|
| 双机 allreduce 带宽低 / 卡死 | 全系 | topo 文件 busid 不匹配 | `export NCCL_TOPO_FILE=null` 自动探测 |
| 多机通信走内核态、性能差 | K100-AI | IB 网卡降速未用 GDR | 更换 IB 网卡；老驱动暂关 `NCCL_NET_GDR_READ=1` 并升级驱动 |
| 重装驱动后 allreduce 仅 30+ | K100-AI | 驱动状态未生效 | 重启物理机 |
| 双机 ACS 导致通信异常 | 主要为 PCIe 卡（K100-AI/BW/BW10/BW100/BW150 等；**OAM 产品 BW1000/BW1100 基本无影响**） | ACS 未关 | PCIe 卡：BIOS 关 ACS；`NCCL_IB_GID_INDEX` unset；`NCCL_IB_TC=160或162`。OAM 机型先排查其他根因 |
| 默认通道数上限导致多卡慢 | BW150/BW1000/多节点BW1100 | `NCCL_SIMPLE_CHANNELS` 默认 16 | `export NCCL_SIMPLE_CHANNELS=32` |
| `NCCL_RINGS` 配置后通信失败 | 全系 | RINGS 只能从 0 编号 GPU 序列 | `HIP_VISIBLE_DEVICES` 不能跳过编号（如 0,1,2,3 而非 0,2,4,6） |
| dcu_megatron 默认 NCCL 不适配本机 | K100-AI/BW | 框架默认是 BW1000 配置 | 改 `dcu_megatron/requirements/env.sh` 按实际机器配 NCCL |
| 创建 ray 集群前 GPU 不可用 | BW | 容器未设可见设备 | 每容器先 `export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` |
| 双机 SSH 免密/网口 IP 冲突 | K100-AI | 主机名绑定错 IP | 修正 `/etc/hostname` 与 hosts；确保双机互信 + 网口 IP 不冲突 |

---

## D. 库缺失 / 编译 / 框架版本

| 现象 | 设备 | 根因 | 规避/修复 |
|---|---|---|---|
| `libamdsmi` 缺失 | 全系 | 自动下载版不兼容 | 装特定 amdsmi whl（非 DAS 自动下载版） |
| `libtorch_cuda.so` / `libcudart` / `librocm_smi64` 缺失 | 全系 | LD_LIBRARY_PATH 未含框架镜像自带路径 | 补 `LD_LIBRARY_PATH` 指向镜像内对应 lib |
| hip 编译报链接 CUDA 目标失败 | 全系 | CMakeLists 未链接 DTK | `find_package()` + `source /opt/dtk/cuda/env.sh` |
| nvcc `All arch params replaced by gfx906,gfx926,gfx928` | 全系 | CMakeLists 未加 DTK 链接 | CMakeLists 加 DTK 路径或 source cuda/env.sh |
| numpy 2.x 与 torch 不匹配 | K100-AI/BW | 镜像 numpy 2.x vs torch 2.5 | 降 `numpy==1.25.0` 或 `1.26.4` |
| transformers 版本不兼容（缺 padding_side / check_model_inputs 误判） | 全系 | 版本与模型要求不符 | 按模型定：4.44.2 / 4.57.1 / 5.3.0 等 |
| 7490 CPU 不支持 AVX512 | 全系 | 产品侧 deepspeed 按 AVX512 编译 | 暂不支持 adam offload 到内存；用改编译版本（咨询深度组） |
| 编译 hip 找不到 torch.h | 全系 | include 路径未加 | Makefile 加 `/usr/local/lib/python3.10/dist-packages/torch/include/...` |
| `bc` 缺失导致脚本失败 | 全系 | 基础工具未装 | `apt/yum install bc` |
| `PYTHONPATH` 污染导致导入错包 | 全系 | 环境变量串味 | `unset PYTHONPATH` 后重跑 |
| 紫光/新支点 OS 限制脚本执行 | 全系 | OS 安全策略 | `bash xxx.sh` + 改用用户目录 tmp |

---

## E. 已知硬件 RMA 提示（非技术可避，走换卡/报修）
以下现象经验证为硬件故障，**不要花时间调软件**：
- dmesg 报 `GCEA err detected at instance`（UE 错误）→ 卡硬件损坏，换卡
- ubb 未正常供电导致 OAM 整体不被识别 → 查供电/换模组
- PCIe 卡槽故障 / 物理机 lspci 找不到显卡 → 换卡槽/报修
- 锁频、电源 VR 固件老、过流丢卡（128cu 瞬时电流高）→ 产线改 hyqual_normal 或换卡
- 旧库存 BM 型号板卡 i2c 上拉电阻不符 → 厂商改硬件或返厂

---

## 关联
- 操作类手册见：`dcu-driver`（驱动安装）、`dcu-docker`（容器/库）、`dcu-rccl-test`（集合通信测试）、`dcu-dtk`（DTK 编译）。
- 本手册是**纯排查索引**，与上面操作手册互补，不重复维护操作步骤。
- `dcu-driver` / `dcu-docker` / `dcu-rccl-test` 正文中已加"遇到具体报错先查本 skill"的提示，**只指路、不抄 env 内容**（避免双源维护）。
- **XID/SXID 错误码正向速查**（错误码→含义/原因/处理，与本文「现象反查」互补）已分离到 `references/xid_sxid_errors.md`，单独存放便于针对性查询，不混入主文。
- **本包维护细则（存量 enrich / references 分离规范、不动根计数与版本）**见 `references/pack-enrich-maintenance.md`。

## 如何维护 / 从 ontrack 工单增量更新本手册

> 本手册源自 Hygon ontrack 系统导出的工单 CSV。增量更新时严格按以下流程，避免把噪声搬进来。

### 增量更新工作流（强制）
1. **全量覆盖，不抽样**：两个 CSV 都要逐行读完（用户明确要求"每个字都要涉及到，不要只读前500行"）。用 `\|CSD-\d+\|` 作为主键切分工单，不要用 csv.reader（易把多行线索拆错）。提取脚本与覆盖校验方法见 `references/ontrack-csv-extraction.md`。
2. **去重**：两份文档并集，按 CSD 编号去重（交集通常仅 1 个左右）。
3. **二次筛选（用户明确要求，不能省）**：只保留"技术可规避、带根因+修复手段"的经验。剔除：
   - 纯进度跟进（"已解决可关闭""待反馈""持续观察"）
   - 纯链接/网盘/截图/密码/提取码/真实 IP
   - 纯"换镜像版本就解决、无根因"
   - 型号空且无报错无修复的空工单
   - 需换模型 / 硬件 RMA（UE 错误、ubb 未供电、PCIe 卡槽故障、锁频）→ 仅进 E 节提示"走换卡/报修"，不写软件规避
4. **不抄只指路**：新增的 env/修复条目**只存在于本手册（唯一样本源）**。在 `dcu-driver`/`dcu-docker`/`dcu-rccl-test` 等老 skill 里加一句"先来本 skill 按现象反查"的提示即可，**不要**把 env 内容再抄进老 skill（用户原话："不是抄一份，可以提示去哪里找有没有处理经验"）。
5. **脱敏自检（落地前必跑）**：grep 确认无 `harbor.sourcefind.cn` / `image.sourcefind.cn` / 真实 IP / `密码` / `提取码` / 用户名（107/242/jixx 等）。内网域统一记 `<internal_registry>`。
6. **不进 skill**：性能数值、逐行 dmesg 日志、基线表、客户名/行业/项目名一律不写。

### 条目格式（新增时遵守）
`现象关键词 → 设备型号 → 根因 → 规避/修复手段`，每条必须带来源工单可反查的根因，不能是孤立命令。

### ⚠️ 通信/拓扑类条目必加「产品形态前置」（防回归）
涉及 IOMMU/ACS、PCIe P2P、卡间拓扑、NUMA 跨 socket 等**与卡形态强相关**的条目，**禁止写"全系/全机型"**——
不同形态影响差异极大：
- **PCIe 形态卡**（K100-AI、BW10、BW100、BW150 等）：走 PCIe P2P 拓扑，BIOS 未关 IOMMU/ACS 会隔离直连、带宽不达标，须关。
- **OAM 形态产品**（BW1000、BW1100）：基本不受 IOMMU/ACS 影响，无需为此改 BIOS/内核；若现象出现在 OAM 机型，先排查其他根因，勿盲目套 PCIe 卡的修复。
新增/修订此类条目时，设备列按形态限定（如"主要为 PCIe 卡（K100-AI/BW/BW10/BW100/BW150 等；OAM 产品 BW1000/BW1100 基本无影响）"），根因列点明形态前提。
（来源：用户 2026-07-23 实测纠正——原"双机 ACS 导致通信异常 | 全系"误述已修正；dcu-env/dcu-driver/dcu-rccl-test 正文同步加前提。）
