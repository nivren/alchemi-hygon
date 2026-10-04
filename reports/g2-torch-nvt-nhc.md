# 固定晶胞 NHC Torch reference 移植

日期：2026-10-04。开发分支：`codex/feat-fixed-cell-nhc`，起点 `7a90fcf`；实现提交 `b4050ec`；本轮交付到本地 `develop`，未推送。

## 实现与框架边界

- 公共入口 `NVTNoseHoover(..., backend="torch_reference")`；保留 BaseDynamics、Batch、
  模型包装器、Hooks、FusedStage 和 sampler refill 状态选择/追加机制。
- ops 提供设备内 functional reference；framework 提供原地 custom_op 与显式 backend
  dispatcher。catalog 注册 `torch_reference.nose_hoover-v1`，使用通用 executor binding；
  framework 只解析一次 BackendSelection。`backend=None/warp` 仍选 Warp，无静默回退。
- FP32/FP64，链长度 1–8，Yoshida 3/5，异构普通 Batch 支持逐体系 dt/temperature/tau；
  晶胞固定，自由度沿用上游每体系 `3N`。公共单位 fs/K，算子单位 internal time/kT(eV)。
- scalar 构造参数支持真实 sampler refill；新体系 ndof/Q 使用其实际原子数。
  vector 参数缺少新体系来源映射时明确报错，不拿旧体系参数冒充。
- `state_dict/load_state_dict` 提供积分器续算；Batch/model/Hook/sampler/sink 仍由调用者保存。
- 合同：[TORCH_NVT_NHC_CONTRACT.md](../docs/TORCH_NVT_NHC_CONTRACT.md)。

## 上游逻辑修正

四项修正的详细原因、默认路径影响和上游反馈状态统一记录在
[UPSTREAM_BUGFIXES.md](../docs/UPSTREAM_BUGFIXES.md) 的 UBF-001–004。

锁定来源 framework `4dfe3723def34df3fadb245981081ccf8c94c257`，
ops `26dbceb61e30cca80e1a5805eebeb51d7dc68fd1`；未更新 lock，external 未修改。

| 注释标记 | 原问题与修正 | 回归依据 |
|---|---|---|
| `NHC-YOSHIDA` | framework 接收 yoshida_order，却不传 ops nloops；现在 pre/post 都传 3/5，DD adapter 保持相同 ABI | spy 实际 executor 参数；可逆链矩阵 |
| `NHC-REFILL` | dummy Batch 算 Q、平均原子数算 ndof 对异构 arrivals 不成立；按 updated Batch 尾部实际计数构造 | 2/4 原子初始化，保留4并追加7/3；真实 SizeAwareSampler refill |
| `NHC-MASK` | BaseDynamics 恢复非活动原子字段，但不恢复私有温控链；现在屏蔽 masked/FusedStage/已退出体系的链推进 | 四种 masked/step 入口、FusedStage 独立轨迹对照、AFTER_STEP 观察 |
| `NHC-ETA` | 单次 thermostat operator 为 dt/2，但两次 eta drift 各用 dt/2，累计多一倍；改为两次 dt/4 | 小步导数 `delta eta/dt → eta_dot/2`、独立单链公式、扩展能量 dt² 收敛 |

以上在对应源码旁写了原因注释，补丁登记为 `UPSTREAM.md` LP-064–068。
Yoshida 接线、refill/mask 修正也作用于 Warp 框架路径；eta 修正同时改锁定 Warp 两个
单体系/批量传播函数。默认后端选择未变，但这些修正会改变相应错误轨迹/状态。
本机无 Warp，不能声称 Warp/NVIDIA 执行已验证；保留三项可运行的 Warp CPU 对照测试。

## CPU 测试与数值证据

- ops focused：`22 passed, 3 skipped`（Warp 未安装）。解析 Q/kick/drift、独立单链
  oracle、FP32/64 × C=1/3/8 × nloops=1/3/5 可逆性、external ke2、空输入/错误。
- framework lifecycle 与 physical oracle 合计 `21 passed`：导入不加载 Warp、公共 run、单次 selection、独立 executor 替换、
  补位/重排、mask、FusedStage/迁移、8 个 Hook 阶段、续算、alias/scale-reset。
- functional v 输入通过 gradcheck/gradgradcheck；不扩大为所有参数或完整 MD 轨迹梯度。
- 原地链 ABI 的 `torch.library.opcheck` 四项 SUCCESS；CPU
  `torch.compile(backend="eager", fullgraph=True)` 通过。未验证 Inductor/HCU compile。
- 独立谐势：2×128 原子、FP64、300/600 K、非同频谐振子，burn-in 25%；
  kinetic/configurational 二阶矩比值约 `1.01319/1.00145`、`1.01509/1.00506`。
  同一物理时长的扩展能量峰峰值：

| internal dt / steps | 300 K (eV) | 600 K (eV) |
|---|---:|---:|
| 0.02 / 2000 | 0.0002580681 | 0.0008741443 |
| 0.01 / 4000 | 0.0000652372 | 0.0002189708 |

时间步减半误差比约 0.253/0.250。修正前使用两次 dt/2 eta drift 的诊断分别约
7.37/11.47 eV，减半时间步不改善；此诊断不能扩大为所有体系都能达到同一误差。

原始日志：`artifacts/nhc/{ops,framework,physics,cpu-gate}.log`；
提交前完整 CPU gate 重跑日志为 `artifacts/nhc/cpu-gate-precommit.log`，退出码0、结果同下。
NHC 回归加入 `scripts/check_cpu_reference.sh`，framework/ops 分进程。
最终验收门退出码0：五个独立进程分别67/1/157/22/24通过，合计271通过、3项Warp缺失跳过；
compileall 和 git diff --check 通过。

## HCU 实测

资源前只读检查：8 卡约 2 MiB、无设备进程；占用确认空闲的物理 HCU4，单卡限时120秒。
DTK 26.04；PyTorch 2.9.0；HIP 6.3.26093；设备名 `BW1000 64G`。
执行方式：

```sh
source scripts/activate_hygon_env.sh project
HIP_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 PYTHONPATH=packages/framework:packages/ops \
  timeout 120 .venv/bin/python probes/nhc_reference.py --device cuda \
  --checkpoint /home/wangleping/.cache/mace/MACE-OFF23_small.model
```

- 退出码0；CPU/HCU FP32/64 × C=1/3/8 × nloops=3/5 共12组全部返回量一致；
  容差 FP32 rtol3e-5/atol3e-7，FP64 rtol1e-11/atol1e-13。
- 真实 MACE-OFF23 small、FP32、水/甲烷，batch_ptr=[0,3,8]，周期12Å晶胞；
  dt0.1fs、300/600K、tau10fs。先邻居/力 priming，再执行3步，与2+1续算对照；
  第4步 mask 第二体系，非活动 eta 不变。cell 逐位不变、force/velocity 有限。
- raw：`artifacts/nhc/hcu.log`；结束后 hy-smi 查询正常、无遗留任务。
- 当前 MACE 未安装 cuEquivariance，走既有 MACE 路径；未新增模型加速后端。
- 本次是数值/执行 smoke，无性能 benchmark，无长 MACE NVT 统计结论。

## 未覆盖

NPT/NPH、变胞 stress/cell-force、DomainParallel、多节点、生产 HIP/Triton dynamics、
vector arrivals 参数映射、完整 workflow checkpoint、atom_ptr/_out、全量上游行为套件、
Warp/NVIDIA 执行、HCU autograd/compile、长真实势统计与生产性能均未验收。
原地 MD custom_op forward-only；functional 梯度证据限上面列出的 CPU 输入。
