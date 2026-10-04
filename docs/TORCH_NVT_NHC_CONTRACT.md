# 固定晶胞 NHC Torch reference 契约

日期：2026-10-04。来源：framework `4dfe3723def34df3fadb245981081ccf8c94c257`
的 `NVTNoseHoover` / `_ops/nose_hoover.py`，ops
`26dbceb61e30cca80e1a5805eebeb51d7dc68fd1` 的 `dynamics/integrators/nose_hoover.py`。

- 保留 `BaseDynamics`、Batch state、Hooks、FusedStage、inflight 状态选择/追加机制。
  公共参数为 fs、Kelvin；算子参数为 internal time、`kT` (eV)。晶胞不更新。
- 输入 positions/velocities/forces `[N,3]`，masses `[N]` 或 `[N,1]`；逐体系
  dt/kT/tau/ndof `[M]`；链 eta/eta_dot/Q `[M,C]`，`1 <= C <= 8`；
  batch_idx `[N]` int32/int64。浮点 FP32/FP64、同设备同 dtype，CPU/CUDA(HIP) eager。
- `Q[:,0]=ndof*kT*tau**2`，其余链质量 `kT*tau**2`。当前自由度沿用上游 `3*N_atoms`，
  不自动减去质心或约束自由度。温度、tau、质量必须正且有限，步长正且有限。
- 双向链 sweep、Yoshida-Suzuki 3/5 子步沿用锁定 ops；底层诊断也允许 nloops=1。
  每次 half-update 重置 total_scale=1，compute_ke=False 使用调用者提供的 ke2。
- 更新原地修改 velocities、eta、eta_dot、ke2、total_scale、step_scale、dt_chain；
  kick 仅改 velocities，drift 仅改 positions。拒绝不支持的 alias；空 M=N=0 为 no-op。
- Torch 功能函数保留连续路径，单独验收一二阶梯度；MD 原地 custom_op 为 forward-only，
  不提供穿过完整轨迹的 autograd。fake/meta、opcheck、compile 另列实测，不由 eager 推断。
- 使用当前 Torch stream；GPU scatter reduction 不承诺逐位确定性。不回退 CPU、不改 dtype。
- `backend=None/warp` 保留 Warp 后端选择；torch_reference 显式登记 executor；不支持的
  backend/DomainParallel 明确失败。上游逻辑修复会影响对应 Warp 接线并独立记录。
- 上游缺陷修复候选：`yoshida_order` 未传至 ops `nloops`；refill 用 dummy 原子数算 Q、
  用平均原子数算 ndof；mask 未保护私有链；eta 两次 drift 合计误推进 dt 而非 dt/2。各补丁在代码旁标记，并在 UPSTREAM.md / 报告记录复现与测试。
- 补位采用 updated Batch 尾部实际新体系计数。scalar 参数可重复使用；逐体系向量在 refill
  后需有明确来源，不能取第一个体系冒充全部新体系。不具备来源时明确失败。
- 续算状态只含积分器状态与 step_count，Batch/model/Hook/sampler/sink 状态由调用者拥有。
- 验收：解析质量/kick/drift、独立标量链 oracle、Warp CPU 对照、异构 Batch、重排/补位、
  FusedStage/Hook 时序、续算、谐势统计/可逆性/能量诊断、限时单 HCU 和真实 MACE 短链。
  NPT/NPH、变胞、跨 rank 和生产 HIP/Triton 为后续任务。

实测与未覆盖范围见 [`g2-torch-nvt-nhc.md`](../reports/g2-torch-nvt-nhc.md)；
CPU/HCU 数值证据只用于上述固定晶胞 reference 范围。
