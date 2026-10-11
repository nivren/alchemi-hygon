# M5-3B 异构 Jacobi / 多 stream 隔离合同

2026-10-10，用户已确认。新增本地实验，不是 NVIDIA 上游算子的移植。
不注册产品后端，不更改默认、版本锁、State、Filter、模型与 Hooks 合同。

输入为同一 HIP device 的非空 Tensor 列表，各项连续 FP64 `[n,n]` 或
`[B,n,n]`，实际正整数 n/B 可不同。仅 LOWER、eager、forward-only、无梯度。
错误 dtype/device/layout/空列表/UPLO/不可表示大小明确失败，无回退或谱 padding。
输出按输入顺序返回五元组 w/V/info/residual/sweeps，设备和 shape 遵循 M5-3A。
输入不变，输出不 alias 输入，批内体系不可截断。

串行复用 M5-3A 数学。多 stream 使用预加载、释放 GIL 的独立扩展入口和
固定 2/4 worker；设备/当前 stream 显式绑定，输入生产 event、输出完成 event、
record_stream 和等待全部 future 后抛错保护生命周期。主机仍参与调度和每 sweep
标量读取，不能宣称全异步/零 CPU 工作。

原生异构临时拼接设备数据，用 int64 n/元素偏移/向量偏移及 tile 工作列表。
保持每矩阵中心化、256线程规约顺序、Jacobi 配对/表达式、稳定设备排序、
原始范数 `1e-15` 收敛和 info=0/1/2。全局 round 上限取实际最大 round 数；
每矩阵在本地 round 结束后跳过核函数。所有 round 数为奇数，相差偶数，
因此空转后的双缓冲与本地结果一致。冻结同时复制两个缓冲；先 CPU 验证。
设备 metadata 拷贝属于控制输入，矩阵无 D2H/H2D 计算回退。

probe 数学 adapter 先产生全部候选 H，再一起分解，计算全部方向后统一提交。
公式逐字复用产品分组数学，失败时坐标/Hessian/历史/计数均不变。
不加入曲率跳过、epsilon、谱 clamp 或不同 maxstep 规则。

谱 atol=2e-10/rtol=2e-11，方程和正交 <1e-10；保持已有 ASE 逐步容差。
验证真未收敛、输入保护、后组失败、跨 stream producer/consumer 及内存复用。
真实 Li八组、T2 561 B8和混合组；三独立进程、预热2次、至少7次交替样本，
按同一完整 checked scope 记录 wall/HIP event、峰值显存、退化与 kernel 重叠。
完整 BFGS 窗口与分解窗口分别记录。共享 MLIP 求值不拆成各体系调用。
T2共享力 ASE数值 oracle 不算独立 CPU MLIP 对照。未完成全收敛不声称完成。

本合同隔离验收完成，结果与未覆盖范围见[M5-3B报告](../reports/g2-ase-bfgs-m5-3b.md)。
2stream为建议接入候选，原生异构和4stream暂留实验；M5-4接入待确认。
