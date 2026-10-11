# M5-2：隔离 HIP round-robin Jacobi 求解器合同

这是 M5-2 的历史验收合同。当前动态原型已取消其尺寸/Batch 上限，见
[M5-3A 合同](HIP_BFGS_JACOBI_DYNAMIC_CONTRACT.md)；旧源码快照保留在
`artifacts/ase-bfgs-m5-3a-2026-10-10/original-m5-2/`，旧结果不改写。

2026-10-10，用户“继续”确认[M5-2方案](ASE_BFGS_M5_2_PROPOSAL.md)。本地新增原型，
不是NVIDIA已有BFGS移植，尚不注册产品后端。保留当前Torch/rocSOLVER及全部
BFGS/UnitCellFilter/Batch数学与默认选择。

当前状态：CPU oracle、HCU19例、10步ASE/重启/失败原子性、真实MPA-0变胞ASE对照
及三进程微基准已通过；初次端到端计时受共享HCU4作业影响，仍排除。后续新窗口
三轮配对全部通过，中位数57.308→46.726秒，见[端到端报告](../reports/g2-ase-bfgs-m5-2-e2e.md)。
M5-2隔离原型验收完成，仅探针验证，不扩大产品能力；见[M5-2报告](../reports/g2-ase-bfgs-m5-2.md)。

## 接口与数值

探针内 `raw_eigh(a, max_sweeps=100)` 返回独立 `(w, V, info, residual, sweeps)`；
`eigh(a, UPLO="L")` 检查输入、info和有限性，返回w/V，供隔离BFGS测试注入。

- 实对称FP64、连续Torch行主序 `[n,n]` 或 `[B,n,n]`；n仅147/285、B=1..4，
  无空输入、无grad、仅当前已验证HCU Torch/DTK26.04/gfx936。CPU/其他dtype/
  非连续/UPPER/未知规模明确拒绝，不回退。单独的CPU准备程序生成oracle。
- LOWER定义矩阵，UPPER允许污染且不读取。原型把选中三角对称展开为内部列主序；
  输出V为Torch连续行主序，**列为特征向量**；w升序，简并时按内部列索引确定排序。
- 无单位/PBC依赖。输入不修改，输出不alias；工作张量归调用所有。CUDAGuard和
  current stream，所有矩阵计算/旋转/排序/诊断规约在HCU。主机只调度固定匹配轮次，
  每个sweep显式读取一个“是否仍有未收敛体系”的标量；不承诺全异步或无host同步。
- round-robin不相交Givens旋转，`A'=R^T A R, V'=V R`。奇数n用调度虚拟索引，
  无padding谱。采用设备内标量中心化 `sigma=trace(A)/n`，分解 `A-sigma*I`，
  排序后w加回sigma；外部Hessian不改，残差阈值仍按未平移矩阵原始norm计算。
  这是对早期近alpha简并Hessian的精度/收敛修正，不放宽阈值或做谱正则化。
  每轮从只读旧A/V生成新A/V，保持对称，避免跨block原地数据竞争。
  不对特征值做clamp、阻尼、曲率跳过或修改ASE的abs谱规则。
- 收敛为 `||offdiag(A)||F <= 1e-15*||A_initial||F`，零矩阵单独精确收敛；
  max_sweeps正整数1..100。info=0成功、1未收敛、2非有限数值诊断；checked路径
  拒绝任何非零info。没有因失败切换算法的路径。零/负特征值保持原值，BFGS
  自身仍按既有零谱错误/负谱abs规则处理；整个Batch失败时状态不提交。
- forward-only eager原型，无torch.library产品注册、fake/autograd/compile支持。
  可重复配对/排序不构成全工作流bitwise确定性承诺。

## 验收顺序

1. CPU先验证配对覆盖、旋转公式，独立torch.linalg.eigh FP64生成捕获147/285
   Hessian、随机、解析谱、近简并、负/零谱和三角污染oracle。不同矩阵B4另外验证。
2. HIP谱atol=2e-10/rtol=2e-11，相对方程/正交<1e-10；保留输入，检查info与
   max_sweeps=1实际失败、非法输入、非默认stream、稳定简并方向。
3. 隔离接入既有BFGS数学：ASE逐步H/方向/坐标/maxstep、restart，异构147/285/147
   分组与失败原子性；使用既有数值门，不通过放宽容差保留候选。
4. 过数值门后按M5-1同设备/输入/FP64/检查范围，三进程、每项预热2次/7样本，
   原型与rocSOLVER交替；冷构建/首调用及profiler另记。保留条件为完整checked
   solver稳态中位数至少下降10%，跨重复可见，并有同配置两CIF MPA-0变胞CSP
   工作流收益。数值或性能门失败，记录并停止接入产品。

范围依据为[M5-1实测](../reports/g2-ase-bfgs-m5-1.md)；并行Jacobi的学术来源参考
[Brent/Luk/Van Loan, 1985](https://maths-people.anu.edu.au/brent/pub/pub084.html)。
代码为本项目新增实现，不复制来源项目。实际结果另记报告，未运行不得写verified。
