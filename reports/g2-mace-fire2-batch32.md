# G2 MACE + FIRE2 批量固定晶胞弛豫

## 范围

本次验证使用项目 `.venv`、DTK 26.04、目标 HCU `gfx936`（设备名
`BW200, UBB BW1000`），在一个 `Batch` 中同时处理 32 个 92 原子周期
CIF。模型是用户提供的本地 `MACE-OFF23_small.model`，优化器是公共
`FIRE2(backend="torch_reference")`，固定晶胞，`fmax=0.01`，最大步数
2000，`dt=0.01`，reference 邻居 skin 为 0.5。

这里的 `dt=0.01` 是 FIRE2 必填的初始步长；其余 FIRE2 超参数保持锁定
上游默认值。32 个结构在同一 Batch 中进行模型前向、邻居写回和 per-system
FIRE2 状态更新，没有 Python 逐结构优化循环。

## 可重跑命令

```bash
source scripts/activate_hygon_env.sh project
HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 timeout 230 \
  python probes/mace_fire2_batch_relaxation.py \
  --device cuda --count 32 --max-steps 2000 --fmax 0.01 \
  --dt 0.01 --skin 0.5 --log-every 20 --max-wall-seconds 190 \
  --checkpoint /home/wangleping/.cache/mace/MACE-OFF23_small.model
```

退出码为 `0`。设备可见性和 DTK 环境必须与命令一致；受限沙箱隐藏
`/dev/kfd` 时不能据此判断 HCU 能力。

## 结果

- `num_structures=32`，`num_nodes=2944`，`batch_ptr` 为 32 个连续的
  92 原子区间，未发生跨体系邻居边。
- 所有 32 个结构在 `step_count=837` 前达到收敛；总用时
  `143.8770088129677 s`，低于 190 秒保护线和 200 秒基线目标。
- 结束后在返回的最终坐标上重新触发 reference 邻居 Hook 并计算 MACE
  力；`max_final_fmax=0.00997547060251236`，32 个结构均满足阈值。
- 最终邻居边数为 `74952`。邻居矩阵、位移、索引和 MACE 输入均留在 HCU
  Torch tensor；skin rebuild 的判断仍有少量 host sync。
- MACE 运行的是 `mace-torch`/`e3nn` 的真实模型组合；本次没有安装或启用
  `cuEquivariance` 加速。

## FIRE2 数值边界

FIRE2 的 reference 三阶段按锁定上游实现：deferred half-step 归约、
`P > 0`/`P <= 0` 状态分支、`delaystep` 后的步长/alpha 更新、更新后
alpha 与更新前 dt 的速度混合、上坡 `-0.5` 修正、per-system `maxstep`
裁剪和同步缩放 dt。与 `/home/wangleping/codes/hyalchemi-ops` 的
独立 Torch 实现进行 5 组异构随机输入交叉运行，位置、速度、alpha、dt
和计数器的最大差均为 0；该工程只作交叉证据，锁定上游仍是规范。

批处理中已收敛图的 `BaseDynamics.step()` 会暂时执行共享前向，然后恢复
其坐标。为避免 `forces`/`energy` 留下临时坐标的结果，默认保存/恢复字段
现在还包括 `forces`、`energy`、`stress`；最终 probe 仍显式重新评估一次，
作为返回状态的独立检查。

## 限制

这证明的是固定晶胞、Torch reference、单 HCU、32×92 周期 MACE/FIRE2
组合，不等同于完整生产后端。变胞 stress、NVT/Langevin、DataSink/重启、
inflight 补位、周期 half-list、cell-list、Triton/HIP kernel 和默认 Warp
路径仍分别排期。
