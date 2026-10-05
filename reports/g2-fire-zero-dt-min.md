# FIRE `dt_min=0` 本地兼容性修正

日期：2026-10-05。用户确认先修复零下限拒绝问题。
前一4000原子测量先收口 `1dba058`，当前分支 `codex/fix-fire-zero-dt-min`
从该提交开始；本修正完成，待用户确认下一阶段后提交，未合并/推送。

## 实现与归因

`packages/ops/nvalchemiops/cell_fire_reference.py::fire_cell_post` 的本地新增
`dt_min<=0` 校验过严。改为 `<0`，允许零下限；实际 `dt>0`、有限值、
shape/dtype/alias 和状态写入前校验保持。非正功率分支仍为
`max(dt*f_dec, dt_min)`，没有替换为一个正数下限。

锁定 NVIDIA ops `26dbceb61e30cca80e1a5805eebeb51d7dc68fd1`
及用户 fork ops `eadc3eeebaeceb832f7341eb59303dc588f857bd` 的
`nvalchemiops/dynamics/optimizers/fire.py` 均已有该 `max` 公式，允许零下限。
这是 **LOCAL-004 / LP-084 本地移植 bug**，不是新的上游 bug，亦非新增算法。
代码注释与 `docs/UPSTREAM_BUGFIXES.md` 分别记录本地归因。

只改一项 reference 参数边界；没有改 framework 产品代码、公共 API、默认参数、
FIRE 步序/质量/晶胞力、FIRE2 或 registry。显式 `torch_reference` 和默认 Warp 选择保持。

## CPU 验证

- 修复前四个新增零下限测试全部因 `invalid FIRE controls` 失败，复现原问题。
- 修复后 ops 专项 `46 passed`、framework 专项 `19 passed`。
- `OMP_NUM_THREADS=1 bash scripts/check_cpu_reference.sh` 退出码0：
  **434 passed, 3 skipped**（156+1+231+22+24；三项原有 Warp 缺依赖 skip）。
  包含 compileall、Git diff whitespace 检查；不是 HCU 全量 gate。
- 新增 FP64 NumPy 对照覆盖正/负/零/纯晶胞功率，以 `dt=.0004/.0006`
  检验零下限下负/零功率应缩为 `.0002/.0003`，避免隐含原 `.001` 下限。
  负下限、实际零步长和负步长仍拒绝，所有输入状态保持；公共解析势160步收敛
  且分数坐标守恒。原有 FP32/FP64 reference 回归保留。
- 三个修改 Python 文件的 Ruff lint 与 format 检查通过。

## HCU4 FP64 验证

运行前只读确认HCU4空闲；`HIP_VISIBLE_DEVICES=4`、OMP=1、外层180秒限时。
普通 Torch/e3nn eager，没有 CuEq/compile；进程退出码0。运行后无 KFD PID。

- 四种联合功率单步与独立 NumPy 的速度、全部晶胞速度、alpha/dt/计数
  最大绝对误差均为0；三种非法边界在写入前拒绝。
- 公共解析势25步 CPU/HCU 对照：位置最大绝对误差 `1.11e-16 Å`，
  晶胞 `9.32e-18 Å`，能量 `8.47e-21 eV`；势能降低、分数坐标守恒。
- 原4000测量抽样清单前三个未经预弛豫CIF：Li3Nb2CrO6、Li4V2Fe3Co3O16、
  Ba2Li3Y3(WO4)8，合计124原子。真实 MACE-MPA-0-medium + 原生完整矩阵 FIRE
  `dt=.1 / dt_max=1 / dt_min=0 / maxstep=.2 / n_min=5` 运行10次实际优化更新；
  真实原子质量、晶胞质量1、模型eval且冻结参数，位置/应变梯度仍用于F/stress。
  显式Torch reference邻居、skin=0、6Å每次重建。E/F/stress/r/H保持设备内FP64且有限，
  原子位置和晶胞发生更新、体积为正。此短步无收敛退出判据，不报告为弛豫完成。

模型SHA256：`75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638`。
原始证据在 `artifacts/fire-zero-dt-min/`：CPU gate与专项日志、
`verify_hcu.py`、`hcu.log`、`hcu.json`、`mpa0-final.pt`、
NVIDIA/fork源码公式审计及作业后资源日志；私有数据与模型不入库。
HCU复核入口（需先确认HCU4资源可用）：

```sh
source scripts/activate_hygon_env.sh project
HIP_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 PYTHONPATH="$PWD:$PYTHONPATH" \
  timeout 180s python artifacts/fire-zero-dt-min/verify_hcu.py
```

## 限制与下一阶段

零下限没有重跑完整4000原子Batch，未新增长期收敛、持续补位吞吐、NVIDIA/Warp数值
对照或完整HCU gate证据；既有Batch性能继续使用 `.002` 历史参数，不能直接归于零下限。

后续 M1–M5 收口审查已完成，见 [审查报告](g2-relaxation-port-closure-review.md)。
本修正已签名提交 `0e354f4`；审查文档待下一阶段确认后收口。
