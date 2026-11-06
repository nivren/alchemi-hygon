# 合作方参考版本第一阶段：BFGS 本地集成与复验

日期：2026-11-06。用户分别确认范围核验、候选复验和本地集成里程碑。
本阶段将已完成的BFGS工作纳入本地develop，不推送、不创建标签、不发布参考版本。

## 实现与集成范围

集成起点：`cfd22ad11fe0e1704dff48f53abee69135f8db99`。
已验证功能源码：`e3605c1856cc9ba19475157df38f8cf1c63ed265`。
保留四个DCO签名提交：Ops `d68ea01`、Framework `bf52bca`、验证 `93955b0`、文档 `e3605c1`。
合计120文件，与[原交付清单](../docs/ASE_BFGS_DELIVERY_FILES.json)及历史交付哈希逐项匹配。
所有本机已知项目分支中，只有上述BFGS提交尚未被develop包含；不含上游参考分支，
也不代表联网发现了所有服务端分支或未提交工作。

在独立分支 `wangleping/integration-partner-reference-stage1` 快进纳入四提交，
复验后补充本报告和当前交付状态，再以fast-forward纳入本地develop。
集成记录提交不修改产品源码、测试、probes、锁文件、包版本或依赖。
功能分支2026-10-11已向三个remote推送；本次develop集成未推送。

原工作树中的三个2026-10-14应用汇报文件保持原样，不加入提交；旧worktrees均保留。

## 本轮测试

所有验证针对上述e3605c1源码，在独立候选worktree运行；两包导入路径已断言位于候选packages。
复用原项目.venv，不重装依赖；候选PYTHONPATH明确，HIP扩展使用独立构建缓存。
环境：Python3.12.13、DTK26.04、海光Torch2.9.0+das.opt1.dtk2604、HIP6.3.26093、ASE3.29.0。
单卡 `HIP_VISIBLE_DEVICES=4`，smoke识别BW1000 64G；每项HCU作业前确认该卡空闲。

| 验证 | 实际结果 |
| --- | --- |
| `bash scripts/check_cpu_reference.sh` | 退出0；669 passed / 5 skipped |
| `probes/torch_probe.py --device cuda` | 退出0 |
| `bash scripts/check_hcu_reference_smoke.sh` | 六项reference smoke全部完成，退出0 |
| Ops Batch与native Jacobi HCU | 退出0；18 passed |
| Framework BFGS HCU | 退出0；24 passed |
| 原rocSOLVER HCU | 退出0；39 passed |
| 公共BFGS对ASE十步 | 退出0；11体系，配对restart逐位一致 |
| Git清单/哈希/差异检查 | 候选120文件与启动前一致，工作树干净 |

CPU跳过含三项缺少Warp的NHC对照及两项HCU溢出保护参数；后两项已在HCU Ops门实际通过。
各组有重叠，不把测试数相加作为独立用例总量；reference smoke不是优化HIP邻居性能验收。

专用回归命令（在候选/已集成源码根目录，先加载project环境，HCU须先确认资源）：

```bash
# CPU gate隐藏设备；HCU命令显式选择已分配或确认空闲的4号卡。
HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= bash scripts/check_cpu_reference.sh
HIP_VISIBLE_DEVICES=4 bash scripts/check_hcu_reference_smoke.sh

HIP_VISIBLE_DEVICES=4 .venv/bin/python -m pytest -q \
  packages/ops/test/torch/test_ase_bfgs_batch.py \
  packages/ops/test/torch/test_bfgs_jacobi_hip_hcu.py

HIP_VISIBLE_DEVICES=4 .venv/bin/python -m pytest -q \
  packages/framework/test/compatibility/test_ase_bfgs_jacobi_hcu.py \
  packages/framework/test/compatibility/test_ase_bfgs_cell_hcu.py \
  packages/framework/test/compatibility/test_ase_bfgs_batch_hcu.py

HIP_VISIBLE_DEVICES=4 .venv/bin/python -m pytest -q \
  packages/ops/test/torch/test_bfgs_eigh_hip_hcu.py

# --output应使用新的artifacts路径。
HIP_VISIBLE_DEVICES=4 .venv/bin/python probes/ase_bfgs_product_acceptance.py \
  --mode bfgs --output artifacts/public-bfgs-ase-rerun.json
```

上面列出复跑入口；本轮每项具体超时、完整环境、命令和退出码以原始记录为准。
本地develop仅增加交付文档，产品源码与已验证候选相同，未重复运行不变的数值门。

## 数值结果与未验证范围

11体系DOF为15/21/30/45/57/93/201/489/189/189/561，运行10步。
最大绝对差：坐标2.673e-14、Hessian7.816e-14、谱1.739e-11、raw step3.517e-14、
applied step5.190e-15。零谱、不收敛、错误特征向量的预期拒绝门通过。

本轮未重跑真实MPA-0完整CSP、wheel/sdist、性能套件或独立完整HIP邻居验收。
此前真实CSP与包装证据见[最终审阅](g2-ase-bfgs-delivery-review.md)，
动态HIP实现和输入相关性能见[M5-4报告](g2-ase-bfgs-m5-4.md)。
FP64/eager/gfx936限定构建、稠密内存开销、标量host同步与原退化案例保持；
不扩大到其他硬件/构建、NVIDIA、compile、梯度、多卡、全目录成功率或生产吞吐。

## 原始记录与后续

- 范围/候选复验：`artifacts/partner-reference-stage1-m2-2026-11-06/`，含status、provenance、final-audit、各任务log及公共ASE结果。
- 本地集成核验：`artifacts/partner-reference-stage1-m3-2026-11-06/`，保留集成前后refs、文档diff与最终提交记录。
- 原阶段记录保持原日期，历史“未提交/未合并/未推送”表述仅对应当时；当前结论见[STATUS](../docs/STATUS.md)。

下一阶段为合作方参考版本的能力清单与范围确认。每阶段、每个小里程碑开始前说明目标，
完成后总结，获得用户明确确认才进入下一里程碑；本次不启动文档全面整理或示例开发。
