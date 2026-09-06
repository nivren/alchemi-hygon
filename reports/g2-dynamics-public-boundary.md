# G2/N2b：dynamics 公共导入与 reference wrapper 边界

日期：2026-09-06  
目标设备：海光 BW200 / UBB BW1000，gfx936

本步完成了不改变默认后端的导入隔离：

- `nvalchemi.dynamics`、`integrators`、`optimizers` 和
  `dynamics.hooks` 改为惰性导出；命名空间导入不再触发 Warp；
- `_bridge` 的 Warp dtype 映射改为函数内按需加载，状态 Batch 和参数广播
  可以在无 Warp 环境导入；
- `_ops.velocity_verlet`、`_ops.fire` 移除顶层 Warp/ops 导入；`backend=None`
  或 `backend="warp"` 仍使用原 custom-op/Warp 路径，
  `backend="torch_reference"`/`"auto"` 委派到顶层 reference；
- NVE 类增加 `backend` 保存并传递给 VV wrapper；`from nvalchemi.dynamics import NVE`
  在无 Warp 进程中可完成导入；
- hook kinetics 工具支持显式 `backend="torch_reference"`，默认仍保持原 Warp
  custom-op 路径；
- `StageTimingHook` 的上游测试已解除收集阻断。

## 验证

项目 `.venv` 导入边界测试：

```bash
PYTHONPATH=packages/framework:packages/ops .venv/bin/python -m pytest -q \
  packages/framework/test/compatibility/test_dynamics_import_boundary.py
```

结果：`3 passed`，退出码 `0`。独立子进程验证了 `warp` 不在
`sys.modules`，并执行了公共 VV、FIRE2 和 kinetics reference wrapper。

StageTimingHook 收集验证：

```bash
PYTHONPATH=packages/framework:packages/ops .venv/bin/python -m pytest \
  --collect-only -q packages/framework/test/hooks/test_stage_timing_hook.py
```

结果：`43 tests collected`，退出码 `0`。这只证明导入/收集阻断解除，尚未
宣称 43 项均在 HCU 运行通过。

## 边界

本步没有把 NVE/FIRE/FIRE2 的完整 `BaseDynamics.run()` 工作流改成 reference，
也没有接入 MACE、DataSink、收敛 Hook 或 inflight batching。旧类请求其默认
路径时仍会按需导入 Warp；reference 只在显式 backend 参数下选择。下一步是
让 NVE 的 reference `pre_update/post_update` 通过完整 Batch→model→forces 链运行，
再以同一模式接入固定晶胞 FIRE/FIRE2。
