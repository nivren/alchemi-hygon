# G2 backend registry 收尾审计

日期：2026-09-08。审计对象为合并到 `develop` 的当前代码。

## 审计命令

```bash
rg -n "_select_backend|resolve_compute_backend|resolve_backend|backend.*(warp|auto|torch_reference)|torch_reference.*backend" \
  packages/framework/nvalchemi/dynamics/_ops \
  packages/framework/nvalchemi/hooks \
  packages/framework/nvalchemi/models \
  packages/framework/nvalchemi/neighbors.py \
  packages/ops/nvalchemiops/backend.py \
  packages/ops/nvalchemiops/torch_backend.py
```

## 结果

- `packages/framework/nvalchemi/dynamics/_ops/fire.py` 和
  `velocity_verlet.py` 均直接使用 `nvalchemi._backend.resolve_compute_backend`；
  `dynamics/_ops` 下不存在 `_select_backend`。
- `neighbors.py`、`NeighborListHook`、LJ、periodic helper、observer utility
  使用同一 framework adapter；ops dispatcher 直接使用
  `nvalchemiops.backend.resolve_backend`。
- `cell_align.py`、`langevin.py`、`nose_hoover.py`、`npt_nph.py`、
  `thermostat_utils.py` 和 `_bridge.py` 属于当前 Warp-only/桥接白名单，不承担
  reference backend 选择，因此不强行导入 resolver。
- `neighbors._write_neighbor_data_to_batch` 中的 `backend == "warp"` 是在中央
  选择完成后的矩阵到 COO 执行分支；`LennardJonesModelWrapper.distribution_spec`
  的 Warp 判断是未实现 reference 域分解时的能力保护。二者都没有维护独立的
  backend 枚举或 `auto` 选择规则。
- `StageTimingHook.timer_backend` 是计时器实现选择（`cuda_event`/
  `perf_counter`），与计算 backend 不同，保持独立是正确的。

## 结论

当前 G2 slice 已满足“计算 backend 由中央 registry 选择”的收敛条件；没有发现
需要在本轮重新拆分的局部 backend selector。新增算子或 framework 接线仍必须先
登记 capability，再接入 framework adapter。

本审计不等同于 Triton/HIP 已注册，也不等同于 Warp-only 模块已经获得 HCU 支持。
