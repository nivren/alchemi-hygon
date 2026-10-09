# Upstream Delta V1 — 2026-10-09

只读静态分析；比较冻结 Git 对象，未运行候选源码/安装依赖/修改产品或版本锁。
本地产品精确提交：`132ff32d90e4cf56b422a34674a5a93c596490af`。候选状态：Candidate，尚非 Validated。

## 方法与复核边界

文件变化采用两端 tree diff（包含 rename 两端），历史采用 merge-base 与双侧独有提交。
同时比较基线上游→候选、基线上游→本地产品子树并取交集；交集仅为潜在冲突。
API 为 AST 可见非私有函数/类/方法、构造参数、注解字段及 __init__ 导出；不等于所有正式公共 API。
记录参数、默认值、返回注解、装饰器；动态导出、实际返回值和数值语义须人工复核。
签名不变但 AST 实现变化单独标记。源码不导入、不执行；parse failure 不被当作无变化。
Markdown 包含全部文件/签名变化；可选 JSON 保留每个同签名实现变化及完整依赖声明。

| 分析环境 | 记录 |
|---|---|
| python | 3.12.13 |
| git | git version 2.34.1 |

| 输入记录 | SHA256 |
|---|---|
| lock | 0a3f1562cde340150c70bdb1367220b111c36277cefb3f25d884e4011ea55ecf |
| candidate | 9ba1fc4e8804d2b0c15d43ce9d22314f5a6911295ab70cccacd4ee27e9520ef7 |
| patches | 816fee9fe50a4246f6abe9d1c95ce2e0e8405220d84b0f75437e2995035d240c |
| bugfixes | 2032fd957b2bf3ecdd5364927e394ffa9d2407051d92152420b9e5657a5bfc5c |
| tool | bc23debcae6c7aa78f25bcb877d2a4907f6749cbcc2a785d57e39221688dbab1 |

## 摘要与使用入口

| 仓库 | 历史关系 | 上游文件变化 | 本地交集 | AST 接口/导出变化 |
|---|---|---|---|---|
| framework | diverged | 467 | 40 | 1236 |
| ops | diverged | 322 | 8 | 357 |

复现命令与独立 Git 对象获取方式见 [UPSTREAM_SYNC.md](../docs/UPSTREAM_SYNC.md#4-差异分析及补丁审查)。
依赖阻塞、未移植特性与 retain/backport/defer 决策见文末人工审阅；详细清单可展开阅读。

## framework

| 版本 | 完整 SHA | 声明版本 |
|---|---|---|
| Baseline | 4dfe3723def34df3fadb245981081ccf8c94c257 | 0.2.0 |
| Candidate (main) | f59e28dc6701583a13bb294e536076c2a7379e98 | 0.2.0 |

官方源码：[基线](https://github.com/NVIDIA/nvalchemi-toolkit/tree/4dfe3723def34df3fadb245981081ccf8c94c257)、[候选](https://github.com/NVIDIA/nvalchemi-toolkit/tree/f59e28dc6701583a13bb294e536076c2a7379e98)。

历史关系：**diverged**；基线独有 24 个提交，候选独有 41 个提交。
merge-base：bdc7fb68efad484062699f9725e8c33448c479e0

### 文件与主要模块

上游文件变化 467：{'A': 239, 'D': 1, 'M': 227}；本地文件变化 88；潜在交集 40。

| 主要模块/路径前缀 | 变化文件 |
|---|---|
| .claude/skills | 10 |
| .github/workflows | 1 |
| .greptile/config.json | 1 |
| .greptile/files.json | 1 |
| AGENTS.md | 1 |
| CHANGELOG.md | 1 |
| benchmark/distributed | 2 |
| docs/_static | 1 |
| docs/index.md | 1 |
| docs/modules | 23 |
| docs/sg_execution_times.rst | 1 |
| docs/userguide | 23 |
| examples/README.rst | 1 |
| examples/advanced | 13 |
| examples/basic | 4 |
| examples/distributed | 4 |
| examples/intermediate | 7 |
| greptile.json | 1 |
| nvalchemi/_checkpoint.py | 1 |
| nvalchemi/_optional.py | 1 |
| nvalchemi/_serialization.py | 1 |
| nvalchemi/_typing.py | 1 |
| nvalchemi/csp | 38 |
| nvalchemi/data | 16 |
| nvalchemi/distributed | 4 |
| nvalchemi/dynamics | 54 |
| nvalchemi/enhanced_sampling | 16 |
| nvalchemi/gen | 5 |
| nvalchemi/hooks | 8 |
| nvalchemi/models | 16 |
| nvalchemi/neighbors.py | 1 |
| nvalchemi/specs.py | 1 |
| nvalchemi/training | 47 |
| pyproject.toml | 1 |
| test/conftest.py | 1 |
| test/csp | 21 |
| test/data | 8 |
| test/distributed | 24 |
| test/dynamics | 33 |
| test/enhanced_sampling | 9 |
| test/gen | 5 |
| test/hooks | 6 |
| test/models | 11 |
| test/test_checkpoint.py | 1 |
| test/test_multigpu_marker.py | 1 |
| test/test_serialization.py | 1 |
| test/training | 38 |
| uv.lock | 1 |

<details><summary>完整上游文件变化（A/M/D/R/T 等 Git 状态）</summary>

| 状态 | 之前 | 之后 |
|---|---|---|
| M | .claude/skills/README.md | .claude/skills/README.md |
| M | .claude/skills/nvalchemi-data-structures/SKILL.md | .claude/skills/nvalchemi-data-structures/SKILL.md |
| A | — | .claude/skills/nvalchemi-distillation/SKILL.md |
| M | .claude/skills/nvalchemi-distributed/SKILL.md | .claude/skills/nvalchemi-distributed/SKILL.md |
| M | .claude/skills/nvalchemi-dynamics-api/SKILL.md | .claude/skills/nvalchemi-dynamics-api/SKILL.md |
| M | .claude/skills/nvalchemi-dynamics-hooks/SKILL.md | .claude/skills/nvalchemi-dynamics-hooks/SKILL.md |
| M | .claude/skills/nvalchemi-dynamics-implementation/SKILL.md | .claude/skills/nvalchemi-dynamics-implementation/SKILL.md |
| A | — | .claude/skills/nvalchemi-mep/SKILL.md |
| M | .claude/skills/nvalchemi-model-wrapping/SKILL.md | .claude/skills/nvalchemi-model-wrapping/SKILL.md |
| M | .claude/skills/nvalchemi-reporting/SKILL.md | .claude/skills/nvalchemi-reporting/SKILL.md |
| M | .github/workflows/ci.yml | .github/workflows/ci.yml |
| A | — | .greptile/config.json |
| A | — | .greptile/files.json |
| M | AGENTS.md | AGENTS.md |
| M | CHANGELOG.md | CHANGELOG.md |
| M | benchmark/distributed/README.md | benchmark/distributed/README.md |
| M | benchmark/distributed/configs/uma.yaml | benchmark/distributed/configs/uma.yaml |
| M | docs/_static/js/install-matrix.js | docs/_static/js/install-matrix.js |
| M | docs/index.md | docs/index.md |
| A | — | docs/modules/csp.rst |
| M | docs/modules/data.rst | docs/modules/data.rst |
| M | docs/modules/distributed_runtime.rst | docs/modules/distributed_runtime.rst |
| M | docs/modules/dynamics/api.rst | docs/modules/dynamics/api.rst |
| M | docs/modules/dynamics/buffers_and_data_flow.rst | docs/modules/dynamics/buffers_and_data_flow.rst |
| M | docs/modules/dynamics/fused_stage.rst | docs/modules/dynamics/fused_stage.rst |
| M | docs/modules/dynamics/hooks.rst | docs/modules/dynamics/hooks.rst |
| M | docs/modules/dynamics/implementing_dynamics.rst | docs/modules/dynamics/implementing_dynamics.rst |
| M | docs/modules/dynamics/index.rst | docs/modules/dynamics/index.rst |
| A | — | docs/modules/dynamics/mep.rst |
| M | docs/modules/dynamics/methods.rst | docs/modules/dynamics/methods.rst |
| M | docs/modules/dynamics/overview.rst | docs/modules/dynamics/overview.rst |
| A | — | docs/modules/gen.rst |
| M | docs/modules/hooks.rst | docs/modules/hooks.rst |
| M | docs/modules/index.md | docs/modules/index.md |
| M | docs/modules/models.rst | docs/modules/models.rst |
| A | — | docs/modules/training/distillation.rst |
| M | docs/modules/training/finetuning.rst | docs/modules/training/finetuning.rst |
| M | docs/modules/training/hooks.rst | docs/modules/training/hooks.rst |
| M | docs/modules/training/index.rst | docs/modules/training/index.rst |
| M | docs/modules/training/losses.rst | docs/modules/training/losses.rst |
| M | docs/modules/training/strategy.rst | docs/modules/training/strategy.rst |
| M | docs/modules/training/validation.rst | docs/modules/training/validation.rst |
| M | docs/sg_execution_times.rst | docs/sg_execution_times.rst |
| M | docs/userguide/about/conventions.md | docs/userguide/about/conventions.md |
| M | docs/userguide/about/install.md | docs/userguide/about/install.md |
| M | docs/userguide/about/intro.md | docs/userguide/about/intro.md |
| M | docs/userguide/agent_skills.md | docs/userguide/agent_skills.md |
| A | — | docs/userguide/csp.md |
| M | docs/userguide/data.md | docs/userguide/data.md |
| M | docs/userguide/datapipes.md | docs/userguide/datapipes.md |
| A | — | docs/userguide/distillation.md |
| M | docs/userguide/distributed.md | docs/userguide/distributed.md |
| M | docs/userguide/distributed_training.md | docs/userguide/distributed_training.md |
| M | docs/userguide/dynamics.md | docs/userguide/dynamics.md |
| A | — | docs/userguide/dynamics_mep.md |
| M | docs/userguide/dynamics_simulations.md | docs/userguide/dynamics_simulations.md |
| M | docs/userguide/dynamics_sinks.md | docs/userguide/dynamics_sinks.md |
| A | — | docs/userguide/enhanced_sampling.md |
| M | docs/userguide/finetuning.md | docs/userguide/finetuning.md |
| A | — | docs/userguide/generative.md |
| M | docs/userguide/hooks.md | docs/userguide/hooks.md |
| M | docs/userguide/index.md | docs/userguide/index.md |
| M | docs/userguide/models.md | docs/userguide/models.md |
| M | docs/userguide/reporting.md | docs/userguide/reporting.md |
| M | docs/userguide/training.md | docs/userguide/training.md |
| M | docs/userguide/training_finetuning.md | docs/userguide/training_finetuning.md |
| M | examples/README.rst | examples/README.rst |
| M | examples/advanced/01_biased_potential.py | examples/advanced/01_biased_potential.py |
| M | examples/advanced/03_custom_convergence.py | examples/advanced/03_custom_convergence.py |
| M | examples/advanced/09_uma_nve.py | examples/advanced/09_uma_nve.py |
| A | — | examples/advanced/11_batched_neb.py |
| A | — | examples/advanced/12_umbrella_sampling.py |
| A | — | examples/advanced/13_replica_exchange.py |
| A | — | examples/advanced/14_metadynamics.py |
| A | — | examples/advanced/15_rmsd_metadynamics.py |
| A | — | examples/advanced/16_adaptive_biasing_force.py |
| M | examples/advanced/README.rst | examples/advanced/README.rst |
| A | — | examples/advanced/_dneb_method.py |
| A | — | examples/advanced/batched_neb_endpoints/dft_reference.json |
| A | — | examples/advanced/batched_neb_endpoints/endpoints.json |
| M | examples/basic/02_geometry_optimization.py | examples/basic/02_geometry_optimization.py |
| M | examples/basic/03_ase_integration.py | examples/basic/03_ase_integration.py |
| A | — | examples/basic/06_geometry_optimization_lbfgs.py |
| M | examples/basic/README.rst | examples/basic/README.rst |
| M | examples/distributed/01_distributed_pipeline.py | examples/distributed/01_distributed_pipeline.py |
| M | examples/distributed/02_distributed_monitoring.py | examples/distributed/02_distributed_monitoring.py |
| M | examples/distributed/07_fire_nvt_dd.py | examples/distributed/07_fire_nvt_dd.py |
| M | examples/distributed/README.rst | examples/distributed/README.rst |
| M | examples/intermediate/01_multistage_pipeline.py | examples/intermediate/01_multistage_pipeline.py |
| M | examples/intermediate/04_inflight_batching.py | examples/intermediate/04_inflight_batching.py |
| A | — | examples/intermediate/08_lora_finetuning.py |
| A | — | examples/intermediate/09_generative_samplers.py |
| A | — | examples/intermediate/09_offline_distillation.py |
| A | — | examples/intermediate/10_onpolicy_distillation.py |
| M | examples/intermediate/README.rst | examples/intermediate/README.rst |
| D | greptile.json | — |
| A | — | nvalchemi/_checkpoint.py |
| M | nvalchemi/_optional.py | nvalchemi/_optional.py |
| M | nvalchemi/_serialization.py | nvalchemi/_serialization.py |
| M | nvalchemi/_typing.py | nvalchemi/_typing.py |
| A | — | nvalchemi/csp/__init__.py |
| A | — | nvalchemi/csp/_batch.py |
| A | — | nvalchemi/csp/_chem/__init__.py |
| A | — | nvalchemi/csp/_chem/_conformers.py |
| A | — | nvalchemi/csp/_chem/_contacts.py |
| A | — | nvalchemi/csp/_chem/_input.py |
| A | — | nvalchemi/csp/_chem/_topology.py |
| A | — | nvalchemi/csp/_chem/_validation.py |
| A | — | nvalchemi/csp/_comparison/__init__.py |
| A | — | nvalchemi/csp/_comparison/descriptor.py |
| A | — | nvalchemi/csp/_comparison/kernels.py |
| A | — | nvalchemi/csp/_comparison/score.py |
| A | — | nvalchemi/csp/_comparison/store.py |
| A | — | nvalchemi/csp/_comparison/stream.py |
| A | — | nvalchemi/csp/_data_tables.py |
| A | — | nvalchemi/csp/_packing_transport.py |
| A | — | nvalchemi/csp/_space_group_tables.py |
| A | — | nvalchemi/csp/_validation.py |
| A | — | nvalchemi/csp/chem.py |
| A | — | nvalchemi/csp/comparison.py |
| A | — | nvalchemi/csp/data.py |
| A | — | nvalchemi/csp/generator.py |
| A | — | nvalchemi/csp/hooks.py |
| A | — | nvalchemi/csp/packer/__init__.py |
| A | — | nvalchemi/csp/packer/_cells.py |
| A | — | nvalchemi/csp/packer/_contacts.py |
| A | — | nvalchemi/csp/packer/_interop.py |
| A | — | nvalchemi/csp/packer/_kernels/__init__.py |
| A | — | nvalchemi/csp/packer/_kernels/relaxation.py |
| A | — | nvalchemi/csp/packer/_kernels/sampling.py |
| A | — | nvalchemi/csp/packer/_state.py |
| A | — | nvalchemi/csp/packer/config.py |
| A | — | nvalchemi/csp/packer/engine.py |
| A | — | nvalchemi/csp/packer/protocol.py |
| A | — | nvalchemi/csp/packer/result.py |
| A | — | nvalchemi/csp/storage.py |
| A | — | nvalchemi/csp/symmetry.py |
| A | — | nvalchemi/csp/volume.py |
| M | nvalchemi/data/__init__.py | nvalchemi/data/__init__.py |
| M | nvalchemi/data/atomic_data.py | nvalchemi/data/atomic_data.py |
| M | nvalchemi/data/batch.py | nvalchemi/data/batch.py |
| M | nvalchemi/data/buffer_kernels.py | nvalchemi/data/buffer_kernels.py |
| M | nvalchemi/data/data.py | nvalchemi/data/data.py |
| M | nvalchemi/data/datapipes/__init__.py | nvalchemi/data/datapipes/__init__.py |
| M | nvalchemi/data/datapipes/backends/__init__.py | nvalchemi/data/datapipes/backends/__init__.py |
| M | nvalchemi/data/datapipes/backends/base.py | nvalchemi/data/datapipes/backends/base.py |
| M | nvalchemi/data/datapipes/backends/zarr.py | nvalchemi/data/datapipes/backends/zarr.py |
| M | nvalchemi/data/datapipes/dataset.py | nvalchemi/data/datapipes/dataset.py |
| M | nvalchemi/data/datapipes/in_memory_dataset.py | nvalchemi/data/datapipes/in_memory_dataset.py |
| M | nvalchemi/data/datapipes/samplers.py | nvalchemi/data/datapipes/samplers.py |
| A | — | nvalchemi/data/group_layout.py |
| M | nvalchemi/data/level_storage.py | nvalchemi/data/level_storage.py |
| M | nvalchemi/data/transforms/__init__.py | nvalchemi/data/transforms/__init__.py |
| A | — | nvalchemi/data/transforms/supercell.py |
| M | nvalchemi/distributed/__init__.py | nvalchemi/distributed/__init__.py |
| M | nvalchemi/distributed/_dynamics_coordinator.py | nvalchemi/distributed/_dynamics_coordinator.py |
| M | nvalchemi/distributed/_runtime.py | nvalchemi/distributed/_runtime.py |
| M | nvalchemi/distributed/domain_parallel.py | nvalchemi/distributed/domain_parallel.py |
| M | nvalchemi/dynamics/__init__.py | nvalchemi/dynamics/__init__.py |
| M | nvalchemi/dynamics/_ops/__init__.py | nvalchemi/dynamics/_ops/__init__.py |
| M | nvalchemi/dynamics/_ops/_bridge.py | nvalchemi/dynamics/_ops/_bridge.py |
| M | nvalchemi/dynamics/_ops/cell_align.py | nvalchemi/dynamics/_ops/cell_align.py |
| M | nvalchemi/dynamics/_ops/fire.py | nvalchemi/dynamics/_ops/fire.py |
| A | — | nvalchemi/dynamics/_ops/lbfgs.py |
| M | nvalchemi/dynamics/_ops/npt_nph.py | nvalchemi/dynamics/_ops/npt_nph.py |
| M | nvalchemi/dynamics/_ops/thermostat_utils.py | nvalchemi/dynamics/_ops/thermostat_utils.py |
| M | nvalchemi/dynamics/_units.py | nvalchemi/dynamics/_units.py |
| M | nvalchemi/dynamics/base.py | nvalchemi/dynamics/base.py |
| M | nvalchemi/dynamics/hooks/__init__.py | nvalchemi/dynamics/hooks/__init__.py |
| M | nvalchemi/dynamics/hooks/_utils.py | nvalchemi/dynamics/hooks/_utils.py |
| M | nvalchemi/dynamics/hooks/cell_align.py | nvalchemi/dynamics/hooks/cell_align.py |
| M | nvalchemi/dynamics/hooks/freeze.py | nvalchemi/dynamics/hooks/freeze.py |
| M | nvalchemi/dynamics/hooks/logging.py | nvalchemi/dynamics/hooks/logging.py |
| M | nvalchemi/dynamics/hooks/monitors.py | nvalchemi/dynamics/hooks/monitors.py |
| M | nvalchemi/dynamics/hooks/safety.py | nvalchemi/dynamics/hooks/safety.py |
| M | nvalchemi/dynamics/hooks/snapshot.py | nvalchemi/dynamics/hooks/snapshot.py |
| A | — | nvalchemi/dynamics/hooks/swap.py |
| M | nvalchemi/dynamics/integrators/nph.py | nvalchemi/dynamics/integrators/nph.py |
| M | nvalchemi/dynamics/integrators/npt.py | nvalchemi/dynamics/integrators/npt.py |
| M | nvalchemi/dynamics/integrators/nvt_langevin.py | nvalchemi/dynamics/integrators/nvt_langevin.py |
| M | nvalchemi/dynamics/integrators/nvt_nose_hoover.py | nvalchemi/dynamics/integrators/nvt_nose_hoover.py |
| A | — | nvalchemi/dynamics/mep/__init__.py |
| A | — | nvalchemi/dynamics/mep/_alignment.py |
| A | — | nvalchemi/dynamics/mep/_geometry.py |
| A | — | nvalchemi/dynamics/mep/hooks/__init__.py |
| A | — | nvalchemi/dynamics/mep/hooks/climbing_image_selection.py |
| A | — | nvalchemi/dynamics/mep/hooks/neb_force.py |
| A | — | nvalchemi/dynamics/mep/hooks/path_diagnostics.py |
| A | — | nvalchemi/dynamics/mep/hooks/path_energy_stats.py |
| A | — | nvalchemi/dynamics/mep/idpp.py |
| A | — | nvalchemi/dynamics/mep/interpolate.py |
| A | — | nvalchemi/dynamics/mep/neb.py |
| A | — | nvalchemi/dynamics/mep/neb_configs.py |
| A | — | nvalchemi/dynamics/mep/neb_equations.py |
| A | — | nvalchemi/dynamics/mep/neb_ops/__init__.py |
| A | — | nvalchemi/dynamics/mep/neb_ops/equations.py |
| A | — | nvalchemi/dynamics/mep/neb_ops/kernels.py |
| A | — | nvalchemi/dynamics/mep/neb_ops/launchers.py |
| A | — | nvalchemi/dynamics/mep/neb_ops/methods.py |
| A | — | nvalchemi/dynamics/mep/neb_ops/modes.py |
| A | — | nvalchemi/dynamics/mep/neb_ops/torch_ops.py |
| A | — | nvalchemi/dynamics/mep/path_ops/__init__.py |
| A | — | nvalchemi/dynamics/mep/path_ops/paths.py |
| A | — | nvalchemi/dynamics/mep/path_ops/torch_paths.py |
| A | — | nvalchemi/dynamics/mep/validate.py |
| M | nvalchemi/dynamics/optimizers/__init__.py | nvalchemi/dynamics/optimizers/__init__.py |
| M | nvalchemi/dynamics/optimizers/fire.py | nvalchemi/dynamics/optimizers/fire.py |
| M | nvalchemi/dynamics/optimizers/fire2.py | nvalchemi/dynamics/optimizers/fire2.py |
| A | — | nvalchemi/dynamics/optimizers/lbfgs.py |
| M | nvalchemi/dynamics/sinks.py | nvalchemi/dynamics/sinks.py |
| A | — | nvalchemi/dynamics/strategy.py |
| A | — | nvalchemi/dynamics/structure_sampler.py |
| A | — | nvalchemi/enhanced_sampling/__init__.py |
| A | — | nvalchemi/enhanced_sampling/_adaptive.py |
| A | — | nvalchemi/enhanced_sampling/_bias.py |
| A | — | nvalchemi/enhanced_sampling/_exchange.py |
| A | — | nvalchemi/enhanced_sampling/_history.py |
| A | — | nvalchemi/enhanced_sampling/_runner.py |
| A | — | nvalchemi/enhanced_sampling/biases/__init__.py |
| A | — | nvalchemi/enhanced_sampling/biases/abf.py |
| A | — | nvalchemi/enhanced_sampling/biases/metadynamics.py |
| A | — | nvalchemi/enhanced_sampling/biases/rmsd_metad.py |
| A | — | nvalchemi/enhanced_sampling/biases/umbrella.py |
| A | — | nvalchemi/enhanced_sampling/biases/walls.py |
| A | — | nvalchemi/enhanced_sampling/cv/__init__.py |
| A | — | nvalchemi/enhanced_sampling/cv/_periodic.py |
| A | — | nvalchemi/enhanced_sampling/cv/pair_distance.py |
| A | — | nvalchemi/enhanced_sampling/hooks.py |
| A | — | nvalchemi/gen/__init__.py |
| A | — | nvalchemi/gen/_device.py |
| A | — | nvalchemi/gen/generator.py |
| A | — | nvalchemi/gen/pipeline.py |
| A | — | nvalchemi/gen/stages.py |
| M | nvalchemi/hooks/__init__.py | nvalchemi/hooks/__init__.py |
| M | nvalchemi/hooks/_context.py | nvalchemi/hooks/_context.py |
| M | nvalchemi/hooks/_protocol.py | nvalchemi/hooks/_protocol.py |
| M | nvalchemi/hooks/_registry.py | nvalchemi/hooks/_registry.py |
| M | nvalchemi/hooks/bias.py | nvalchemi/hooks/bias.py |
| M | nvalchemi/hooks/neighbor_list.py | nvalchemi/hooks/neighbor_list.py |
| M | nvalchemi/hooks/periodic.py | nvalchemi/hooks/periodic.py |
| M | nvalchemi/hooks/stage_timing.py | nvalchemi/hooks/stage_timing.py |
| M | nvalchemi/models/__init__.py | nvalchemi/models/__init__.py |
| A | — | nvalchemi/models/_derivatives.py |
| M | nvalchemi/models/_utils.py | nvalchemi/models/_utils.py |
| M | nvalchemi/models/aimnet2.py | nvalchemi/models/aimnet2.py |
| M | nvalchemi/models/base.py | nvalchemi/models/base.py |
| M | nvalchemi/models/demo.py | nvalchemi/models/demo.py |
| M | nvalchemi/models/dftd3.py | nvalchemi/models/dftd3.py |
| M | nvalchemi/models/ewald.py | nvalchemi/models/ewald.py |
| A | — | nvalchemi/models/gen/__init__.py |
| A | — | nvalchemi/models/gen/base.py |
| A | — | nvalchemi/models/gen/demo.py |
| M | nvalchemi/models/lj.py | nvalchemi/models/lj.py |
| M | nvalchemi/models/mace.py | nvalchemi/models/mace.py |
| M | nvalchemi/models/pipeline.py | nvalchemi/models/pipeline.py |
| M | nvalchemi/models/pme.py | nvalchemi/models/pme.py |
| M | nvalchemi/models/uma.py | nvalchemi/models/uma.py |
| M | nvalchemi/neighbors.py | nvalchemi/neighbors.py |
| A | — | nvalchemi/specs.py |
| M | nvalchemi/training/__init__.py | nvalchemi/training/__init__.py |
| M | nvalchemi/training/_checkpoint.py | nvalchemi/training/_checkpoint.py |
| M | nvalchemi/training/_spec.py | nvalchemi/training/_spec.py |
| M | nvalchemi/training/_spec_utils.py | nvalchemi/training/_spec_utils.py |
| M | nvalchemi/training/_validation.py | nvalchemi/training/_validation.py |
| A | — | nvalchemi/training/distillation/__init__.py |
| A | — | nvalchemi/training/distillation/_attach.py |
| A | — | nvalchemi/training/distillation/config.py |
| A | — | nvalchemi/training/distillation/evaluation/__init__.py |
| A | — | nvalchemi/training/distillation/evaluation/_export.py |
| A | — | nvalchemi/training/distillation/evaluation/accuracy.py |
| A | — | nvalchemi/training/distillation/evaluation/report.py |
| A | — | nvalchemi/training/distillation/evaluation/stability.py |
| A | — | nvalchemi/training/distillation/hooks.py |
| A | — | nvalchemi/training/distillation/labeling.py |
| A | — | nvalchemi/training/distillation/losses/__init__.py |
| A | — | nvalchemi/training/distillation/losses/atomic_energy.py |
| A | — | nvalchemi/training/distillation/losses/distribution.py |
| A | — | nvalchemi/training/distillation/losses/embedding.py |
| A | — | nvalchemi/training/distillation/losses/hessian.py |
| A | — | nvalchemi/training/distillation/replay.py |
| A | — | nvalchemi/training/distillation/scoring.py |
| A | — | nvalchemi/training/distillation/seeding.py |
| A | — | nvalchemi/training/distillation/strategy.py |
| M | nvalchemi/training/distributed.py | nvalchemi/training/distributed.py |
| M | nvalchemi/training/finetune.py | nvalchemi/training/finetune.py |
| M | nvalchemi/training/hooks/__init__.py | nvalchemi/training/hooks/__init__.py |
| M | nvalchemi/training/hooks/checkpoint.py | nvalchemi/training/hooks/checkpoint.py |
| M | nvalchemi/training/hooks/ddp.py | nvalchemi/training/hooks/ddp.py |
| M | nvalchemi/training/hooks/ema.py | nvalchemi/training/hooks/ema.py |
| M | nvalchemi/training/hooks/finetune.py | nvalchemi/training/hooks/finetune.py |
| M | nvalchemi/training/losses/__init__.py | nvalchemi/training/losses/__init__.py |
| M | nvalchemi/training/losses/composition.py | nvalchemi/training/losses/composition.py |
| M | nvalchemi/training/losses/reductions.py | nvalchemi/training/losses/reductions.py |
| M | nvalchemi/training/losses/terms.py | nvalchemi/training/losses/terms.py |
| A | — | nvalchemi/training/peft/__init__.py |
| A | — | nvalchemi/training/peft/_peft.py |
| A | — | nvalchemi/training/peft/config.py |
| A | — | nvalchemi/training/peft/fingerprints.py |
| A | — | nvalchemi/training/peft/loading.py |
| A | — | nvalchemi/training/peft/lora.py |
| A | — | nvalchemi/training/peft/lora_hook.py |
| A | — | nvalchemi/training/peft/lora_wrappers.py |
| A | — | nvalchemi/training/peft/registry.py |
| A | — | nvalchemi/training/reference_energies.py |
| M | nvalchemi/training/runtime.py | nvalchemi/training/runtime.py |
| M | nvalchemi/training/strategy.py | nvalchemi/training/strategy.py |
| M | pyproject.toml | pyproject.toml |
| M | test/conftest.py | test/conftest.py |
| A | — | test/csp/fixtures/contact_oracle.py |
| A | — | test/csp/fixtures/generate_packer_stage2_cpu.py |
| A | — | test/csp/fixtures/packer_stage2_cpu.py |
| A | — | test/csp/test_batch.py |
| A | — | test/csp/test_chem.py |
| A | — | test/csp/test_comparison.py |
| A | — | test/csp/test_comparison_batched_descriptor.py |
| A | — | test/csp/test_comparison_hydrogen_selection.py |
| A | — | test/csp/test_comparison_stream.py |
| A | — | test/csp/test_data.py |
| A | — | test/csp/test_generator.py |
| A | — | test/csp/test_generator_native_distributed.py |
| A | — | test/csp/test_hooks.py |
| A | — | test/csp/test_molecular_topology.py |
| A | — | test/csp/test_packer.py |
| A | — | test/csp/test_packer_config.py |
| A | — | test/csp/test_packer_distributed.py |
| A | — | test/csp/test_pipeline.py |
| A | — | test/csp/test_storage.py |
| A | — | test/csp/test_symmetry.py |
| A | — | test/csp/test_volume.py |
| M | test/data/test_atomic_data.py | test/data/test_atomic_data.py |
| M | test/data/test_batch.py | test/data/test_batch.py |
| A | — | test/data/test_dataset_device.py |
| A | — | test/data/test_group_layout.py |
| M | test/data/test_level_storage.py | test/data/test_level_storage.py |
| M | test/data/test_multidataset_samplers.py | test/data/test_multidataset_samplers.py |
| A | — | test/data/test_transforms_supercell.py |
| M | test/data/test_zarr_datapipe.py | test/data/test_zarr_datapipe.py |
| M | test/distributed/_core/test_compile_halo_scatter.py | test/distributed/_core/test_compile_halo_scatter.py |
| M | test/distributed/_gloo_harness.py | test/distributed/_gloo_harness.py |
| M | test/distributed/model/test_aimnet2_compile_recompile_gate.py | test/distributed/model/test_aimnet2_compile_recompile_gate.py |
| M | test/distributed/model/test_compile_recompile_gate.py | test/distributed/model/test_compile_recompile_gate.py |
| M | test/distributed/model/test_dftd3_multigpu.py | test/distributed/model/test_dftd3_multigpu.py |
| M | test/distributed/model/test_distributed_models.py | test/distributed/model/test_distributed_models.py |
| M | test/distributed/model/test_distributed_pipeline_multigpu.py | test/distributed/model/test_distributed_pipeline_multigpu.py |
| M | test/distributed/model/test_electrostatics_stress_multigpu.py | test/distributed/model/test_electrostatics_stress_multigpu.py |
| M | test/distributed/model/test_ewald_multigpu.py | test/distributed/model/test_ewald_multigpu.py |
| M | test/distributed/model/test_mace_cueq_compile_gate.py | test/distributed/model/test_mace_cueq_compile_gate.py |
| M | test/distributed/model/test_mace_cueq_multigpu.py | test/distributed/model/test_mace_cueq_multigpu.py |
| M | test/distributed/model/test_mace_nocueq_compile_gate.py | test/distributed/model/test_mace_nocueq_compile_gate.py |
| M | test/distributed/model/test_multigpu.py | test/distributed/model/test_multigpu.py |
| M | test/distributed/model/test_pme_multigpu.py | test/distributed/model/test_pme_multigpu.py |
| M | test/distributed/model/test_shared_partition_multigpu.py | test/distributed/model/test_shared_partition_multigpu.py |
| M | test/distributed/model/test_slab_multigpu.py | test/distributed/model/test_slab_multigpu.py |
| M | test/distributed/model/test_uma_gp_partition_multigpu.py | test/distributed/model/test_uma_gp_partition_multigpu.py |
| M | test/distributed/model/test_uma_multigpu.py | test/distributed/model/test_uma_multigpu.py |
| M | test/distributed/test_deferred_migration_halo.py | test/distributed/test_deferred_migration_halo.py |
| M | test/distributed/test_domain_parallel.py | test/distributed/test_domain_parallel.py |
| A | — | test/distributed/test_gloo_harness.py |
| M | test/distributed/test_pipeline_dd_mesh.py | test/distributed/test_pipeline_dd_mesh.py |
| M | test/distributed/test_public_api_stability.py | test/distributed/test_public_api_stability.py |
| A | — | test/distributed/test_runtime_context.py |
| M | test/dynamics/conftest.py | test/dynamics/conftest.py |
| M | test/dynamics/test_base.py | test/dynamics/test_base.py |
| M | test/dynamics/test_bias_hook.py | test/dynamics/test_bias_hook.py |
| M | test/dynamics/test_cell_align.py | test/dynamics/test_cell_align.py |
| M | test/dynamics/test_demo_dynamics.py | test/dynamics/test_demo_dynamics.py |
| M | test/dynamics/test_distributed_correctness.py | test/dynamics/test_distributed_correctness.py |
| A | — | test/dynamics/test_fire2.py |
| M | test/dynamics/test_freeze_hook.py | test/dynamics/test_freeze_hook.py |
| M | test/dynamics/test_hook_utils.py | test/dynamics/test_hook_utils.py |
| A | — | test/dynamics/test_idpp.py |
| M | test/dynamics/test_inflight.py | test/dynamics/test_inflight.py |
| A | — | test/dynamics/test_lbfgs.py |
| A | — | test/dynamics/test_mep_hooks.py |
| A | — | test/dynamics/test_neb.py |
| A | — | test/dynamics/test_neb_hooks.py |
| A | — | test/dynamics/test_neb_ops.py |
| M | test/dynamics/test_npt_nph.py | test/dynamics/test_npt_nph.py |
| M | test/dynamics/test_observer_hooks.py | test/dynamics/test_observer_hooks.py |
| M | test/dynamics/test_ops.py | test/dynamics/test_ops.py |
| A | — | test/dynamics/test_path_alignment.py |
| A | — | test/dynamics/test_path_geometry.py |
| A | — | test/dynamics/test_path_ops.py |
| A | — | test/dynamics/test_paths.py |
| A | — | test/dynamics/test_per_system_params.py |
| M | test/dynamics/test_periodic_hook.py | test/dynamics/test_periodic_hook.py |
| M | test/dynamics/test_pipeline_stage.py | test/dynamics/test_pipeline_stage.py |
| M | test/dynamics/test_safety_hooks.py | test/dynamics/test_safety_hooks.py |
| M | test/dynamics/test_single_loop.py | test/dynamics/test_single_loop.py |
| M | test/dynamics/test_sinks.py | test/dynamics/test_sinks.py |
| M | test/dynamics/test_state_management.py | test/dynamics/test_state_management.py |
| A | — | test/dynamics/test_strategy.py |
| A | — | test/dynamics/test_structure_sampler.py |
| A | — | test/dynamics/test_swap_hook.py |
| A | — | test/enhanced_sampling/__init__.py |
| A | — | test/enhanced_sampling/test_abf.py |
| A | — | test/enhanced_sampling/test_bias_core.py |
| A | — | test/enhanced_sampling/test_builtin_biases.py |
| A | — | test/enhanced_sampling/test_checkpoint.py |
| A | — | test/enhanced_sampling/test_exchange.py |
| A | — | test/enhanced_sampling/test_hooks.py |
| A | — | test/enhanced_sampling/test_metadynamics.py |
| A | — | test/enhanced_sampling/test_runner.py |
| A | — | test/gen/__init__.py |
| A | — | test/gen/conftest.py |
| A | — | test/gen/test_gen_generator.py |
| A | — | test/gen/test_gen_pipeline.py |
| A | — | test/gen/test_gen_serialization.py |
| M | test/hooks/test_context.py | test/hooks/test_context.py |
| M | test/hooks/test_neighbor_list_hook.py | test/hooks/test_neighbor_list_hook.py |
| M | test/hooks/test_protocol.py | test/hooks/test_protocol.py |
| M | test/hooks/test_registry.py | test/hooks/test_registry.py |
| M | test/hooks/test_shared_hooks.py | test/hooks/test_shared_hooks.py |
| M | test/hooks/test_stage_timing_hook.py | test/hooks/test_stage_timing_hook.py |
| M | test/models/test_aimnet2.py | test/models/test_aimnet2.py |
| M | test/models/test_base.py | test/models/test_base.py |
| M | test/models/test_dftd3.py | test/models/test_dftd3.py |
| M | test/models/test_ewald.py | test/models/test_ewald.py |
| A | — | test/models/test_gen_demo.py |
| A | — | test/models/test_generative_base.py |
| M | test/models/test_mace.py | test/models/test_mace.py |
| A | — | test/models/test_model_utils.py |
| M | test/models/test_pipeline.py | test/models/test_pipeline.py |
| M | test/models/test_pme.py | test/models/test_pme.py |
| M | test/models/test_uma.py | test/models/test_uma.py |
| A | — | test/test_checkpoint.py |
| A | — | test/test_multigpu_marker.py |
| A | — | test/test_serialization.py |
| M | test/training/conftest.py | test/training/conftest.py |
| A | — | test/training/distillation/__init__.py |
| A | — | test/training/distillation/conftest.py |
| A | — | test/training/distillation/test_advanced_objectives.py |
| A | — | test/training/distillation/test_atomic_energy_loss.py |
| A | — | test/training/distillation/test_attach.py |
| A | — | test/training/distillation/test_config.py |
| A | — | test/training/distillation/test_distribution_loss.py |
| A | — | test/training/distillation/test_embedding_loss.py |
| A | — | test/training/distillation/test_evaluation_accuracy.py |
| A | — | test/training/distillation/test_evaluation_report.py |
| A | — | test/training/distillation/test_evaluation_stability.py |
| A | — | test/training/distillation/test_hessian_loss.py |
| A | — | test/training/distillation/test_hooks.py |
| A | — | test/training/distillation/test_labeling.py |
| A | — | test/training/distillation/test_multi_gpu.py |
| A | — | test/training/distillation/test_on_policy.py |
| A | — | test/training/distillation/test_relaxation.py |
| A | — | test/training/distillation/test_replay.py |
| A | — | test/training/distillation/test_scoring.py |
| A | — | test/training/distillation/test_seeding.py |
| A | — | test/training/distillation/test_strategy.py |
| M | test/training/test_checkpoint.py | test/training/test_checkpoint.py |
| M | test/training/test_ddp_hook.py | test/training/test_ddp_hook.py |
| A | — | test/training/test_distributed.py |
| M | test/training/test_ema_hook.py | test/training/test_ema_hook.py |
| M | test/training/test_finetune.py | test/training/test_finetune.py |
| M | test/training/test_losses.py | test/training/test_losses.py |
| A | — | test/training/test_peft_lora.py |
| A | — | test/training/test_peft_registry.py |
| A | — | test/training/test_reference_energies.py |
| M | test/training/test_runtime.py | test/training/test_runtime.py |
| M | test/training/test_spec.py | test/training/test_spec.py |
| A | — | test/training/test_spec_utils.py |
| M | test/training/test_strategy.py | test/training/test_strategy.py |
| M | test/training/test_strategy_validate.py | test/training/test_strategy_validate.py |
| M | test/training/test_validation_config.py | test/training/test_validation_config.py |
| M | test/training/test_validation_loop.py | test/training/test_validation_loop.py |
| M | uv.lock | uv.lock |

</details>

### 潜在冲突与本地补丁

| 交集路径 | 登记 LP |
|---|---|
| nvalchemi/data/batch.py | LP-004 |
| nvalchemi/data/level_storage.py | LP-070 |
| nvalchemi/distributed/_dynamics_coordinator.py | 无路径匹配，人工复核 |
| nvalchemi/distributed/domain_parallel.py | 无路径匹配，人工复核 |
| nvalchemi/dynamics/__init__.py | LP-003 |
| nvalchemi/dynamics/_ops/_bridge.py | 无路径匹配，人工复核 |
| nvalchemi/dynamics/_ops/cell_align.py | LP-075 |
| nvalchemi/dynamics/_ops/fire.py | 无路径匹配，人工复核 |
| nvalchemi/dynamics/_ops/npt_nph.py | LP-073 |
| nvalchemi/dynamics/base.py | LP-001 |
| nvalchemi/dynamics/hooks/__init__.py | 无路径匹配，人工复核 |
| nvalchemi/dynamics/hooks/_utils.py | LP-011 |
| nvalchemi/dynamics/hooks/cell_align.py | 无路径匹配，人工复核 |
| nvalchemi/dynamics/hooks/logging.py | 无路径匹配，人工复核 |
| nvalchemi/dynamics/hooks/monitors.py | 无路径匹配，人工复核 |
| nvalchemi/dynamics/integrators/nvt_langevin.py | LP-071 |
| nvalchemi/dynamics/integrators/nvt_nose_hoover.py | LP-065, LP-066, LP-067 |
| nvalchemi/dynamics/optimizers/__init__.py | 无路径匹配，人工复核 |
| nvalchemi/dynamics/optimizers/fire.py | 无路径匹配，人工复核 |
| nvalchemi/dynamics/optimizers/fire2.py | 无路径匹配，人工复核 |
| nvalchemi/dynamics/sinks.py | LP-008 |
| nvalchemi/hooks/__init__.py | LP-006 |
| nvalchemi/hooks/neighbor_list.py | LP-060 |
| nvalchemi/hooks/periodic.py | LP-012 |
| nvalchemi/hooks/stage_timing.py | 无路径匹配，人工复核 |
| nvalchemi/models/base.py | 无路径匹配，人工复核 |
| nvalchemi/models/lj.py | 无路径匹配，人工复核 |
| nvalchemi/models/mace.py | LP-069 |
| nvalchemi/neighbors.py | LP-005 |
| nvalchemi/training/__init__.py | 无路径匹配，人工复核 |
| nvalchemi/training/hooks/__init__.py | 无路径匹配，人工复核 |
| pyproject.toml | 无路径匹配，人工复核 |
| test/dynamics/conftest.py | 无路径匹配，人工复核 |
| test/dynamics/test_hook_utils.py | LP-012 |
| test/dynamics/test_inflight.py | LP-009 |
| test/dynamics/test_observer_hooks.py | LP-011 |
| test/dynamics/test_ops.py | LP-002 |
| test/dynamics/test_periodic_hook.py | LP-012 |
| test/dynamics/test_state_management.py | LP-010 |
| test/models/test_mace.py | 无路径匹配，人工复核 |

LP 按显式 packages/ 或 nvalchemi/nvalchemiops 路径匹配；简写、通配和语义关联需人工复核。

<details><summary>完整本地变更文件（相对导入基线）</summary>

| 状态 | 之前 | 之后 |
|---|---|---|
| A | — | nvalchemi/_backend.py |
| A | — | nvalchemi/_dynamics_reference/__init__.py |
| A | — | nvalchemi/_dynamics_reference/fire.py |
| A | — | nvalchemi/_dynamics_reference/kinetics.py |
| A | — | nvalchemi/_dynamics_reference/langevin.py |
| A | — | nvalchemi/_dynamics_reference/nose_hoover.py |
| A | — | nvalchemi/_dynamics_reference/periodic.py |
| A | — | nvalchemi/_dynamics_reference/segmented_reduce.py |
| A | — | nvalchemi/_dynamics_reference/velocity_verlet.py |
| A | — | nvalchemi/_dynamics_stage.py |
| M | nvalchemi/data/batch.py | nvalchemi/data/batch.py |
| M | nvalchemi/data/level_storage.py | nvalchemi/data/level_storage.py |
| A | — | nvalchemi/data/storage_backend.py |
| A | — | nvalchemi/data/warp_storage_backend.py |
| M | nvalchemi/distributed/_dynamics_coordinator.py | nvalchemi/distributed/_dynamics_coordinator.py |
| M | nvalchemi/distributed/domain_parallel.py | nvalchemi/distributed/domain_parallel.py |
| M | nvalchemi/dynamics/__init__.py | nvalchemi/dynamics/__init__.py |
| M | nvalchemi/dynamics/_ops/_bridge.py | nvalchemi/dynamics/_ops/_bridge.py |
| M | nvalchemi/dynamics/_ops/cell_align.py | nvalchemi/dynamics/_ops/cell_align.py |
| M | nvalchemi/dynamics/_ops/fire.py | nvalchemi/dynamics/_ops/fire.py |
| A | — | nvalchemi/dynamics/_ops/fire_cell.py |
| M | nvalchemi/dynamics/_ops/langevin.py | nvalchemi/dynamics/_ops/langevin.py |
| M | nvalchemi/dynamics/_ops/nose_hoover.py | nvalchemi/dynamics/_ops/nose_hoover.py |
| M | nvalchemi/dynamics/_ops/npt_nph.py | nvalchemi/dynamics/_ops/npt_nph.py |
| A | — | nvalchemi/dynamics/_ops/stress.py |
| M | nvalchemi/dynamics/_ops/velocity_verlet.py | nvalchemi/dynamics/_ops/velocity_verlet.py |
| M | nvalchemi/dynamics/base.py | nvalchemi/dynamics/base.py |
| M | nvalchemi/dynamics/hooks/__init__.py | nvalchemi/dynamics/hooks/__init__.py |
| M | nvalchemi/dynamics/hooks/_utils.py | nvalchemi/dynamics/hooks/_utils.py |
| M | nvalchemi/dynamics/hooks/cell_align.py | nvalchemi/dynamics/hooks/cell_align.py |
| M | nvalchemi/dynamics/hooks/logging.py | nvalchemi/dynamics/hooks/logging.py |
| M | nvalchemi/dynamics/hooks/monitors.py | nvalchemi/dynamics/hooks/monitors.py |
| M | nvalchemi/dynamics/integrators/__init__.py | nvalchemi/dynamics/integrators/__init__.py |
| M | nvalchemi/dynamics/integrators/nve.py | nvalchemi/dynamics/integrators/nve.py |
| M | nvalchemi/dynamics/integrators/nvt_langevin.py | nvalchemi/dynamics/integrators/nvt_langevin.py |
| M | nvalchemi/dynamics/integrators/nvt_nose_hoover.py | nvalchemi/dynamics/integrators/nvt_nose_hoover.py |
| M | nvalchemi/dynamics/optimizers/__init__.py | nvalchemi/dynamics/optimizers/__init__.py |
| M | nvalchemi/dynamics/optimizers/fire.py | nvalchemi/dynamics/optimizers/fire.py |
| M | nvalchemi/dynamics/optimizers/fire2.py | nvalchemi/dynamics/optimizers/fire2.py |
| M | nvalchemi/dynamics/sinks.py | nvalchemi/dynamics/sinks.py |
| M | nvalchemi/hooks/__init__.py | nvalchemi/hooks/__init__.py |
| M | nvalchemi/hooks/neighbor_list.py | nvalchemi/hooks/neighbor_list.py |
| M | nvalchemi/hooks/periodic.py | nvalchemi/hooks/periodic.py |
| M | nvalchemi/hooks/reporting/_scalars.py | nvalchemi/hooks/reporting/_scalars.py |
| M | nvalchemi/hooks/stage_timing.py | nvalchemi/hooks/stage_timing.py |
| M | nvalchemi/models/base.py | nvalchemi/models/base.py |
| M | nvalchemi/models/lj.py | nvalchemi/models/lj.py |
| M | nvalchemi/models/mace.py | nvalchemi/models/mace.py |
| M | nvalchemi/neighbors.py | nvalchemi/neighbors.py |
| M | nvalchemi/training/__init__.py | nvalchemi/training/__init__.py |
| M | nvalchemi/training/hooks/__init__.py | nvalchemi/training/hooks/__init__.py |
| M | pyproject.toml | pyproject.toml |
| A | — | test/compatibility/test_backend_default_boundary.py |
| A | — | test/compatibility/test_backend_selection_propagation.py |
| A | — | test/compatibility/test_cell_stress_force_reference.py |
| A | — | test/compatibility/test_cell_tools_reference.py |
| A | — | test/compatibility/test_dynamics_import_boundary.py |
| A | — | test/compatibility/test_dynamics_reference_fire.py |
| A | — | test/compatibility/test_dynamics_reference_langevin.py |
| A | — | test/compatibility/test_dynamics_reference_langevin_restart.py |
| A | — | test/compatibility/test_dynamics_reference_langevin_statistics.py |
| A | — | test/compatibility/test_dynamics_reference_nhc.py |
| A | — | test/compatibility/test_dynamics_reference_nhc_physics.py |
| A | — | test/compatibility/test_dynamics_reference_velocity_verlet.py |
| A | — | test/compatibility/test_executor_binding.py |
| A | — | test/compatibility/test_fire2_skin_functional.py |
| A | — | test/compatibility/test_fire2_upstream_reference.py |
| A | — | test/compatibility/test_heterogeneous_trajectory_reference.py |
| A | — | test/compatibility/test_hip_coo_neighbors.py |
| A | — | test/compatibility/test_hip_current_cell_neighbors.py |
| A | — | test/compatibility/test_hip_variable_cell_workflow.py |
| A | — | test/compatibility/test_inactive_dynamics_outputs.py |
| A | — | test/compatibility/test_public_dynamics_reference.py |
| A | — | test/compatibility/test_variable_cell_fire_reference.py |
| A | — | test/compatibility/test_variable_cell_workflow.py |
| A | — | test/data/test_storage_backend.py |
| M | test/dynamics/conftest.py | test/dynamics/conftest.py |
| M | test/dynamics/test_hook_utils.py | test/dynamics/test_hook_utils.py |
| M | test/dynamics/test_inflight.py | test/dynamics/test_inflight.py |
| M | test/dynamics/test_observer_hooks.py | test/dynamics/test_observer_hooks.py |
| M | test/dynamics/test_ops.py | test/dynamics/test_ops.py |
| M | test/dynamics/test_periodic_hook.py | test/dynamics/test_periodic_hook.py |
| M | test/dynamics/test_state_management.py | test/dynamics/test_state_management.py |
| A | — | test/hooks/test_neighbor_list_torch_reference.py |
| A | — | test/hooks/test_optional_imports.py |
| A | — | test/models/test_lj_torch_reference.py |
| M | test/models/test_mace.py | test/models/test_mace.py |
| A | — | test/models/test_neighbors_torch_reference.py |

</details>

### 静态 API 变化

1236 个 AST 接口/导出条目变化；97 个签名相同的实现变化，实际返回值/行为 unknown，需复核。

<details><summary>完整静态接口变化（含参数、默认值和返回注解）</summary>

| 状态 | 路径:符号 | 基线 | 候选 |
|---|---|---|---|
| added | nvalchemi/_checkpoint.py:CheckpointContents | — | class() @ dataclass(frozen=True) |
| added | nvalchemi/_checkpoint.py:CheckpointContents.batch | — | Batch \| None = None |
| added | nvalchemi/_checkpoint.py:CheckpointContents.manifest | — | CheckpointManifest = required |
| added | nvalchemi/_checkpoint.py:CheckpointContents.states | — | dict[str, dict[str, Any]] = field(default_factory=dict) |
| added | nvalchemi/_checkpoint.py:CheckpointManifest | — | class(BaseModel) |
| added | nvalchemi/_checkpoint.py:CheckpointManifest.batch_checksum | — | str = '' |
| added | nvalchemi/_checkpoint.py:CheckpointManifest.batch_field_dtypes | — | dict[str, str] = Field(default_factory=dict) |
| added | nvalchemi/_checkpoint.py:CheckpointManifest.batch_fields | — | list[str] = Field(default_factory=list) |
| added | nvalchemi/_checkpoint.py:CheckpointManifest.checksums | — | dict[str, str] = Field(default_factory=dict) |
| added | nvalchemi/_checkpoint.py:CheckpointManifest.compatibility | — | dict[str, Any] = Field(default_factory=dict) |
| added | nvalchemi/_checkpoint.py:CheckpointManifest.components | — | list[str] = Field(default_factory=list) |
| added | nvalchemi/_checkpoint.py:CheckpointManifest.format_version | — | int = CHECKPOINT_FORMAT_VERSION |
| added | nvalchemi/_checkpoint.py:CheckpointManifest.metadata | — | dict[str, Any] = Field(default_factory=dict) |
| added | nvalchemi/_checkpoint.py:CheckpointManifest.num_graphs | — | int \| None = None |
| added | nvalchemi/_checkpoint.py:Stateful | — | class(Protocol) @ runtime_checkable |
| added | nvalchemi/_checkpoint.py:Stateful.load_state_dict | — | (self, state: Mapping[str, Any]) -> None |
| added | nvalchemi/_checkpoint.py:Stateful.state_dict | — | (self) -> Mapping[str, Any] |
| added | nvalchemi/_checkpoint.py:load_checkpoint | — | (path: str \| Path, components: Mapping[str, Stateful] \| None=None, *, device: torch.device \| str='cpu', validate: Callable[[CheckpointManifest], None] \| None=None) -> CheckpointContents |
| added | nvalchemi/_checkpoint.py:save_checkpoint | — | (path: str \| Path, components: Mapping[str, Stateful], *, batch: Batch \| None=None, batch_fields: Sequence[str]=(), compatibility: Mapping[str, Any] \| None=None, metadata: Mapping[str, Any] \| None=None) -> CheckpointManifest |
| added | nvalchemi/_serialization.py:MeasurementRecord | — | class(BaseModel) |
| added | nvalchemi/_serialization.py:MeasurementRecord.from_dict | — | (cls, data: Mapping[str, Any]) -> Self @ classmethod |
| added | nvalchemi/_serialization.py:MeasurementRecord.to_dict | — | (self) -> dict[str, Any] |
| added | nvalchemi/_serialization.py:capture_callable_spec | — | (fn: Callable[..., Any], *, field_name: str) -> 'BaseSpec' |
| added | nvalchemi/csp/__init__.py:exports | — | __all__ = ['CrystalSystem', 'RigidMoleculeASUZarrReader', 'RigidMoleculeASUZarrWriter', 'CSPGenerator', 'CSP_OUTPUT_FIELDS', 'MolecularPackingInput', 'RigidMoleculeASUBatch', 'SpaceGroupPolicy', 'csd_space_group_probabilities', 'estimate_formula_unit_volume', 'get_crystal_system', 'get_default_atomic_volumes', 'get_space_group_candidates', 'get_space_group_operation_count', 'get_space_group_operations', 'is_sohncke_space_group', 'sample_space_groups'] ; from nvalchemi.csp.data import MolecularPackingInput, RigidMoleculeASUBatch ; from nvalchemi.csp.generator import CSP_OUTPUT_FIELDS, CSPGenerator ; from nvalchemi.csp.storage import RigidMoleculeASUZarrReader, RigidMoleculeASUZarrWriter ; from nvalchemi.csp.symmetry import CrystalSystem, SpaceGroupPolicy, csd_space_group_probabilities, get_crystal_system, get_space_group_candidates, get_space_group_operation_count, get_space_group_operations, is_sohncke_space_group, sample_space_groups ; from nvalchemi.csp.volume import estimate_formula_unit_volume, get_default_atomic_volumes |
| added | nvalchemi/csp/_batch.py:expand_asu_batch | — | (compact: RigidMoleculeASUBatch, *, indices: Tensor \| None=None, device: torch.device \| str \| None=None) -> Batch |
| added | nvalchemi/csp/_chem/__init__.py:exports | — | — |
| added | nvalchemi/csp/_chem/_conformers.py:generate_conformers_from_smiles | — | (smiles: str, config: RDKitConformerConfig) -> Mol |
| added | nvalchemi/csp/_chem/_contacts.py:contact_distances | — | (molecules: Sequence[Mol]) -> torch.Tensor |
| added | nvalchemi/csp/_chem/_input.py:build_molecular_packing_input | — | (molecules: Sequence[Mol], *, contact_distances: Tensor \| None=None, formula_unit_volume: float \| None=None, component_index: Tensor \| None=None, metadata: Mapping[str, Any] \| None=None) -> MolecularPackingInput |
| added | nvalchemi/csp/_chem/_topology.py:connected_components | — | (adjacency: list[list[bool]]) -> list[list[int]] |
| added | nvalchemi/csp/_chem/_topology.py:find_isomorphism | — | (source_numbers: list[int], source_adjacency: list[list[bool]], target_numbers: list[int], target_adjacency: list[list[bool]]) -> list[int] \| None |
| added | nvalchemi/csp/_chem/_topology.py:topological_atom_types | — | (atomic_numbers: list[int], adjacency: list[list[bool]]) -> list[int] |
| added | nvalchemi/csp/_chem/_validation.py:connected_molecules | — | (molecules: Sequence[Mol]) -> tuple[Mol, ...] |
| added | nvalchemi/csp/_comparison/__init__.py:exports | — | — |
| added | nvalchemi/csp/_comparison/descriptor.py:DescriptorTile | — | class() @ dataclass(frozen=True) |
| added | nvalchemi/csp/_comparison/descriptor.py:DescriptorTile.atom_offsets | — | tuple[int, ...] = required |
| added | nvalchemi/csp/_comparison/descriptor.py:DescriptorTile.capacity | — | int = required |
| added | nvalchemi/csp/_comparison/descriptor.py:DescriptorTile.center_types | — | Tensor \| None = required |
| added | nvalchemi/csp/_comparison/descriptor.py:DescriptorTile.distances | — | Tensor = required |
| added | nvalchemi/csp/_comparison/descriptor.py:DescriptorTile.estimated_peak_bytes | — | int = required |
| added | nvalchemi/csp/_comparison/descriptor.py:DescriptorTile.geometry_bytes | — | int = required |
| added | nvalchemi/csp/_comparison/descriptor.py:DescriptorTile.layout | — | (self, *, typed_neighbors: bool) -> tuple[Tensor, Tensor \| None] |
| added | nvalchemi/csp/_comparison/descriptor.py:DescriptorTile.neighbor_types | — | Tensor \| None = required |
| added | nvalchemi/csp/_comparison/descriptor.py:DescriptorTile.valid_width | — | int = required |
| added | nvalchemi/csp/_comparison/descriptor.py:DescriptorTile.widths | — | tuple[int, ...] = required |
| added | nvalchemi/csp/_comparison/descriptor.py:build_descriptor_tiles | — | (positions: Tensor, pointers: Iterable[int], cells: Tensor \| None, pbc: Tensor, atom_types: Tensor \| None, *, cutoff: float, device: torch.device, memory_budget_bytes: int \| None) -> Iterator[DescriptorTile] |
| added | nvalchemi/csp/_comparison/score.py:score_descriptor_pairs | — | (left: DescriptorBlock, right: DescriptorBlock, pairs: Tensor, cutoff: float, has_center_types: bool, typed_neighbors: bool) -> tuple[Tensor, Tensor] |
| added | nvalchemi/csp/_comparison/score.py:threshold_score | — | (threshold: float, device: torch.device) -> Tensor |
| added | nvalchemi/csp/_comparison/score.py:threshold_score_descriptor_pairs | — | (left: DescriptorBlock, right: DescriptorBlock, pairs: Tensor, cutoff: float, log_bound: float, accept_log_bound: float, has_center_types: bool, typed_neighbors: bool) -> tuple[Tensor, Tensor] |
| added | nvalchemi/csp/_comparison/store.py:DescriptorBlock | — | class() @ dataclass(frozen=True) |
| added | nvalchemi/csp/_comparison/store.py:DescriptorBlock.atom_counts | — | Tensor = required |
| added | nvalchemi/csp/_comparison/store.py:DescriptorBlock.atom_offsets | — | Tensor = required |
| added | nvalchemi/csp/_comparison/store.py:DescriptorBlock.center_types | — | Tensor = required |
| added | nvalchemi/csp/_comparison/store.py:DescriptorBlock.distances | — | Tensor = required |
| added | nvalchemi/csp/_comparison/store.py:DescriptorBlock.neighbor_types | — | Tensor = required |
| added | nvalchemi/csp/_comparison/store.py:DescriptorBlock.row_offsets | — | Tensor = required |
| added | nvalchemi/csp/_comparison/store.py:DescriptorBlock.source_ids | — | tuple[int, ...] \| None = required |
| added | nvalchemi/csp/_comparison/store.py:DescriptorBlock.widths | — | Tensor = required |
| added | nvalchemi/csp/_comparison/store.py:DescriptorStore | — | class() @ dataclass(frozen=True) |
| added | nvalchemi/csp/_comparison/store.py:DescriptorStore.atom_counts | — | Tensor = required |
| added | nvalchemi/csp/_comparison/store.py:DescriptorStore.atom_offsets | — | Tensor = required |
| added | nvalchemi/csp/_comparison/store.py:DescriptorStore.block | — | (self, structure_ids: set[int], device: torch.device) -> DescriptorBlock |
| added | nvalchemi/csp/_comparison/store.py:DescriptorStore.bytes_for | — | (self, structure_ids: set[int]) -> int |
| added | nvalchemi/csp/_comparison/store.py:DescriptorStore.center_type_presence | — | Tensor = required |
| added | nvalchemi/csp/_comparison/store.py:DescriptorStore.center_types | — | Tensor = required |
| added | nvalchemi/csp/_comparison/store.py:DescriptorStore.device | — | torch.device = required |
| added | nvalchemi/csp/_comparison/store.py:DescriptorStore.distances | — | Tensor = required |
| added | nvalchemi/csp/_comparison/store.py:DescriptorStore.neighbor_types | — | Tensor = required |
| added | nvalchemi/csp/_comparison/store.py:DescriptorStore.num_structures | — | (self) -> int @ property |
| added | nvalchemi/csp/_comparison/store.py:DescriptorStore.row_offsets | — | Tensor = required |
| added | nvalchemi/csp/_comparison/store.py:DescriptorStore.storage_bytes | — | (self) -> int @ property |
| added | nvalchemi/csp/_comparison/store.py:DescriptorStore.summaries | — | Tensor = required |
| added | nvalchemi/csp/_comparison/store.py:DescriptorStore.typed_rank_indices | — | tuple[int, ...] = required |
| added | nvalchemi/csp/_comparison/store.py:DescriptorStore.typed_summaries | — | Tensor = required |
| added | nvalchemi/csp/_comparison/store.py:DescriptorStore.typed_type_vocab | — | tuple[int, ...] = required |
| added | nvalchemi/csp/_comparison/store.py:DescriptorStore.widths | — | Tensor = required |
| added | nvalchemi/csp/_comparison/store.py:available_cuda_bytes | — | (device: torch.device \| str) -> int |
| added | nvalchemi/csp/_comparison/store.py:build_descriptor_store | — | (batch: Batch, *, cutoff: float, atom_types: Tensor \| None, typed_neighbors: bool, has_atom_types: bool, device: torch.device, cuda_memory_budget_bytes: int \| None, require_device_residency: bool=False) -> DescriptorStore |
| added | nvalchemi/csp/_comparison/store.py:build_descriptor_stores | — | (batch: Batch, *, cutoff: float, atom_types: Tensor \| None, modes: Iterable[str], device: torch.device, cuda_memory_budget_bytes: int \| None, require_device_residency: bool=False) -> dict[str, DescriptorStore] |
| added | nvalchemi/csp/_comparison/store.py:comparison_geometry | — | (batch: Batch, atom_types: Tensor \| None, *, include_hydrogens: bool, atom_types_validated: bool=False) -> tuple[Batch, Tensor \| None] |
| added | nvalchemi/csp/_comparison/store.py:validate_include_hydrogens | — | (include_hydrogens: bool) -> None |
| added | nvalchemi/csp/_comparison/store.py:workspace_reserve | — | (memory_budget_bytes: int) -> int |
| added | nvalchemi/csp/_comparison/stream.py:deduplicate_stream | — | (count: int, read_typed_batch: Callable[[Tensor], tuple[Batch, Tensor \| None]], *, cutoff: float, threshold: float, type_vocabulary: Sequence[int] \| None=None, priority_order: Sequence[int] \| Tensor \| None=None, device: torch.device \| str='cpu', input_batch_size: int \| None=None, pair_block_size: int \| None=None, max_batch_atoms: int=_MAX_BATCH_ATOMS, max_memory_fraction: float=0.85, summary_coordinate_count: int=_DEFAULT_SUMMARY_COORDINATE_COUNT, include_hydrogens: bool=True, confirm: Callable[[Tensor], Tensor] \| None=None) -> DeduplicationResult |
| added | nvalchemi/csp/_comparison/stream.py:iter_matches_stream | — | (count: int, read_typed_batch: Callable[[Tensor], tuple[Batch, Tensor \| None]], *, cutoff: float, threshold: float, type_vocabulary: Sequence[int] \| None=None, other_count: int \| None=None, read_other_typed_batch: Callable[[Tensor], tuple[Batch, Tensor \| None]] \| None=None, device: torch.device \| str='cpu', max_batch_atoms: int=_MAX_BATCH_ATOMS, max_memory_fraction: float=0.85, pair_chunk_size: int=8192, summary_coordinate_count: int=_DEFAULT_SUMMARY_COORDINATE_COUNT, include_hydrogens: bool=True) -> Iterator[Tensor] |
| added | nvalchemi/csp/_validation.py:finite_nonnegative | — | (value: Any, name: str) -> float |
| added | nvalchemi/csp/_validation.py:finite_positive | — | (value: Any, name: str) -> float |
| added | nvalchemi/csp/_validation.py:nonnegative_int | — | (value: Any, name: str) -> int |
| added | nvalchemi/csp/_validation.py:positive_integer | — | (value: Any, *, name: str) -> int |
| added | nvalchemi/csp/chem.py:RDKitConformerConfig | — | class(BaseModel) |
| added | nvalchemi/csp/chem.py:RDKitConformerConfig.embed_force_tol | — | float = 0.05 |
| added | nvalchemi/csp/chem.py:RDKitConformerConfig.max_embed_attempts | — | int = 1000 |
| added | nvalchemi/csp/chem.py:RDKitConformerConfig.num_conformers | — | int = 100 |
| added | nvalchemi/csp/chem.py:RDKitConformerConfig.num_threads | — | int = -1 |
| added | nvalchemi/csp/chem.py:RDKitConformerConfig.prune_rms_thresh | — | float = -1.0 |
| added | nvalchemi/csp/chem.py:RDKitConformerConfig.random_seed | — | int \| None = None |
| added | nvalchemi/csp/chem.py:TopologicalAtomTypeMap | — | class() |
| added | nvalchemi/csp/chem.py:TopologicalAtomTypeMap.__init__ | — | (self, atomic_numbers: Tensor, adjacency: Tensor) -> None |
| added | nvalchemi/csp/chem.py:TopologicalAtomTypeMap.for_batch | — | (self, batch: Batch) -> Tensor |
| added | nvalchemi/csp/chem.py:TopologicalAtomTypeMap.for_connectivity | — | (self, atomic_numbers: Tensor, adjacency: Tensor) -> Tensor |
| added | nvalchemi/csp/chem.py:build_contact_distance_matrix | — | (molecules: Sequence[Mol], *, device: torch.device \| str='cpu') -> Tensor @ OptionalDependency.RDKIT.require |
| added | nvalchemi/csp/chem.py:build_molecular_packing_input | — | (molecules: Sequence[Mol], *, contact_distances: Tensor \| None=None, formula_unit_volume: float \| None=None, component_index: Tensor \| None=None, metadata: Mapping[str, JSONValue] \| None=None) -> MolecularPackingInput @ OptionalDependency.RDKIT.require |
| added | nvalchemi/csp/chem.py:generate_conformers_from_smiles | — | (smiles: str, *, config: RDKitConformerConfig \| None=None) -> Mol @ OptionalDependency.RDKIT.require |
| added | nvalchemi/csp/chem.py:topological_atom_types_from_connectivity | — | (atomic_numbers: Tensor, adjacency: Tensor) -> Tensor |
| added | nvalchemi/csp/chem.py:topological_atom_types_from_mol | — | (mol: Mol) -> Tensor @ OptionalDependency.RDKIT.require |
| added | nvalchemi/csp/comparison.py:DeduplicationResult | — | class() @ dataclass(frozen=True) |
| added | nvalchemi/csp/comparison.py:DeduplicationResult.multiplicities | — | Tensor = required |
| added | nvalchemi/csp/comparison.py:DeduplicationResult.representative_indices | — | Tensor = required |
| added | nvalchemi/csp/comparison.py:DeduplicationResult.retained_indices | — | Tensor = required |
| added | nvalchemi/csp/comparison.py:RadialComparisonIndex | — | class() |
| added | nvalchemi/csp/comparison.py:RadialComparisonIndex.__init__ | — | (self, descriptor_store: DescriptorStore, cutoff: float, atom_types: Tensor \| None, typed_neighbors: bool, device: torch.device, block_size: int, max_memory_fraction: float, cuda_memory_budget_bytes: int \| None, summary: Tensor, typed_summary: Tensor, center_type_presence: Tensor, include_hydrogens: bool=True) -> None |
| added | nvalchemi/csp/comparison.py:RadialComparisonIndex.build | — | (cls, batch: Batch, *, cutoff: float, atom_types: Tensor \| None=None, typed_neighbors: bool=True, dtype: torch.dtype=torch.float32, device: torch.device \| str \| None=None, max_memory_fraction: float=0.7, structure_block_size: int \| None=None, include_hydrogens: bool=True) -> RadialComparisonIndex @ classmethod |
| added | nvalchemi/csp/comparison.py:RadialComparisonIndex.deduplicate | — | (self, *, threshold: float, confirm: Callable[[Tensor], Tensor] \| None=None, pair_chunk_size: int \| None=None) -> DeduplicationResult |
| added | nvalchemi/csp/comparison.py:RadialComparisonIndex.find_matches | — | (self, other: RadialComparisonIndex \| None=None, *, threshold: float, pair_indices: Tensor \| None=None, pair_chunk_size: int \| None=None) -> Tensor |
| added | nvalchemi/csp/comparison.py:RadialComparisonIndex.include_hydrogens | — | (self) -> bool @ property |
| added | nvalchemi/csp/comparison.py:RadialComparisonIndex.iter_matches | — | (self, other: RadialComparisonIndex \| None=None, *, threshold: float, pair_indices: Tensor \| None=None, pair_chunk_size: int \| None=None) -> Iterator[Tensor] |
| added | nvalchemi/csp/comparison.py:RadialComparisonIndex.num_structures | — | (self) -> int @ property |
| added | nvalchemi/csp/comparison.py:RadialComparisonIndex.score_pairs | — | (self, pair_indices: Tensor, other: RadialComparisonIndex \| None=None, *, pair_chunk_size: int \| None=None) -> Tensor |
| added | nvalchemi/csp/comparison.py:deduplicate_batch | — | (batch: Batch, *, atom_types: Tensor \| None=None, cutoff: float, threshold: float, priority_order: Sequence[int] \| Tensor \| None=None, device: torch.device \| str \| None=None, max_batch_atoms: int=200000, max_memory_fraction: float=0.85, summary_coordinate_count: int=32, include_hydrogens: bool=True, confirm: Callable[[Tensor], Tensor] \| None=None) -> DeduplicationResult |
| added | nvalchemi/csp/comparison.py:deduplicate_stream | — | (count: int, read_typed_batch: Callable[[Tensor], tuple[Batch, Tensor \| None]], *, cutoff: float, threshold: float, type_vocabulary: Sequence[int] \| None=None, priority_order: Sequence[int] \| Tensor \| None=None, device: torch.device \| str='cpu', input_batch_size: int \| None=None, pair_block_size: int \| None=None, max_batch_atoms: int=200000, max_memory_fraction: float=0.85, summary_coordinate_count: int=32, include_hydrogens: bool=True, confirm: Callable[[Tensor], Tensor] \| None=None) -> DeduplicationResult |
| added | nvalchemi/csp/comparison.py:iter_matches_stream | — | (count: int, read_typed_batch: Callable[[Tensor], tuple[Batch, Tensor \| None]], *, cutoff: float, threshold: float, type_vocabulary: Sequence[int] \| None=None, other_count: int \| None=None, read_other_typed_batch: Callable[[Tensor], tuple[Batch, Tensor \| None]] \| None=None, device: torch.device \| str='cpu', max_batch_atoms: int=200000, max_memory_fraction: float=0.85, pair_chunk_size: int=8192, summary_coordinate_count: int=32, include_hydrogens: bool=True) -> Iterator[Tensor] |
| added | nvalchemi/csp/data.py:MolecularPackingInput | — | class(BaseModel) |
| added | nvalchemi/csp/data.py:MolecularPackingInput.atomic_numbers | — | Tensor = required |
| added | nvalchemi/csp/data.py:MolecularPackingInput.component_charge | — | Tensor = required |
| added | nvalchemi/csp/data.py:MolecularPackingInput.component_index | — | Tensor = required |
| added | nvalchemi/csp/data.py:MolecularPackingInput.conformer_positions | — | Tensor = required |
| added | nvalchemi/csp/data.py:MolecularPackingInput.conformer_ptr | — | Tensor = required |
| added | nvalchemi/csp/data.py:MolecularPackingInput.contact_distances | — | Tensor = required |
| added | nvalchemi/csp/data.py:MolecularPackingInput.formula_unit_charge | — | (self) -> int @ property |
| added | nvalchemi/csp/data.py:MolecularPackingInput.formula_unit_volume | — | float = required |
| added | nvalchemi/csp/data.py:MolecularPackingInput.from_state_dict | — | (cls, state: Mapping[str, object]) -> MolecularPackingInput @ classmethod |
| added | nvalchemi/csp/data.py:MolecularPackingInput.metadata | — | Mapping[str, Any] \| None = None |
| added | nvalchemi/csp/data.py:MolecularPackingInput.molecule_atom_ptr | — | Tensor = required |
| added | nvalchemi/csp/data.py:MolecularPackingInput.molecule_charge | — | (self) -> Tensor @ property |
| added | nvalchemi/csp/data.py:MolecularPackingInput.molecule_conformer_ptr | — | Tensor = required |
| added | nvalchemi/csp/data.py:MolecularPackingInput.num_atoms | — | (self) -> int @ property |
| added | nvalchemi/csp/data.py:MolecularPackingInput.num_components | — | (self) -> int @ property |
| added | nvalchemi/csp/data.py:MolecularPackingInput.num_conformers | — | (self) -> int @ property |
| added | nvalchemi/csp/data.py:MolecularPackingInput.num_molecules | — | (self) -> int @ property |
| added | nvalchemi/csp/data.py:MolecularPackingInput.sha256 | — | (self) -> str @ property |
| added | nvalchemi/csp/data.py:MolecularPackingInput.state_dict | — | (self) -> dict[str, object] |
| added | nvalchemi/csp/data.py:MolecularPackingInput.to | — | (self, device: torch.device \| str) -> MolecularPackingInput |
| added | nvalchemi/csp/data.py:RigidMoleculeASUBatch | — | class(BaseModel) |
| added | nvalchemi/csp/data.py:RigidMoleculeASUBatch.cells | — | Tensor = required |
| added | nvalchemi/csp/data.py:RigidMoleculeASUBatch.check_integrity | — | (self) -> None |
| added | nvalchemi/csp/data.py:RigidMoleculeASUBatch.concatenate | — | (cls, batches: Sequence[RigidMoleculeASUBatch]) -> RigidMoleculeASUBatch @ classmethod |
| added | nvalchemi/csp/data.py:RigidMoleculeASUBatch.conformer_indices | — | Tensor = required |
| added | nvalchemi/csp/data.py:RigidMoleculeASUBatch.fractional_centers | — | Tensor = required |
| added | nvalchemi/csp/data.py:RigidMoleculeASUBatch.num_structures | — | (self) -> int @ property |
| added | nvalchemi/csp/data.py:RigidMoleculeASUBatch.packing_input | — | MolecularPackingInput = required |
| added | nvalchemi/csp/data.py:RigidMoleculeASUBatch.properties | — | Mapping[str, Tensor] = Field(default_factory=dict) |
| added | nvalchemi/csp/data.py:RigidMoleculeASUBatch.rotations | — | Tensor = required |
| added | nvalchemi/csp/data.py:RigidMoleculeASUBatch.select | — | (self, indices: Tensor) -> RigidMoleculeASUBatch |
| added | nvalchemi/csp/data.py:RigidMoleculeASUBatch.space_groups | — | Tensor = required |
| added | nvalchemi/csp/data.py:RigidMoleculeASUBatch.structure_ids | — | Tensor = required |
| added | nvalchemi/csp/data.py:RigidMoleculeASUBatch.structure_molecule_ptr | — | Tensor = required |
| added | nvalchemi/csp/data.py:RigidMoleculeASUBatch.to | — | (self, device: torch.device \| str) -> RigidMoleculeASUBatch |
| added | nvalchemi/csp/data.py:RigidMoleculeASUBatch.to_batch | — | (self, indices: Tensor \| None=None, *, device: torch.device \| str \| None=None) -> Batch |
| added | nvalchemi/csp/data.py:RigidMoleculeASUBatch.z | — | Tensor = required |
| added | nvalchemi/csp/data.py:RigidMoleculeASUBatch.z_prime | — | Tensor = required |
| added | nvalchemi/csp/generator.py:CSPGenerator | — | class(AtomisticGenerator) |
| added | nvalchemi/csp/generator.py:CSPGenerator.__init__ | — | (self, packer: CrystalPacker, *, process_group: ProcessGroup \| None=None, gather_to_rank: int \| None=None, expand: bool=True, outputs: frozenset[str] \| None=None, on_result: Callable[[PackingResult], None] \| None=None, **generator_options: Any) -> None |
| added | nvalchemi/csp/generator.py:CSPGenerator.compile | — | (self, **kwargs: Any) -> CSPGenerator |
| added | nvalchemi/csp/generator.py:CSPGenerator.generator_func | — | GeneratingFunction = Field(repr=False) |
| added | nvalchemi/csp/generator.py:CSPGenerator.to_spec | — | (self) -> BaseSpec |
| added | nvalchemi/csp/hooks.py:DeduplicateHook | — | class() |
| added | nvalchemi/csp/hooks.py:DeduplicateHook.__init__ | — | (self, engine: DeduplicationEngine, *, frequency: int=1) -> None |
| added | nvalchemi/csp/hooks.py:DeduplicateHook.radial | — | (cls, *, cutoff: float, threshold: float, confirm: Callable[[Tensor], Tensor] \| None=None, frequency: int=1) -> DeduplicateHook @ classmethod |
| added | nvalchemi/csp/hooks.py:DeduplicationEngine | — | class(Protocol) |
| added | nvalchemi/csp/hooks.py:DeduplicationEngine.deduplicate | — | (self, batch: Batch) -> Tensor |
| added | nvalchemi/csp/packer/__init__.py:exports | — | __all__ = ['CandidateBudgetPacker', 'CrystalPacker', 'OverlapReliefConfig', 'OverlapReliefPacker', 'OverlapReliefProgress', 'PackingContext', 'PackingReport', 'PackingResult', 'PackingStopReason'] ; from nvalchemi.csp.packer.config import OverlapReliefConfig ; from nvalchemi.csp.packer.engine import OverlapReliefPacker ; from nvalchemi.csp.packer.protocol import CandidateBudgetPacker, CrystalPacker, PackingContext ; from nvalchemi.csp.packer.result import OverlapReliefProgress, PackingReport, PackingResult, PackingStopReason |
| added | nvalchemi/csp/packer/_cells.py:sample_valid_cells | — | (*, count: int, config: object, groups: Tensor, probabilities: Tensor, formula_unit_volume: float, seed: int) -> tuple[Tensor, Tensor] |
| added | nvalchemi/csp/packer/_cells.py:sampling_tables | — | (*, config: object, device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor] |
| added | nvalchemi/csp/packer/_contacts.py:ContactEvaluation | — | class() @ dataclass(frozen=True) |
| added | nvalchemi/csp/packer/_contacts.py:ContactEvaluation.forces | — | Tensor = required |
| added | nvalchemi/csp/packer/_contacts.py:ContactEvaluation.max_overlap | — | Tensor = required |
| added | nvalchemi/csp/packer/_contacts.py:ContactEvaluation.torques | — | Tensor = required |
| added | nvalchemi/csp/packer/_contacts.py:ContactEvaluation.total_overlap | — | Tensor = required |
| added | nvalchemi/csp/packer/_contacts.py:ContactEvaluation.virial | — | Tensor = required |
| added | nvalchemi/csp/packer/_contacts.py:ContactMaps | — | class() @ dataclass(frozen=True) |
| added | nvalchemi/csp/packer/_contacts.py:ContactMaps.asu_atom | — | Tensor = required |
| added | nvalchemi/csp/packer/_contacts.py:ContactMaps.expanded_meta | — | Tensor = required |
| added | nvalchemi/csp/packer/_contacts.py:ContactMaps.local_atom | — | Tensor = required |
| added | nvalchemi/csp/packer/_contacts.py:ContactMaps.molecule | — | Tensor = required |
| added | nvalchemi/csp/packer/_contacts.py:ContactMaps.num_asu_atoms | — | int = required |
| added | nvalchemi/csp/packer/_contacts.py:ContactMaps.num_molecules | — | int = required |
| added | nvalchemi/csp/packer/_contacts.py:ContactMaps.num_symmetry_operations | — | int = required |
| added | nvalchemi/csp/packer/_contacts.py:ContactMaps.symmetry | — | Tensor = required |
| added | nvalchemi/csp/packer/_contacts.py:ContactWorkspace | — | class() @ dataclass |
| added | nvalchemi/csp/packer/_contacts.py:ContactWorkspace.allocate | — | (cls, *, batch_size: int, num_molecules: int, expanded_atom_count: int, device: torch.device) -> ContactWorkspace @ classmethod |
| added | nvalchemi/csp/packer/_contacts.py:ContactWorkspace.atom_periodic_shifts | — | Tensor \| None = None |
| added | nvalchemi/csp/packer/_contacts.py:ContactWorkspace.atom_to_cell_mapping | — | Tensor \| None = None |
| added | nvalchemi/csp/packer/_contacts.py:ContactWorkspace.atoms_per_cell_count | — | Tensor \| None = None |
| added | nvalchemi/csp/packer/_contacts.py:ContactWorkspace.batch_idx | — | Tensor = required |
| added | nvalchemi/csp/packer/_contacts.py:ContactWorkspace.batch_size | — | int = required |
| added | nvalchemi/csp/packer/_contacts.py:ContactWorkspace.build_cell_list | — | (self, cells: Tensor, cutoff: float) -> None |
| added | nvalchemi/csp/packer/_contacts.py:ContactWorkspace.cell_atom_list | — | Tensor \| None = None |
| added | nvalchemi/csp/packer/_contacts.py:ContactWorkspace.cell_atom_start_indices | — | Tensor \| None = None |
| added | nvalchemi/csp/packer/_contacts.py:ContactWorkspace.cell_offsets | — | Tensor \| None = None |
| added | nvalchemi/csp/packer/_contacts.py:ContactWorkspace.cells_per_dimension | — | Tensor \| None = None |
| added | nvalchemi/csp/packer/_contacts.py:ContactWorkspace.ensure_cell_capacity | — | (self, cells: Tensor, cutoff: float) -> None |
| added | nvalchemi/csp/packer/_contacts.py:ContactWorkspace.expanded_atom_count | — | int = required |
| added | nvalchemi/csp/packer/_contacts.py:ContactWorkspace.forces | — | Tensor = required |
| added | nvalchemi/csp/packer/_contacts.py:ContactWorkspace.image_offsets | — | Tensor = required |
| added | nvalchemi/csp/packer/_contacts.py:ContactWorkspace.image_rel_asu | — | Tensor = required |
| added | nvalchemi/csp/packer/_contacts.py:ContactWorkspace.max_cell_count | — | int = required |
| added | nvalchemi/csp/packer/_contacts.py:ContactWorkspace.max_overlap | — | Tensor = required |
| added | nvalchemi/csp/packer/_contacts.py:ContactWorkspace.neighbor_search_radius | — | Tensor \| None = None |
| added | nvalchemi/csp/packer/_contacts.py:ContactWorkspace.num_molecules | — | int = required |
| added | nvalchemi/csp/packer/_contacts.py:ContactWorkspace.pbc | — | Tensor = required |
| added | nvalchemi/csp/packer/_contacts.py:ContactWorkspace.positions | — | Tensor = required |
| added | nvalchemi/csp/packer/_contacts.py:ContactWorkspace.torques | — | Tensor = required |
| added | nvalchemi/csp/packer/_contacts.py:ContactWorkspace.total_overlap | — | Tensor = required |
| added | nvalchemi/csp/packer/_contacts.py:ContactWorkspace.virial | — | Tensor = required |
| added | nvalchemi/csp/packer/_contacts.py:contact_forces | — | (*, conformer_positions: Tensor, conformer_ptr: Tensor, conformer_ids: Tensor, centers: Tensor, rotations: Tensor, cells: Tensor, inverse_cells: Tensor, symmetry_table: Tensor, selected_symmetry_ops: Tensor, contact_distances: Tensor, max_contact_distance: float, maps: ContactMaps, workspace: ContactWorkspace, active_mask: Tensor \| None=None) -> ContactEvaluation |
| added | nvalchemi/csp/packer/_contacts.py:make_contact_maps | — | (molecule_atom_ptr: Tensor, num_symmetry_operations: int) -> ContactMaps |
| added | nvalchemi/csp/packer/_interop.py:as_warp | — | (value: torch.Tensor, dtype: Any) -> wp.array |
| added | nvalchemi/csp/packer/_interop.py:launch_device | — | (value: torch.Tensor) -> str |
| added | nvalchemi/csp/packer/_interop.py:scoped_warp_stream | — | (value: torch.Tensor) -> unknown |
| added | nvalchemi/csp/packer/_kernels/__init__.py:exports | — | — |
| added | nvalchemi/csp/packer/_kernels/relaxation.py:relax_step | — | (*, active: torch.Tensor, conformer_positions: torch.Tensor, conformer_ptr: torch.Tensor, conformer_ids: torch.Tensor, molecule_atom_ptr: torch.Tensor, forces: torch.Tensor, torques: torch.Tensor, max_overlap: torch.Tensor, virial: torch.Tensor, cells: torch.Tensor, inverse_cells: torch.Tensor, centers: torch.Tensor, rotations: torch.Tensor, space_groups: torch.Tensor, reference_volumes: torch.Tensor, steps: torch.Tensor, expanded_atom_count: int, step_scale: float, max_step: float, cell_step_scale: float, max_cell_strain: float, volume_compression_scale: float) -> None |
| added | nvalchemi/csp/packer/_kernels/sampling.py:generate_cells | — | (*, count: int, candidate_groups: torch.Tensor, probabilities: torch.Tensor, volume_range: tuple[float, float], min_cell_height: float, max_axis_ratio: float, oversample_factor: float, seed: int, max_rounds: int=128) -> tuple[torch.Tensor, torch.Tensor] |
| added | nvalchemi/csp/packer/_kernels/sampling.py:initialize_rows | — | (*, rows: torch.Tensor, sampled_cells: torch.Tensor, sampled_groups: torch.Tensor, op_indices: torch.Tensor, op_ptr: torch.Tensor, conformer_starts: torch.Tensor, conformer_stops: torch.Tensor, seed: int, conformer_ids: torch.Tensor, centers: torch.Tensor, rotations: torch.Tensor, cells: torch.Tensor, inverse_cells: torch.Tensor, reference_volumes: torch.Tensor, groups: torch.Tensor, selected_ops: torch.Tensor, steps: torch.Tensor) -> None |
| added | nvalchemi/csp/packer/_kernels/sampling.py:sample_cell_trials | — | (*, num_trials: int, candidate_groups: torch.Tensor, probabilities: torch.Tensor, volume_range: tuple[float, float], min_cell_height: float, max_axis_ratio: float, seed: int, round_index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] |
| added | nvalchemi/csp/packer/_state.py:WorkingState | — | class() @ dataclass |
| added | nvalchemi/csp/packer/_state.py:WorkingState.active | — | Tensor = required |
| added | nvalchemi/csp/packer/_state.py:WorkingState.allocate | — | (cls, *, batch_size: int, molecule_count: int, symmetry_operation_count: int, device: torch.device) -> WorkingState @ classmethod |
| added | nvalchemi/csp/packer/_state.py:WorkingState.cells | — | Tensor = required |
| added | nvalchemi/csp/packer/_state.py:WorkingState.centers | — | Tensor = required |
| added | nvalchemi/csp/packer/_state.py:WorkingState.conformer_ids | — | Tensor = required |
| added | nvalchemi/csp/packer/_state.py:WorkingState.initialize | — | (self, *, rows: Tensor, sampled_cells: Tensor, sampled_groups: Tensor, op_indices: Tensor, op_ptr: Tensor, conformer_starts: Tensor, conformer_stops: Tensor, seed: int) -> None |
| added | nvalchemi/csp/packer/_state.py:WorkingState.inverse_cells | — | Tensor = required |
| added | nvalchemi/csp/packer/_state.py:WorkingState.reference_volumes | — | Tensor = required |
| added | nvalchemi/csp/packer/_state.py:WorkingState.rotations | — | Tensor = required |
| added | nvalchemi/csp/packer/_state.py:WorkingState.space_groups | — | Tensor = required |
| added | nvalchemi/csp/packer/_state.py:WorkingState.steps | — | Tensor = required |
| added | nvalchemi/csp/packer/_state.py:WorkingState.symmetry_ops | — | Tensor = required |
| added | nvalchemi/csp/packer/_state.py:relax_step | — | (*, state: WorkingState, conformer_positions: Tensor, conformer_ptr: Tensor, molecule_atom_ptr: Tensor, forces: Tensor, torques: Tensor, virial: Tensor, max_overlap: Tensor, expanded_atom_count: int, step_scale: float, max_step: float, cell_step_scale: float, max_cell_strain: float, volume_compression_scale: float, active: Tensor \| None=None) -> None |
| added | nvalchemi/csp/packer/config.py:OverlapReliefConfig | — | class(BaseModel) |
| added | nvalchemi/csp/packer/config.py:OverlapReliefConfig.batch_size | — | int = Field(strict=True) |
| added | nvalchemi/csp/packer/config.py:OverlapReliefConfig.cell_oversample_factor | — | float = 1.5 |
| added | nvalchemi/csp/packer/config.py:OverlapReliefConfig.cell_step_scale | — | float = 0.01 |
| added | nvalchemi/csp/packer/config.py:OverlapReliefConfig.cell_volume_range | — | tuple[float, float] \| None = None |
| added | nvalchemi/csp/packer/config.py:OverlapReliefConfig.cell_volume_scale_range | — | tuple[float, float] \| None = None |
| added | nvalchemi/csp/packer/config.py:OverlapReliefConfig.convergence_check_interval | — | int = Field(default=10, strict=True) |
| added | nvalchemi/csp/packer/config.py:OverlapReliefConfig.effective | — | (self, **overrides: Any) -> OverlapReliefConfig |
| added | nvalchemi/csp/packer/config.py:OverlapReliefConfig.max_axis_ratio | — | float = 10.0 |
| added | nvalchemi/csp/packer/config.py:OverlapReliefConfig.max_candidates | — | Annotated[int, Field(strict=True)] \| Literal['auto'] \| None = 'auto' |
| added | nvalchemi/csp/packer/config.py:OverlapReliefConfig.max_cell_strain | — | float = 0.003 |
| added | nvalchemi/csp/packer/config.py:OverlapReliefConfig.max_step | — | float = 0.3 |
| added | nvalchemi/csp/packer/config.py:OverlapReliefConfig.max_steps_per_candidate | — | int = Field(default=500, strict=True) |
| added | nvalchemi/csp/packer/config.py:OverlapReliefConfig.min_cell_height | — | float = 3.8 |
| added | nvalchemi/csp/packer/config.py:OverlapReliefConfig.overlap_tolerance | — | float = 0.05 |
| added | nvalchemi/csp/packer/config.py:OverlapReliefConfig.space_groups | — | SpaceGroupPolicy = Field(default_factory=SpaceGroupPolicy.sampled) |
| added | nvalchemi/csp/packer/config.py:OverlapReliefConfig.step_scale | — | float = 0.12 |
| added | nvalchemi/csp/packer/config.py:OverlapReliefConfig.volume_compression_scale | — | float = 1e-05 |
| added | nvalchemi/csp/packer/config.py:OverlapReliefConfig.z | — | int = Field(strict=True) |
| added | nvalchemi/csp/packer/config.py:OverlapReliefConfig.z_prime | — | int = Field(strict=True) |
| added | nvalchemi/csp/packer/engine.py:OverlapReliefPacker | — | class() |
| added | nvalchemi/csp/packer/engine.py:OverlapReliefPacker.__init__ | — | (self, config: OverlapReliefConfig, *, device: torch.device \| str) -> None |
| added | nvalchemi/csp/packer/engine.py:OverlapReliefPacker.pack | — | (self, inputs: MolecularPackingInput, *, num_samples: int=1, rng: torch.Generator \| None=None, context: PackingContext \| None=None, run_id: int \| None=None, candidate_budget: int \| None \| Literal['config']='config', progress_callback: Callable[[OverlapReliefProgress], None] \| None=None, **config_overrides: object) -> PackingResult[RigidMoleculeASUBatch] |
| added | nvalchemi/csp/packer/engine.py:OverlapReliefPacker.resolve_candidate_budget | — | (self, *, num_samples: int, **pack_options: object) -> int \| None |
| added | nvalchemi/csp/packer/engine.py:OverlapReliefPacker.to_spec | — | (self) -> BaseSpec |
| added | nvalchemi/csp/packer/protocol.py:CandidateBudgetPacker | — | class(Protocol) @ runtime_checkable |
| added | nvalchemi/csp/packer/protocol.py:CandidateBudgetPacker.resolve_candidate_budget | — | (self, *, num_samples: int, **pack_options: Any) -> int \| None |
| added | nvalchemi/csp/packer/protocol.py:CrystalPacker | — | class(Protocol) @ runtime_checkable |
| added | nvalchemi/csp/packer/protocol.py:CrystalPacker.device | — | torch.device = required |
| added | nvalchemi/csp/packer/protocol.py:CrystalPacker.pack | — | (self, inputs: MolecularPackingInput, *, num_samples: int=1, rng: torch.Generator \| None=None, context: PackingContext \| None=None, **options: Any) -> PackingResult |
| added | nvalchemi/csp/packer/protocol.py:PackingContext | — | class() @ dataclass(frozen=True) |
| added | nvalchemi/csp/packer/protocol.py:PackingContext.rank | — | int = 0 |
| added | nvalchemi/csp/packer/protocol.py:PackingContext.run_id | — | int = required |
| added | nvalchemi/csp/packer/protocol.py:PackingContext.structure_ids | — | (self, count: int, *, device: torch.device \| str) -> torch.Tensor |
| added | nvalchemi/csp/packer/protocol.py:PackingContext.world_size | — | int = 1 |
| added | nvalchemi/csp/packer/result.py:OverlapReliefProgress | — | class() @ dataclass(frozen=True) |
| added | nvalchemi/csp/packer/result.py:OverlapReliefProgress.accepted_count | — | int = required |
| added | nvalchemi/csp/packer/result.py:OverlapReliefProgress.active_count | — | int = required |
| added | nvalchemi/csp/packer/result.py:OverlapReliefProgress.converged_count | — | int = required |
| added | nvalchemi/csp/packer/result.py:OverlapReliefProgress.expired_count | — | int = required |
| added | nvalchemi/csp/packer/result.py:OverlapReliefProgress.generated_count | — | int = required |
| added | nvalchemi/csp/packer/result.py:OverlapReliefProgress.iteration | — | int = required |
| added | nvalchemi/csp/packer/result.py:OverlapReliefProgress.max_overlap | — | float = required |
| added | nvalchemi/csp/packer/result.py:OverlapReliefProgress.rank | — | int = 0 |
| added | nvalchemi/csp/packer/result.py:OverlapReliefProgress.replaced_count | — | int = required |
| added | nvalchemi/csp/packer/result.py:OverlapReliefProgress.total_overlap | — | float = required |
| added | nvalchemi/csp/packer/result.py:OverlapReliefProgress.world_size | — | int = 1 |
| added | nvalchemi/csp/packer/result.py:PackingReport | — | class() @ dataclass(frozen=True) |
| added | nvalchemi/csp/packer/result.py:PackingReport.accepted_count | — | int = required |
| added | nvalchemi/csp/packer/result.py:PackingReport.complete | — | (self) -> bool @ property |
| added | nvalchemi/csp/packer/result.py:PackingReport.generated_count | — | int \| None = None |
| added | nvalchemi/csp/packer/result.py:PackingReport.rank | — | int = required |
| added | nvalchemi/csp/packer/result.py:PackingReport.requested_count | — | int = required |
| added | nvalchemi/csp/packer/result.py:PackingReport.stop_reason | — | str = required |
| added | nvalchemi/csp/packer/result.py:PackingResult | — | class(Generic[_StructureT]) @ dataclass(frozen=True) |
| added | nvalchemi/csp/packer/result.py:PackingResult.accepted_count | — | (self) -> int @ property |
| added | nvalchemi/csp/packer/result.py:PackingResult.complete | — | (self) -> bool @ property |
| added | nvalchemi/csp/packer/result.py:PackingResult.generated_count | — | (self) -> int \| None @ property |
| added | nvalchemi/csp/packer/result.py:PackingResult.reports | — | tuple[PackingReport, ...] = required |
| added | nvalchemi/csp/packer/result.py:PackingResult.requested_count | — | (self) -> int @ property |
| added | nvalchemi/csp/packer/result.py:PackingResult.run_id | — | int = required |
| added | nvalchemi/csp/packer/result.py:PackingResult.scope | — | Literal['local', 'gathered'] = 'local' |
| added | nvalchemi/csp/packer/result.py:PackingResult.stop_reason | — | (self) -> PackingStopReason @ property |
| added | nvalchemi/csp/packer/result.py:PackingResult.structures | — | _StructureT = required |
| added | nvalchemi/csp/packer/result.py:PackingStopReason | — | class(str, Enum) |
| added | nvalchemi/csp/storage.py:RigidMoleculeASUZarrReader | — | class() |
| added | nvalchemi/csp/storage.py:RigidMoleculeASUZarrReader.__init__ | — | (self, store: StoreLike) -> None |
| added | nvalchemi/csp/storage.py:RigidMoleculeASUZarrReader.close | — | (self) -> None |
| added | nvalchemi/csp/storage.py:RigidMoleculeASUZarrReader.field_names | — | (self) -> list[str] @ property |
| added | nvalchemi/csp/storage.py:RigidMoleculeASUZarrReader.get_metadata | — | (self, indices: torch.Tensor) -> torch.Tensor |
| added | nvalchemi/csp/storage.py:RigidMoleculeASUZarrReader.metadata | — | (self) -> Mapping[str, object] @ property |
| added | nvalchemi/csp/storage.py:RigidMoleculeASUZarrReader.read | — | (self, indices: torch.Tensor \| None=None, *, device: torch.device \| str='cpu') -> RigidMoleculeASUBatch |
| added | nvalchemi/csp/storage.py:RigidMoleculeASUZarrReader.read_batch | — | (self, indices: torch.Tensor, *, device: torch.device \| str='cpu') -> Batch |
| added | nvalchemi/csp/storage.py:RigidMoleculeASUZarrReader.refresh | — | (self) -> None |
| added | nvalchemi/csp/storage.py:RigidMoleculeASUZarrWriter | — | class() |
| added | nvalchemi/csp/storage.py:RigidMoleculeASUZarrWriter.__init__ | — | (self, store: StoreLike, config: ZarrWriteConfig \| Mapping[str, Any] \| None=None) -> None |
| added | nvalchemi/csp/storage.py:RigidMoleculeASUZarrWriter.append | — | (self, structures: RigidMoleculeASUBatch) -> None |
| added | nvalchemi/csp/storage.py:RigidMoleculeASUZarrWriter.close | — | (self) -> None |
| added | nvalchemi/csp/storage.py:RigidMoleculeASUZarrWriter.defragment | — | (self, config: ZarrWriteConfig \| Mapping[str, Any] \| None=None) -> None |
| added | nvalchemi/csp/storage.py:RigidMoleculeASUZarrWriter.delete | — | (self, indices: torch.Tensor) -> None |
| added | nvalchemi/csp/storage.py:RigidMoleculeASUZarrWriter.write | — | (self, structures: RigidMoleculeASUBatch) -> None |
| added | nvalchemi/csp/symmetry.py:CrystalSystem | — | class(str, Enum) |
| added | nvalchemi/csp/symmetry.py:SpaceGroupPolicy | — | class(BaseModel) |
| added | nvalchemi/csp/symmetry.py:SpaceGroupPolicy.crystal_system | — | CrystalSystem \| None = None |
| added | nvalchemi/csp/symmetry.py:SpaceGroupPolicy.draw | — | (self, num_samples: int, *, num_operations: int, seed: int \| None=None, device: Device=None) -> Tensor |
| added | nvalchemi/csp/symmetry.py:SpaceGroupPolicy.fixed | — | (cls, group: int, *, sohncke_only: bool=False) -> SpaceGroupPolicy @ classmethod |
| added | nvalchemi/csp/symmetry.py:SpaceGroupPolicy.group | — | int \| None = Field(default=None, strict=True) |
| added | nvalchemi/csp/symmetry.py:SpaceGroupPolicy.mode | — | Literal['fixed', 'sampled'] = required |
| added | nvalchemi/csp/symmetry.py:SpaceGroupPolicy.probabilities | — | Mapping[int, float] \| None = Field(default=None, repr=False) |
| added | nvalchemi/csp/symmetry.py:SpaceGroupPolicy.sampled | — | (cls, *, probabilities: Mapping[int, float] \| None=None, crystal_system: CrystalSystem \| None=None, sohncke_only: bool=False) -> SpaceGroupPolicy @ classmethod |
| added | nvalchemi/csp/symmetry.py:SpaceGroupPolicy.sohncke_only | — | bool = Field(default=False, strict=True) |
| added | nvalchemi/csp/symmetry.py:csd_space_group_probabilities | — | () -> dict[int, float] |
| added | nvalchemi/csp/symmetry.py:get_crystal_system | — | (space_group: int) -> CrystalSystem |
| added | nvalchemi/csp/symmetry.py:get_space_group_candidates | — | (num_operations: int, *, crystal_system: CrystalSystem \| None=None, sohncke_only: bool=False) -> Tensor |
| added | nvalchemi/csp/symmetry.py:get_space_group_operation_count | — | (space_group: int) -> int |
| added | nvalchemi/csp/symmetry.py:get_space_group_operations | — | (space_group: int, *, dtype: torch.dtype=torch.float32, device: Device=None) -> Tensor |
| added | nvalchemi/csp/symmetry.py:is_sohncke_space_group | — | (space_group: int) -> bool |
| added | nvalchemi/csp/symmetry.py:sample_space_groups | — | (num_samples: int, num_operations: int, *, probabilities: Mapping[int, float] \| None=None, crystal_system: CrystalSystem \| None=None, sohncke_only: bool=False, seed: int \| None=None, device: Device=None) -> Tensor |
| added | nvalchemi/csp/volume.py:estimate_formula_unit_volume | — | (atomic_numbers: Tensor, *, atomic_volumes: Mapping[int, float] \| None=None) -> float |
| added | nvalchemi/csp/volume.py:get_default_atomic_volumes | — | () -> dict[int, float] |
| changed | nvalchemi/data/__init__.py:exports | __all__ = ['AtomicData', 'Batch', 'Reader', 'AtomicDataZarrReader', 'AtomicDataZarrWriter', 'BatchDatasetProtocol', 'Dataset', 'InMemoryDataset', 'DataLoader', 'Compose', 'transforms'] ; from __future__ import annotations ; from nvalchemi.data import transforms ; from nvalchemi.data.atomic_data import AtomicData ; from nvalchemi.data.batch import Batch ; from nvalchemi.data.datapipes import AtomicDataZarrReader, AtomicDataZarrWriter, BatchDatasetProtocol, DataLoader, Dataset, InMemoryDataset, Reader ; from nvalchemi.data.transforms import Compose | __all__ = ['AtomicData', 'Batch', 'LevelSchema', 'GroupLayout', 'resolve_device', 'Reader', 'AtomicDataZarrReader', 'AtomicDataZarrWriter', 'BatchDatasetProtocol', 'FieldSchema', 'Dataset', 'InMemoryDataset', 'DataLoader', 'Compose', 'transforms'] ; from __future__ import annotations ; from nvalchemi.data import transforms ; from nvalchemi.data.atomic_data import AtomicData ; from nvalchemi.data.batch import Batch ; from nvalchemi.data.datapipes import AtomicDataZarrReader, AtomicDataZarrWriter, BatchDatasetProtocol, DataLoader, Dataset, FieldSchema, InMemoryDataset, Reader ; from nvalchemi.data.group_layout import GroupLayout ; from nvalchemi.data.level_storage import LevelSchema, resolve_device ; from nvalchemi.data.transforms import Compose |
| added | nvalchemi/data/atomic_data.py:AtomicData.clone | — | (self) -> AtomicData |
| added | nvalchemi/data/atomic_data.py:AtomicData.precision_preserving_keys | — | ClassVar[frozenset[str]] = frozenset() |
| added | nvalchemi/data/atomic_data.py:AtomicData.to | — | (self, device: torch.device \| str, dtype: torch.dtype \| None=None, non_blocking: bool=False) -> AtomicData |
| changed | nvalchemi/data/batch.py:Batch.add_key | (self, key: str, values: list[Tensor], level: str='node', overwrite: bool=False) -> None | (self, key: str, values: list[Tensor], level: str='node', overwrite: bool=False, *, dtype: torch.dtype \| None=None, payload_shape: tuple[int, ...] \| None=None) -> None |
| added | nvalchemi/data/batch.py:Batch.add_level | — | (self, name: str, *, segmented: bool) -> None |
| added | nvalchemi/data/batch.py:Batch.add_product_level | — | (self, name: str, *, left: str, right: str) -> None |
| changed | nvalchemi/data/batch.py:Batch.clone | (self) -> Batch | (self, *, drop: Iterable[str]=()) -> Batch |
| added | nvalchemi/data/batch.py:Batch.drop_level | — | (self, name: str) -> None |
| changed | nvalchemi/data/batch.py:Batch.empty | (cls, *, num_systems: int, num_nodes: int, num_edges: int, template: AtomicData \| Batch \| None=None, device: torch.device \| str='cpu', attr_map: LevelSchema \| None=None) -> Batch @ classmethod | (cls, *, num_systems: int, num_nodes: int, num_edges: int, template: AtomicData \| Batch \| None=None, device: torch.device \| str='cpu', attr_map: LevelSchema \| None=None, level_capacities: dict[str, int] \| None=None) -> Batch @ classmethod |
| added | nvalchemi/data/batch.py:Batch.extend_level_schema | — | (self, schema: LevelSchema) -> None |
| added | nvalchemi/data/batch.py:Batch.get_level_schema | — | (self) -> LevelSchema |
| added | nvalchemi/data/batch.py:Batch.group_layout | — | (self) -> GroupLayout @ property |
| changed | nvalchemi/data/batch.py:Batch.index_select | (self, idx: int \| slice \| Tensor \| list[int] \| np.ndarray \| Sequence[int]) -> Batch | (self, idx: int \| slice \| Tensor \| list[int] \| np.ndarray \| Sequence[int], *, drop: Iterable[str]=()) -> Batch |
| added | nvalchemi/data/batch.py:Batch.level_keys | — | (self) -> dict[str, set[str]] @ property |
| added | nvalchemi/data/batch.py:Batch.level_ptr | — | (self, name: str) -> Tensor |
| added | nvalchemi/data/batch.py:Batch.pop_level | — | (self, name: str) -> LevelStorage \| None |
| added | nvalchemi/data/batch.py:Batch.set_group_layout | — | (self, group_idx: Tensor) -> None |
| added | nvalchemi/data/batch.py:Batch.set_level | — | (self, name: str, storage: LevelStorage) -> None |
| added | nvalchemi/data/batch.py:Batch.without_keys | — | (self, *keys: str) -> Iterator[None] @ contextmanager |
| changed | nvalchemi/data/datapipes/__init__.py:exports | __all__ = ['Reader', 'AtomicDataZarrReader', 'AtomicDataZarrWriter', 'ZarrArrayConfig', 'ZarrWriteConfig', 'BatchDatasetProtocol', 'Dataset', 'InMemoryDataset', 'MultiDataset', 'DistributedSamplerProtocol', 'MultiDatasetSampler', 'MultiDatasetBatchSampler', 'DataLoader'] ; from __future__ import annotations ; from nvalchemi.data.datapipes.backends.base import Reader ; from nvalchemi.data.datapipes.backends.zarr import AtomicDataZarrReader, AtomicDataZarrWriter, ZarrArrayConfig, ZarrWriteConfig ; from nvalchemi.data.datapipes.dataloader import DataLoader ; from nvalchemi.data.datapipes.dataset import BatchDatasetProtocol, Dataset ; from nvalchemi.data.datapipes.in_memory_dataset import InMemoryDataset ; from nvalchemi.data.datapipes.multidataset import MultiDataset ; from nvalchemi.data.datapipes.samplers import DistributedSamplerProtocol, MultiDatasetBatchSampler, MultiDatasetSampler | __all__ = ['Reader', 'AtomicDataZarrReader', 'AtomicDataZarrWriter', 'FieldSchema', 'ZarrArrayConfig', 'ZarrWriteConfig', 'BatchDatasetProtocol', 'Dataset', 'InMemoryDataset', 'MultiDataset', 'DistributedSamplerProtocol', 'MultiDatasetSampler', 'MultiDatasetBatchSampler', 'distributed_shard', 'dataset_device', 'same_device', 'DataLoader'] ; from __future__ import annotations ; from nvalchemi.data.datapipes.backends.base import Reader ; from nvalchemi.data.datapipes.backends.zarr import AtomicDataZarrReader, AtomicDataZarrWriter, FieldSchema, ZarrArrayConfig, ZarrWriteConfig ; from nvalchemi.data.datapipes.dataloader import DataLoader ; from nvalchemi.data.datapipes.dataset import BatchDatasetProtocol, Dataset, dataset_device, same_device ; from nvalchemi.data.datapipes.in_memory_dataset import InMemoryDataset ; from nvalchemi.data.datapipes.multidataset import MultiDataset ; from nvalchemi.data.datapipes.samplers import DistributedSamplerProtocol, MultiDatasetBatchSampler, MultiDatasetSampler, distributed_shard |
| changed | nvalchemi/data/datapipes/backends/__init__.py:exports | __all__ = ['AtomicDataZarrReader', 'AtomicDataZarrWriter'] ; from __future__ import annotations ; from nvalchemi.data.datapipes.backends.zarr import AtomicDataZarrReader, AtomicDataZarrWriter | __all__ = ['AtomicDataZarrReader', 'AtomicDataZarrWriter', 'FieldSchema'] ; from __future__ import annotations ; from nvalchemi.data.datapipes.backends.zarr import AtomicDataZarrReader, AtomicDataZarrWriter, FieldSchema |
| added | nvalchemi/data/datapipes/backends/base.py:Reader.level_schema | — | (self) -> LevelSchema \| None @ property |
| changed | nvalchemi/data/datapipes/backends/zarr.py:AtomicDataZarrReader.__init__ | (self, store: StoreLike, *, pin_memory: bool=False, include_index_in_metadata: bool=True) -> None | (self, store: StoreLike, *, pin_memory: bool=False, include_index_in_metadata: bool=True, fields: Sequence[str] \| None=None) -> None |
| added | nvalchemi/data/datapipes/backends/zarr.py:AtomicDataZarrReader.check_integrity | — | (self) -> None |
| added | nvalchemi/data/datapipes/backends/zarr.py:AtomicDataZarrReader.field_array | — | (self, field: str) -> zarr.Array |
| added | nvalchemi/data/datapipes/backends/zarr.py:AtomicDataZarrReader.level_schema | — | (self) -> LevelSchema \| None @ property |
| added | nvalchemi/data/datapipes/backends/zarr.py:AtomicDataZarrReader.level_sizes | — | (self) -> dict[str, int] |
| added | nvalchemi/data/datapipes/backends/zarr.py:AtomicDataZarrReader.num_samples | — | (self) -> int @ property |
| added | nvalchemi/data/datapipes/backends/zarr.py:AtomicDataZarrReader.schema | — | (self) -> dict[str, FieldSchema] |
| added | nvalchemi/data/datapipes/backends/zarr.py:AtomicDataZarrReader.store | — | (self) -> StoreLike @ property |
| changed | nvalchemi/data/datapipes/backends/zarr.py:AtomicDataZarrWriter.add_custom | (self, key: str, data: torch.Tensor, level: Literal['atom', 'edge', 'system']) -> None | (self, key: str, data: torch.Tensor, level: str, *, attr_map: LevelSchema \| None=None, level_ptrs: Mapping[str, torch.Tensor] \| None=None) -> None |
| added | nvalchemi/data/datapipes/backends/zarr.py:FieldSchema | — | class() @ dataclasses.dataclass(frozen=True) |
| added | nvalchemi/data/datapipes/backends/zarr.py:FieldSchema.dtype | — | torch.dtype = required |
| added | nvalchemi/data/datapipes/backends/zarr.py:FieldSchema.level | — | str = required |
| added | nvalchemi/data/datapipes/backends/zarr.py:FieldSchema.row_shape | — | tuple[int, ...] = required |
| added | nvalchemi/data/datapipes/dataset.py:dataset_device | — | (dataset: BatchDatasetProtocol, probe: Batch \| None=None) -> torch.device |
| added | nvalchemi/data/datapipes/dataset.py:same_device | — | (left: torch.device \| None, right: torch.device \| None) -> bool |
| added | nvalchemi/data/datapipes/samplers.py:distributed_shard | — | (indices: list, *, num_replicas: int, rank: int, drop_last: bool, pad: bool=True) -> list |
| added | nvalchemi/data/group_layout.py:GroupLayout | — | class() @ dataclass(frozen=True) |
| added | nvalchemi/data/group_layout.py:GroupLayout.broadcast | — | (self, group_values: Tensor) -> Tensor |
| added | nvalchemi/data/group_layout.py:GroupLayout.device | — | (self) -> torch.device @ property |
| added | nvalchemi/data/group_layout.py:GroupLayout.from_batch | — | (cls, batch: Batch) -> GroupLayout @ classmethod |
| added | nvalchemi/data/group_layout.py:GroupLayout.graph_rank | — | Tensor = required |
| added | nvalchemi/data/group_layout.py:GroupLayout.group_idx | — | Tensor = required |
| added | nvalchemi/data/group_layout.py:GroupLayout.group_ptr | — | Tensor = required |
| added | nvalchemi/data/group_layout.py:GroupLayout.node_to_group | — | Tensor = required |
| added | nvalchemi/data/group_layout.py:GroupLayout.num_graphs_per_group | — | Tensor = required |
| added | nvalchemi/data/group_layout.py:GroupLayout.num_groups | — | (self) -> int @ property |
| added | nvalchemi/data/group_layout.py:GroupLayout.reduce_all | — | (self, graph_mask: Tensor) -> Tensor |
| added | nvalchemi/data/group_layout.py:GroupLayout.reduce_any | — | (self, graph_mask: Tensor) -> Tensor |
| changed | nvalchemi/data/level_storage.py:BaseLevelStorage.clone | (self) -> BaseLevelStorage | (self, *, drop: Collection[str]=()) -> BaseLevelStorage |
| changed | nvalchemi/data/level_storage.py:BaseLevelStorage.select | (self, idx: IndexType) -> BaseLevelStorage @ abstractmethod | (self, idx: IndexType, *, drop: Collection[str]=()) -> BaseLevelStorage @ abstractmethod |
| added | nvalchemi/data/level_storage.py:LevelSchema.add_level | — | (self, name: str, *, segmented: bool) -> None |
| added | nvalchemi/data/level_storage.py:LevelSchema.add_product_level | — | (self, name: str, *, left: str, right: str) -> None |
| added | nvalchemi/data/level_storage.py:LevelSchema.level_kind | — | (self, name: str) -> LevelKind |
| added | nvalchemi/data/level_storage.py:LevelSchema.level_names | — | (self) -> tuple[str, ...] @ property |
| changed | nvalchemi/data/level_storage.py:MultiLevelStorage.clone | (self) -> MultiLevelStorage | (self, *, drop: Collection[str]=()) -> MultiLevelStorage |
| changed | nvalchemi/data/level_storage.py:MultiLevelStorage.select | (self, idx: IndexType) -> MultiLevelStorage | (self, idx: IndexType, *, drop: Collection[str]=()) -> MultiLevelStorage |
| changed | nvalchemi/data/level_storage.py:SegmentedLevelStorage.clone | (self) -> SegmentedLevelStorage | (self, *, drop: Collection[str]=()) -> SegmentedLevelStorage |
| changed | nvalchemi/data/level_storage.py:SegmentedLevelStorage.select | (self, idx: IndexType) -> SegmentedLevelStorage | (self, idx: IndexType, *, drop: Collection[str]=()) -> SegmentedLevelStorage |
| changed | nvalchemi/data/level_storage.py:UniformLevelStorage.clone | (self) -> UniformLevelStorage | (self, *, drop: Collection[str]=()) -> UniformLevelStorage |
| changed | nvalchemi/data/level_storage.py:UniformLevelStorage.select | (self, idx: IndexType) -> UniformLevelStorage | (self, idx: IndexType, *, drop: Collection[str]=()) -> UniformLevelStorage |
| added | nvalchemi/data/level_storage.py:effective_dtype | — | (declared: str \| torch.dtype \| None) -> torch.dtype \| str \| None |
| added | nvalchemi/data/level_storage.py:resolve_device | — | (device: DeviceType \| None) -> torch.device |
| changed | nvalchemi/data/transforms/__init__.py:exports | __all__ = ['Compose'] ; from __future__ import annotations ; from nvalchemi.data.transforms.compose import Compose | __all__ = ['DEFAULT_EXTENSIVE_SYSTEM_KEYS', 'DEFAULT_INTENSIVE_SYSTEM_KEYS', 'Compose', 'make_supercell'] ; from __future__ import annotations ; from nvalchemi.data.transforms.compose import Compose ; from nvalchemi.data.transforms.supercell import DEFAULT_EXTENSIVE_SYSTEM_KEYS, DEFAULT_INTENSIVE_SYSTEM_KEYS, make_supercell |
| added | nvalchemi/data/transforms/supercell.py:make_supercell | — | (data: AtomicData, repeats: Sequence[int], *, extensive_keys: Collection[str]=DEFAULT_EXTENSIVE_SYSTEM_KEYS, intensive_keys: Collection[str]=DEFAULT_INTENSIVE_SYSTEM_KEYS, drop_keys: Collection[str]=()) -> AtomicData |
| changed | nvalchemi/distributed/__init__.py:exports | __all__ = ['AdapterRegistry', 'AdapterStatus', 'CompilePolicy', 'ForceStrategy', 'DistributedModel', 'DistributedPipelineModel', 'DistributionSpec', 'DomainConfig', 'DomainParallel', 'HookScope', 'JitAdapter', 'FunctionAdapter', 'GraphPadder', 'COOPadder', 'DensePadder', 'DenseBatchPadder', 'resolve_cap', 'MLIPSpec', 'MethodAdapter', 'OpAdapter', 'OutputKind', 'OutputSpec', 'ParticleHaloConfig', 'pin_fp32', 'PythonAdapter', 'Reduce', 'Scope', 'ShardedBatch', 'SpatialPartitioner', 'autograd_target', 'collective_device', 'current_dd_context', 'DistributedManager', 'PhysicsNeMoUninitializedDistributedManagerWarning', 'resolve_global_rank', 'resolve_world_size', 'neighbor_refresh_adapters', 'distributed_method', 'localize', 'refresh_neighbors', 'reshard_by_destination', 'scatter_to_owners', 'system_sum', 'to_local', 'trace_and_validate'] ; from __future__ import annotations ; from nvalchemi.distributed._core.particle_halo import ParticleHaloConfig as ParticleHaloConfig ; from nvalchemi.distributed._core.reshard import reshard_by_destination as reshard_by_destination ; from nvalchemi.distributed.config import DomainConfig as DomainConfig ; from nvalchemi.distributed.config import HookScope as HookScope ; from nvalchemi.distributed.domain_parallel import DomainParallel as DomainParallel ; from nvalchemi.distributed.partitioner import SpatialPartitioner as SpatialPartitioner ; from nvalchemi.distributed.sharded_batch import ShardedBatch as ShardedBatch ; from typing import TYPE_CHECKING, Any ; import torch | __all__ = ['AdapterRegistry', 'AdapterStatus', 'CompilePolicy', 'ForceStrategy', 'DistributedModel', 'DistributedPipelineModel', 'DistributionSpec', 'DomainConfig', 'DomainParallel', 'HookScope', 'JitAdapter', 'FunctionAdapter', 'GraphPadder', 'COOPadder', 'DensePadder', 'DenseBatchPadder', 'resolve_cap', 'MLIPSpec', 'MethodAdapter', 'OpAdapter', 'OutputKind', 'OutputSpec', 'ParticleHaloConfig', 'pin_fp32', 'PythonAdapter', 'Reduce', 'Scope', 'ShardedBatch', 'SpatialPartitioner', 'autograd_target', 'collective_device', 'collective_error_sync', 'CollectivePhase', 'current_dd_context', 'DistributedManager', 'PhysicsNeMoUninitializedDistributedManagerWarning', 'ProcessGroupContext', 'resolve_global_rank', 'resolve_world_size', 'neighbor_refresh_adapters', 'distributed_method', 'localize', 'refresh_neighbors', 'reshard_by_destination', 'scatter_to_owners', 'system_sum', 'to_local', 'trace_and_validate'] ; from __future__ import annotations ; from nvalchemi.distributed._core.particle_halo import ParticleHaloConfig as ParticleHaloConfig ; from nvalchemi.distributed._core.reshard import reshard_by_destination as reshard_by_destination ; from nvalchemi.distributed._runtime import CollectivePhase as CollectivePhase ; from nvalchemi.distributed._runtime import ProcessGroupContext as ProcessGroupContext ; from nvalchemi.distributed._runtime import collective_error_sync as collective_error_sync ; from nvalchemi.distributed.config import DomainConfig as DomainConfig ; from nvalchemi.distributed.config import HookScope as HookScope ; from nvalchemi.distributed.domain_parallel import DomainParallel as DomainParallel ; from nvalchemi.distributed.partitioner import SpatialPartitioner as SpatialPartitioner ; from nvalchemi.distributed.sharded_batch import ShardedBatch as ShardedBatch ; from typing import TYPE_CHECKING, Any ; import torch |
| added | nvalchemi/distributed/_runtime.py:CollectivePhase | — | class(Generic[_T]) |
| added | nvalchemi/distributed/_runtime.py:CollectivePhase.__init__ | — | (self) -> None |
| added | nvalchemi/distributed/_runtime.py:CollectivePhase.metadata | — | (self) -> _T \| None @ property ; (self, value: _T \| None) -> None @ metadata.setter |
| added | nvalchemi/distributed/_runtime.py:CollectivePhase.phase | — | (self) -> str @ property ; (self, value: str) -> None @ phase.setter |
| added | nvalchemi/distributed/_runtime.py:CollectivePhase.records | — | (self) -> tuple[_T \| None, ...] @ property |
| added | nvalchemi/distributed/_runtime.py:ProcessGroupContext | — | class() @ dataclass(frozen=True, init=False, slots=True) |
| added | nvalchemi/distributed/_runtime.py:ProcessGroupContext.__init__ | — | (self, process_group: dist.ProcessGroup, *, execution_device: torch.device \| str \| None=None) -> None |
| added | nvalchemi/distributed/_runtime.py:ProcessGroupContext.backend | — | str = required |
| added | nvalchemi/distributed/_runtime.py:ProcessGroupContext.collective_device | — | torch.device = required |
| added | nvalchemi/distributed/_runtime.py:ProcessGroupContext.communication_scope | — | (self) -> Iterator[None] @ contextlib.contextmanager |
| added | nvalchemi/distributed/_runtime.py:ProcessGroupContext.global_rank | — | (self, group_rank: int) -> int |
| added | nvalchemi/distributed/_runtime.py:ProcessGroupContext.process_group | — | dist.ProcessGroup = required |
| added | nvalchemi/distributed/_runtime.py:ProcessGroupContext.rank | — | int = required |
| added | nvalchemi/distributed/_runtime.py:ProcessGroupContext.world_size | — | int = required |
| added | nvalchemi/distributed/_runtime.py:collective_error_sync | — | (context: ProcessGroupContext, *, phase: str, error_type: type[Exception]=RuntimeError) -> Iterator[CollectivePhase[Any]] @ contextlib.contextmanager |
| changed | nvalchemi/dynamics/__init__.py:exports | __all__ = ['BaseDynamics', 'ConvergenceHook', 'DataSink', 'DemoDynamics', 'DistributedPipeline', 'DynamicsStage', 'FIRE', 'FIRE2', 'FIRE2VariableCell', 'FIREVariableCell', 'FusedStage', 'GPUBuffer', 'Hook', 'HostMemory', 'NPH', 'NPT', 'NVE', 'NVTLangevin', 'NVTNoseHoover', 'SizeAwareSampler', 'ZarrData', 'hooks', 'initialize_velocities', 'integrators', 'optimizers'] ; from __future__ import annotations ; from nvalchemi.dynamics import hooks, integrators, optimizers ; from nvalchemi.dynamics._ops.thermostat_utils import initialize_velocities ; from nvalchemi.dynamics.base import BaseDynamics, ConvergenceHook, DistributedPipeline, DynamicsStage, FusedStage, Hook ; from nvalchemi.dynamics.demo import DemoDynamics ; from nvalchemi.dynamics.integrators import NPH, NPT, NVE, NVTLangevin, NVTNoseHoover ; from nvalchemi.dynamics.optimizers import FIRE, FIRE2, FIRE2VariableCell, FIREVariableCell ; from nvalchemi.dynamics.sampler import SizeAwareSampler ; from nvalchemi.dynamics.sinks import DataSink, GPUBuffer, HostMemory, ZarrData | __all__ = ['BaseDynamics', 'ConvergenceHook', 'DataSink', 'DemoDynamics', 'DistributedPipeline', 'DynamicsStage', 'DynamicsStrategy', 'FIRE', 'FIRE2', 'FIRE2VariableCell', 'FIREVariableCell', 'FitPolicy', 'FusedStage', 'GPUBuffer', 'Hook', 'HostMemory', 'KB_EV', 'LBFGS', 'LBFGSVariableCell', 'ResizableSink', 'NPH', 'NPT', 'NVE', 'NVTLangevin', 'NVTNoseHoover', 'OrderedStructureSampler', 'SizeAwareSampler', 'StructureSource', 'WithinBudget', 'ZarrData', 'hooks', 'initialize_velocities', 'integrators', 'optimizers', 'requires_grad_ctx'] ; from __future__ import annotations ; from nvalchemi.dynamics import hooks, integrators, optimizers ; from nvalchemi.dynamics._ops.thermostat_utils import initialize_velocities ; from nvalchemi.dynamics._units import KB_EV ; from nvalchemi.dynamics.base import BaseDynamics, ConvergenceHook, DistributedPipeline, DynamicsStage, FusedStage, Hook, requires_grad_ctx ; from nvalchemi.dynamics.demo import DemoDynamics ; from nvalchemi.dynamics.integrators import NPH, NPT, NVE, NVTLangevin, NVTNoseHoover ; from nvalchemi.dynamics.optimizers import FIRE, FIRE2, LBFGS, FIRE2VariableCell, FIREVariableCell, LBFGSVariableCell ; from nvalchemi.dynamics.sampler import SizeAwareSampler ; from nvalchemi.dynamics.sinks import DataSink, GPUBuffer, HostMemory, ResizableSink, ZarrData ; from nvalchemi.dynamics.strategy import DynamicsStrategy ; from nvalchemi.dynamics.structure_sampler import FitPolicy, OrderedStructureSampler, StructureSource, WithinBudget |
| changed | nvalchemi/dynamics/_ops/cell_align.py:align_cell | (positions: torch.Tensor, cell: torch.Tensor, batch_idx: torch.Tensor \| None=None) -> None | (positions: torch.Tensor, cell: torch.Tensor, batch_idx: torch.Tensor \| None=None, *, transform: torch.Tensor \| None=None) -> torch.Tensor |
| added | nvalchemi/dynamics/_ops/cell_align.py:cell_alignment_offenders | — | (cell: torch.Tensor, atol: float=ALIGN_ATOL) -> torch.Tensor |
| changed | nvalchemi/dynamics/_ops/fire.py:fire2_step_coord_cell | (positions: torch.Tensor, velocities: torch.Tensor, forces: torch.Tensor, cell: torch.Tensor, cell_velocities: torch.Tensor, cell_force: torch.Tensor, batch_idx: torch.Tensor, alpha: torch.Tensor, dt: torch.Tensor, nsteps_inc: torch.Tensor, *, vf: torch.Tensor \| None=None, v_sumsq: torch.Tensor \| None=None, f_sumsq: torch.Tensor \| None=None, max_norm: torch.Tensor \| None=None, delaystep: int=60, dtgrow: float=1.05, dtshrink: float=0.75, alphashrink: float=0.985, alpha0: float=0.09, tmax: float=0.08, tmin: float=0.005, maxstep: float=0.1) -> None | (positions: torch.Tensor, velocities: torch.Tensor, forces: torch.Tensor, cell: torch.Tensor, cell_velocities: torch.Tensor, cell_force: torch.Tensor, batch_idx: torch.Tensor, alpha: torch.Tensor, dt: torch.Tensor, nsteps_inc: torch.Tensor, *, vf: torch.Tensor \| None=None, v_sumsq: torch.Tensor \| None=None, f_sumsq: torch.Tensor \| None=None, max_norm: torch.Tensor \| None=None, delaystep: int=60, dtgrow: float=1.05, dtshrink: float=0.75, alphashrink: float=0.985, alpha0: float=0.09, tmax: float=0.08, tmin: float=0.005, maxstep: float=0.1, cell_force_scale: float=1.0) -> None |
| added | nvalchemi/dynamics/_ops/lbfgs.py:lbfgs_step_coord | — | (positions: torch.Tensor, forces: torch.Tensor, state: LBFGSState, batch_idx: torch.Tensor, *, maxstep: float=0.2, curvature_eps: float \| None=None) -> None |
| added | nvalchemi/dynamics/_ops/lbfgs.py:lbfgs_step_coord_cell | — | (positions: torch.Tensor, cell: torch.Tensor, forces: torch.Tensor, stress: torch.Tensor, state: LBFGSState, cell_state: LBFGSCellState, batch_idx: torch.Tensor, *, maxstep: float=0.2, curvature_eps: float \| None=None) -> None |
| changed | nvalchemi/dynamics/base.py:BaseDynamics.__init__ | (self, model: BaseModelMixin, hooks: list[Hook] \| None=None, convergence_hook: Any=None, n_steps: int \| None=None, exit_status: int=1, **kwargs: Any) -> None | (self, model: BaseModelMixin, hooks: list[Hook] \| None=None, convergence_hook: Any=None, n_steps: int \| None=None, exit_status: int=1, by_group: bool=False, **kwargs: Any) -> None |
| added | nvalchemi/dynamics/base.py:BaseDynamics.active_graph_mask | — | (batch: Batch, exit_status: int) -> Bool[torch.Tensor, ' G'] \| None @ staticmethod |
| added | nvalchemi/dynamics/base.py:BaseDynamics.apply_per_system_params | — | (self, params: Mapping[str, torch.Tensor], batch: Batch) -> None |
| added | nvalchemi/dynamics/base.py:BaseDynamics.bookkeeping_keys | — | (self) -> frozenset[str] |
| added | nvalchemi/dynamics/base.py:BaseDynamics.check_initial_batch | — | (self, batch: Batch) -> None |
| changed | nvalchemi/dynamics/base.py:BaseDynamics.compute | (self, batch: Batch \| AtomsLike) -> ModelOutputs | (self, batch: Batch \| AtomsLike, active_graph_mask: Bool[torch.Tensor, 'B'] \| None=None) -> ModelOutputs |
| added | nvalchemi/dynamics/base.py:BaseDynamics.load_state_dict | — | (self, state: Mapping[str, Any]) -> None |
| added | nvalchemi/dynamics/base.py:BaseDynamics.redistribute_state | — | (self, walker_ids: torch.Tensor) -> None |
| added | nvalchemi/dynamics/base.py:BaseDynamics.required_input_keys | — | (self) -> frozenset[str] |
| added | nvalchemi/dynamics/base.py:BaseDynamics.samples_equilibrium | — | ClassVar[bool] = True |
| added | nvalchemi/dynamics/base.py:BaseDynamics.seed_offset | — | (self, offset: int) -> tuple[str, ...] |
| added | nvalchemi/dynamics/base.py:BaseDynamics.state_dict | — | (self) -> dict[str, Any] |
| changed | nvalchemi/dynamics/base.py:ConvergenceHook.__init__ | (self, criteria: _ConvergenceCriterion \| list[_ConvergenceCriterion] \| dict \| list[dict] \| None=None, source_status: int \| None=None, target_status: int \| None=None, frequency: int=1) -> None | (self, criteria: _ConvergenceCriterion \| list[_ConvergenceCriterion] \| dict \| list[dict] \| None=None, source_status: int \| None=None, target_status: int \| None=None, frequency: int=1, by_group: bool=False) -> None |
| added | nvalchemi/dynamics/base.py:ConvergenceHook.evaluate_mask | — | (self, batch: Batch) -> torch.Tensor |
| changed | nvalchemi/dynamics/base.py:ConvergenceHook.from_fmax | (cls, threshold: float=0.05, source_status: int \| None=None, target_status: int \| None=None, frequency: int=1) -> ConvergenceHook @ classmethod | (cls, threshold: float=0.05, source_status: int \| None=None, target_status: int \| None=None, frequency: int=1, by_group: bool=False) -> ConvergenceHook @ classmethod |
| changed | nvalchemi/dynamics/base.py:ConvergenceHook.from_forces | (cls, threshold: float, frequency: int=1, source_status: int \| None=None, target_status: int \| None=None) -> ConvergenceHook @ classmethod | (cls, threshold: float, frequency: int=1, source_status: int \| None=None, target_status: int \| None=None, by_group: bool=False) -> ConvergenceHook @ classmethod |
| added | nvalchemi/dynamics/base.py:ConvergenceHook.on_register | — | (self, workflow: object) -> None |
| changed | nvalchemi/dynamics/base.py:FusedStage.__init__ | (self, sub_stages: list[tuple[int, BaseDynamics]], *, entry_status: int=0, exit_status: int=-1, compile_step: bool=False, compile_kwargs: dict[str, Any] \| None=None, init_fn: Callable[[Batch], None] \| None=None, **kwargs: Any) -> None | (self, sub_stages: list[tuple[int, BaseDynamics]], *, entry_status: int=0, exit_status: int=-1, compile_step: bool=False, compile_kwargs: dict[str, Any] \| None=None, init_fn: Callable[[Batch], None] \| None=None, reprime_on_entry: set[int] \| None=None, **kwargs: Any) -> None |
| added | nvalchemi/dynamics/base.py:FusedStage.register_hook | — | (self, hook: Hook, stage: Enum \| None=None) -> None |
| added | nvalchemi/dynamics/base.py:requires_grad_ctx | — | (*tensors: torch.Tensor, requires_grad: bool=True) -> Iterator[None] @ contextmanager |
| changed | nvalchemi/dynamics/hooks/__init__.py:exports | __all__ = ['AlignCellHook', 'ConvergedSnapshotHook', 'EnergyDriftMonitorHook', 'FreezeAtomsHook', 'LoggingHook', 'MaxForceClampHook', 'NaNDetectorHook', 'SnapshotHook', 'StageTimingHook', 'TorchProfilerHook'] ; from __future__ import annotations ; from nvalchemi.dynamics.hooks.cell_align import AlignCellHook ; from nvalchemi.dynamics.hooks.freeze import FreezeAtomsHook ; from nvalchemi.dynamics.hooks.logging import LoggingHook ; from nvalchemi.dynamics.hooks.monitors import EnergyDriftMonitorHook ; from nvalchemi.dynamics.hooks.safety import MaxForceClampHook, NaNDetectorHook ; from nvalchemi.dynamics.hooks.snapshot import ConvergedSnapshotHook, SnapshotHook ; from nvalchemi.hooks.physicsnemo_profiling import TorchProfilerHook ; from nvalchemi.hooks.stage_timing import StageTimingHook | __all__ = ['AlignCellHook', 'ConvergedSnapshotHook', 'EnergyDriftMonitorHook', 'FreezeAtomsHook', 'LoggingHook', 'MaxForceClampHook', 'NaNDetectorHook', 'PairSwapHook', 'SnapshotHook', 'StabilityMetrics', 'StabilityMonitor', 'StageTimingHook', 'TorchProfilerHook', 'apply_pair_swaps', 'even_odd_pairs', 'kinetic_energy_per_graph', 'nonfinite_graph_mask', 'total_momentum'] ; from __future__ import annotations ; from nvalchemi.dynamics.hooks._utils import kinetic_energy_per_graph ; from nvalchemi.dynamics.hooks.cell_align import AlignCellHook ; from nvalchemi.dynamics.hooks.freeze import FreezeAtomsHook ; from nvalchemi.dynamics.hooks.logging import LoggingHook ; from nvalchemi.dynamics.hooks.monitors import EnergyDriftMonitorHook, StabilityMetrics, StabilityMonitor, total_momentum ; from nvalchemi.dynamics.hooks.safety import MaxForceClampHook, NaNDetectorHook, nonfinite_graph_mask ; from nvalchemi.dynamics.hooks.snapshot import ConvergedSnapshotHook, SnapshotHook ; from nvalchemi.dynamics.hooks.swap import PairSwapHook, apply_pair_swaps, even_odd_pairs ; from nvalchemi.hooks.physicsnemo_profiling import TorchProfilerHook ; from nvalchemi.hooks.stage_timing import StageTimingHook |
| changed | nvalchemi/dynamics/hooks/freeze.py:FreezeAtomsHook.__init__ | (self, frequency: int=1, freeze_category: int=AtomCategory.SPECIAL.value, zero_forces: bool=True, stage: Enum=DynamicsStage.BEFORE_PRE_UPDATE) -> None | (self, frequency: int=1, freeze_category: int=AtomCategory.SPECIAL.value, mask_key: str \| None=None, zero_forces: bool=True, zero_velocities: bool=True, stage: Enum=DynamicsStage.BEFORE_PRE_UPDATE) -> None |
| changed | nvalchemi/dynamics/hooks/logging.py:LoggingHook.__init__ | (self, backend: LogBackend, frequency: int=1, log_path: str \| Path \| None=None, custom_scalars: dict[str, Callable[[DynamicsContext], float \| torch.Tensor]] \| None=None, writer_fn: Callable[[int, list[dict[str, float]]], None] \| None=None, stage: Enum=DynamicsStage.AFTER_STEP) -> None | (self, backend: LogBackend, frequency: int=1, log_path: str \| Path \| None=None, custom_scalars: dict[str, Callable[[DynamicsContext], float \| torch.Tensor]] \| None=None, writer_fn: Callable[[int, list[dict[str, float]]], None] \| None=None, stage: Enum=DynamicsStage.AFTER_STEP, *, by_group: bool=False) -> None |
| added | nvalchemi/dynamics/hooks/monitors.py:StabilityMetrics | — | class(MeasurementRecord) |
| added | nvalchemi/dynamics/hooks/monitors.py:StabilityMetrics.aggregate | — | Literal['max', 'mean'] = 'max' |
| added | nvalchemi/dynamics/hooks/monitors.py:StabilityMetrics.energy_drift_per_atom | — | float = required |
| added | nvalchemi/dynamics/hooks/monitors.py:StabilityMetrics.energy_drift_per_atom_per_ns | — | float \| None = required |
| added | nvalchemi/dynamics/hooks/monitors.py:StabilityMetrics.energy_drift_per_atom_per_step | — | float = required |
| added | nvalchemi/dynamics/hooks/monitors.py:StabilityMetrics.energy_fluctuation_per_atom | — | float \| None = None |
| added | nvalchemi/dynamics/hooks/monitors.py:StabilityMetrics.first_divergence_step | — | int \| None = None |
| added | nvalchemi/dynamics/hooks/monitors.py:StabilityMetrics.first_step | — | int = required |
| added | nvalchemi/dynamics/hooks/monitors.py:StabilityMetrics.last_step | — | int = required |
| added | nvalchemi/dynamics/hooks/monitors.py:StabilityMetrics.max_energy_excursion_per_atom | — | float \| None = None |
| added | nvalchemi/dynamics/hooks/monitors.py:StabilityMetrics.max_momentum_drift | — | float = required |
| added | nvalchemi/dynamics/hooks/monitors.py:StabilityMetrics.num_samples | — | int = required |
| added | nvalchemi/dynamics/hooks/monitors.py:StabilityMetrics.timestep_fs | — | float \| None = required |
| added | nvalchemi/dynamics/hooks/monitors.py:StabilityMonitor | — | class() |
| added | nvalchemi/dynamics/hooks/monitors.py:StabilityMonitor.__init__ | — | (self, *, frequency: int=1, stage: Enum=DynamicsStage.AFTER_STEP, timestep_fs: float \| None=None, include_kinetic: bool=True, warmup_steps: int=0, divergence: Callable[[Batch], Bool[torch.Tensor, 'G']] \| None=None, aggregate: Literal['max', 'mean']='max', stop_on_composition_change: bool=True) -> None |
| added | nvalchemi/dynamics/hooks/monitors.py:StabilityMonitor.metrics | — | (self) -> StabilityMetrics |
| added | nvalchemi/dynamics/hooks/monitors.py:total_momentum | — | (batch: Batch) -> Float[torch.Tensor, 'G 3'] |
| added | nvalchemi/dynamics/hooks/safety.py:nonfinite_graph_mask | — | (batch: Batch, keys: Iterable[str]=('positions', 'forces')) -> Bool[torch.Tensor, 'G'] |
| added | nvalchemi/dynamics/hooks/swap.py:PairSwapHook | — | class() |
| added | nvalchemi/dynamics/hooks/swap.py:PairSwapHook.__init__ | — | (self, accept_fn: Callable[[Batch, torch.Tensor, torch.Tensor], torch.Tensor], params_fn: Callable[[torch.Tensor], Mapping[str, torch.Tensor]] \| None=None, *, slot_field: str, n_slots: int, pairing: Literal['even_odd'] \| Callable[[int, int], list[tuple[int, int]]]='even_odd', frequency: int=100, on_swap: Callable[[Batch], None] \| None=None, record_fn: Callable[[list[tuple[int, int]], torch.Tensor], None] \| None=None) -> None |
| added | nvalchemi/dynamics/hooks/swap.py:PairSwapHook.attempt_segment | — | (self, batch: Batch, segment: int) -> None |
| added | nvalchemi/dynamics/hooks/swap.py:PairSwapHook.load_state_dict | — | (self, state: Mapping[str, Any]) -> None |
| added | nvalchemi/dynamics/hooks/swap.py:PairSwapHook.on_register | — | (self, workflow: Any) -> None |
| added | nvalchemi/dynamics/hooks/swap.py:PairSwapHook.state_dict | — | (self) -> Mapping[str, Any] |
| added | nvalchemi/dynamics/hooks/swap.py:apply_pair_swaps | — | (slots: torch.Tensor, rows_i: torch.Tensor, rows_j: torch.Tensor, accepted: torch.Tensor \| None=None) -> torch.Tensor |
| added | nvalchemi/dynamics/hooks/swap.py:even_odd_pairs | — | (segment: int, n_slots: int) -> list[tuple[int, int]] |
| added | nvalchemi/dynamics/integrators/nvt_langevin.py:NVTLangevin.apply_per_system_params | — | (self, params: Mapping[str, torch.Tensor], batch: Batch) -> None |
| added | nvalchemi/dynamics/integrators/nvt_langevin.py:NVTLangevin.random_seed | — | (self) -> int @ property ; (self, value: int) -> None @ random_seed.setter |
| added | nvalchemi/dynamics/integrators/nvt_nose_hoover.py:NVTNoseHoover.apply_per_system_params | — | (self, params: Mapping[str, torch.Tensor], batch: Batch) -> None |
| added | nvalchemi/dynamics/mep/__init__.py:exports | — | __all__ = ['ClimbingImageConfig', 'ConstantSpringConfig', 'IDPPModel', 'NEB', 'NEBMethod', 'PositionAlignment', 'SpringConfig', 'SpringContext', 'TorchNEBMethod', 'align_batch_positions', 'interpolate_paths', 'prepare_idpp_targets', 'validate_paths'] ; from nvalchemi.dynamics.mep._alignment import PositionAlignment, align_batch_positions ; from nvalchemi.dynamics.mep.idpp import IDPPModel, prepare_idpp_targets ; from nvalchemi.dynamics.mep.interpolate import interpolate_paths ; from nvalchemi.dynamics.mep.neb import NEB, ClimbingImageConfig ; from nvalchemi.dynamics.mep.neb_configs import ConstantSpringConfig, NEBMethod, SpringConfig, SpringContext, TorchNEBMethod ; from nvalchemi.dynamics.mep.validate import validate_paths |
| added | nvalchemi/dynamics/mep/_alignment.py:PositionAlignment | — | class(NamedTuple) |
| added | nvalchemi/dynamics/mep/_alignment.py:PositionAlignment.positions | — | Tensor = required |
| added | nvalchemi/dynamics/mep/_alignment.py:PositionAlignment.rotation | — | Tensor = required |
| added | nvalchemi/dynamics/mep/_alignment.py:PositionAlignment.translation | — | Tensor = required |
| added | nvalchemi/dynamics/mep/_alignment.py:align_batch_positions | — | (reference_positions: Tensor, mobile_positions: Tensor, batch_idx: Tensor, num_atoms_per_graph: Tensor, cell: Tensor \| None=None, pbc: Tensor \| None=None, fit_mask: Tensor \| None=None, *, prepared_mic: PreparedMIC \| None=None, skip_extra_checks: bool=False) -> PositionAlignment |
| added | nvalchemi/dynamics/mep/_geometry.py:PreparedMIC | — | class() @ dataclass(frozen=True, slots=True) |
| added | nvalchemi/dynamics/mep/_geometry.py:PreparedMIC.candidate_count | — | Tensor = required |
| added | nvalchemi/dynamics/mep/_geometry.py:PreparedMIC.candidate_shifts | — | Tensor = required |
| added | nvalchemi/dynamics/mep/_geometry.py:PreparedMIC.cartesian_to_fractional | — | Tensor = required |
| added | nvalchemi/dynamics/mep/_geometry.py:PreparedMIC.execution_mode | — | int = required |
| added | nvalchemi/dynamics/mep/_geometry.py:PreparedMIC.max_candidate_count | — | int = required |
| added | nvalchemi/dynamics/mep/_geometry.py:PreparedMIC.mode | — | Tensor = required |
| added | nvalchemi/dynamics/mep/_geometry.py:PreparedMIC.periodic_basis | — | Tensor = required |
| added | nvalchemi/dynamics/mep/_geometry.py:PreparedMIC.rank | — | Tensor = required |
| added | nvalchemi/dynamics/mep/_geometry.py:minimum_image_displacement | — | (displacement: Tensor, graph_idx: Tensor, cell: Tensor \| None=None, pbc: Tensor \| None=None, *, prepared: PreparedMIC \| None=None) -> Tensor |
| added | nvalchemi/dynamics/mep/_geometry.py:prepare_batch_mic | — | (batch: Batch, cell: Tensor, pbc: Tensor) -> PreparedMIC |
| added | nvalchemi/dynamics/mep/_geometry.py:prepare_mic | — | (cell: Tensor, pbc: Tensor) -> PreparedMIC |
| added | nvalchemi/dynamics/mep/hooks/__init__.py:exports | — | __all__ = ['ClimbingImageSelectionHook', 'NEBForceHook', 'PathDiagnostics', 'PathDiagnosticsHook', 'PathEnergyStats', 'PathEnergyStatsHook'] ; from nvalchemi.dynamics.mep.hooks.climbing_image_selection import ClimbingImageSelectionHook ; from nvalchemi.dynamics.mep.hooks.neb_force import NEBForceHook ; from nvalchemi.dynamics.mep.hooks.path_diagnostics import PathDiagnostics, PathDiagnosticsHook ; from nvalchemi.dynamics.mep.hooks.path_energy_stats import PathEnergyStats, PathEnergyStatsHook |
| added | nvalchemi/dynamics/mep/hooks/climbing_image_selection.py:ClimbingImageSelectionHook | — | class() |
| added | nvalchemi/dynamics/mep/hooks/climbing_image_selection.py:ClimbingImageSelectionHook.__init__ | — | (self, *, energy_stats_hook: PathEnergyStatsHook, selection: Literal['fixed', 'dynamic']='fixed', status_code: int \| None=None, frequency: int=1) -> None |
| added | nvalchemi/dynamics/mep/hooks/climbing_image_selection.py:ClimbingImageSelectionHook.on_register | — | (self, workflow: object) -> None |
| added | nvalchemi/dynamics/mep/hooks/neb_force.py:NEBForceHook | — | class() |
| added | nvalchemi/dynamics/mep/hooks/neb_force.py:NEBForceHook.__init__ | — | (self, *, energy_stats_hook: PathEnergyStatsHook, spring: float \| SpringConfig=0.1, method: str \| NEBMethod \| TorchNEBMethod='improved_tangent', endpoint_mode: Literal['fixed', 'relaxed']='fixed', fixed_atom_indices: Mapping[int, Sequence[int]] \| None=None) -> None |
| added | nvalchemi/dynamics/mep/hooks/neb_force.py:NEBForceHook.on_register | — | (self, workflow: object) -> None |
| added | nvalchemi/dynamics/mep/hooks/path_diagnostics.py:PathDiagnostics | — | class() @ dataclass(frozen=True) |
| added | nvalchemi/dynamics/mep/hooks/path_diagnostics.py:PathDiagnostics.energy_barrier | — | Tensor = required |
| added | nvalchemi/dynamics/mep/hooks/path_diagnostics.py:PathDiagnostics.fmax | — | Tensor = required |
| added | nvalchemi/dynamics/mep/hooks/path_diagnostics.py:PathDiagnostics.highest_interior_image_idx | — | Tensor = required |
| added | nvalchemi/dynamics/mep/hooks/path_diagnostics.py:PathDiagnostics.path_length | — | Tensor = required |
| added | nvalchemi/dynamics/mep/hooks/path_diagnostics.py:PathDiagnosticsHook | — | class() |
| added | nvalchemi/dynamics/mep/hooks/path_diagnostics.py:PathDiagnosticsHook.__init__ | — | (self, *, energy_stats_hook: PathEnergyStatsHook, frequency: int=1) -> None |
| added | nvalchemi/dynamics/mep/hooks/path_diagnostics.py:PathDiagnosticsHook.get_diagnostics | — | (self) -> PathDiagnostics |
| added | nvalchemi/dynamics/mep/hooks/path_diagnostics.py:PathDiagnosticsHook.on_register | — | (self, workflow: object) -> None |
| added | nvalchemi/dynamics/mep/hooks/path_energy_stats.py:PathEnergyStats | — | class() @ dataclass(frozen=True) |
| added | nvalchemi/dynamics/mep/hooks/path_energy_stats.py:PathEnergyStats.endpoint_reference_energy | — | Tensor = required |
| added | nvalchemi/dynamics/mep/hooks/path_energy_stats.py:PathEnergyStats.highest_interior_energy | — | Tensor = required |
| added | nvalchemi/dynamics/mep/hooks/path_energy_stats.py:PathEnergyStats.highest_interior_image_idx | — | Tensor = required |
| added | nvalchemi/dynamics/mep/hooks/path_energy_stats.py:PathEnergyStatsHook | — | class() |
| added | nvalchemi/dynamics/mep/hooks/path_energy_stats.py:PathEnergyStatsHook.__init__ | — | (self) -> None |
| added | nvalchemi/dynamics/mep/hooks/path_energy_stats.py:PathEnergyStatsHook.get_stats | — | (self, ctx: DynamicsContext) -> PathEnergyStats |
| added | nvalchemi/dynamics/mep/idpp.py:IDPPModel | — | class(nn.Module, BaseModelMixin) |
| added | nvalchemi/dynamics/mep/idpp.py:IDPPModel.__init__ | — | (self) -> None |
| added | nvalchemi/dynamics/mep/idpp.py:IDPPModel.compute_embeddings | — | (self, data: AtomicData \| Batch, **kwargs: Any) -> AtomicData \| Batch |
| added | nvalchemi/dynamics/mep/idpp.py:IDPPModel.direct_derivative_keys | — | (self) -> set[str] |
| added | nvalchemi/dynamics/mep/idpp.py:IDPPModel.embedding_shapes | — | (self) -> dict[str, tuple[int, ...]] @ property |
| added | nvalchemi/dynamics/mep/idpp.py:IDPPModel.forward | — | (self, data: Batch, **kwargs: Any) -> ModelOutputs |
| added | nvalchemi/dynamics/mep/idpp.py:prepare_idpp_targets | — | (paths: Batch) -> Batch |
| added | nvalchemi/dynamics/mep/interpolate.py:interpolate_paths | — | (initial: Batch, final: Batch, num_images: int \| Sequence[int], remove_translation_and_rotation: bool=False, fit_mask: Tensor \| None=None) -> Batch |
| added | nvalchemi/dynamics/mep/neb.py:ClimbingImageConfig | — | class(BaseModel) |
| added | nvalchemi/dynamics/mep/neb.py:ClimbingImageConfig.max_climbing_steps | — | PositiveInt \| None = Field(default=None, description='Per-path step budget for climbing-image optimization.') |
| added | nvalchemi/dynamics/mep/neb.py:ClimbingImageConfig.max_regular_steps | — | PositiveInt \| None = Field(default=None, description="Per-path step budget for regular NEB. When reached, the path enters the climbing-image stage even if the regular-stage convergence criterion has not been satisfied. None waits for regular-stage convergence, subject to the overall NEB.n_steps limit. Only valid with mode='after_regular'. ") |
| added | nvalchemi/dynamics/mep/neb.py:ClimbingImageConfig.mode | — | Literal['after_regular', 'immediate'] = Field(default='after_regular', description='Activate climbing forces after regular NEB convergence or from the initial evaluation.') |
| added | nvalchemi/dynamics/mep/neb.py:ClimbingImageConfig.regular_fmax | — | float \| None = Field(default=None, gt=0, description="Force threshold for entering the climbing stage; None uses the final NEB fmax. Only valid with mode='after_regular'.") |
| added | nvalchemi/dynamics/mep/neb.py:ClimbingImageConfig.selection | — | Literal['fixed', 'dynamic'] = Field(default='fixed', description='Keep the initially selected climbing image or reselect the highest-energy image after every evaluation.') |
| added | nvalchemi/dynamics/mep/neb.py:NEB | — | class(DynamicsStrategy) |
| added | nvalchemi/dynamics/mep/neb.py:NEB.build_engine | — | (self) -> FusedStage |
| added | nvalchemi/dynamics/mep/neb.py:NEB.climbing | — | ClimbingImageConfig \| None = Field(default=None, description='Climbing-image configuration; None runs regular NEB only.') |
| added | nvalchemi/dynamics/mep/neb.py:NEB.compile | — | bool = Field(default=False, strict=True, description='Compile the fused NEB step with torch.compile.') |
| added | nvalchemi/dynamics/mep/neb.py:NEB.compile_kwargs | — | dict[str, Any] = Field(default_factory=dict, description='Keyword arguments forwarded to torch.compile.') |
| added | nvalchemi/dynamics/mep/neb.py:NEB.convergence_hook | — | ConvergenceHook \| None = Field(default=None, exclude=True, description='Final-stage convergence override; takes precedence over fmax.') |
| added | nvalchemi/dynamics/mep/neb.py:NEB.diagnostics_frequency | — | PositiveInt = Field(default=1, description='Step frequency for computing and logging per-path diagnostics.') |
| added | nvalchemi/dynamics/mep/neb.py:NEB.diagnostics_log_path | — | Path \| None = Field(default=None, description='CSV output path for per-path diagnostics. None disables both path diagnostics and their logging hook.') |
| added | nvalchemi/dynamics/mep/neb.py:NEB.endpoint_mode | — | Literal['fixed', 'relaxed'] = Field(default='fixed', description='Whether path endpoint coordinates remain fixed or are relaxed.') |
| added | nvalchemi/dynamics/mep/neb.py:NEB.fixed_atom_indices | — | dict[int, tuple[int, ...]] \| None = Field(default=None, description='Mapping from path indices to atom indices held fixed in every image of that path.') |
| added | nvalchemi/dynamics/mep/neb.py:NEB.fmax | — | float = Field(default=0.05, gt=0, description='Force threshold for convergence of the final or only stage; ignored when convergence_hook is provided.') |
| added | nvalchemi/dynamics/mep/neb.py:NEB.from_spec_dict | — | (cls, spec: Mapping[str, Any], *, model: BaseModelMixin, extra_hooks: Sequence[Hook] \| None=None) -> Self @ classmethod |
| added | nvalchemi/dynamics/mep/neb.py:NEB.method | — | str \| NEBMethod \| TorchNEBMethod = Field(default='improved_tangent', description="Method for computing NEB forces: 'improved_tangent', custom Warp equations, or a Torch callable. 'improved_tangent' is equivalent to NEBMethod() without overrides.") |
| added | nvalchemi/dynamics/mep/neb.py:NEB.neighbor_hooks | — | list[Hook] \| None = Field(default=None, exclude=True, description='Neighbor-list hooks for the fused NEB engine. None uses hooks generated by model.make_neighbor_hooks().') |
| added | nvalchemi/dynamics/mep/neb.py:NEB.optimizer | — | SerializableClass = Field(default=FIRE2, frozen=True, description='Optimizer class used by internal stages. Must subclass BaseDynamics and support fixed-cell, group-aware updates through the FusedStage contract. Cannot be reassigned after construction.') |
| added | nvalchemi/dynamics/mep/neb.py:NEB.optimizer_kwargs | — | dict[str, Any] = Field(default_factory=dict, description='Keyword arguments forwarded to each internal optimizer stage. When using FIRE2, NEB-tuned defaults are applied to unspecified parameters.') |
| added | nvalchemi/dynamics/mep/neb.py:NEB.regular_convergence_hook | — | ConvergenceHook \| None = Field(default=None, exclude=True, description='Preliminary-stage convergence override; takes precedence over climbing.regular_fmax and fmax.') |
| added | nvalchemi/dynamics/mep/neb.py:NEB.spring | — | float \| SpringConfig = Field(default=0.1, description='Spring configuration used by the shared NEB force hook.') |
| added | nvalchemi/dynamics/mep/neb.py:NEB.to_spec_dict | — | (self) -> dict[str, Any] |
| added | nvalchemi/dynamics/mep/neb_configs.py:ConstantSpringConfig | — | class() @ dataclass(frozen=True, slots=True) |
| added | nvalchemi/dynamics/mep/neb_configs.py:ConstantSpringConfig.refresh | — | Literal[DynamicsStage.ON_ADMISSION] = field(default=DynamicsStage.ON_ADMISSION, init=False) |
| added | nvalchemi/dynamics/mep/neb_configs.py:ConstantSpringConfig.resolve | — | (self, context: SpringContext) -> Tensor |
| added | nvalchemi/dynamics/mep/neb_configs.py:ConstantSpringConfig.value | — | float = required |
| added | nvalchemi/dynamics/mep/neb_configs.py:NEBMethod | — | class() @ dataclass(frozen=True, slots=True) |
| added | nvalchemi/dynamics/mep/neb_configs.py:NEBMethod.climbing_force_fn | — | wp.Function = climbing_image_effective_force |
| added | nvalchemi/dynamics/mep/neb_configs.py:NEBMethod.effective_force_fn | — | wp.Function = neb_effective_force |
| added | nvalchemi/dynamics/mep/neb_configs.py:NEBMethod.from_key | — | (cls, key: str) -> Self @ classmethod |
| added | nvalchemi/dynamics/mep/neb_configs.py:NEBMethod.tangent_weights_fn | — | wp.Function = improved_tangent_weights |
| added | nvalchemi/dynamics/mep/neb_configs.py:NEBMethod.to_key | — | (self) -> str |
| added | nvalchemi/dynamics/mep/neb_configs.py:SpringConfig | — | class(Protocol) @ runtime_checkable |
| added | nvalchemi/dynamics/mep/neb_configs.py:SpringConfig.refresh | — | Literal[DynamicsStage.ON_ADMISSION, DynamicsStage.AFTER_COMPUTE] = required |
| added | nvalchemi/dynamics/mep/neb_configs.py:SpringConfig.resolve | — | (self, context: SpringContext) -> Tensor |
| added | nvalchemi/dynamics/mep/neb_configs.py:SpringContext | — | class() @ dataclass(frozen=True, slots=True) |
| added | nvalchemi/dynamics/mep/neb_configs.py:SpringContext.cell | — | Tensor = required |
| added | nvalchemi/dynamics/mep/neb_configs.py:SpringContext.energies | — | Tensor \| None = required |
| added | nvalchemi/dynamics/mep/neb_configs.py:SpringContext.image_ptr | — | Tensor = required |
| added | nvalchemi/dynamics/mep/neb_configs.py:SpringContext.layout | — | GroupLayout = required |
| added | nvalchemi/dynamics/mep/neb_configs.py:SpringContext.num_links | — | (self) -> int @ property |
| added | nvalchemi/dynamics/mep/neb_configs.py:SpringContext.pbc | — | Tensor = required |
| added | nvalchemi/dynamics/mep/neb_configs.py:SpringContext.physical_forces | — | Tensor \| None = required |
| added | nvalchemi/dynamics/mep/neb_configs.py:SpringContext.positions | — | Tensor = required |
| added | nvalchemi/dynamics/mep/neb_configs.py:SpringContext.step_count | — | int = required |
| added | nvalchemi/dynamics/mep/neb_configs.py:TorchNEBMethod | — | class(Protocol) @ runtime_checkable |
| added | nvalchemi/dynamics/mep/neb_ops/__init__.py:exports | — | __all__ = ['neb_forces', 'prepare_neb_method_key'] ; from __future__ import annotations ; from nvalchemi.dynamics.mep.neb_ops.methods import prepare_neb_method_key ; from nvalchemi.dynamics.mep.neb_ops.torch_ops import neb_forces |
| added | nvalchemi/dynamics/mep/neb_ops/equations.py:climbing_image_effective_force | — | (physical_force: Any, tangent: Any, force_dot_tangent: Any) -> unknown @ wp.func |
| added | nvalchemi/dynamics/mep/neb_ops/equations.py:improved_tangent_weights | — | (energy_prev: Any, energy_curr: Any, energy_next: Any) -> unknown @ wp.func |
| added | nvalchemi/dynamics/mep/neb_ops/equations.py:neb_effective_force | — | (physical_force: Any, tangent: Any, force_dot_tangent: Any, k_plus: Any, k_minus: Any, norm_d_plus: Any, norm_d_minus: Any, energy_prev: Any, energy_curr: Any, energy_next: Any, path_energy_ref: Any, path_energy_max: Any) -> unknown @ wp.func |
| added | nvalchemi/dynamics/mep/neb_ops/equations.py:neb_effective_force_from_gram_stats | — | (physical_force: Any, tangent: Any, d_plus: Any, d_minus: Any, force_dot_tangent: Any, dplus_dot_tangent: Any, dminus_dot_tangent: Any, norm_d_plus: Any, norm_d_minus: Any, dplus_dot_dminus: Any, force_dot_dplus: Any, force_dot_dminus: Any, force_squared_norm: Any, k_plus: Any, k_minus: Any, energy_prev: Any, energy_curr: Any, energy_next: Any, path_energy_ref: Any, path_energy_max: Any) -> unknown @ wp.func |
| added | nvalchemi/dynamics/mep/neb_ops/kernels.py:build_gram_stats_neb_kernel | — | (tangent_weights_fn: wp.Function, effective_force_fn: wp.Function, climbing_force_fn: wp.Function) -> wp.Kernel |
| added | nvalchemi/dynamics/mep/neb_ops/kernels.py:build_stored_tangent_neb_kernel | — | (tangent_weights_fn: wp.Function, effective_force_fn: wp.Function, climbing_force_fn: wp.Function) -> wp.Kernel |
| added | nvalchemi/dynamics/mep/neb_ops/launchers.py:launch_neb_forces_kernel | — | (method: str, args: list[object], *, scalar_dtype: object, num_images: int) -> None |
| added | nvalchemi/dynamics/mep/neb_ops/methods.py:prepare_neb_method_key | — | (tangent_fn: wp.Function, force_fn: wp.Function, climbing_force_fn: wp.Function) -> str |
| added | nvalchemi/dynamics/mep/neb_ops/methods.py:resolve_neb_method | — | (method: str) -> _StoredTangentMethod \| _GramStatsMethod @ cache |
| added | nvalchemi/dynamics/mep/neb_ops/torch_ops.py:gram_stats_neb_forces | — | (positions: torch.Tensor, physical_forces: torch.Tensor, image_energies: torch.Tensor, image_ptr: torch.Tensor, path_ptr: torch.Tensor, image_path_idx: torch.Tensor, spring_constants: torch.Tensor, image_force_mode: torch.Tensor, path_energy_ref: torch.Tensor, path_energy_max: torch.Tensor, mic_mode: torch.Tensor, periodic_basis: torch.Tensor, cartesian_to_fractional: torch.Tensor, mic_candidate_count: torch.Tensor, candidate_shifts: torch.Tensor, *, method: str, effective_forces: torch.Tensor \| None=None, link_lengths: torch.Tensor \| None=None) -> tuple[torch.Tensor, torch.Tensor] |
| added | nvalchemi/dynamics/mep/neb_ops/torch_ops.py:neb_forces | — | (positions: torch.Tensor, physical_forces: torch.Tensor, image_energies: torch.Tensor, image_ptr: torch.Tensor, path_ptr: torch.Tensor, image_path_idx: torch.Tensor, spring_constants: torch.Tensor, image_force_mode: torch.Tensor, path_energy_ref: torch.Tensor, path_energy_max: torch.Tensor, mic_mode: torch.Tensor, periodic_basis: torch.Tensor, cartesian_to_fractional: torch.Tensor, mic_candidate_count: torch.Tensor, candidate_shifts: torch.Tensor, *, method: str=DEFAULT_NEB_METHOD_KEY, vector_scratch: torch.Tensor \| None=None, effective_forces: torch.Tensor \| None=None, link_lengths: torch.Tensor \| None=None) -> tuple[torch.Tensor, torch.Tensor] |
| added | nvalchemi/dynamics/mep/neb_ops/torch_ops.py:stored_tangent_neb_forces | — | (positions: torch.Tensor, physical_forces: torch.Tensor, image_energies: torch.Tensor, image_ptr: torch.Tensor, path_ptr: torch.Tensor, image_path_idx: torch.Tensor, spring_constants: torch.Tensor, image_force_mode: torch.Tensor, path_energy_ref: torch.Tensor, path_energy_max: torch.Tensor, mic_mode: torch.Tensor, periodic_basis: torch.Tensor, cartesian_to_fractional: torch.Tensor, mic_candidate_count: torch.Tensor, candidate_shifts: torch.Tensor, *, method: str, tangent_buffer: torch.Tensor \| None=None, effective_forces: torch.Tensor \| None=None, link_lengths: torch.Tensor \| None=None) -> tuple[torch.Tensor, torch.Tensor] |
| added | nvalchemi/dynamics/mep/path_ops/__init__.py:exports | — | from __future__ import annotations |
| added | nvalchemi/dynamics/mep/path_ops/torch_paths.py:path_energy_stats | — | (image_energies: torch.Tensor, path_ptr: torch.Tensor, endpoint_reference_energy: torch.Tensor, highest_interior_energy: torch.Tensor, highest_interior_image_idx: torch.Tensor) -> None |
| added | nvalchemi/dynamics/mep/validate.py:validate_paths | — | (batch: Batch) -> None |
| changed | nvalchemi/dynamics/optimizers/__init__.py:exports | __all__ = ['FIRE', 'FIREVariableCell', 'FIRE2', 'FIRE2VariableCell'] ; from nvalchemi.dynamics.optimizers.fire import FIRE, FIREVariableCell ; from nvalchemi.dynamics.optimizers.fire2 import FIRE2, FIRE2VariableCell | __all__ = ['FIRE', 'FIREVariableCell', 'FIRE2', 'FIRE2VariableCell', 'LBFGS', 'LBFGSVariableCell'] ; from nvalchemi.dynamics.optimizers.fire import FIRE, FIREVariableCell ; from nvalchemi.dynamics.optimizers.fire2 import FIRE2, FIRE2VariableCell ; from nvalchemi.dynamics.optimizers.lbfgs import LBFGS, LBFGSVariableCell |
| added | nvalchemi/dynamics/optimizers/fire.py:FIRE.samples_equilibrium | — | ClassVar[bool] = False |
| added | nvalchemi/dynamics/optimizers/fire.py:FIREVariableCell.samples_equilibrium | — | ClassVar[bool] = False |
| added | nvalchemi/dynamics/optimizers/fire2.py:FIRE2.samples_equilibrium | — | ClassVar[bool] = False |
| changed | nvalchemi/dynamics/optimizers/fire2.py:FIRE2VariableCell.__init__ | (self, model: BaseModelMixin, dt: float \| torch.Tensor, delaystep: int=_FIRE2_DEFAULTS['delaystep'], dtgrow: float=_FIRE2_DEFAULTS['dtgrow'], dtshrink: float=_FIRE2_DEFAULTS['dtshrink'], alphashrink: float=_FIRE2_DEFAULTS['alphashrink'], alpha0: float=_FIRE2_DEFAULTS['alpha0'], tmax: float=_FIRE2_DEFAULTS['tmax'], tmin: float=_FIRE2_DEFAULTS['tmin'], maxstep: float=_FIRE2_DEFAULTS['maxstep'], n_steps: int \| None=None, hooks: list[Hook] \| None=None, convergence_hook: ConvergenceHook \| dict \| None=None, **kwargs: Any) -> None | (self, model: BaseModelMixin, dt: float \| torch.Tensor, delaystep: int=_FIRE2_DEFAULTS['delaystep'], dtgrow: float=_FIRE2_DEFAULTS['dtgrow'], dtshrink: float=_FIRE2_DEFAULTS['dtshrink'], alphashrink: float=_FIRE2_DEFAULTS['alphashrink'], alpha0: float=_FIRE2_DEFAULTS['alpha0'], tmax: float=_FIRE2_DEFAULTS['tmax'], tmin: float=_FIRE2_DEFAULTS['tmin'], maxstep: float=_FIRE2_DEFAULTS['maxstep'], n_steps: int \| None=None, hooks: list[Hook] \| None=None, convergence_hook: ConvergenceHook \| dict \| None=None, *, cell_force_scale: float=1.0, **kwargs: Any) -> None |
| added | nvalchemi/dynamics/optimizers/fire2.py:FIRE2VariableCell.samples_equilibrium | — | ClassVar[bool] = False |
| added | nvalchemi/dynamics/optimizers/lbfgs.py:LBFGS | — | class(_LBFGSMixin, BaseDynamics) |
| added | nvalchemi/dynamics/optimizers/lbfgs.py:LBFGS.pre_update | — | (self, batch: Batch) -> None |
| added | nvalchemi/dynamics/optimizers/lbfgs.py:LBFGS.samples_equilibrium | — | ClassVar[bool] = False |
| added | nvalchemi/dynamics/optimizers/lbfgs.py:LBFGSVariableCell | — | class(_LBFGSMixin, BaseDynamics) |
| added | nvalchemi/dynamics/optimizers/lbfgs.py:LBFGSVariableCell.__init__ | — | (self, model: BaseModelMixin, history_size: int=_LBFGS_DEFAULTS['history_size'], curvature_eps: float \| None=_LBFGS_DEFAULTS['curvature_eps'], maxstep: float=_LBFGS_DEFAULTS['maxstep'], n_steps: int \| None=None, hooks: list[Hook] \| None=None, convergence_hook: ConvergenceHook \| dict \| None=None, *, cell_force_scale: float=1.0, **kwargs: Any) -> None |
| added | nvalchemi/dynamics/optimizers/lbfgs.py:LBFGSVariableCell.cell_force_scale | — | (self) -> float @ property ; (self, value: float) -> None @ cell_force_scale.setter |
| added | nvalchemi/dynamics/optimizers/lbfgs.py:LBFGSVariableCell.pre_update | — | (self, batch: Batch) -> None |
| added | nvalchemi/dynamics/optimizers/lbfgs.py:LBFGSVariableCell.samples_equilibrium | — | ClassVar[bool] = False |
| added | nvalchemi/dynamics/sinks.py:ResizableSink | — | class(Protocol) @ runtime_checkable |
| added | nvalchemi/dynamics/sinks.py:ResizableSink.capacity | — | int = required |
| added | nvalchemi/dynamics/sinks.py:ResizableSink.resize | — | (self, capacity: int) -> None |
| added | nvalchemi/dynamics/strategy.py:DynamicsStrategy | — | class(BaseModel) |
| added | nvalchemi/dynamics/strategy.py:DynamicsStrategy.build_engine | — | (self) -> BaseDynamics |
| added | nvalchemi/dynamics/strategy.py:DynamicsStrategy.build_hooks | — | (self) -> list[Hook] |
| added | nvalchemi/dynamics/strategy.py:DynamicsStrategy.cache_engine | — | bool = Field(default=False, description='Reuse the engine and its runtime state across run() calls.') |
| added | nvalchemi/dynamics/strategy.py:DynamicsStrategy.engine | — | SerializableOptionalClass = Field(default=None, description='Dynamics engine class for the default builder; None requires a build_engine override. Round-trips via an importable dotted path.') |
| added | nvalchemi/dynamics/strategy.py:DynamicsStrategy.engine_kwargs | — | dict[str, Any] = Field(default_factory=dict, description='Additional keyword arguments forwarded to the engine constructor.') |
| added | nvalchemi/dynamics/strategy.py:DynamicsStrategy.extra_hooks | — | list[Hook] = Field(default_factory=list, exclude=True, description='Runtime hooks represented by ordered constructor specs during strategy serialization.') |
| added | nvalchemi/dynamics/strategy.py:DynamicsStrategy.from_spec_dict | — | (cls, spec: Mapping[str, Any], *, model: BaseModelMixin, extra_hooks: Sequence[Hook] \| None=None) -> DynamicsStrategy @ classmethod |
| added | nvalchemi/dynamics/strategy.py:DynamicsStrategy.model | — | BaseModelMixin = Field(exclude=True, description='Live potential model used by the dynamics engine; excluded from strategy serialization.') |
| added | nvalchemi/dynamics/strategy.py:DynamicsStrategy.n_steps | — | PositiveInt \| None = Field(default=None, description='Default dynamics step limit, or None for no fixed limit.') |
| added | nvalchemi/dynamics/strategy.py:DynamicsStrategy.run | — | (self, batch: Batch, n_steps: int \| None=None) -> Batch |
| added | nvalchemi/dynamics/strategy.py:DynamicsStrategy.to_spec_dict | — | (self) -> dict[str, Any] |
| added | nvalchemi/dynamics/structure_sampler.py:FitPolicy | — | class(Protocol) |
| added | nvalchemi/dynamics/structure_sampler.py:OrderedStructureSampler | — | class() |
| added | nvalchemi/dynamics/structure_sampler.py:OrderedStructureSampler.__init__ | — | (self, dataset: BatchDatasetProtocol, *, max_atoms: int \| None=None, max_edges: int \| None=None, max_batch_size: int \| None=None, recycle: bool=False) -> None |
| added | nvalchemi/dynamics/structure_sampler.py:OrderedStructureSampler.draw | — | (self, *, limit: int \| None=None, fits: FitPolicy \| None=None, on_miss: Literal['stop', 'skip']='stop') -> list[AtomicData] |
| added | nvalchemi/dynamics/structure_sampler.py:OrderedStructureSampler.exhausted | — | (self) -> bool @ property |
| added | nvalchemi/dynamics/structure_sampler.py:OrderedStructureSampler.initial_batch | — | (self) -> Batch |
| added | nvalchemi/dynamics/structure_sampler.py:OrderedStructureSampler.load_state_dict | — | (self, state: Mapping[str, int]) -> None |
| added | nvalchemi/dynamics/structure_sampler.py:OrderedStructureSampler.next_row | — | (self) -> int @ property |
| added | nvalchemi/dynamics/structure_sampler.py:OrderedStructureSampler.next_system_id | — | (self) -> int @ property |
| added | nvalchemi/dynamics/structure_sampler.py:OrderedStructureSampler.probe | — | (self) -> Batch |
| added | nvalchemi/dynamics/structure_sampler.py:OrderedStructureSampler.rank | — | (self) -> int @ property |
| added | nvalchemi/dynamics/structure_sampler.py:OrderedStructureSampler.rows | — | (self) -> tuple[int, ...] @ property |
| added | nvalchemi/dynamics/structure_sampler.py:OrderedStructureSampler.shard | — | (self, rank: int, world_size: int) -> None |
| added | nvalchemi/dynamics/structure_sampler.py:OrderedStructureSampler.state_dict | — | (self) -> dict[str, int] |
| added | nvalchemi/dynamics/structure_sampler.py:OrderedStructureSampler.world_size | — | (self) -> int @ property |
| added | nvalchemi/dynamics/structure_sampler.py:OrderedStructureSampler.wraps | — | (self) -> int @ property |
| added | nvalchemi/dynamics/structure_sampler.py:StructureSource | — | class(Protocol) @ runtime_checkable |
| added | nvalchemi/dynamics/structure_sampler.py:StructureSource.draw | — | (self, *, limit: int \| None=None, fits: FitPolicy \| None=None, on_miss: Literal['stop', 'skip']='stop') -> list[AtomicData] |
| added | nvalchemi/dynamics/structure_sampler.py:StructureSource.exhausted | — | (self) -> bool @ property |
| added | nvalchemi/dynamics/structure_sampler.py:StructureSource.initial_batch | — | (self) -> Batch |
| added | nvalchemi/dynamics/structure_sampler.py:StructureSource.load_state_dict | — | (self, state: Mapping[str, Any]) -> None |
| added | nvalchemi/dynamics/structure_sampler.py:StructureSource.probe | — | (self) -> Batch |
| added | nvalchemi/dynamics/structure_sampler.py:StructureSource.shard | — | (self, rank: int, world_size: int) -> None |
| added | nvalchemi/dynamics/structure_sampler.py:StructureSource.state_dict | — | (self) -> dict[str, Any] |
| added | nvalchemi/dynamics/structure_sampler.py:WithinBudget | — | class() @ dataclasses.dataclass(frozen=True) |
| added | nvalchemi/dynamics/structure_sampler.py:WithinBudget.atoms | — | int \| None = None |
| added | nvalchemi/dynamics/structure_sampler.py:WithinBudget.edges | — | int \| None = None |
| added | nvalchemi/enhanced_sampling/__init__.py:exports | — | __all__ = ['ConservativeBias', 'AdaptivePotentialMixin', 'EnhancedSampling', 'WalkerIdentityHook', 'BiasHook', 'EpochCommitHook', 'ThermodynamicState', 'ReplicaExchange', 'pair_distance', 'pair_displacement', 'periodic_difference', 'HarmonicUmbrellaBias', 'UpperWall', 'LowerWall', 'FlatBottomRestraint', 'WellTemperedMetaDynamicsBias', 'RMSDMetaDynamicsBias', 'AdaptiveBiasingForce'] ; from nvalchemi.enhanced_sampling._adaptive import AdaptivePotentialMixin ; from nvalchemi.enhanced_sampling._bias import ConservativeBias ; from nvalchemi.enhanced_sampling._exchange import ReplicaExchange, ThermodynamicState ; from nvalchemi.enhanced_sampling._runner import EnhancedSampling ; from nvalchemi.enhanced_sampling.biases import AdaptiveBiasingForce, FlatBottomRestraint, HarmonicUmbrellaBias, LowerWall, RMSDMetaDynamicsBias, UpperWall, WellTemperedMetaDynamicsBias ; from nvalchemi.enhanced_sampling.cv import pair_displacement, pair_distance, periodic_difference ; from nvalchemi.enhanced_sampling.hooks import BiasHook, EpochCommitHook, WalkerIdentityHook |
| added | nvalchemi/enhanced_sampling/_adaptive.py:AdaptivePotentialMixin | — | class() |
| added | nvalchemi/enhanced_sampling/_adaptive.py:AdaptivePotentialMixin.bump_state_version | — | (self) -> None |
| added | nvalchemi/enhanced_sampling/_adaptive.py:AdaptivePotentialMixin.commit | — | (self) -> None |
| added | nvalchemi/enhanced_sampling/_adaptive.py:AdaptivePotentialMixin.config_fingerprint | — | (self) -> Mapping[str, Any] |
| added | nvalchemi/enhanced_sampling/_adaptive.py:AdaptivePotentialMixin.forward | — | (self, data: AtomicData \| Batch, **kwargs: Any) -> ModelOutputs |
| added | nvalchemi/enhanced_sampling/_adaptive.py:AdaptivePotentialMixin.frequency | — | int = 1 |
| added | nvalchemi/enhanced_sampling/_adaptive.py:AdaptivePotentialMixin.load_state_dict | — | (self, state: Mapping[str, Any], *args: Any, **kwargs: Any) -> Any |
| added | nvalchemi/enhanced_sampling/_adaptive.py:AdaptivePotentialMixin.read_only | — | bool = False |
| added | nvalchemi/enhanced_sampling/_adaptive.py:AdaptivePotentialMixin.stage | — | DynamicsStage = DynamicsStage.AFTER_STEP |
| added | nvalchemi/enhanced_sampling/_adaptive.py:AdaptivePotentialMixin.stamp_state_version | — | (self, outputs: ModelOutputs, data: AtomicData \| Batch) -> ModelOutputs |
| added | nvalchemi/enhanced_sampling/_adaptive.py:AdaptivePotentialMixin.state_dict | — | (self, *args: Any, **kwargs: Any) -> dict[str, Any] |
| added | nvalchemi/enhanced_sampling/_adaptive.py:AdaptivePotentialMixin.state_version | — | (self) -> int @ property |
| added | nvalchemi/enhanced_sampling/_adaptive.py:AdaptivePotentialMixin.update | — | (self, ctx: BiasContext, stage: DynamicsStage) -> None |
| added | nvalchemi/enhanced_sampling/_bias.py:ConservativeBias | — | class(nn.Module, BaseModelMixin) |
| added | nvalchemi/enhanced_sampling/_bias.py:ConservativeBias.__init__ | — | (self, name: str, *, compute_stress: bool=True) -> None |
| added | nvalchemi/enhanced_sampling/_bias.py:ConservativeBias.compute_embeddings | — | (self, data: AtomicData \| Batch, **kwargs: Any) -> AtomicData \| Batch |
| added | nvalchemi/enhanced_sampling/_bias.py:ConservativeBias.distribution_spec | — | (self, strategy: StrategyKind \| None=None) -> MLIPSpec \| None |
| added | nvalchemi/enhanced_sampling/_bias.py:ConservativeBias.embedding_shapes | — | (self) -> dict[str, tuple[int, ...]] @ property |
| added | nvalchemi/enhanced_sampling/_bias.py:ConservativeBias.energy | — | (self, current: Batch) -> Tensor |
| added | nvalchemi/enhanced_sampling/_bias.py:ConservativeBias.export_model | — | (self, path: Path, as_state_dict: bool=False) -> None |
| added | nvalchemi/enhanced_sampling/_bias.py:ConservativeBias.forward | — | (self, data: AtomicData \| Batch, **kwargs: Any) -> ModelOutputs |
| added | nvalchemi/enhanced_sampling/_exchange.py:ReplicaExchange | — | class() |
| added | nvalchemi/enhanced_sampling/_exchange.py:ReplicaExchange.__init__ | — | (self, states: Sequence[ThermodynamicState], initial_state_ids: torch.Tensor, *, mode: Literal['synchronous']='synchronous', attempt_interval: int=100, random_seed: int=1234) -> None |
| added | nvalchemi/enhanced_sampling/_exchange.py:ReplicaExchange.acceptance | — | (self) -> str @ property |
| added | nvalchemi/enhanced_sampling/_exchange.py:ReplicaExchange.acceptance_rate | — | (self) -> float @ property |
| added | nvalchemi/enhanced_sampling/_exchange.py:ReplicaExchange.config_fingerprint | — | (self) -> dict[str, Any] |
| added | nvalchemi/enhanced_sampling/_exchange.py:ReplicaExchange.decide | — | (self, segment: int, state_ids: torch.Tensor, energies: torch.Tensor, bias_current: torch.Tensor \| None=None, bias_swapped: torch.Tensor \| None=None) -> tuple[torch.Tensor, list[tuple[int, int]], torch.Tensor] |
| added | nvalchemi/enhanced_sampling/_exchange.py:ReplicaExchange.describe_config_mismatch | — | (saved: Mapping[str, Any] \| None, actual: Mapping[str, Any] \| None) -> list[str] @ staticmethod |
| added | nvalchemi/enhanced_sampling/_exchange.py:ReplicaExchange.load_state_dict | — | (self, state: Mapping[str, Any]) -> None |
| added | nvalchemi/enhanced_sampling/_exchange.py:ReplicaExchange.pair_acceptance_rates | — | (self) -> list[float] |
| added | nvalchemi/enhanced_sampling/_exchange.py:ReplicaExchange.pair_schedule | — | (self, segment: int) -> list[tuple[int, int]] |
| added | nvalchemi/enhanced_sampling/_exchange.py:ReplicaExchange.per_system_params | — | (self, state_ids: torch.Tensor) -> dict[str, torch.Tensor] |
| added | nvalchemi/enhanced_sampling/_exchange.py:ReplicaExchange.proposed_assignment | — | (self, segment: int, state_ids: torch.Tensor) -> torch.Tensor |
| added | nvalchemi/enhanced_sampling/_exchange.py:ReplicaExchange.state_dict | — | (self) -> dict[str, Any] |
| added | nvalchemi/enhanced_sampling/_exchange.py:ReplicaExchange.swap_hook | — | (self, *, bias_energy_fn: Callable[[Batch, torch.Tensor], torch.Tensor] \| None=None, on_swap: Callable[[Batch], None] \| None=None, slot_field: str='thermodynamic_state_id') -> PairSwapHook |
| added | nvalchemi/enhanced_sampling/_exchange.py:ReplicaExchange.temperatures | — | (self) -> torch.Tensor @ property |
| added | nvalchemi/enhanced_sampling/_exchange.py:ReplicaExchange.validate_assignment | — | (self, state_ids: torch.Tensor, num_graphs: int \| None=None, *, source: str='assignment') -> torch.Tensor |
| added | nvalchemi/enhanced_sampling/_exchange.py:ReplicaExchange.validate_for | — | (self, biases: Mapping[str, Any]) -> None |
| added | nvalchemi/enhanced_sampling/_exchange.py:ThermodynamicState | — | class(BaseModel) |
| added | nvalchemi/enhanced_sampling/_exchange.py:ThermodynamicState.state_id | — | int = Field(ge=0) |
| added | nvalchemi/enhanced_sampling/_exchange.py:ThermodynamicState.temperature | — | float = Field(gt=0.0) |
| added | nvalchemi/enhanced_sampling/_exchange.py:log_acceptance_is_accepted | — | (log_alpha: torch.Tensor, uniforms: torch.Tensor) -> torch.Tensor |
| added | nvalchemi/enhanced_sampling/_history.py:DepositHistoryMixin | — | class() |
| added | nvalchemi/enhanced_sampling/_history.py:DepositHistoryMixin.capacity | — | (self) -> int @ property |
| added | nvalchemi/enhanced_sampling/_history.py:DepositHistoryMixin.history | — | str = required |
| added | nvalchemi/enhanced_sampling/_history.py:DepositHistoryMixin.name | — | str = required |
| added | nvalchemi/enhanced_sampling/_history.py:DepositHistoryMixin.storage | — | str = required |
| added | nvalchemi/enhanced_sampling/_history.py:DepositHistoryMixin.validate_history_options | — | (cls, storage: str, history: str) -> None @ classmethod |
| added | nvalchemi/enhanced_sampling/_runner.py:EnhancedSampling | — | class(DynamicsStrategy) |
| added | nvalchemi/enhanced_sampling/_runner.py:EnhancedSampling.biases | — | dict[str, Any] = Field(default_factory=dict, exclude=True, description='Unique name to bias; each a BaseModelMixin whose name matches.') |
| added | nvalchemi/enhanced_sampling/_runner.py:EnhancedSampling.build_hooks | — | (self) -> list[Hook] |
| added | nvalchemi/enhanced_sampling/_runner.py:EnhancedSampling.cache_engine | — | bool = Field(default=True, description='Always True: the hook family belongs to one engine.') |
| added | nvalchemi/enhanced_sampling/_runner.py:EnhancedSampling.checkpoint | — | (self, path: str \| Path, batch: Batch \| None=None) -> None |
| added | nvalchemi/enhanced_sampling/_runner.py:EnhancedSampling.compile_biases | — | bool = Field(default=False, description="torch.compile each conservative bias's energy().") |
| added | nvalchemi/enhanced_sampling/_runner.py:EnhancedSampling.dynamics | — | (self) -> BaseDynamics |
| added | nvalchemi/enhanced_sampling/_runner.py:EnhancedSampling.from_spec_dict | — | (cls, spec: Mapping[str, Any], *, model: BaseModelMixin, biases: Mapping[str, BaseModelMixin] \| None=None, replica_exchange: ReplicaExchange \| None=None, extra_hooks: Sequence[Hook] \| None=None) -> EnhancedSampling @ classmethod |
| added | nvalchemi/enhanced_sampling/_runner.py:EnhancedSampling.last_outputs | — | (self) -> dict[str, torch.Tensor] @ property |
| added | nvalchemi/enhanced_sampling/_runner.py:EnhancedSampling.prime_after_update | — | bool = Field(default=True, description='Re-prime forces when an update() changes a bias.') |
| added | nvalchemi/enhanced_sampling/_runner.py:EnhancedSampling.prime_forces | — | (self, batch: Batch) -> Batch |
| added | nvalchemi/enhanced_sampling/_runner.py:EnhancedSampling.replica_exchange | — | ReplicaExchange \| None = Field(default=None, exclude=True, description='Optional ladder; a runtime object, not serialised.') |
| added | nvalchemi/enhanced_sampling/_runner.py:EnhancedSampling.restore | — | (self, path: str \| Path, device: torch.device \| str \| None=None) -> Batch |
| added | nvalchemi/enhanced_sampling/_runner.py:EnhancedSampling.run | — | (self, batch: Batch, n_steps: int \| None=None, *, prime: bool=True) -> Batch |
| added | nvalchemi/enhanced_sampling/_runner.py:EnhancedSampling.state_dict | — | (self) -> Mapping[str, Any] |
| added | nvalchemi/enhanced_sampling/_runner.py:EnhancedSampling.steps_per_epoch | — | int = Field(default=10000, description='Steps per consistency epoch.') |
| added | nvalchemi/enhanced_sampling/_runner.py:EnhancedSampling.warm_start | — | (self, frames: Batch) -> None |
| added | nvalchemi/enhanced_sampling/biases/__init__.py:exports | — | __all__ = ['AdaptiveBiasingForce', 'FlatBottomRestraint', 'HarmonicUmbrellaBias', 'LowerWall', 'RMSDMetaDynamicsBias', 'UpperWall', 'WellTemperedMetaDynamicsBias'] ; from nvalchemi.enhanced_sampling.biases.abf import AdaptiveBiasingForce ; from nvalchemi.enhanced_sampling.biases.metadynamics import WellTemperedMetaDynamicsBias ; from nvalchemi.enhanced_sampling.biases.rmsd_metad import RMSDMetaDynamicsBias ; from nvalchemi.enhanced_sampling.biases.umbrella import HarmonicUmbrellaBias ; from nvalchemi.enhanced_sampling.biases.walls import FlatBottomRestraint, LowerWall, UpperWall |
| added | nvalchemi/enhanced_sampling/biases/abf.py:AdaptiveBiasingForce | — | class(AdaptivePotentialMixin, nn.Module, BaseModelMixin) |
| added | nvalchemi/enhanced_sampling/biases/abf.py:AdaptiveBiasingForce.__init__ | — | (self, atom_indices: Tensor, temperature: float, cv_range: tuple[float, float], *, name: str='abf', n_bins: int=100, min_samples: int=200, full_samples: int \| None=None, max_force: float \| None=None, frequency: int=1) -> None |
| added | nvalchemi/enhanced_sampling/biases/abf.py:AdaptiveBiasingForce.bin_centers | — | (self) -> Tensor @ property |
| added | nvalchemi/enhanced_sampling/biases/abf.py:AdaptiveBiasingForce.bin_index | — | (self, values: Tensor) -> Tensor |
| added | nvalchemi/enhanced_sampling/biases/abf.py:AdaptiveBiasingForce.bin_width | — | (self) -> float @ property |
| added | nvalchemi/enhanced_sampling/biases/abf.py:AdaptiveBiasingForce.compute_embeddings | — | (self, data: AtomicData \| Batch, **kwargs: Any) -> AtomicData \| Batch |
| added | nvalchemi/enhanced_sampling/biases/abf.py:AdaptiveBiasingForce.config_fingerprint | — | (self) -> dict[str, Any] |
| added | nvalchemi/enhanced_sampling/biases/abf.py:AdaptiveBiasingForce.embedding_shapes | — | (self) -> dict[str, tuple[int, ...]] @ property |
| added | nvalchemi/enhanced_sampling/biases/abf.py:AdaptiveBiasingForce.forward | — | (self, data: AtomicData \| Batch, **kwargs: Any) -> ModelOutputs |
| added | nvalchemi/enhanced_sampling/biases/abf.py:AdaptiveBiasingForce.free_energy | — | (self) -> Tensor |
| added | nvalchemi/enhanced_sampling/biases/abf.py:AdaptiveBiasingForce.mean_force | — | (self) -> Tensor |
| added | nvalchemi/enhanced_sampling/biases/abf.py:AdaptiveBiasingForce.ramp_fraction | — | (self) -> Tensor |
| added | nvalchemi/enhanced_sampling/biases/abf.py:AdaptiveBiasingForce.supplies_exchange_energy | — | bool = False |
| added | nvalchemi/enhanced_sampling/biases/abf.py:AdaptiveBiasingForce.update | — | (self, ctx: BiasContext, stage: DynamicsStage) -> None |
| added | nvalchemi/enhanced_sampling/biases/metadynamics.py:WellTemperedMetaDynamicsBias | — | class(AdaptivePotentialMixin, DepositHistoryMixin, ConservativeBias) |
| added | nvalchemi/enhanced_sampling/biases/metadynamics.py:WellTemperedMetaDynamicsBias.__init__ | — | (self, cv: Callable[[Batch], Tensor], height: float, sigma: Tensor \| float, temperature: float, *, bias_factor: float \| None=None, name: str='metadynamics', frequency: int=500, storage: Literal['preallocated', 'grow', 'fifo']='preallocated', max_hills: int \| None=None, history: Literal['shared', 'state', 'walker']='shared', periods: Tensor \| None=None, ramp_depositions: int=0, compute_stress: bool=True) -> None |
| added | nvalchemi/enhanced_sampling/biases/metadynamics.py:WellTemperedMetaDynamicsBias.config_fingerprint | — | (self) -> dict[str, Any] |
| added | nvalchemi/enhanced_sampling/biases/metadynamics.py:WellTemperedMetaDynamicsBias.energy | — | (self, current: Batch) -> Tensor |
| added | nvalchemi/enhanced_sampling/biases/metadynamics.py:WellTemperedMetaDynamicsBias.free_energy | — | (self, values: Tensor, owner_key: Tensor \| None=None) -> Tensor |
| added | nvalchemi/enhanced_sampling/biases/metadynamics.py:WellTemperedMetaDynamicsBias.gaussian_sum | — | (self, values: Tensor, owner_key: Tensor) -> Tensor |
| added | nvalchemi/enhanced_sampling/biases/metadynamics.py:WellTemperedMetaDynamicsBias.load_state_dict | — | (self, state: Mapping[str, Any], *args: Any, **kwargs: Any) -> Any |
| added | nvalchemi/enhanced_sampling/biases/metadynamics.py:WellTemperedMetaDynamicsBias.update | — | (self, ctx: BiasContext, stage: DynamicsStage) -> None |
| added | nvalchemi/enhanced_sampling/biases/rmsd_metad.py:RMSDMetaDynamicsBias | — | class(AdaptivePotentialMixin, DepositHistoryMixin, ConservativeBias) |
| added | nvalchemi/enhanced_sampling/biases/rmsd_metad.py:RMSDMetaDynamicsBias.__init__ | — | (self, k_push: float, alpha: float, *, name: str='rmsd_metadynamics', atom_indices: Tensor \| None=None, frequency: int=500, storage: Literal['preallocated', 'grow', 'fifo']='fifo', max_references: int \| None=None, history: Literal['shared', 'state', 'walker']='shared', ramp_depositions: int=1, references: Tensor \| None=None, compute_stress: bool=False) -> None |
| added | nvalchemi/enhanced_sampling/biases/rmsd_metad.py:RMSDMetaDynamicsBias.config_fingerprint | — | (self) -> dict[str, Any] |
| added | nvalchemi/enhanced_sampling/biases/rmsd_metad.py:RMSDMetaDynamicsBias.energy | — | (self, current: Batch) -> Tensor |
| added | nvalchemi/enhanced_sampling/biases/rmsd_metad.py:RMSDMetaDynamicsBias.forward | — | (self, data: Batch, **kwargs: Any) -> ModelOutputs |
| added | nvalchemi/enhanced_sampling/biases/rmsd_metad.py:RMSDMetaDynamicsBias.load_state_dict | — | (self, state: Mapping[str, Any], *args: Any, **kwargs: Any) -> Any |
| added | nvalchemi/enhanced_sampling/biases/rmsd_metad.py:RMSDMetaDynamicsBias.repulsion | — | (self, coords: Tensor, owner_key: Tensor) -> Tensor |
| added | nvalchemi/enhanced_sampling/biases/rmsd_metad.py:RMSDMetaDynamicsBias.update | — | (self, ctx: BiasContext, stage: DynamicsStage) -> None |
| added | nvalchemi/enhanced_sampling/biases/umbrella.py:HarmonicUmbrellaBias | — | class(ConservativeBias) |
| added | nvalchemi/enhanced_sampling/biases/umbrella.py:HarmonicUmbrellaBias.__init__ | — | (self, cv: Callable[[Batch], Tensor], centers: Tensor \| float, stiffness: Tensor \| float, *, name: str='umbrella', periods: Tensor \| None=None, compute_stress: bool=True) -> None |
| added | nvalchemi/enhanced_sampling/biases/umbrella.py:HarmonicUmbrellaBias.energy | — | (self, current: Batch) -> Tensor |
| added | nvalchemi/enhanced_sampling/biases/umbrella.py:HarmonicUmbrellaBias.forward | — | (self, data: Batch, **kwargs: Any) -> ModelOutputs |
| added | nvalchemi/enhanced_sampling/biases/walls.py:FlatBottomRestraint | — | class(_WallBase) |
| added | nvalchemi/enhanced_sampling/biases/walls.py:FlatBottomRestraint.__init__ | — | (self, cv: Callable[[Batch], Tensor], lower: Tensor \| float, upper: Tensor \| float, stiffness: Tensor \| float=10.0, *, name: str='flat_bottom', exponent: float=2.0, compute_stress: bool=True) -> None |
| added | nvalchemi/enhanced_sampling/biases/walls.py:FlatBottomRestraint.energy | — | (self, current: Batch) -> Tensor |
| added | nvalchemi/enhanced_sampling/biases/walls.py:LowerWall | — | class(_WallBase) |
| added | nvalchemi/enhanced_sampling/biases/walls.py:LowerWall.__init__ | — | (self, cv: Callable[[Batch], Tensor], threshold: Tensor \| float, stiffness: Tensor \| float=10.0, *, name: str='lower_wall', exponent: float=2.0, compute_stress: bool=True) -> None |
| added | nvalchemi/enhanced_sampling/biases/walls.py:UpperWall | — | class(_WallBase) |
| added | nvalchemi/enhanced_sampling/biases/walls.py:UpperWall.__init__ | — | (self, cv: Callable[[Batch], Tensor], threshold: Tensor \| float, stiffness: Tensor \| float=10.0, *, name: str='upper_wall', exponent: float=2.0, compute_stress: bool=True) -> None |
| added | nvalchemi/enhanced_sampling/cv/__init__.py:exports | — | __all__ = ['pair_displacement', 'pair_distance', 'periodic_difference'] ; from nvalchemi.enhanced_sampling.cv._periodic import periodic_difference ; from nvalchemi.enhanced_sampling.cv.pair_distance import pair_displacement, pair_distance |
| added | nvalchemi/enhanced_sampling/cv/_periodic.py:periodic_difference | — | (values: Tensor, centers: Tensor, periods: Tensor \| None=None) -> Tensor |
| added | nvalchemi/enhanced_sampling/cv/pair_distance.py:pair_displacement | — | (batch: Batch, atom_indices: Tensor) -> Tensor |
| added | nvalchemi/enhanced_sampling/cv/pair_distance.py:pair_distance | — | (batch: Batch, atom_indices: Tensor) -> Tensor |
| added | nvalchemi/enhanced_sampling/hooks.py:BiasHook | — | class() |
| added | nvalchemi/enhanced_sampling/hooks.py:BiasHook.__init__ | — | (self, biases: Mapping[str, BaseModelMixin] \| None=None, *, prime_after_update: bool=True, compile_biases: bool=False) -> None |
| added | nvalchemi/enhanced_sampling/hooks.py:BiasHook.adaptive_biases | — | (self) -> dict[str, BaseModelMixin] |
| added | nvalchemi/enhanced_sampling/hooks.py:BiasHook.bias_energy | — | (self, batch: Batch, state_ids: torch.Tensor) -> torch.Tensor |
| added | nvalchemi/enhanced_sampling/hooks.py:BiasHook.commit | — | (self) -> None |
| added | nvalchemi/enhanced_sampling/hooks.py:BiasHook.evaluate_and_apply | — | (self, batch: Batch) -> None |
| added | nvalchemi/enhanced_sampling/hooks.py:BiasHook.load_state_dict | — | (self, state: Mapping[str, Any]) -> None |
| added | nvalchemi/enhanced_sampling/hooks.py:BiasHook.observe_and_update | — | (self, batch: Batch) -> None |
| added | nvalchemi/enhanced_sampling/hooks.py:BiasHook.on_register | — | (self, workflow: Any) -> None |
| added | nvalchemi/enhanced_sampling/hooks.py:BiasHook.replay | — | (self, frames: Batch) -> None |
| added | nvalchemi/enhanced_sampling/hooks.py:BiasHook.reprime_from_scratch | — | (self, batch: Batch) -> None |
| added | nvalchemi/enhanced_sampling/hooks.py:BiasHook.state_dict | — | (self) -> Mapping[str, Any] |
| added | nvalchemi/enhanced_sampling/hooks.py:BiasHook.sync_seen_versions | — | (self) -> None |
| added | nvalchemi/enhanced_sampling/hooks.py:EpochCommitHook | — | class() |
| added | nvalchemi/enhanced_sampling/hooks.py:EpochCommitHook.__init__ | — | (self, targets: list[StatefulHook], *, frequency: int) -> None |
| added | nvalchemi/enhanced_sampling/hooks.py:EpochCommitHook.commit_epoch | — | (self, epoch: int) -> None |
| added | nvalchemi/enhanced_sampling/hooks.py:EpochCommitHook.load_state_dict | — | (self, state: Mapping[str, Any]) -> None |
| added | nvalchemi/enhanced_sampling/hooks.py:EpochCommitHook.state_dict | — | (self) -> Mapping[str, Any] |
| added | nvalchemi/enhanced_sampling/hooks.py:WalkerIdentityHook | — | class() |
| added | nvalchemi/enhanced_sampling/hooks.py:WalkerIdentityHook.__init__ | — | (self, *, steps_per_epoch: int, exchange: ReplicaExchange \| None=None) -> None |
| added | nvalchemi/enhanced_sampling/hooks.py:WalkerIdentityHook.load_state_dict | — | (self, state: Mapping[str, Any]) -> None |
| added | nvalchemi/enhanced_sampling/hooks.py:WalkerIdentityHook.on_register | — | (self, workflow: Any) -> None |
| added | nvalchemi/enhanced_sampling/hooks.py:WalkerIdentityHook.stamp | — | (self, batch: Batch, step: int) -> None |
| added | nvalchemi/enhanced_sampling/hooks.py:WalkerIdentityHook.state_dict | — | (self) -> Mapping[str, Any] |
| added | nvalchemi/gen/__init__.py:exports | — | __all__ = ['AtomisticGenerator', 'ConditionFunction', 'GenerationContext', 'GenerationPipeline', 'GenerationStage', 'GeneratingFunction'] ; from __future__ import annotations ; from nvalchemi.gen.generator import AtomisticGenerator, ConditionFunction, GeneratingFunction ; from nvalchemi.gen.pipeline import GenerationPipeline ; from nvalchemi.gen.stages import GenerationStage ; from nvalchemi.hooks import GenerationContext |
| added | nvalchemi/gen/_device.py:normalize_device | — | (value: torch.device \| str) -> torch.device |
| added | nvalchemi/gen/generator.py:AtomisticGenerator | — | class(BaseModel, HookRegistryMixin) |
| added | nvalchemi/gen/generator.py:AtomisticGenerator.compile | — | (self, **kwargs: Any) -> AtomisticGenerator |
| added | nvalchemi/gen/generator.py:AtomisticGenerator.compile_generate | — | bool = Field(default=False, description='Compile the generating function with torch.compile (immediate via compile(), or lazily at session entry).') |
| added | nvalchemi/gen/generator.py:AtomisticGenerator.compile_kwargs | — | dict[str, Any] = Field(default_factory=dict, description='Keyword arguments forwarded to torch.compile; validated against its signature at construction and in compile().') |
| added | nvalchemi/gen/generator.py:AtomisticGenerator.condition_func | — | ConditionFunction \| None = None |
| added | nvalchemi/gen/generator.py:AtomisticGenerator.dedicated_stream | — | bool = Field(default=True, description='Create a dedicated CUDA stream at session entry when the resolved device is CUDA. Set False to suppress stream creation.') |
| added | nvalchemi/gen/generator.py:AtomisticGenerator.device | — | _SerializableOptionalDevice = Field(default=None, description='Execution device, defaulting to generator_func.device. Bare CUDA is pinned to the current GPU at construction; CUDA availability and index bounds are validated.') |
| added | nvalchemi/gen/generator.py:AtomisticGenerator.enable_inference_mode | — | bool = Field(default=False, description='Run generation under `torch.inference_mode` in the context manager. This disables gradient tracking; enable it for workflows that do not need gradients.') |
| added | nvalchemi/gen/generator.py:AtomisticGenerator.generator_func | — | GeneratingFunction = required |
| added | nvalchemi/gen/generator.py:AtomisticGenerator.hooks | — | list[Hook] = Field(default_factory=list, description='Generation hooks, fired at GenerationStage points with a shared GenerationContext per call.') |
| added | nvalchemi/gen/generator.py:AtomisticGenerator.model_post_init | — | (self, __context: Any) -> None |
| added | nvalchemi/gen/generator.py:AtomisticGenerator.num_samples | — | int = Field(default=1, ge=1, description='Independent draws per call, passed straight through to the generating function.') |
| added | nvalchemi/gen/generator.py:AtomisticGenerator.outputs | — | frozenset[str] \| None = Field(default=None, description='Batch fields the output carries (written or forwarded). Defaults from generator_func.outputs when available. Checked against the materialized batch before each return.') |
| added | nvalchemi/gen/generator.py:AtomisticGenerator.required_inputs | — | frozenset[str] \| None = Field(default=None, description='Batch fields the input carries (empty = unconditional). Defaults from generator_func.required_inputs when available. Checked against the (conditioned) inputs on every call.') |
| added | nvalchemi/gen/generator.py:AtomisticGenerator.sample | — | (self, inputs: Any=None, *, num_samples: int \| None=None, **kwargs: Any) -> Any |
| added | nvalchemi/gen/generator.py:AtomisticGenerator.seed | — | int \| None = Field(default=None, description='Base seed for per-draw RNGs (seed + step_count per call).') |
| added | nvalchemi/gen/generator.py:AtomisticGenerator.step_count | — | int = Field(default=0, ge=0, exclude=True, description='Runtime counter of generation calls (hook frequency).') |
| added | nvalchemi/gen/generator.py:AtomisticGenerator.stream | — | (self, inputs: Any=None, *, max_batches: int \| None=None, **kwargs: Any) -> Iterator[Any] |
| added | nvalchemi/gen/generator.py:ConditionFunction | — | class(Protocol) @ runtime_checkable |
| added | nvalchemi/gen/generator.py:GeneratingFunction | — | class(Protocol[InputT, SampleT]) @ runtime_checkable |
| added | nvalchemi/gen/pipeline.py:GenerationPipeline | — | class(BaseModel) |
| added | nvalchemi/gen/pipeline.py:GenerationPipeline.compile | — | (self, **kwargs: Any) -> GenerationPipeline |
| added | nvalchemi/gen/pipeline.py:GenerationPipeline.model_post_init | — | (self, __context: Any) -> None |
| added | nvalchemi/gen/pipeline.py:GenerationPipeline.stages | — | list[Any] = Field(min_length=1, description='Ordered stages: Generators, dynamics engines, or Batch -> Batch callables.') |
| added | nvalchemi/gen/pipeline.py:GenerationPipeline.stream | — | (self, inputs: Any=None, *, max_batches: int \| None=None, stage_kwargs: Mapping[str, Any] \| Sequence[Mapping[str, Any] \| None] \| None=None) -> Iterator[Any] |
| added | nvalchemi/gen/stages.py:GenerationStage | — | class(Enum) |
| changed | nvalchemi/hooks/__init__.py:exports | __all__ = ['BaseRichLayout', 'BiasedPotentialHook', 'CheckpointableHook', 'DynamicsContext', 'DynamicsRichLayout', 'Hook', 'HookContext', 'HookRegistryMixin', 'NeighborListHook', 'Reporter', 'ReporterMessage', 'ReportingErrorPolicy', 'ReportingOrchestrator', 'ReportingState', 'RichLayout', 'RichReporter', 'ScalarCallback', 'ScalarSnapshot', 'TensorBoardReporter', 'TensorBoardWriter', 'StageTimingHook', 'TorchProfilerHook', 'TrainContext', 'TrainingRichLayout', 'WrapPeriodicHook', 'collect_scalars', 'extract_dynamics_scalars', 'extract_loss_scalars', 'extract_optimizer_lr_scalars', 'extract_scalars'] ; from __future__ import annotations ; from nvalchemi.hooks._context import DynamicsContext, HookContext, TrainContext ; from nvalchemi.hooks._protocol import CheckpointableHook, Hook ; from nvalchemi.hooks._registry import HookRegistryMixin ; from nvalchemi.hooks.bias import BiasedPotentialHook ; from nvalchemi.hooks.neighbor_list import NeighborListHook ; from nvalchemi.hooks.periodic import WrapPeriodicHook ; from nvalchemi.hooks.physicsnemo_profiling import TorchProfilerHook ; from nvalchemi.hooks.reporting import BaseRichLayout, DynamicsRichLayout, Reporter, ReporterMessage, ReportingErrorPolicy, ReportingOrchestrator, ReportingState, RichLayout, RichReporter, ScalarCallback, ScalarSnapshot, TensorBoardReporter, TensorBoardWriter, TrainingRichLayout, collect_scalars, extract_dynamics_scalars, extract_loss_scalars, extract_optimizer_lr_scalars, extract_scalars ; from nvalchemi.hooks.stage_timing import StageTimingHook | __all__ = ['BaseRichLayout', 'BiasContext', 'BiasedPotentialHook', 'CheckpointableHook', 'DynamicsContext', 'DynamicsRichLayout', 'GenerationContext', 'Hook', 'HookContext', 'HookRegistryMixin', 'NeighborListHook', 'Reporter', 'ReporterMessage', 'ReportingErrorPolicy', 'ReportingOrchestrator', 'ReportingState', 'RichLayout', 'RichReporter', 'ScalarCallback', 'ScalarSnapshot', 'StatefulHook', 'TensorBoardReporter', 'TensorBoardWriter', 'StageTimingHook', 'TorchProfilerHook', 'TrainContext', 'TrainingRichLayout', 'WrapPeriodicHook', 'collect_scalars', 'extract_dynamics_scalars', 'extract_loss_scalars', 'extract_optimizer_lr_scalars', 'extract_scalars'] ; from __future__ import annotations ; from nvalchemi.hooks._context import BiasContext, DynamicsContext, GenerationContext, HookContext, TrainContext ; from nvalchemi.hooks._protocol import CheckpointableHook, Hook, StatefulHook ; from nvalchemi.hooks._registry import HookRegistryMixin ; from nvalchemi.hooks.bias import BiasedPotentialHook ; from nvalchemi.hooks.neighbor_list import NeighborListHook ; from nvalchemi.hooks.periodic import WrapPeriodicHook ; from nvalchemi.hooks.physicsnemo_profiling import TorchProfilerHook ; from nvalchemi.hooks.reporting import BaseRichLayout, DynamicsRichLayout, Reporter, ReporterMessage, ReportingErrorPolicy, ReportingOrchestrator, ReportingState, RichLayout, RichReporter, ScalarCallback, ScalarSnapshot, TensorBoardReporter, TensorBoardWriter, TrainingRichLayout, collect_scalars, extract_dynamics_scalars, extract_loss_scalars, extract_optimizer_lr_scalars, extract_scalars ; from nvalchemi.hooks.stage_timing import StageTimingHook |
| added | nvalchemi/hooks/_context.py:BiasContext | — | class(DynamicsContext) @ dataclass(kw_only=True) |
| added | nvalchemi/hooks/_context.py:BiasContext.contribution | — | ModelOutputs = field(default_factory=OrderedDict) |
| added | nvalchemi/hooks/_context.py:DynamicsContext.active_graph_mask | — | Bool[torch.Tensor, 'B'] \| None = None |
| changed | nvalchemi/hooks/_context.py:DynamicsContext.converged_mask | torch.Tensor \| None = None | Bool[torch.Tensor, 'B'] \| None = None |
| added | nvalchemi/hooks/_context.py:DynamicsContext.graduated_mask | — | Bool[torch.Tensor, 'B'] \| None = None |
| added | nvalchemi/hooks/_context.py:GenerationContext | — | class(HookContext) @ dataclass(kw_only=True) |
| added | nvalchemi/hooks/_context.py:GenerationContext.accepted_mask | — | torch.Tensor \| None = None |
| added | nvalchemi/hooks/_context.py:GenerationContext.batch | — | Batch \| None = None |
| added | nvalchemi/hooks/_context.py:GenerationContext.inputs | — | Any = None |
| added | nvalchemi/hooks/_context.py:GenerationContext.intermediates | — | dict[str, Any] = field(default_factory=dict) |
| added | nvalchemi/hooks/_context.py:GenerationContext.sample | — | Any = None |
| added | nvalchemi/hooks/_context.py:GenerationContext.step_count | — | int = 0 |
| added | nvalchemi/hooks/_protocol.py:StatefulHook | — | class(Hook, CheckpointableHook, Protocol) @ runtime_checkable |
| added | nvalchemi/hooks/_protocol.py:StatefulHook.commit | — | (self) -> None |
| added | nvalchemi/hooks/_protocol.py:StatefulHook.read_only | — | bool = False |
| changed | nvalchemi/models/__init__.py:exports | __all__ = ['DemoModelWrapper', 'DFTD3ModelWrapper', 'EwaldModelWrapper', 'LennardJonesModelWrapper', 'PMEModelWrapper', 'AIMNet2Wrapper', 'MACEWrapper', 'UMAWrapper', 'PipelineModelWrapper', 'PipelineStep', 'PipelineGroup'] ; from __future__ import annotations ; from nvalchemi.models.aimnet2 import AIMNet2Wrapper ; from nvalchemi.models.demo import DemoModelWrapper ; from nvalchemi.models.dftd3 import DFTD3ModelWrapper ; from nvalchemi.models.ewald import EwaldModelWrapper ; from nvalchemi.models.lj import LennardJonesModelWrapper ; from nvalchemi.models.mace import MACEWrapper ; from nvalchemi.models.pipeline import PipelineGroup, PipelineModelWrapper, PipelineStep ; from nvalchemi.models.pme import PMEModelWrapper ; from nvalchemi.models.uma import UMAWrapper ; from typing import TYPE_CHECKING | __all__ = ['DerivativeNotSupported', 'HessianOperator', 'DemoModelWrapper', 'DFTD3ModelWrapper', 'EwaldModelWrapper', 'LennardJonesModelWrapper', 'PMEModelWrapper', 'AIMNet2Wrapper', 'MACEWrapper', 'UMAWrapper', 'PipelineModelWrapper', 'PipelineStep', 'PipelineGroup'] ; from __future__ import annotations ; from nvalchemi.models._derivatives import DerivativeNotSupported, HessianOperator ; from nvalchemi.models.aimnet2 import AIMNet2Wrapper ; from nvalchemi.models.demo import DemoModelWrapper ; from nvalchemi.models.dftd3 import DFTD3ModelWrapper ; from nvalchemi.models.ewald import EwaldModelWrapper ; from nvalchemi.models.lj import LennardJonesModelWrapper ; from nvalchemi.models.mace import MACEWrapper ; from nvalchemi.models.pipeline import PipelineGroup, PipelineModelWrapper, PipelineStep ; from nvalchemi.models.pme import PMEModelWrapper ; from nvalchemi.models.uma import UMAWrapper ; from typing import TYPE_CHECKING |
| added | nvalchemi/models/_derivatives.py:DerivativeNotSupported | — | class(NotImplementedError) |
| added | nvalchemi/models/_derivatives.py:DerivativeNotSupported.__init__ | — | (self, *, model_name: str, operation: _DerivativeOperation, execution: _DerivativeExecutionKind, strategy: _DerivativeStrategy \| None, reason: str) -> None |
| added | nvalchemi/models/_derivatives.py:HessianOperator | — | class() |
| added | nvalchemi/models/_derivatives.py:HessianOperator.__init__ | — | (self, context: AbstractContextManager[_DerivativeGraph]) -> None |
| added | nvalchemi/models/_derivatives.py:HessianOperator.close | — | (self) -> None |
| added | nvalchemi/models/_derivatives.py:HessianOperator.from_energy | — | (cls, energy: Tensor, positions: Tensor) -> HessianOperator @ classmethod |
| added | nvalchemi/models/_derivatives.py:HessianOperator.matvec | — | (self, vector: Tensor, *, create_graph: bool=False) -> Tensor |
| added | nvalchemi/models/_utils.py:aggregate_contributions | — | (contributions: list[ModelOutputs]) -> ModelOutputs |
| changed | nvalchemi/models/_utils.py:autograd_forces | (energy: Energy, positions: NodePositions, training: bool=False, retain_graph: bool=False) -> Forces | (energy: Energy, positions: NodePositions, training: bool=False, retain_graph: bool=False, allow_unused: bool=False) -> Forces |
| changed | nvalchemi/models/_utils.py:autograd_forces_and_stresses | (energy: Energy, positions: NodePositions, displacement: StrainDisplacement, cell: LatticeVectors, num_graphs: int, training: bool=False, retain_graph: bool=False) -> tuple[Forces, Stress] | (energy: Energy, positions: NodePositions, displacement: StrainDisplacement, cell: LatticeVectors, num_graphs: int, training: bool=False, retain_graph: bool=False, allow_unused: bool=False) -> tuple[Forces, Stress] |
| changed | nvalchemi/models/_utils.py:autograd_stresses | (energy: Energy, displacement: StrainDisplacement, cell: LatticeVectors, num_graphs: int, training: bool=False, retain_graph: bool=False) -> Stress | (energy: Energy, displacement: StrainDisplacement, cell: LatticeVectors, num_graphs: int, training: bool=False, retain_graph: bool=False, allow_unused: bool=False) -> Stress |
| added | nvalchemi/models/_utils.py:isolated_energy_derivatives | — | (energy_fn: Callable[[Batch], Energy], batch: Batch, *, want_forces: bool=True, want_stress: bool=True, allow_unused: bool=False) -> ModelOutputs |
| added | nvalchemi/models/_utils.py:validate_contribution | — | (outputs: ModelOutputs, *, source: str='ModelOutputs', num_atoms: int \| None=None, num_graphs: int \| None=None) -> None |
| added | nvalchemi/models/base.py:BaseModelMixin.compute_hessian | — | (self, batch: Batch, *, strategy: Literal['vmap', 'loop']='vmap', row_chunk_size: int \| None=None) -> Batch |
| added | nvalchemi/models/base.py:BaseModelMixin.hessian_vector_product | — | (self, batch: Batch, vectors: torch.Tensor, *, create_graph: bool=False) -> torch.Tensor |
| added | nvalchemi/models/base.py:BaseModelMixin.narrowed_outputs | — | (self, outputs: Iterable[str]) -> Iterator[None] @ contextmanager |
| added | nvalchemi/models/base.py:BaseModelMixin.prepare_hessian | — | (self, batch: Batch) -> HessianOperator |
| added | nvalchemi/models/base.py:BaseModelMixin.requires_autograd | — | (self) -> bool @ property |
| added | nvalchemi/models/dftd3.py:DFTD3ModelWrapper.direct_derivative_keys | — | (self) -> set[str] |
| added | nvalchemi/models/gen/__init__.py:exports | — | __all__ = ['DemoDiffusionModel', 'DemoGANModel', 'GenerativeModelConfig', 'GenerativeModelMixin'] ; from __future__ import annotations ; from nvalchemi.models.gen.base import GenerativeModelConfig, GenerativeModelMixin ; from nvalchemi.models.gen.demo import DemoDiffusionModel, DemoGANModel |
| added | nvalchemi/models/gen/base.py:GenerativeModelConfig | — | class(BaseModel) |
| added | nvalchemi/models/gen/base.py:GenerativeModelConfig.outputs | — | Annotated[frozenset[str], Field(description="Batch fields the model's generated output carries (written or forwarded). Required: feeds GenerationPipeline link validation.")] = required |
| added | nvalchemi/models/gen/base.py:GenerativeModelConfig.prediction_outputs | — | Annotated[set[str] \| None, Field(default=None, description="Output keys the model predicts (e.g. {'flow'}). None defaults to {'flow'} at use time.")] = None |
| added | nvalchemi/models/gen/base.py:GenerativeModelConfig.required_inputs | — | Annotated[frozenset[str], Field(description="Batch fields the model's conditioning reads (empty = unconditional). Required: feeds GenerationPipeline link validation.")] = required |
| added | nvalchemi/models/gen/base.py:GenerativeModelMixin | — | class(abc.ABC) |
| added | nvalchemi/models/gen/base.py:GenerativeModelMixin.adapt_output | — | (self, raw: Any, data: AtomicData \| Batch) -> ModelOutputs |
| added | nvalchemi/models/gen/base.py:GenerativeModelMixin.model_config | — | GenerativeModelConfig = required |
| added | nvalchemi/models/gen/demo.py:DemoDiffusionModel | — | class(nn.Module, GenerativeModelMixin) |
| added | nvalchemi/models/gen/demo.py:DemoDiffusionModel.__init__ | — | (self, num_atoms: int=3, hidden: int=32) -> None |
| added | nvalchemi/models/gen/demo.py:DemoDiffusionModel.forward | — | (self, x: torch.Tensor, sigma: torch.Tensor, class_labels: torch.Tensor \| None=None) -> torch.Tensor |
| added | nvalchemi/models/gen/demo.py:DemoDiffusionModel.to_batch | — | (self, sample: TensorDict, cond_batch: Batch \| None=None) -> Batch |
| added | nvalchemi/models/gen/demo.py:DemoGANModel | — | class(nn.Module, GenerativeModelMixin) |
| added | nvalchemi/models/gen/demo.py:DemoGANModel.__init__ | — | (self, num_atoms: int=3, latent_dim: int=4, hidden: int=32) -> None |
| added | nvalchemi/models/gen/demo.py:DemoGANModel.decode | — | (self, z: torch.Tensor) -> torch.Tensor |
| added | nvalchemi/models/gen/demo.py:DemoGANModel.forward | — | (self, data: Any, *, x: torch.Tensor, **kwargs: Any) -> torch.Tensor |
| added | nvalchemi/models/gen/demo.py:DemoGANModel.to_batch | — | (self, sample: TensorDict, cond_batch: Batch \| None=None) -> Batch |
| added | nvalchemi/models/lj.py:LennardJonesModelWrapper.direct_derivative_keys | — | (self) -> set[str] |
| added | nvalchemi/specs.py:BaseSpec | — | class(BaseModel) |
| added | nvalchemi/specs.py:BaseSpec.accepts_kwarg | — | (self, name: str) -> bool |
| added | nvalchemi/specs.py:BaseSpec.build | — | (self, *args: Any, strict: bool=False, **extra_kwargs: Any) -> object |
| added | nvalchemi/specs.py:BaseSpec.cls_path | — | Annotated[str, AfterValidator(_ensure_importable), Field(description='Dotted import path of the target callable.')] = required |
| added | nvalchemi/specs.py:BaseSpec.timestamp | — | Annotated[str, Field(description='ISO-8601 UTC timestamp of spec creation.')] = required |
| added | nvalchemi/specs.py:create_model_spec | — | (target: Any, **kwargs: Any) -> BaseSpec |
| added | nvalchemi/specs.py:create_model_spec_from_json | — | (spec: dict[str, Any]) -> BaseSpec |
| changed | nvalchemi/training/__init__.py:exports | __all__ = ['BaseLossFunction', 'BaseSpec', 'CheckpointManifest', 'CheckpointHook', 'CheckpointValidator', 'ComposedLossFunction', 'ComposedLossOutput', 'DTypePolicy', 'ConstantWeight', 'CosineWeight', 'EnergyHuberLoss', 'EnergyMAELoss', 'EnergyMSELoss', 'ForceHuberLoss', 'ForceL2NormLoss', 'ForceMSELoss', 'DDPHook', 'EMAHook', 'FineTuningStrategy', 'LinearWeight', 'LossWeightSchedule', 'OptimizerConfig', 'PiecewiseWeight', 'ReductionContext', 'StressHuberLoss', 'StressMSELoss', 'LossTargetAssemblyProtocol', 'TrainingStage', 'TrainingStrategy', 'BatchValidationCallback', 'ValidationConfig', 'ValidationLoop', 'assemble_loss_targets', 'configure_dataloader', 'configure_parallelism', 'create_model_spec', 'create_model_spec_from_json', 'default_training_fn', 'freeze_unconfigured_models', 'loss_component_to_spec', 'load_checkpoint', 'move_to_devices', 'register_type_serializer', 'save_checkpoint', 'setup_optimizers', 'step_lr_schedulers', 'step_optimizers', 'zero_gradients'] ; from __future__ import annotations ; from nvalchemi.training._checkpoint import CheckpointManifest, CheckpointValidator, load_checkpoint, save_checkpoint ; from nvalchemi.training._spec import BaseSpec, create_model_spec, create_model_spec_from_json, register_type_serializer ; from nvalchemi.training._stages import TrainingStage ; from nvalchemi.training._validation import BatchValidationCallback, ValidationConfig, ValidationLoop ; from nvalchemi.training.finetune import FineTuningStrategy ; from nvalchemi.training.hooks import CheckpointHook, DDPHook, EMAHook ; from nvalchemi.training.losses import BaseLossFunction, ComposedLossFunction, ComposedLossOutput, ConstantWeight, CosineWeight, DTypePolicy, EnergyHuberLoss, EnergyMAELoss, EnergyMSELoss, ForceHuberLoss, ForceL2NormLoss, ForceMSELoss, LinearWeight, LossTargetAssemblyProtocol, LossWeightSchedule, PiecewiseWeight, ReductionContext, StressHuberLoss, StressMSELoss, assemble_loss_targets, loss_component_to_spec ; from nvalchemi.training.optimizers import OptimizerConfig, setup_optimizers, step_lr_schedulers, step_optimizers, zero_gradients ; from nvalchemi.training.runtime import configure_dataloader, configure_parallelism, freeze_unconfigured_models, move_to_devices ; from nvalchemi.training.strategy import TrainingStrategy, default_training_fn | __all__ = ['BaseLossFunction', 'BaseSpec', 'CheckpointManifest', 'CheckpointHook', 'CheckpointValidator', 'ComposedLossFunction', 'ComposedLossOutput', 'DTypePolicy', 'ConstantWeight', 'CosineWeight', 'EnergyHuberLoss', 'EnergyMAELoss', 'EnergyMSELoss', 'ForceHuberLoss', 'ForceL2NormLoss', 'ForceMSELoss', 'DDPHook', 'EMAHook', 'FineTuningStrategy', 'LinearWeight', 'LossWeightSchedule', 'OptimizerConfig', 'PiecewiseWeight', 'ReductionContext', 'StressHuberLoss', 'StressMSELoss', 'LossTargetAssemblyProtocol', 'TrainingStage', 'TrainingStrategy', 'BatchValidationCallback', 'ValidationConfig', 'ValidationLoop', 'assemble_loss_targets', 'configure_dataloader', 'configure_parallelism', 'create_model_spec', 'create_model_spec_from_json', 'default_training_fn', 'ensure_reiterable_validation_data', 'eval_configured_models', 'evaluating', 'freeze_unconfigured_models', 'loss_component_to_spec', 'load_checkpoint', 'move_to_devices', 'register_type_serializer', 'rehome_optimizer_state', 'save_checkpoint', 'setup_optimizers', 'step_lr_schedulers', 'step_optimizers', 'unwrap_model', 'zero_gradients'] ; from __future__ import annotations ; from nvalchemi.training._checkpoint import CheckpointManifest, CheckpointValidator, load_checkpoint, save_checkpoint ; from nvalchemi.training._spec import BaseSpec, create_model_spec, create_model_spec_from_json, register_type_serializer ; from nvalchemi.training._stages import TrainingStage ; from nvalchemi.training._validation import BatchValidationCallback, ValidationConfig, ValidationLoop, ensure_reiterable_validation_data ; from nvalchemi.training.finetune import FineTuningStrategy ; from nvalchemi.training.hooks import CheckpointHook, DDPHook, EMAHook ; from nvalchemi.training.losses import BaseLossFunction, ComposedLossFunction, ComposedLossOutput, ConstantWeight, CosineWeight, DTypePolicy, EnergyHuberLoss, EnergyMAELoss, EnergyMSELoss, ForceHuberLoss, ForceL2NormLoss, ForceMSELoss, LinearWeight, LossTargetAssemblyProtocol, LossWeightSchedule, PiecewiseWeight, ReductionContext, StressHuberLoss, StressMSELoss, assemble_loss_targets, loss_component_to_spec ; from nvalchemi.training.optimizers import OptimizerConfig, setup_optimizers, step_lr_schedulers, step_optimizers, zero_gradients ; from nvalchemi.training.runtime import configure_dataloader, configure_parallelism, eval_configured_models, evaluating, freeze_unconfigured_models, move_to_devices, rehome_optimizer_state, unwrap_model ; from nvalchemi.training.strategy import TrainingStrategy, default_training_fn |
| changed | nvalchemi/training/_checkpoint.py:load_checkpoint | (root_folder: Path \| str, checkpoint_index: int=-1, map_location: str \| torch.device \| None=None, model_names: Iterable[str] \| None=None, *, adapter: str \| None=None, adapter_kwargs: Mapping[str, Any] \| None=None, validators: Sequence[CheckpointValidator] \| None=None, hooks: Sequence[Any] \| None=None, training_fn: Any=None, strategy: Any \| None=None) -> CheckpointManifest \| dict[str, Any] | (root_folder: Path \| str, checkpoint_index: int=-1, map_location: str \| torch.device \| None=None, model_names: Iterable[str] \| None=None, *, adapter: str \| None=None, adapter_kwargs: Mapping[str, Any] \| None=None, validators: Sequence[CheckpointValidator] \| None=None, hooks: Sequence[Any] \| None=None, training_fn: Any=None, strategy: Any \| None=None, models: ModelInput \| None=None, **runtime_overrides: Any) -> CheckpointManifest \| dict[str, Any] |
| changed | nvalchemi/training/_checkpoint.py:save_checkpoint | (root_folder: Path \| str, models: dict[str, tuple[nn.Module, BaseSpec]] \| Any \| None=None, optimizers: dict[str, tuple[torch.optim.Optimizer, BaseSpec]] \| None=None, schedulers: dict[str, tuple[torch.optim.lr_scheduler.LRScheduler, BaseSpec]] \| None=None, associations: _Associations \| None=None, checkpoint_index: int=-1, strategy: Any \| None=None) -> int | (root_folder: Path \| str, models: dict[str, tuple[nn.Module, BaseSpec]] \| Any \| None=None, optimizers: dict[str, tuple[torch.optim.Optimizer, BaseSpec]] \| None=None, schedulers: dict[str, tuple[torch.optim.lr_scheduler.LRScheduler, BaseSpec]] \| None=None, associations: _Associations \| None=None, checkpoint_index: int=-1, strategy: Any \| None=None, save_trainable_state_only: bool=False) -> int |
| removed | nvalchemi/training/_spec.py:BaseSpec | class(BaseModel) | — |
| removed | nvalchemi/training/_spec.py:BaseSpec.accepts_kwarg | (self, name: str) -> bool | — |
| removed | nvalchemi/training/_spec.py:BaseSpec.build | (self, *args: Any, strict: bool=False, **extra_kwargs: Any) -> object | — |
| removed | nvalchemi/training/_spec.py:BaseSpec.cls_path | Annotated[str, AfterValidator(_ensure_importable), Field(description='Dotted import path of the target callable.')] = required | — |
| removed | nvalchemi/training/_spec.py:BaseSpec.timestamp | Annotated[str, Field(description='ISO-8601 UTC timestamp of spec creation.')] = required | — |
| removed | nvalchemi/training/_spec.py:create_model_spec | (target: Any, **kwargs: Any) -> BaseSpec | — |
| removed | nvalchemi/training/_spec.py:create_model_spec_from_json | (spec: dict[str, Any]) -> BaseSpec | — |
| added | nvalchemi/training/_spec_utils.py:DatasetRef | — | class(BaseModel) |
| added | nvalchemi/training/_spec_utils.py:DatasetRef.device | — | Annotated[str, Field(default='cpu', description='Device the dataset collates the rows it serves onto.')] = 'cpu' |
| added | nvalchemi/training/_spec_utils.py:DatasetRef.path | — | Annotated[str, Field(description='Filesystem path or URI of the store to read.')] = required |
| added | nvalchemi/training/_spec_utils.py:dataset_from_spec_dict | — | (spec: Mapping[str, Any], *, field: str) -> Dataset |
| added | nvalchemi/training/_spec_utils.py:dataset_spec_dict | — | (dataset: BatchDatasetProtocol, *, field: str, remedy: str \| None=None) -> dict[str, Any] |
| changed | nvalchemi/training/_validation.py:ValidationConfig.validation_data | Annotated[Iterable[Batch], PlainValidator(_ensure_reiterable_validation_data), Field(description='Re-iterable container (e.g. ``list``, ``DataLoader``, ``Dataset``) yielding :class:`~nvalchemi.data.Batch` instances. The strategy re-iterates this on every validation pass; one-shot generators and bare iterators are rejected at construction time.')] = required | Annotated[Iterable[Batch], PlainValidator(ensure_reiterable_validation_data), Field(description='Re-iterable container (e.g. ``list``, ``DataLoader``, ``Dataset``) yielding :class:`~nvalchemi.data.Batch` instances. The strategy re-iterates this on every validation pass; one-shot generators and bare iterators are rejected at construction time.')] = required |
| added | nvalchemi/training/_validation.py:ensure_reiterable_validation_data | — | (value: Any) -> Any |
| added | nvalchemi/training/distillation/__init__.py:exports | — | __all__ = ['BUILTIN_SIGNALS', 'FIFO', 'AdmissionPolicy', 'AtomicEnergyMatchingLoss', 'BoltzmannMatchingLoss', 'DistillationStrategy', 'EmbeddingMatchingLoss', 'EmbeddingProjector', 'EvictionPolicy', 'FitPolicy', 'HessianMatchingLoss', 'InProcessTeacherScorer', 'InitialStructures', 'InitialStructuresSource', 'NeighborListPolicy', 'OnPolicyConfig', 'OnPolicySettings', 'ReplayBuffer', 'ReplayEviction', 'ResizableSink', 'SUPPORTED_SIGNALS', 'SignalLevel', 'SignalNormalizer', 'TeacherLabelHook', 'TeacherLabels', 'TeacherScorer', 'TeacherSignal', 'WithinBudget', 'build_mixed_loader', 'default_distillation_fn', 'embedding_distillation_fn', 'hessian_distillation_fn', 'label_dataset', 'nonfinite_divergence', 'scorer_fields', 'signal_fields', 'signal_for_field'] ; from __future__ import annotations ; from nvalchemi.training.distillation.config import OnPolicyConfig, OnPolicySettings, ResizableSink ; from nvalchemi.training.distillation.hooks import TeacherLabelHook, nonfinite_divergence ; from nvalchemi.training.distillation.labeling import label_dataset ; from nvalchemi.training.distillation.losses import AtomicEnergyMatchingLoss, BoltzmannMatchingLoss, EmbeddingMatchingLoss, EmbeddingProjector, HessianMatchingLoss ; from nvalchemi.training.distillation.replay import FIFO, AdmissionPolicy, EvictionPolicy, ReplayBuffer, ReplayEviction, build_mixed_loader ; from nvalchemi.training.distillation.scoring import BUILTIN_SIGNALS, SUPPORTED_SIGNALS, InProcessTeacherScorer, NeighborListPolicy, SignalLevel, SignalNormalizer, TeacherLabels, TeacherScorer, TeacherSignal, scorer_fields, signal_fields, signal_for_field ; from nvalchemi.training.distillation.seeding import FitPolicy, InitialStructures, InitialStructuresSource, WithinBudget ; from nvalchemi.training.distillation.strategy import DistillationStrategy, default_distillation_fn, embedding_distillation_fn, hessian_distillation_fn |
| added | nvalchemi/training/distillation/config.py:OnPolicyConfig | — | class(OnPolicySettings) |
| added | nvalchemi/training/distillation/config.py:OnPolicyConfig.capture_sink | — | Annotated[DataSink \| None, Field(default=None, description="Runtime-only sink the labeling hook stages each segment's labeled frames in until the segment boundary drains them into the replay buffer; None builds a host-memory sink per segment, and a GPUBuffer keeps the staging on the generation device. The loop sizes it to (generation_steps + 1) frames per trajectory, growing a ResizableSink through resize(capacity) and refusing a smaller sink otherwise.")] = None |
| added | nvalchemi/training/distillation/config.py:OnPolicyConfig.convergence_criterion | — | (self) -> ConvergenceHook \| None @ property |
| added | nvalchemi/training/distillation/config.py:OnPolicyConfig.convergence_hook | — | Annotated[ConvergenceHook \| None, Field(default=None, description='Live criterion deciding when a generated trajectory is finished, in place of the fmax threshold. No recipe describes it, so it is runtime-only.')] = None |
| added | nvalchemi/training/distillation/config.py:OnPolicyConfig.divergence | — | Annotated[Callable[[Batch], Bool[torch.Tensor, 'G']] \| None, Field(default=None, description="Predicate over the live frame returning one boolean per graph, set where the trajectory diverged; the lifecycle moves the verdict onto the batch's device, then freezes and retires those graphs uncaptured. None flags non-finite positions or forces. Runtime-only: no recipe names it.")] = None |
| added | nvalchemi/training/distillation/config.py:OnPolicyConfig.dynamics | — | Annotated[BaseDynamics, Field(description='Propagator generating on-policy frames from the student. Any BaseDynamics: an integrator for trajectories, an optimizer for relaxation paths.')] = required |
| added | nvalchemi/training/distillation/config.py:OnPolicyConfig.initial_structures | — | Annotated[InitialStructuresSource, Field(description='Structures the generated trajectories start from: any InitialStructuresSource, of which InitialStructures is the reference. The initial batch, the backfill, and a restart all read from its one position. A bare dataset is wrapped in an unbudgeted InitialStructures.')] = required |
| added | nvalchemi/training/distillation/config.py:OnPolicyConfig.replay_admission | — | Annotated[AdmissionPolicy \| None, Field(default=None, description='Predicate over a batch of captured frames returning one boolean per graph; frames it refuses never enter the replay buffer. Runtime-only: no recipe names it.')] = None |
| added | nvalchemi/training/distillation/config.py:OnPolicyConfig.replay_eviction | — | Annotated[ReplayEviction \| EvictionPolicy, Field(default='fifo', description="Policy retiring frames from a full replay buffer: 'fifo', or a live EvictionPolicy instance, which is runtime-only and recorded as 'fifo' in the declarative settings.")] = 'fifo' |
| added | nvalchemi/training/distillation/config.py:OnPolicyConfig.settings | — | (self) -> OnPolicySettings @ property |
| added | nvalchemi/training/distillation/config.py:OnPolicyConfig.teacher_scorer | — | Annotated[TeacherScorer, Field(description='Scorer producing the teacher signals for generated frames. A label_fields declaration on a custom one lets the strategy check its fields against reference_dataset up front and makes a teacher_* field of its own usable as a loss target.')] = required |
| added | nvalchemi/training/distillation/config.py:OnPolicySettings | — | class(BaseModel) |
| added | nvalchemi/training/distillation/config.py:OnPolicySettings.batch_size | — | Annotated[int, Field(default=8, gt=0, description='Samples per training batch, split between the reference dataset and the replay buffer at replay_ratio.')] = 8 |
| added | nvalchemi/training/distillation/config.py:OnPolicySettings.fmax | — | Annotated[float \| None, Field(default=None, gt=0.0, description='Max force norm below which a generated trajectory counts as finished, which is what turns a relaxation run into a lifecycle. None runs no trajectory lifecycle: nothing graduates and nothing is backfilled, which is what a molecular-dynamics run wants.')] = None |
| added | nvalchemi/training/distillation/config.py:OnPolicySettings.generation_steps | — | Annotated[int, Field(default=100, gt=0, description='Propagator steps generated per segment.')] = 100 |
| added | nvalchemi/training/distillation/config.py:OnPolicySettings.label_frequency | — | Annotated[int, Field(default=100, gt=0, description="Propagator steps between teacher labelings, on top of the segment's own last frame. Larger values trade label density for generation throughput.")] = 100 |
| added | nvalchemi/training/distillation/config.py:OnPolicySettings.probe | — | Annotated[bool, Field(default=True, description="Whether the propagator's compute() is run on one initial structure at construction to check its declared keys against what it does. False skips that forward and defers a mismatch to the first step.")] = True |
| added | nvalchemi/training/distillation/config.py:OnPolicySettings.rank_seed_stride | — | Annotated[int, Field(default=1000003, gt=0, description="Seed-space distance between neighboring ranks: rank r moves seed, and every random_seed the propagator exposes, by r * rank_seed_stride. Both streams add a step counter to the base seed, so keep it above the run's step count. Set a different stride when a replicate launch's seeds would land on another rank's stride.")] = 1000003 |
| added | nvalchemi/training/distillation/config.py:OnPolicySettings.replay_capacity | — | Annotated[int \| None, Field(default=None, gt=0, description='Frames the replay buffer keeps; None leaves it unbounded.')] = None |
| added | nvalchemi/training/distillation/config.py:OnPolicySettings.replay_device | — | Annotated[str \| None, Field(default=None, description="Device the replay buffer keeps frames on, named as a string. None uses the device the reference dataset emits its batches on, so the mixture collates on one device, or host memory when the run has no reference dataset. An index-less 'cuda' names the device this rank has made current, which under a launcher is the one it pinned this rank to.")] = None |
| added | nvalchemi/training/distillation/config.py:OnPolicySettings.replay_eviction | — | Annotated[ReplayEviction, Field(default='fifo', description="Policy retiring frames from a full replay buffer, named as a recipe spells it; 'fifo' drops the oldest frames first.")] = 'fifo' |
| added | nvalchemi/training/distillation/config.py:OnPolicySettings.replay_ratio | — | Annotated[float, Field(ge=0.0, le=1.0, description='Fraction of every training batch drawn from the replay buffer; the rest comes from the reference dataset.')] = required |
| added | nvalchemi/training/distillation/config.py:OnPolicySettings.require_wrapped_student | — | Annotated[bool, Field(default=True, description="Whether a multi-rank run refuses to start unless the SETUP stage replaced models['student'] with a wrapper owning it, the way a DDPHook does. False skips that check with a warning for wrappers working in place (FSDP2 fully_shard, hook-based gradient synchronization) and leaves keeping the ranks' students in step to the caller.")] = True |
| added | nvalchemi/training/distillation/config.py:OnPolicySettings.samples_equilibrium | — | Annotated[bool \| None, Field(default=None, description="Whether the propagator samples an equilibrium ensemble. None infers it: a stage declaring samples_equilibrium=False on its class, as the relaxation optimizers do, or a convergence criterion is treated as not sampling equilibrium. True or False overrides that inference for a distribution-matching objective's guard.")] = None |
| added | nvalchemi/training/distillation/config.py:OnPolicySettings.seed | — | Annotated[int, Field(default=0, ge=0, description="Base seed of every segment's mixture sampler, combined with the segment index so consecutive segments draw different reference samples and replicate runs can be made independent.")] = 0 |
| added | nvalchemi/training/distillation/config.py:OnPolicySettings.training_steps_per_segment | — | Annotated[int, Field(gt=0, description="Training batches drawn from each segment's mixture, one optimizer step each unless an update hook vetoes the step.")] = required |
| added | nvalchemi/training/distillation/config.py:OnPolicySettings.weight_sync_frequency | — | Annotated[int, Field(default=1, gt=0, description='Segments between weight syncs to the propagator. Reserved: must be 1 while the propagator shares the student module.')] = 1 |
| added | nvalchemi/training/distillation/evaluation/__init__.py:exports | — | __all__ = ['AcceptanceBar', 'AcceptanceCheck', 'AcceptanceReport', 'AcceptanceThresholds', 'AccuracyMetrics', 'AccuracyQuantity', 'AccuracyQuantitySpec', 'BAR_FAMILIES', 'BUILTIN_ACCURACY_QUANTITIES', 'DEFAULT_BARS', 'ExtensivityMetrics', 'MeasurementRecord', 'MetricFamily', 'NonConservativeResidual', 'RDFComparison', 'RadialDistribution', 'StabilityMetrics', 'StabilityMonitor', 'StudentEvaluation', 'StudentVerdict', 'build_acceptance_report', 'compare_radial_distributions', 'evaluate_accuracy', 'extensivity_error', 'measured_bars', 'non_conservative_residual', 'radial_distribution', 'total_momentum'] ; from __future__ import annotations ; from nvalchemi.training.distillation.evaluation._export import MeasurementRecord ; from nvalchemi.training.distillation.evaluation.accuracy import BUILTIN_ACCURACY_QUANTITIES, AccuracyMetrics, AccuracyQuantity, AccuracyQuantitySpec, NonConservativeResidual, evaluate_accuracy, non_conservative_residual ; from nvalchemi.training.distillation.evaluation.report import BAR_FAMILIES, DEFAULT_BARS, AcceptanceBar, AcceptanceCheck, AcceptanceReport, AcceptanceThresholds, MetricFamily, StudentEvaluation, StudentVerdict, build_acceptance_report, measured_bars ; from nvalchemi.training.distillation.evaluation.stability import ExtensivityMetrics, RadialDistribution, RDFComparison, StabilityMetrics, StabilityMonitor, compare_radial_distributions, extensivity_error, radial_distribution, total_momentum |
| added | nvalchemi/training/distillation/evaluation/accuracy.py:AccuracyMetrics | — | class(MeasurementRecord) |
| added | nvalchemi/training/distillation/evaluation/accuracy.py:AccuracyMetrics.atomic_energies_mae | — | float \| None = None |
| added | nvalchemi/training/distillation/evaluation/accuracy.py:AccuracyMetrics.atomic_energies_rmse | — | float \| None = None |
| added | nvalchemi/training/distillation/evaluation/accuracy.py:AccuracyMetrics.energy_mae | — | float \| None = None |
| added | nvalchemi/training/distillation/evaluation/accuracy.py:AccuracyMetrics.energy_per_atom_mae | — | float \| None = None |
| added | nvalchemi/training/distillation/evaluation/accuracy.py:AccuracyMetrics.energy_per_atom_rmse | — | float \| None = None |
| added | nvalchemi/training/distillation/evaluation/accuracy.py:AccuracyMetrics.energy_rmse | — | float \| None = None |
| added | nvalchemi/training/distillation/evaluation/accuracy.py:AccuracyMetrics.errors | — | dict[str, dict[str, float]] = Field(default_factory=dict) |
| added | nvalchemi/training/distillation/evaluation/accuracy.py:AccuracyMetrics.force_cosine_aggregate | — | float \| None = None |
| added | nvalchemi/training/distillation/evaluation/accuracy.py:AccuracyMetrics.force_cosine_mean | — | float \| None = None |
| added | nvalchemi/training/distillation/evaluation/accuracy.py:AccuracyMetrics.force_nonfinite_atoms | — | int = 0 |
| added | nvalchemi/training/distillation/evaluation/accuracy.py:AccuracyMetrics.forces_mae | — | float \| None = None |
| added | nvalchemi/training/distillation/evaluation/accuracy.py:AccuracyMetrics.forces_rmse | — | float \| None = None |
| added | nvalchemi/training/distillation/evaluation/accuracy.py:AccuracyMetrics.name | — | str = required |
| added | nvalchemi/training/distillation/evaluation/accuracy.py:AccuracyMetrics.num_atoms | — | int = required |
| added | nvalchemi/training/distillation/evaluation/accuracy.py:AccuracyMetrics.num_graphs | — | int = required |
| added | nvalchemi/training/distillation/evaluation/accuracy.py:AccuracyMetrics.stress_mae | — | float \| None = None |
| added | nvalchemi/training/distillation/evaluation/accuracy.py:AccuracyMetrics.stress_rmse | — | float \| None = None |
| added | nvalchemi/training/distillation/evaluation/accuracy.py:AccuracyMetrics.to_dict | — | (self) -> dict[str, Any] |
| added | nvalchemi/training/distillation/evaluation/accuracy.py:AccuracyQuantitySpec | — | class() @ dataclasses.dataclass(frozen=True) |
| added | nvalchemi/training/distillation/evaluation/accuracy.py:AccuracyQuantitySpec.name | — | str = required |
| added | nvalchemi/training/distillation/evaluation/accuracy.py:AccuracyQuantitySpec.prediction_key | — | str = required |
| added | nvalchemi/training/distillation/evaluation/accuracy.py:AccuracyQuantitySpec.reference_key | — | str = required |
| added | nvalchemi/training/distillation/evaluation/accuracy.py:AccuracyQuantitySpec.signal | — | str \| None = None |
| added | nvalchemi/training/distillation/evaluation/accuracy.py:AccuracyQuantitySpec.supervised_loss | — | Callable[..., BaseLossFunction] \| None = None |
| added | nvalchemi/training/distillation/evaluation/accuracy.py:NonConservativeResidual | — | class(MeasurementRecord) |
| added | nvalchemi/training/distillation/evaluation/accuracy.py:NonConservativeResidual.amplitude | — | float = required |
| added | nvalchemi/training/distillation/evaluation/accuracy.py:NonConservativeResidual.force_floor | — | float = required |
| added | nvalchemi/training/distillation/evaluation/accuracy.py:NonConservativeResidual.force_floor_max | — | float = required |
| added | nvalchemi/training/distillation/evaluation/accuracy.py:NonConservativeResidual.force_rms | — | float = required |
| added | nvalchemi/training/distillation/evaluation/accuracy.py:NonConservativeResidual.loop_work_max_abs | — | float = required |
| added | nvalchemi/training/distillation/evaluation/accuracy.py:NonConservativeResidual.loop_work_mean_abs | — | float = required |
| added | nvalchemi/training/distillation/evaluation/accuracy.py:NonConservativeResidual.num_probes | — | int = required |
| added | nvalchemi/training/distillation/evaluation/accuracy.py:NonConservativeResidual.relative_floor | — | float = required |
| added | nvalchemi/training/distillation/evaluation/accuracy.py:NonConservativeResidual.relative_floor_max | — | float = required |
| added | nvalchemi/training/distillation/evaluation/accuracy.py:NonConservativeResidual.segments | — | int = required |
| added | nvalchemi/training/distillation/evaluation/accuracy.py:evaluate_accuracy | — | (model: BaseModelMixin, data: Iterable[Batch], *, targets: Literal['reference', 'teacher']='reference', quantities: Sequence[str \| AccuracyQuantitySpec] \| None=None, scorer: TeacherScorer \| BaseModelMixin \| None=None, target_keys: Mapping[str, str] \| None=None, loss_fn: ComposedLossFunction \| None=None, validation_fn: Callable[..., Any]=default_training_fn, grad_mode: Literal['auto', 'enabled', 'disabled']='auto', label_dtype: torch.dtype \| None=None, device: torch.device \| str \| None=None, distributed_manager: Any \| None=None, name: str='accuracy') -> AccuracyMetrics |
| added | nvalchemi/training/distillation/evaluation/accuracy.py:non_conservative_residual | — | (teacher: TeacherScorer \| BaseModelMixin, data: Iterable[Batch] \| Batch, *, num_loops: int=4, amplitude: float=0.05, segments: int=4, generator: torch.Generator \| None=None) -> NonConservativeResidual |
| added | nvalchemi/training/distillation/evaluation/report.py:AcceptanceBar | — | class() @ dataclasses.dataclass(frozen=True) |
| added | nvalchemi/training/distillation/evaluation/report.py:AcceptanceBar.attribute | — | str = '' |
| added | nvalchemi/training/distillation/evaluation/report.py:AcceptanceBar.check | — | str = '' |
| added | nvalchemi/training/distillation/evaluation/report.py:AcceptanceBar.comparison | — | Literal['<=', '>='] = '<=' |
| added | nvalchemi/training/distillation/evaluation/report.py:AcceptanceBar.families | — | tuple[str, ...] = required |
| added | nvalchemi/training/distillation/evaluation/report.py:AcceptanceBar.missing | — | str = '' |
| added | nvalchemi/training/distillation/evaluation/report.py:AcceptanceBar.name | — | str = required |
| added | nvalchemi/training/distillation/evaluation/report.py:AcceptanceBar.quantities | — | tuple[str, ...] = () |
| added | nvalchemi/training/distillation/evaluation/report.py:AcceptanceCheck | — | class() @ dataclasses.dataclass(frozen=True) |
| added | nvalchemi/training/distillation/evaluation/report.py:AcceptanceCheck.comparison | — | Literal['<=', '>='] = required |
| added | nvalchemi/training/distillation/evaluation/report.py:AcceptanceCheck.detail | — | str = '' |
| added | nvalchemi/training/distillation/evaluation/report.py:AcceptanceCheck.limit | — | float \| None = required |
| added | nvalchemi/training/distillation/evaluation/report.py:AcceptanceCheck.name | — | str = required |
| added | nvalchemi/training/distillation/evaluation/report.py:AcceptanceCheck.passed | — | bool = required |
| added | nvalchemi/training/distillation/evaluation/report.py:AcceptanceCheck.to_dict | — | (self) -> dict[str, Any] |
| added | nvalchemi/training/distillation/evaluation/report.py:AcceptanceCheck.value | — | float \| None = required |
| added | nvalchemi/training/distillation/evaluation/report.py:AcceptanceReport | — | class() @ dataclasses.dataclass(frozen=True) |
| added | nvalchemi/training/distillation/evaluation/report.py:AcceptanceReport.accepted | — | (self) -> bool @ property |
| added | nvalchemi/training/distillation/evaluation/report.py:AcceptanceReport.evaluations | — | tuple[StudentEvaluation, ...] = required |
| added | nvalchemi/training/distillation/evaluation/report.py:AcceptanceReport.scalars | — | (self) -> dict[str, float] |
| added | nvalchemi/training/distillation/evaluation/report.py:AcceptanceReport.thresholds | — | AcceptanceThresholds = required |
| added | nvalchemi/training/distillation/evaluation/report.py:AcceptanceReport.to_dict | — | (self) -> dict[str, Any] |
| added | nvalchemi/training/distillation/evaluation/report.py:AcceptanceReport.verdicts | — | tuple[StudentVerdict, ...] = required |
| added | nvalchemi/training/distillation/evaluation/report.py:AcceptanceThresholds | — | class(BaseModel) |
| added | nvalchemi/training/distillation/evaluation/report.py:AcceptanceThresholds.extra | — | Annotated[dict[str, float], Field(default_factory=dict, description='Limits of custom acceptance bars, keyed by the AcceptanceBar name they are applied under in the table the report is built with.')] = required |
| added | nvalchemi/training/distillation/evaluation/report.py:AcceptanceThresholds.max_energy_drift_per_atom_per_ns | — | Annotated[float \| None, Field(default=None, gt=0, description='Largest accepted fitted energy drift rate, in eV/atom/ns.')] = None |
| added | nvalchemi/training/distillation/evaluation/report.py:AcceptanceThresholds.max_energy_drift_per_atom_per_step | — | Annotated[float \| None, Field(default=None, gt=0, description='Largest accepted energy drift per atom per step.')] = None |
| added | nvalchemi/training/distillation/evaluation/report.py:AcceptanceThresholds.max_energy_per_atom_mae | — | Annotated[float \| None, Field(default=None, gt=0, description='Largest accepted energy MAE per atom.')] = None |
| added | nvalchemi/training/distillation/evaluation/report.py:AcceptanceThresholds.max_extensivity_error_per_atom | — | Annotated[float \| None, Field(default=None, gt=0, description='Largest accepted supercell energy-scaling error per atom.')] = None |
| added | nvalchemi/training/distillation/evaluation/report.py:AcceptanceThresholds.max_forces_mae | — | Annotated[float \| None, Field(default=None, gt=0, description='Largest accepted force MAE per Cartesian component.')] = None |
| added | nvalchemi/training/distillation/evaluation/report.py:AcceptanceThresholds.max_from_scratch_ratio | — | Annotated[float \| None, Field(default=None, gt=0, description="Largest accepted ratio of the distilled student's error to the equal-size from-scratch student's error on the same holdout. The ratio is taken on energy_per_atom_mae, forces_mae, and stress_mae, whichever both carry, and the worst one is kept. A limit of 1.0 demands a match and a limit below 1.0 demands a margin.")] = None |
| added | nvalchemi/training/distillation/evaluation/report.py:AcceptanceThresholds.max_momentum_drift | — | Annotated[float \| None, Field(default=None, gt=0, description="Largest accepted deviation of a graph's total momentum. Only meaningful under a momentum-conserving integrator: a stochastic thermostat exchanges momentum with its bath by design.")] = None |
| added | nvalchemi/training/distillation/evaluation/report.py:AcceptanceThresholds.max_rdf_jensen_shannon | — | Annotated[float \| None, Field(default=None, gt=0, le=1.0, description="Largest accepted Jensen-Shannon divergence between the student's and the reference trajectory's pair-distance histograms. Blind to which species a pair joins unless the compared distributions were themselves resolved to one pair of species.")] = None |
| added | nvalchemi/training/distillation/evaluation/report.py:AcceptanceThresholds.max_stress_mae | — | Annotated[float \| None, Field(default=None, gt=0, description='Largest accepted stress MAE.')] = None |
| added | nvalchemi/training/distillation/evaluation/report.py:AcceptanceThresholds.min_force_cosine | — | Annotated[float \| None, Field(default=None, ge=-1.0, le=1.0, description="Smallest accepted magnitude-weighted cosine similarity between the student's and the target force fields. It is read off force_cosine_aggregate rather than the per-atom mean, which the holdout's near-zero forces dominate.")] = None |
| added | nvalchemi/training/distillation/evaluation/report.py:AcceptanceThresholds.no_divergence | — | Annotated[bool, Field(default=False, description="Fail a student whose stability series stopped on a divergence, that is, whose StabilityMetrics carry a first_divergence_step. The drift bars alone cannot catch one: the series ends at the step the monitor's predicate fired, so they describe the segment before it, which can look perfectly conserved.")] = False |
| added | nvalchemi/training/distillation/evaluation/report.py:StudentEvaluation | — | class(MeasurementRecord) |
| added | nvalchemi/training/distillation/evaluation/report.py:StudentEvaluation.accuracy | — | AccuracyMetrics = required |
| added | nvalchemi/training/distillation/evaluation/report.py:StudentEvaluation.baseline_accuracy | — | AccuracyMetrics \| None = None |
| added | nvalchemi/training/distillation/evaluation/report.py:StudentEvaluation.extensivity | — | ExtensivityMetrics \| None = None |
| added | nvalchemi/training/distillation/evaluation/report.py:StudentEvaluation.extra | — | dict[str, dict[str, float]] = Field(default_factory=dict) |
| added | nvalchemi/training/distillation/evaluation/report.py:StudentEvaluation.from_dict | — | (cls, data: Mapping[str, Any]) -> StudentEvaluation @ classmethod |
| added | nvalchemi/training/distillation/evaluation/report.py:StudentEvaluation.name | — | str = required |
| added | nvalchemi/training/distillation/evaluation/report.py:StudentEvaluation.num_parameters | — | int \| None = None |
| added | nvalchemi/training/distillation/evaluation/report.py:StudentEvaluation.rdf | — | RDFComparison \| None = None |
| added | nvalchemi/training/distillation/evaluation/report.py:StudentEvaluation.stability | — | StabilityMetrics \| None = None |
| added | nvalchemi/training/distillation/evaluation/report.py:StudentEvaluation.to_dict | — | (self) -> dict[str, Any] |
| added | nvalchemi/training/distillation/evaluation/report.py:StudentEvaluation.weights | — | Literal['ema', 'raw'] \| None = None |
| added | nvalchemi/training/distillation/evaluation/report.py:StudentVerdict | — | class() @ dataclasses.dataclass(frozen=True) |
| added | nvalchemi/training/distillation/evaluation/report.py:StudentVerdict.accepted | — | bool = required |
| added | nvalchemi/training/distillation/evaluation/report.py:StudentVerdict.checks | — | tuple[AcceptanceCheck, ...] = required |
| added | nvalchemi/training/distillation/evaluation/report.py:StudentVerdict.name | — | str = required |
| added | nvalchemi/training/distillation/evaluation/report.py:StudentVerdict.to_dict | — | (self) -> dict[str, Any] |
| added | nvalchemi/training/distillation/evaluation/report.py:build_acceptance_report | — | (evaluations: Sequence[StudentEvaluation], thresholds: AcceptanceThresholds \| None=None, *, bars: Sequence[AcceptanceBar]=DEFAULT_BARS) -> AcceptanceReport |
| added | nvalchemi/training/distillation/evaluation/report.py:measured_bars | — | (*families: str, accuracy_quantities: Sequence[AccuracyQuantity] \| None=None, bars: Sequence[AcceptanceBar]=DEFAULT_BARS) -> frozenset[str] |
| added | nvalchemi/training/distillation/evaluation/stability.py:ExtensivityMetrics | — | class(MeasurementRecord) |
| added | nvalchemi/training/distillation/evaluation/stability.py:ExtensivityMetrics.max_error_per_atom | — | float = required |
| added | nvalchemi/training/distillation/evaluation/stability.py:ExtensivityMetrics.max_relative_error | — | float = required |
| added | nvalchemi/training/distillation/evaluation/stability.py:ExtensivityMetrics.mean_error_per_atom | — | float = required |
| added | nvalchemi/training/distillation/evaluation/stability.py:ExtensivityMetrics.num_graphs | — | int = required |
| added | nvalchemi/training/distillation/evaluation/stability.py:ExtensivityMetrics.repeats | — | tuple[int, int, int] = required |
| added | nvalchemi/training/distillation/evaluation/stability.py:RDFComparison | — | class(MeasurementRecord) |
| added | nvalchemi/training/distillation/evaluation/stability.py:RDFComparison.jensen_shannon | — | float = required |
| added | nvalchemi/training/distillation/evaluation/stability.py:RDFComparison.l1 | — | float = required |
| added | nvalchemi/training/distillation/evaluation/stability.py:RDFComparison.max_deviation | — | float = required |
| added | nvalchemi/training/distillation/evaluation/stability.py:RDFComparison.num_bins | — | int = required |
| added | nvalchemi/training/distillation/evaluation/stability.py:RDFComparison.pair | — | tuple[int, int] \| None = None |
| added | nvalchemi/training/distillation/evaluation/stability.py:RadialDistribution | — | class() @ dataclasses.dataclass(frozen=True) |
| added | nvalchemi/training/distillation/evaluation/stability.py:RadialDistribution.counts | — | torch.Tensor = required |
| added | nvalchemi/training/distillation/evaluation/stability.py:RadialDistribution.edges | — | torch.Tensor = required |
| added | nvalchemi/training/distillation/evaluation/stability.py:RadialDistribution.g_r | — | torch.Tensor = required |
| added | nvalchemi/training/distillation/evaluation/stability.py:RadialDistribution.num_atoms | — | int = required |
| added | nvalchemi/training/distillation/evaluation/stability.py:RadialDistribution.num_frames | — | int = required |
| added | nvalchemi/training/distillation/evaluation/stability.py:RadialDistribution.pair | — | tuple[int, int] \| None = None |
| added | nvalchemi/training/distillation/evaluation/stability.py:RadialDistribution.r_max | — | float = required |
| added | nvalchemi/training/distillation/evaluation/stability.py:RadialDistribution.to_dict | — | (self) -> dict[str, Any] |
| added | nvalchemi/training/distillation/evaluation/stability.py:compare_radial_distributions | — | (reference: RadialDistribution, candidate: RadialDistribution) -> RDFComparison |
| added | nvalchemi/training/distillation/evaluation/stability.py:extensivity_error | — | (model: TeacherScorer \| BaseModelMixin, data: Iterable[Batch] \| Batch, *, repeats: Sequence[int]=(2, 1, 1), extensive_keys: Collection[str]=DEFAULT_EXTENSIVE_SYSTEM_KEYS, intensive_keys: Collection[str]=DEFAULT_INTENSIVE_SYSTEM_KEYS, drop_keys: Collection[str]=()) -> ExtensivityMetrics |
| added | nvalchemi/training/distillation/evaluation/stability.py:radial_distribution | — | (frames: Iterable[Batch] \| Batch, *, r_max: float=6.0, num_bins: int=60, pair: Sequence[int] \| None=None) -> RadialDistribution |
| added | nvalchemi/training/distillation/hooks.py:TeacherLabelHook | — | class() |
| added | nvalchemi/training/distillation/hooks.py:TeacherLabelHook.__init__ | — | (self, teacher_scorer: TeacherScorer, sink: DataSink \| None=None, frequency: int=1, exit_status: int \| None=None) -> None |
| added | nvalchemi/training/distillation/hooks.py:TeacherLabelHook.labeled_step | — | (self) -> int \| None @ property |
| added | nvalchemi/training/distillation/hooks.py:TeacherLabelHook.stored_graphs | — | (self, step_count: int, batch: Batch) -> Bool[torch.Tensor, 'G'] \| None |
| added | nvalchemi/training/distillation/labeling.py:label_dataset | — | (dataset: BatchDatasetProtocol, scorer: TeacherScorer, store: StoreLike, *, batch_size: int=32, device: torch.device \| str \| None=None, resume: bool=True, keep_neighbors: bool=False, prefetch: bool \| Literal['auto']='auto') -> int |
| added | nvalchemi/training/distillation/losses/__init__.py:exports | — | __all__ = ['AtomicEnergyMatchingLoss', 'BoltzmannMatchingLoss', 'EmbeddingMatchingLoss', 'EmbeddingProjector', 'HessianMatchingLoss'] ; from __future__ import annotations ; from nvalchemi.training.distillation.losses.atomic_energy import AtomicEnergyMatchingLoss ; from nvalchemi.training.distillation.losses.distribution import BoltzmannMatchingLoss ; from nvalchemi.training.distillation.losses.embedding import EmbeddingMatchingLoss, EmbeddingProjector ; from nvalchemi.training.distillation.losses.hessian import HessianMatchingLoss |
| added | nvalchemi/training/distillation/losses/atomic_energy.py:AtomicEnergyMatchingLoss | — | class(BaseLossFunction) |
| added | nvalchemi/training/distillation/losses/atomic_energy.py:AtomicEnergyMatchingLoss.__init__ | — | (self, *, target_key: str='teacher_atomic_energies', prediction_key: str='predicted_atomic_energies', normalize_by_atom_count: bool=True, ignore_nonfinite: bool=True, dtype_policy: DTypePolicy='strict') -> None |
| added | nvalchemi/training/distillation/losses/atomic_energy.py:AtomicEnergyMatchingLoss.compute_residual | — | (self, pred: _NodeEnergies, target: _NodeEnergies, valid: _NodeMask) -> _NodeEnergies |
| added | nvalchemi/training/distillation/losses/atomic_energy.py:AtomicEnergyMatchingLoss.extra_repr | — | (self) -> str |
| added | nvalchemi/training/distillation/losses/atomic_energy.py:AtomicEnergyMatchingLoss.mask | — | (self, pred: _NodeEnergies, target: _NodeEnergies, ctx: ReductionContext, **kwargs: Any) -> _NodeMask |
| added | nvalchemi/training/distillation/losses/atomic_energy.py:AtomicEnergyMatchingLoss.reduce | — | (self, residual: _NodeEnergies, valid: _NodeMask, ctx: ReductionContext, **kwargs: Any) -> torch.Tensor |
| added | nvalchemi/training/distillation/losses/atomic_energy.py:AtomicEnergyMatchingLoss.requires_eval_grad | — | bool = False |
| added | nvalchemi/training/distillation/losses/distribution.py:BoltzmannMatchingLoss | — | class(BaseLossFunction) |
| added | nvalchemi/training/distillation/losses/distribution.py:BoltzmannMatchingLoss.__init__ | — | (self, *, target_key: str='teacher_energy', prediction_key: str='predicted_energy', beta: float=0.5, temperature: float=300.0, ignore_nonfinite: bool=True, dtype_policy: DTypePolicy='strict', world_batch: bool \| None=None, check_one_system: bool=True) -> None |
| added | nvalchemi/training/distillation/losses/distribution.py:BoltzmannMatchingLoss.compute_residual | — | (self, pred: Energy, target: Energy, valid: _EnergyMask) -> Energy |
| added | nvalchemi/training/distillation/losses/distribution.py:BoltzmannMatchingLoss.extra_repr | — | (self) -> str |
| added | nvalchemi/training/distillation/losses/distribution.py:BoltzmannMatchingLoss.mask | — | (self, pred: Energy, target: Energy, ctx: ReductionContext, **kwargs: Any) -> _EnergyMask |
| added | nvalchemi/training/distillation/losses/distribution.py:BoltzmannMatchingLoss.normalize | — | (self, pred: Energy, target: Energy, **kwargs: Any) -> tuple[Energy, Energy, ReductionContext] |
| added | nvalchemi/training/distillation/losses/distribution.py:BoltzmannMatchingLoss.reduce | — | (self, residual: Energy, valid: _EnergyMask, ctx: ReductionContext, **kwargs: Any) -> torch.Tensor |
| added | nvalchemi/training/distillation/losses/distribution.py:BoltzmannMatchingLoss.requires_eval_grad | — | bool = False |
| added | nvalchemi/training/distillation/losses/distribution.py:BoltzmannMatchingLoss.thermal_energy | — | (self) -> float @ property |
| added | nvalchemi/training/distillation/losses/embedding.py:EmbeddingMatchingLoss | — | class(BaseLossFunction) |
| added | nvalchemi/training/distillation/losses/embedding.py:EmbeddingMatchingLoss.__init__ | — | (self, *, target_key: str='teacher_node_embeddings', prediction_key: str='predicted_node_embeddings', normalize_by_atom_count: bool=True, ignore_nonfinite: bool=True, dtype_policy: DTypePolicy='strict') -> None |
| added | nvalchemi/training/distillation/losses/embedding.py:EmbeddingMatchingLoss.compute_residual | — | (self, pred: NodeEmbeddings, target: NodeEmbeddings, valid: _NodeMask) -> NodeEmbeddings |
| added | nvalchemi/training/distillation/losses/embedding.py:EmbeddingMatchingLoss.extra_repr | — | (self) -> str |
| added | nvalchemi/training/distillation/losses/embedding.py:EmbeddingMatchingLoss.mask | — | (self, pred: NodeEmbeddings, target: NodeEmbeddings, ctx: ReductionContext, **kwargs: Any) -> _NodeMask |
| added | nvalchemi/training/distillation/losses/embedding.py:EmbeddingMatchingLoss.reduce | — | (self, residual: NodeEmbeddings, valid: _NodeMask, ctx: ReductionContext, **kwargs: Any) -> torch.Tensor |
| added | nvalchemi/training/distillation/losses/embedding.py:EmbeddingMatchingLoss.requires_eval_grad | — | bool = False |
| added | nvalchemi/training/distillation/losses/embedding.py:EmbeddingMatchingLoss.validate | — | (self, pred: NodeEmbeddings, target: NodeEmbeddings) -> None |
| added | nvalchemi/training/distillation/losses/embedding.py:EmbeddingProjector | — | class(torch.nn.Module, BaseModelMixin) |
| added | nvalchemi/training/distillation/losses/embedding.py:EmbeddingProjector.__init__ | — | (self, in_features: int, out_features: int, *, hidden_features: int \| None=None, bias: bool=True, frozen_student: bool=False) -> None |
| added | nvalchemi/training/distillation/losses/embedding.py:EmbeddingProjector.compute_embeddings | — | (self, data: AtomicData \| Batch, **kwargs: Any) -> AtomicData \| Batch |
| added | nvalchemi/training/distillation/losses/embedding.py:EmbeddingProjector.embedding_shapes | — | (self) -> dict[str, tuple[int, ...]] @ property |
| added | nvalchemi/training/distillation/losses/embedding.py:EmbeddingProjector.extra_repr | — | (self) -> str |
| added | nvalchemi/training/distillation/losses/embedding.py:EmbeddingProjector.forward | — | (self, embeddings: NodeEmbeddings) -> NodeEmbeddings |
| added | nvalchemi/training/distillation/losses/hessian.py:HessianMatchingLoss | — | class(BaseLossFunction) |
| added | nvalchemi/training/distillation/losses/hessian.py:HessianMatchingLoss.__init__ | — | (self, *, target_key: str='teacher_hvp', prediction_key: str='predicted_hvp', normalize_by_atom_count: bool=True, ignore_nonfinite: bool=True, dtype_policy: DTypePolicy='strict') -> None |
| added | nvalchemi/training/distillation/losses/hessian.py:HessianMatchingLoss.compute_residual | — | (self, pred: Forces, target: Forces, valid: _ForceMask) -> Forces |
| added | nvalchemi/training/distillation/losses/hessian.py:HessianMatchingLoss.extra_repr | — | (self) -> str |
| added | nvalchemi/training/distillation/losses/hessian.py:HessianMatchingLoss.mask | — | (self, pred: Forces, target: Forces, ctx: ReductionContext, **kwargs: Any) -> _ForceMask |
| added | nvalchemi/training/distillation/losses/hessian.py:HessianMatchingLoss.reduce | — | (self, residual: Forces, valid: _ForceMask, ctx: ReductionContext, **kwargs: Any) -> torch.Tensor |
| added | nvalchemi/training/distillation/losses/hessian.py:HessianMatchingLoss.requires_eval_grad | — | bool = True |
| added | nvalchemi/training/distillation/replay.py:AdmissionPolicy | — | class(Protocol) @ runtime_checkable |
| added | nvalchemi/training/distillation/replay.py:EvictionPolicy | — | class(Protocol) @ runtime_checkable |
| added | nvalchemi/training/distillation/replay.py:EvictionPolicy.select | — | (self, buffer: Batch, incoming: Batch, capacity: int) -> _DropIndices |
| added | nvalchemi/training/distillation/replay.py:FIFO | — | class() |
| added | nvalchemi/training/distillation/replay.py:FIFO.select | — | (self, buffer: Batch, incoming: Batch, capacity: int) -> _DropIndices |
| added | nvalchemi/training/distillation/replay.py:ReplayBuffer | — | class() |
| added | nvalchemi/training/distillation/replay.py:ReplayBuffer.__init__ | — | (self, *, capacity: int \| None=None, eviction: ReplayEviction \| EvictionPolicy='fifo', admission: AdmissionPolicy \| None=None, device: torch.device \| str \| None=None) -> None |
| added | nvalchemi/training/distillation/replay.py:ReplayBuffer.dataset | — | (self) -> InMemoryDataset @ property |
| added | nvalchemi/training/distillation/replay.py:ReplayBuffer.extend | — | (self, frames: Batch) -> None |
| added | nvalchemi/training/distillation/replay.py:ReplayBuffer.schema | — | (self) -> frozenset[str] @ property |
| added | nvalchemi/training/distillation/replay.py:build_mixed_loader | — | (reference_dataset: BatchDatasetProtocol \| None, replay_buffer: ReplayBuffer, *, replay_ratio: float, batch_size: int, num_batches: int \| None=None, shuffle: bool=True, generator: torch.Generator \| None=None, seed: int=0) -> DataLoader |
| added | nvalchemi/training/distillation/scoring.py:InProcessTeacherScorer | — | class() |
| added | nvalchemi/training/distillation/scoring.py:InProcessTeacherScorer.__init__ | — | (self, teacher: BaseModelMixin, signals: Iterable[str \| TeacherSignal], *, dtype: torch.dtype \| None=None, neighbor_list: NeighborListPolicy='rebuild', autocast: bool \| torch.dtype \| None=False, probe_seed: int \| None=None) -> None |
| added | nvalchemi/training/distillation/scoring.py:InProcessTeacherScorer.label | — | (self, batch: Batch, *, probe_seed: int \| None=None) -> TeacherLabels |
| added | nvalchemi/training/distillation/scoring.py:InProcessTeacherScorer.label_hvp | — | (self, batch: Batch, probe: NodePositions) -> Forces |
| added | nvalchemi/training/distillation/scoring.py:TeacherScorer | — | class(Protocol) @ runtime_checkable |
| added | nvalchemi/training/distillation/scoring.py:TeacherScorer.label | — | (self, batch: Batch) -> TeacherLabels |
| added | nvalchemi/training/distillation/scoring.py:TeacherScorer.signals | — | frozenset[str] = required |
| added | nvalchemi/training/distillation/scoring.py:TeacherSignal | — | class() @ dataclasses.dataclass(frozen=True) |
| added | nvalchemi/training/distillation/scoring.py:TeacherSignal.extra_fields | — | tuple[str, ...] = () |
| added | nvalchemi/training/distillation/scoring.py:TeacherSignal.field | — | str = required |
| added | nvalchemi/training/distillation/scoring.py:TeacherSignal.fields | — | (self) -> tuple[str, ...] @ property |
| added | nvalchemi/training/distillation/scoring.py:TeacherSignal.level | — | SignalLevel = required |
| added | nvalchemi/training/distillation/scoring.py:TeacherSignal.model_output | — | str \| None = required |
| added | nvalchemi/training/distillation/scoring.py:TeacherSignal.name | — | str = required |
| added | nvalchemi/training/distillation/scoring.py:TeacherSignal.normalize | — | SignalNormalizer \| None = None |
| added | nvalchemi/training/distillation/scoring.py:scorer_fields | — | (scorer: TeacherScorer) -> tuple[str, ...] \| None |
| added | nvalchemi/training/distillation/scoring.py:signal_fields | — | (signals: Iterable[str \| TeacherSignal]) -> tuple[str, ...] |
| added | nvalchemi/training/distillation/scoring.py:signal_for_field | — | (field: str, signals: Iterable[str \| TeacherSignal] \| None=None) -> str \| None |
| added | nvalchemi/training/distillation/seeding.py:InitialStructures | — | class(OrderedStructureSampler) |
| added | nvalchemi/training/distillation/seeding.py:InitialStructures.from_spec_dict | — | (cls, spec: Mapping[str, Any]) -> InitialStructures @ classmethod |
| added | nvalchemi/training/distillation/seeding.py:InitialStructures.to_spec_dict | — | (self) -> dict[str, Any] |
| added | nvalchemi/training/distillation/strategy.py:DistillationStrategy | — | class(TrainingStrategy) |
| added | nvalchemi/training/distillation/strategy.py:DistillationStrategy.attach_teacher_labels | — | (self, batch: Batch) -> bool |
| added | nvalchemi/training/distillation/strategy.py:DistillationStrategy.from_checkpoint_dict | — | (cls, spec: Mapping[str, Any], *, models: strategy_validation.ModelInput \| None=None, hooks: Sequence[Any] \| None=None, training_fn: Any=None, validation_config: ValidationConfig \| None=None, on_policy: OnPolicyConfig \| None=None, reference_dataset: BatchDatasetProtocol \| None=None, **runtime_overrides: Any) -> DistillationStrategy @ classmethod |
| added | nvalchemi/training/distillation/strategy.py:DistillationStrategy.from_spec_dict | — | (cls, spec: Mapping[str, Any], *, models: strategy_validation.ModelInput \| None=None, hooks: Sequence[Any] \| None=None, training_fn: Any=None, validation_config: ValidationConfig \| None=None, on_policy: OnPolicyConfig \| None=None, reference_dataset: BatchDatasetProtocol \| None=None) -> DistillationStrategy @ classmethod |
| added | nvalchemi/training/distillation/strategy.py:DistillationStrategy.label_dtype | — | Annotated[torch.dtype \| None, Field(description="Floating-point dtype teacher labels are cast to when a batch is labeled on the fly. ``None`` infers it from the student's first floating-point parameter, never below float32; an explicit dtype is passed to the teacher scorer verbatim.")] = None |
| added | nvalchemi/training/distillation/strategy.py:DistillationStrategy.label_missing | — | Annotated[bool, Field(description='Whether a batch lacking the required ``teacher_*`` fields is labeled on the fly by a teacher forward pass, in training and validation alike. ``False`` skips the teacher, so an unlabeled batch surfaces as a missing loss target.')] = True |
| added | nvalchemi/training/distillation/strategy.py:DistillationStrategy.load_checkpoint | — | (cls, root_folder: Path \| str, checkpoint_index: int=-1, map_location: str \| torch.device \| None=None, *, models: strategy_validation.ModelInput \| None=None, hooks: Sequence[Any] \| None=None, training_fn: Any=None, validators: Sequence[Any] \| None=None, validation_config: ValidationConfig \| None=None, on_policy: OnPolicyConfig \| None=None, reference_dataset: BatchDatasetProtocol \| None=None, **runtime_overrides: Any) -> DistillationStrategy @ classmethod |
| added | nvalchemi/training/distillation/strategy.py:DistillationStrategy.on_policy | — | Annotated[OnPolicyConfig \| None, Field(default=None, exclude=True, description='Segment-loop configuration turning ``run`` into on-policy distillation. ``None`` keeps the offline loop over the dataloader the caller passes to ``run``.')] = None |
| added | nvalchemi/training/distillation/strategy.py:DistillationStrategy.reference_dataset | — | Annotated[BatchDatasetProtocol \| None, Field(default=None, exclude=True, description='Teacher-labeled dataset the on-policy mixture draws its ``1 - replay_ratio`` share from. Required whenever the ratio is below 1, and read only in on-policy mode.')] = None |
| added | nvalchemi/training/distillation/strategy.py:DistillationStrategy.replay_buffer | — | (self) -> ReplayBuffer \| None @ property |
| added | nvalchemi/training/distillation/strategy.py:DistillationStrategy.run | — | (self, dataloader: Iterable[Batch] \| None=None) -> None |
| added | nvalchemi/training/distillation/strategy.py:DistillationStrategy.structure_shard | — | (self) -> tuple[int, ...] @ property |
| added | nvalchemi/training/distillation/strategy.py:DistillationStrategy.teacher_scorer | — | (self) -> InProcessTeacherScorer @ property |
| added | nvalchemi/training/distillation/strategy.py:DistillationStrategy.teacher_signals | — | Annotated[frozenset[str] \| None, Field(description='Teacher signals produced for every scored batch. ``None`` derives them from the ``teacher_*`` targets the training and validation losses read; an explicit set must cover those and may request more, at the cost of re-scoring every batch a store labeled without the extra fields delivers.')] = None |
| added | nvalchemi/training/distillation/strategy.py:DistillationStrategy.to_spec_dict | — | (self) -> dict[str, Any] |
| added | nvalchemi/training/distillation/strategy.py:DistillationStrategy.validate | — | (self) -> dict[str, Any] \| None |
| added | nvalchemi/training/distillation/strategy.py:default_distillation_fn | — | (models: Mapping[str, BaseModelMixin], batch: Batch) -> dict[str, torch.Tensor] |
| added | nvalchemi/training/distillation/strategy.py:embedding_distillation_fn | — | (models: Mapping[str, BaseModelMixin], batch: Batch) -> dict[str, torch.Tensor] |
| added | nvalchemi/training/distillation/strategy.py:hessian_distillation_fn | — | (models: Mapping[str, BaseModelMixin], batch: Batch) -> dict[str, torch.Tensor] |
| added | nvalchemi/training/distributed.py:all_gather_objects | — | (obj: Any, manager: DistributedManager \| None=None) -> list[Any] |
| added | nvalchemi/training/distributed.py:all_gather_rows | — | (tensor: torch.Tensor, manager: DistributedManager \| None=None, *, differentiable: bool=True) -> tuple[torch.Tensor, slice] |
| added | nvalchemi/training/distributed.py:all_reduce_flags | — | (flag: bool \| int, manager: DistributedManager \| None=None) -> torch.Tensor |
| added | nvalchemi/training/finetune.py:FineTuningStrategy.compute_base_fingerprints | — | bool = Field(default=True, description='Used only when peft_config is provided. If True, compute base-model fingerprints and include them in strategy metadata for compatibility checks when loading PEFT checkpoints into a fresh base model.') |
| changed | nvalchemi/training/finetune.py:FineTuningStrategy.freeze_patterns | Annotated[tuple[str, ...], Field(description='Glob patterns excluded from training. Exclusions can be re-included by ``trainable_patterns``.')] = () | Annotated[tuple[str, ...], Field(description='Glob patterns excluded from training. Exclusions can be re-included by ``trainable_patterns``. Must be empty when ``peft_config`` is provided.')] = () |
| changed | nvalchemi/training/finetune.py:FineTuningStrategy.from_spec_dict | (cls, spec: dict[str, Any], *, models: strategy_validation.ModelInput \| None=None, hooks: list[Any] \| None=None, training_fn: Any=None) -> FineTuningStrategy @ classmethod | (cls, spec: dict[str, Any], *, models: strategy_validation.ModelInput \| None=None, hooks: list[Any] \| None=None, training_fn: Any=None, **runtime_overrides: Any) -> FineTuningStrategy @ classmethod |
| added | nvalchemi/training/finetune.py:FineTuningStrategy.get_flattened_registered_managed_parameter_names | — | (self) -> frozenset[str] |
| added | nvalchemi/training/finetune.py:FineTuningStrategy.get_flattened_registered_trainable_parameter_names | — | (self) -> frozenset[str] |
| added | nvalchemi/training/finetune.py:FineTuningStrategy.get_registered_managed_parameter_names | — | (self) -> Mapping[str, frozenset[str]] |
| added | nvalchemi/training/finetune.py:FineTuningStrategy.get_registered_trainable_parameter_names | — | (self) -> Mapping[str, frozenset[str]] |
| added | nvalchemi/training/finetune.py:FineTuningStrategy.peft_config | — | PeftConfig \| None = Field(default=None, description='Parameter-efficient fine-tuning configuration. When provided, its setup hooks run before module patches and trainable parameter selection.') |
| added | nvalchemi/training/finetune.py:FineTuningStrategy.register_base_fingerprints | — | (self, fingerprints: Mapping[str, str]) -> None |
| added | nvalchemi/training/finetune.py:FineTuningStrategy.register_managed_parameter_names | — | (self, names: Sequence[str], *, source: str) -> None |
| added | nvalchemi/training/finetune.py:FineTuningStrategy.register_trainable_parameter_names | — | (self, names: Sequence[str], *, source: str) -> None |
| added | nvalchemi/training/finetune.py:FineTuningStrategy.register_trainable_parameter_summary | — | (self, summary: Mapping[str, Mapping[str, int]]) -> None |
| added | nvalchemi/training/finetune.py:FineTuningStrategy.trainable_parameter_summary | — | (self) -> Mapping[str, dict[str, int]] @ property |
| changed | nvalchemi/training/finetune.py:FineTuningStrategy.trainable_patterns | Annotated[tuple[str, ...], Field(description='Glob patterns included in the trainable parameter allow-list. When no ``freeze_patterns`` are supplied, this is the complete allow-list.')] = () | Annotated[tuple[str, ...], Field(description='Glob patterns for model parameters to keep trainable. Without ``freeze_patterns``, matching parameters form the allow-list, together with parameters registered as trainable by setup hooks. Module patches and, when ``peft_config`` is configured, PEFT adapters register their parameters automatically; use these patterns to select any additional existing model parameters.')] = () |
| changed | nvalchemi/training/hooks/__init__.py:exports | __all__ = ['CheckpointHook', 'DDPHook', 'EMAHook', 'MixedPrecisionHook', 'TorchProfilerHook', 'ModulePatchHook', 'TrainableParameterHook', 'TrainingUpdateHook', 'TrainingUpdateOrchestrator'] ; from __future__ import annotations ; from nvalchemi.hooks import TorchProfilerHook ; from nvalchemi.training.hooks.checkpoint import CheckpointHook ; from nvalchemi.training.hooks.ddp import DDPHook ; from nvalchemi.training.hooks.ema import EMAHook ; from nvalchemi.training.hooks.finetune import ModulePatchHook, TrainableParameterHook ; from nvalchemi.training.hooks.mixed_precision import MixedPrecisionHook ; from nvalchemi.training.hooks.update import TrainingUpdateHook, TrainingUpdateOrchestrator | __all__ = ['CheckpointHook', 'BaseFingerprintHook', 'DDPHook', 'EMAHook', 'MixedPrecisionHook', 'TorchProfilerHook', 'FineTuningSummaryHook', 'ModulePatchHook', 'TrainableParameterHook', 'TrainingUpdateHook', 'TrainingUpdateOrchestrator'] ; from __future__ import annotations ; from nvalchemi.hooks import TorchProfilerHook ; from nvalchemi.training.hooks.checkpoint import CheckpointHook ; from nvalchemi.training.hooks.ddp import DDPHook ; from nvalchemi.training.hooks.ema import EMAHook ; from nvalchemi.training.hooks.finetune import FineTuningSummaryHook, ModulePatchHook, TrainableParameterHook ; from nvalchemi.training.hooks.mixed_precision import MixedPrecisionHook ; from nvalchemi.training.hooks.update import TrainingUpdateHook, TrainingUpdateOrchestrator ; from nvalchemi.training.peft.fingerprints import BaseFingerprintHook |
| added | nvalchemi/training/hooks/checkpoint.py:CheckpointHook.save_trainable_state_only | — | Annotated[bool, Field(description='Save only optimizer-selected model parameters plus buffers instead of full model state and track non-strict model state loading behavior.')] = False |
| added | nvalchemi/training/hooks/ddp.py:DDPHook.wrapped_keys | — | (self) -> frozenset[str] @ property |
| added | nvalchemi/training/hooks/finetune.py:FineTuningSummaryHook | — | class(BaseModel) |
| added | nvalchemi/training/hooks/finetune.py:FineTuningSummaryHook.frequency | — | ClassVar[int] = 1 |
| added | nvalchemi/training/hooks/finetune.py:FineTuningSummaryHook.on_register | — | (self, workflow: Any) -> None |
| added | nvalchemi/training/hooks/finetune.py:FineTuningSummaryHook.stage | — | ClassVar[None] = None |
| added | nvalchemi/training/hooks/finetune.py:ModulePatchHook.register_parameters | — | bool = Field(default=True, description="If True, register the patched module parameters as both trainable and managed under the fixed 'patch' source. Registered patch parameters are included in the final trainable allow-list by default and are protected from being overridden by other sources modifying the model.") |
| changed | nvalchemi/training/hooks/finetune.py:TrainableParameterHook.trainable_patterns | Annotated[tuple[str, ...], Field(description='Glob patterns for parameters to keep trainable. On their own they define the full trainable set (everything else is frozen).')] = () | Annotated[tuple[str, ...], Field(description='Glob patterns for parameters to keep trainable. When supplied without ``freeze_patterns``, they define the full trainable set, along with the trainable parameters registered by earlier hooks, e.g., ``ModulePatchHook``. Everything else is frozen.')] = () |
| changed | nvalchemi/training/losses/__init__.py:exports | __all__ = ['BaseLossFunction', 'ComposedLossFunction', 'ComposedLossOutput', 'DTypePolicy', 'ConstantWeight', 'CosineWeight', 'EnergyHuberLoss', 'EnergyMAELoss', 'EnergyMSELoss', 'ForceHuberLoss', 'ForceL2NormLoss', 'ForceMSELoss', 'LinearWeight', 'LossWeightSchedule', 'LossTargetAssemblyProtocol', 'PiecewiseWeight', 'ReductionContext', 'StressHuberLoss', 'StressMSELoss', 'assemble_loss_targets', 'assert_same_shape', 'frobenius_mse', 'loss_component_to_spec', 'per_graph_mean', 'per_graph_sum'] ; from __future__ import annotations ; from nvalchemi.training.losses.base import LossWeightSchedule ; from nvalchemi.training.losses.composition import BaseLossFunction, ComposedLossFunction, ComposedLossOutput, DTypePolicy, LossTargetAssemblyProtocol, ReductionContext, assemble_loss_targets, assert_same_shape, loss_component_to_spec ; from nvalchemi.training.losses.reductions import frobenius_mse, per_graph_mean, per_graph_sum ; from nvalchemi.training.losses.schedules import ConstantWeight, CosineWeight, LinearWeight, PiecewiseWeight ; from nvalchemi.training.losses.terms import EnergyHuberLoss, EnergyMAELoss, EnergyMSELoss, ForceHuberLoss, ForceL2NormLoss, ForceMSELoss, StressHuberLoss, StressMSELoss | __all__ = ['BaseLossFunction', 'ComposedLossFunction', 'ComposedLossOutput', 'DTypePolicy', 'ConstantWeight', 'CosineWeight', 'EnergyHuberLoss', 'EnergyMAELoss', 'EnergyMSELoss', 'ForceHuberLoss', 'ForceL2NormLoss', 'ForceMSELoss', 'LinearWeight', 'LossWeightSchedule', 'LossTargetAssemblyProtocol', 'PiecewiseWeight', 'ReductionContext', 'StressHuberLoss', 'StressMSELoss', 'assemble_loss_targets', 'assert_same_shape', 'frobenius_mse', 'graph_balanced_mean', 'loss_component_to_spec', 'masked_mean', 'per_graph_mean', 'per_graph_sum'] ; from __future__ import annotations ; from nvalchemi.training.losses.base import LossWeightSchedule ; from nvalchemi.training.losses.composition import BaseLossFunction, ComposedLossFunction, ComposedLossOutput, DTypePolicy, LossTargetAssemblyProtocol, ReductionContext, assemble_loss_targets, assert_same_shape, loss_component_to_spec ; from nvalchemi.training.losses.reductions import frobenius_mse, graph_balanced_mean, masked_mean, per_graph_mean, per_graph_sum ; from nvalchemi.training.losses.schedules import ConstantWeight, CosineWeight, LinearWeight, PiecewiseWeight ; from nvalchemi.training.losses.terms import EnergyHuberLoss, EnergyMAELoss, EnergyMSELoss, ForceHuberLoss, ForceL2NormLoss, ForceMSELoss, StressHuberLoss, StressMSELoss |
| added | nvalchemi/training/losses/reductions.py:graph_balanced_mean | — | (residual: Float[torch.Tensor, 'V ...'], valid: Num[torch.Tensor, 'V ...'], batch_idx: BatchIndices \| None, num_graphs: int \| None, *, loss_name: str) -> tuple[Float[torch.Tensor, ''], Float[torch.Tensor, 'B']] |
| added | nvalchemi/training/losses/reductions.py:masked_mean | — | (residual: Float[torch.Tensor, 'V ...'], valid: Num[torch.Tensor, 'V ...'], *, graph_balanced: bool, batch_idx: BatchIndices \| None=None, num_graphs: int \| None=None, batch: Any \| None=None, loss_name: str='masked_mean') -> tuple[Float[torch.Tensor, ''], Float[torch.Tensor, 'B'] \| None] |
| added | nvalchemi/training/peft/__init__.py:exports | — | __all__ = ['BaseFingerprintHook', 'PeftConfig', 'PeftMethodRegistration', 'available_peft_methods', 'compute_base_fingerprints', 'load_peft_checkpoint_into_model', 'register_peft_method', 'validate_base_fingerprints'] ; from __future__ import annotations ; from nvalchemi.training.peft.config import PeftConfig ; from nvalchemi.training.peft.fingerprints import BaseFingerprintHook, compute_base_fingerprints, validate_base_fingerprints ; from nvalchemi.training.peft.loading import load_peft_checkpoint_into_model ; from nvalchemi.training.peft.registry import PeftMethodRegistration, available_peft_methods, register_peft_method |
| added | nvalchemi/training/peft/config.py:PeftConfig | — | class(BaseModel) |
| added | nvalchemi/training/peft/config.py:PeftConfig.from_spec_dict | — | (cls, spec: Mapping[str, Any], *, import_path_validator: Callable[[str, str], None] \| None=None) -> PeftConfig @ classmethod |
| added | nvalchemi/training/peft/config.py:PeftConfig.peft_method | — | str = Field(frozen=True) |
| added | nvalchemi/training/peft/config.py:PeftConfig.to_spec_dict | — | (self) -> dict[str, Any] |
| added | nvalchemi/training/peft/config.py:build_peft_setup_hooks | — | (config: PeftConfig, strategy_data: Mapping[str, Any]) -> list[Any] |
| added | nvalchemi/training/peft/fingerprints.py:BaseFingerprintHook | — | class(BaseModel) |
| added | nvalchemi/training/peft/fingerprints.py:BaseFingerprintHook.frequency | — | ClassVar[int] = 1 |
| added | nvalchemi/training/peft/fingerprints.py:BaseFingerprintHook.on_register | — | (self, workflow: Any) -> None |
| added | nvalchemi/training/peft/fingerprints.py:BaseFingerprintHook.stage | — | ClassVar[None] = None |
| added | nvalchemi/training/peft/fingerprints.py:compute_base_fingerprints | — | (models: Mapping[str, nn.Module]) -> dict[str, str] |
| added | nvalchemi/training/peft/fingerprints.py:validate_base_fingerprints | — | (models: strategy_validation.ModelInput, saved_fingerprints: Mapping[str, Any]) -> None |
| added | nvalchemi/training/peft/loading.py:load_peft_checkpoint_into_model | — | (model: nn.Module, checkpoint_dir: Path \| str, *, checkpoint_index: int=-1, model_name: str='main', map_location: str \| torch.device \| None='cpu', merge: bool=True, strict: bool=True, allowed_import_paths: Collection[str] \| None=None, trust_remote_code: bool=False) -> nn.Module |
| added | nvalchemi/training/peft/lora.py:LoRAConfig | — | class(PeftConfig) |
| added | nvalchemi/training/peft/lora.py:LoRAConfig.alpha | — | float = 1.0 |
| added | nvalchemi/training/peft/lora.py:LoRAConfig.lora_dropout | — | float = 0.0 |
| added | nvalchemi/training/peft/lora.py:LoRAConfig.lora_target_patterns | — | tuple[str, ...] = Field(min_length=1) |
| added | nvalchemi/training/peft/lora.py:LoRAConfig.peft_method | — | Literal['lora'] = Field(default=LORA_PEFT_METHOD, frozen=True) |
| added | nvalchemi/training/peft/lora.py:LoRAConfig.rank | — | int = 8 |
| added | nvalchemi/training/peft/lora.py:LoRAConfig.to_spec_dict | — | (self) -> dict[str, Any] |
| added | nvalchemi/training/peft/lora.py:LoRAConfig.wrap_mlp | — | bool = False |
| added | nvalchemi/training/peft/lora.py:LoRAConfig.wrapper_registrations | — | LoRAWrapperRegistrations \| None = None |
| added | nvalchemi/training/peft/lora.py:merge_lora_into_model | — | (model: nn.Module, *, strict: bool=True) -> nn.Module |
| added | nvalchemi/training/peft/lora.py:register_lora_method | — | () -> None |
| added | nvalchemi/training/peft/lora_hook.py:LoRAHook | — | class(BaseModel) |
| added | nvalchemi/training/peft/lora_hook.py:LoRAHook.frequency | — | ClassVar[int] = 1 |
| added | nvalchemi/training/peft/lora_hook.py:LoRAHook.lora_config | — | LoRAConfig = required |
| added | nvalchemi/training/peft/lora_hook.py:LoRAHook.on_register | — | (self, workflow: Any) -> None |
| added | nvalchemi/training/peft/lora_hook.py:LoRAHook.register_parameters | — | bool = True |
| added | nvalchemi/training/peft/lora_hook.py:LoRAHook.stage | — | ClassVar[None] = None |
| added | nvalchemi/training/peft/lora_wrappers.py:CuEquivarianceLoRALinear | — | class(nn.Module, LoRALayer) |
| added | nvalchemi/training/peft/lora_wrappers.py:CuEquivarianceLoRALinear.__init__ | — | (self, base_layer: nn.Module, rank: int, alpha: float, dropout: float=0.0, init: _peft.LoRAInit='default', **kwargs: Any) -> None |
| added | nvalchemi/training/peft/lora_wrappers.py:CuEquivarianceLoRALinear.forward | — | (self, x: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor |
| added | nvalchemi/training/peft/lora_wrappers.py:CuEquivarianceLoRALinear.is_compatible | — | (cls, base_layer: nn.Module) -> bool @ classmethod |
| added | nvalchemi/training/peft/lora_wrappers.py:CuEquivarianceLoRALinear.lora_A | — | (self) -> nn.Parameter @ property |
| added | nvalchemi/training/peft/lora_wrappers.py:CuEquivarianceLoRALinear.lora_B | — | (self) -> nn.Parameter @ property |
| added | nvalchemi/training/peft/lora_wrappers.py:CuEquivarianceLoRALinear.merge_into_base | — | (self) -> None @ torch.no_grad() |
| added | nvalchemi/training/peft/lora_wrappers.py:E3NNFullyConnectedLoRALayer | — | class(nn.Module, LoRALayer) |
| added | nvalchemi/training/peft/lora_wrappers.py:E3NNFullyConnectedLoRALayer.__init__ | — | (self, base_layer: nn.Module, rank: int, alpha: float, dropout: float=0.0, init: _peft.LoRAInit='default', **kwargs: Any) -> None |
| added | nvalchemi/training/peft/lora_wrappers.py:E3NNFullyConnectedLoRALayer.forward | — | (self, x: torch.Tensor) -> torch.Tensor |
| added | nvalchemi/training/peft/lora_wrappers.py:E3NNFullyConnectedLoRALayer.is_compatible | — | (cls, base_layer: nn.Module) -> bool @ classmethod |
| added | nvalchemi/training/peft/lora_wrappers.py:E3NNFullyConnectedLoRALayer.merge_into_base | — | (self) -> None @ torch.no_grad() |
| added | nvalchemi/training/peft/lora_wrappers.py:EquivariantLoRALinear | — | class(nn.Module, LoRALayer) |
| added | nvalchemi/training/peft/lora_wrappers.py:EquivariantLoRALinear.__init__ | — | (self, base_layer: nn.Module, rank: int, alpha: float, dropout: float=0.0, init: _peft.LoRAInit='default', **kwargs: Any) -> None |
| added | nvalchemi/training/peft/lora_wrappers.py:EquivariantLoRALinear.forward | — | (self, x: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor |
| added | nvalchemi/training/peft/lora_wrappers.py:EquivariantLoRALinear.is_compatible | — | (cls, base_layer: nn.Module) -> bool @ classmethod |
| added | nvalchemi/training/peft/lora_wrappers.py:EquivariantLoRALinear.lora_A | — | (self) -> nn.Parameter @ property |
| added | nvalchemi/training/peft/lora_wrappers.py:EquivariantLoRALinear.lora_B | — | (self) -> nn.Parameter @ property |
| added | nvalchemi/training/peft/lora_wrappers.py:EquivariantLoRALinear.merge_into_base | — | (self) -> None @ torch.no_grad() |
| added | nvalchemi/training/peft/lora_wrappers.py:available_lora_wrappers | — | () -> LoRAWrapperRegistrations |
| added | nvalchemi/training/peft/registry.py:PeftMethodRegistration | — | class() @ dataclass(frozen=True) |
| added | nvalchemi/training/peft/registry.py:PeftMethodRegistration.apply_peft_from_checkpoint_metadata | — | _ApplyPeftFromCheckpointMetadata = required |
| added | nvalchemi/training/peft/registry.py:PeftMethodRegistration.build_peft_setup_hooks | — | Callable[[PeftConfig, Mapping[str, Any]], list[Any]] = required |
| added | nvalchemi/training/peft/registry.py:PeftMethodRegistration.config_cls | — | type[PeftConfig] = required |
| added | nvalchemi/training/peft/registry.py:PeftMethodRegistration.merge_peft | — | Callable[..., nn.Module] = required |
| added | nvalchemi/training/peft/registry.py:PeftMethodRegistration.peft_method | — | str = required |
| added | nvalchemi/training/peft/registry.py:available_peft_methods | — | () -> tuple[str, ...] |
| added | nvalchemi/training/peft/registry.py:get_peft_registration_by_config | — | (config: PeftConfig) -> PeftMethodRegistration |
| added | nvalchemi/training/peft/registry.py:get_peft_registration_by_method | — | (method: Any) -> PeftMethodRegistration |
| added | nvalchemi/training/peft/registry.py:register_peft_method | — | (method: str, *, config_cls: type[PeftConfig], build_peft_setup_hooks: Callable[[PeftConfig, Mapping[str, Any]], list[Any]], apply_peft_from_checkpoint_metadata: _ApplyPeftFromCheckpointMetadata, merge_peft: Callable[..., nn.Module]) -> PeftMethodRegistration |
| added | nvalchemi/training/reference_energies.py:AtomicReferenceEnergyFit | — | class() @ dataclasses.dataclass(frozen=True, slots=True) |
| added | nvalchemi/training/reference_energies.py:AtomicReferenceEnergyFit.atomic_numbers | — | tuple[int, ...] = required |
| added | nvalchemi/training/reference_energies.py:AtomicReferenceEnergyFit.intercept | — | float = required |
| added | nvalchemi/training/reference_energies.py:AtomicReferenceEnergyFit.num_structures | — | int = required |
| added | nvalchemi/training/reference_energies.py:AtomicReferenceEnergyFit.rank | — | int = required |
| added | nvalchemi/training/reference_energies.py:AtomicReferenceEnergyFit.reference_energies | — | dict[int, float] = required |
| added | nvalchemi/training/reference_energies.py:AtomicReferenceEnergyFit.residual_sum_squares | — | float = required |
| added | nvalchemi/training/reference_energies.py:fit_atomic_reference_energies | — | (batches: Iterable[Batch], *, atomic_numbers: Iterable[int] \| None=None, energy_key: str='energy', baseline_energy_fn: BaselineEnergyFn \| None=None, device: torch.device \| str \| None=None, max_batches: int \| None=None, fit_intercept: bool=False) -> AtomicReferenceEnergyFit |
| added | nvalchemi/training/runtime.py:eval_configured_models | — | (models: dict[str, torch.nn.Module] \| torch.nn.ModuleDict, optimizer_configs: Mapping[str, object]) -> Iterator[None] @ contextmanager |
| added | nvalchemi/training/runtime.py:evaluating | — | (module: torch.nn.Module) -> Iterator[None] @ contextmanager |
| added | nvalchemi/training/runtime.py:rehome_optimizer_state | — | (optimizer: torch.optim.Optimizer) -> None |
| added | nvalchemi/training/runtime.py:unwrap_model | — | (model: _ModelT) -> _ModelT |
| changed | nvalchemi/training/strategy.py:TrainingStrategy.devices | list[torch.device] = Field(default_factory=lambda: [torch.device('cpu')], description='One device shared by all models, or one device per model for helper placement. Named-model ``run`` currently supports one device only.') | list[torch.device] = Field(default_factory=lambda: [torch.device('cpu')], description="One device shared by all models, or one entry per model naming that same device; named-model ``run`` stages its batch on the first, so two distinct devices are refused at run time. An index-less 'cuda' is resolved to the process's current device before the entries are compared.") |
| changed | nvalchemi/training/strategy.py:TrainingStrategy.from_checkpoint_dict | (cls, spec: Mapping[str, Any], *, models: strategy_validation.ModelInput \| None=None, hooks: Sequence[Hook \| TrainingUpdateHook \| TrainingUpdateOrchestrator] \| None=None, training_fn: Callable[..., Mapping[str, torch.Tensor]] \| str \| None=None) -> TrainingStrategy @ classmethod | (cls, spec: Mapping[str, Any], *, models: strategy_validation.ModelInput \| None=None, hooks: Sequence[Hook \| TrainingUpdateHook \| TrainingUpdateOrchestrator] \| None=None, training_fn: Callable[..., Mapping[str, torch.Tensor]] \| str \| None=None, **runtime_overrides: Any) -> TrainingStrategy @ classmethod |
| changed | nvalchemi/training/strategy.py:TrainingStrategy.from_spec_dict | (cls, spec: Mapping[str, Any], *, models: strategy_validation.ModelInput \| None=None, hooks: Sequence[Hook \| TrainingUpdateHook \| TrainingUpdateOrchestrator] \| None=None, training_fn: Callable[..., Mapping[str, torch.Tensor]] \| str \| None=None) -> TrainingStrategy @ classmethod | (cls, spec: Mapping[str, Any], *, models: strategy_validation.ModelInput \| None=None, hooks: Sequence[Hook \| TrainingUpdateHook \| TrainingUpdateOrchestrator] \| None=None, training_fn: Callable[..., Mapping[str, torch.Tensor]] \| str \| None=None, **runtime_overrides: Any) -> TrainingStrategy @ classmethod |
| changed | nvalchemi/training/strategy.py:TrainingStrategy.load_checkpoint | (cls, root_folder: Path \| str, checkpoint_index: int=-1, map_location: str \| torch.device \| None=None, *, hooks: Sequence[Hook \| TrainingUpdateHook \| TrainingUpdateOrchestrator] \| None=None, training_fn: Callable[..., Mapping[str, torch.Tensor]] \| str \| None=None, validators: Sequence[CheckpointValidator] \| None=None) -> TrainingStrategy @ classmethod | (cls, root_folder: Path \| str, checkpoint_index: int=-1, map_location: str \| torch.device \| None=None, *, models: strategy_validation.ModelInput \| None=None, hooks: Sequence[Hook \| TrainingUpdateHook \| TrainingUpdateOrchestrator] \| None=None, training_fn: Callable[..., Mapping[str, torch.Tensor]] \| str \| None=None, validators: Sequence[CheckpointValidator] \| None=None, **runtime_overrides: Any) -> TrainingStrategy @ classmethod |
| changed | nvalchemi/training/strategy.py:TrainingStrategy.save_checkpoint | (self, root_folder: Path \| str, *, checkpoint_index: int=-1) -> int | (self, root_folder: Path \| str, *, checkpoint_index: int=-1, save_trainable_state_only: bool=False) -> int |

</details>

解析失败：{"baseline": [], "candidate": []}

### 依赖变化

| 依赖范围（保留 marker） | 基线 | 候选 |
|---|---|---|
| requires-python | >=3.11,<3.14 | >=3.11,<3.15 |
| dependencies | ["nvalchemi-toolkit-ops>=0.4.1", "torch>=2.8", "click", "loguru", "numpy<2.4", "jaxtyping>=0.3.2", "tensordict>=0.11.0", "pydantic>=2.11.7", "dm-tree>=0.1.8", "plum-dispatch>=2.5.7", "zarr>=3", "periodictable==2.0.2", "rich>=13.0.0", "plotext", "nvidia-physicsnemo>=2.0.0"] | ["nvalchemi-toolkit-ops>=0.5.0", "torch>=2.8", "click", "loguru", "numpy>=2,<3", "jaxtyping>=0.3.2", "tensordict>=0.11.0", "pydantic>=2.11.7", "plum-dispatch>=2.5.7", "zarr>=3", "periodictable==2.0.2", "rich>=13.0.0", "plotext<6", "nvidia-physicsnemo>=2.2.0"] |
| extra:aimnet | ["aimnet"] | ["aimnet"] |
| extra:ase | ["ase>=3.27.0"] | ["ase>=3.27.0"] |
| extra:cu12 | ["nvalchemi-toolkit-ops[torch-cu12]>=0.4.1; sys_platform != 'darwin'", "nvidia-physicsnemo[cu12]>=2.0.0; sys_platform != 'darwin'", "cuml-cu12>=25.6.0; sys_platform != 'darwin'", "torch; sys_platform != 'darwin'", "torchvision; sys_platform != 'darwin'", "cuequivariance-ops-torch-cu12>=0.8.0; sys_platform != 'darwin'"] | ["nvalchemi-toolkit-ops[torch-cu12]>=0.5.0; sys_platform != 'darwin'", "nvidia-physicsnemo>=2.2.0; sys_platform != 'darwin'", "torch; sys_platform != 'darwin'", "torchvision; sys_platform != 'darwin'", "cuequivariance-ops-torch-cu12>=0.8.0; sys_platform != 'darwin'"] |
| extra:cu13 | ["nvalchemi-toolkit-ops[torch-cu13]>=0.4.1; sys_platform != 'darwin'", "nvidia-physicsnemo[cu13]>=2.0.0; sys_platform != 'darwin'", "cuml-cu13>=25.6.0; sys_platform != 'darwin'", "torch; sys_platform != 'darwin'", "torchvision; sys_platform != 'darwin'", "cuequivariance-ops-torch-cu13>=0.8.0; sys_platform != 'darwin'"] | ["nvalchemi-toolkit-ops[torch-cu13]>=0.5.0; sys_platform != 'darwin'", "nvidia-physicsnemo>=2.2.0; sys_platform != 'darwin'", "torch; sys_platform != 'darwin'", "torchvision; sys_platform != 'darwin'", "cuequivariance-ops-torch-cu13>=0.8.0; sys_platform != 'darwin'"] |
| extra:mace | ["cuequivariance-torch>=0.8.0", "mace-torch==0.3.15"] | ["cuequivariance-torch>=0.8.0", "mace-torch==0.3.15"] |
| extra:pymatgen | ["pymatgen>=2025.10.7"] | ["pymatgen>=2025.10.7"] |
| extra:rdkit | [] | ["rdkit>=2025.3.3"] |
| extra:tensorboard | ["tensorboard"] | ["tensorboard"] |
| extra:uma | ["fairchem-core>=2.0.0"] | ["fairchem-core>=2.22.0", "setuptools>=70.0.0,<81"] |
| extra:uma-cu12 | [] | ["fairchem-core>=2.22.0", "setuptools>=70.0.0,<81", "nvalchemi-toolkit-ops[torch-cu12]>=0.5.0; sys_platform != 'darwin'", "torch; sys_platform != 'darwin'", "torchvision; sys_platform != 'darwin'", "cuequivariance-ops-torch-cu12>=0.8.0; sys_platform != 'darwin'"] |
| extra:uma-cu13 | [] | ["fairchem-core>=2.22.0", "setuptools>=70.0.0,<81", "nvalchemi-toolkit-ops[torch-cu13]>=0.5.0; sys_platform != 'darwin'", "torch; sys_platform != 'darwin'", "torchvision; sys_platform != 'darwin'", "cuequivariance-ops-torch-cu13>=0.8.0; sys_platform != 'darwin'"] |

uv sources/metadata/override/conflicts、build-system 和开发依赖组完整保留于 JSON；
以上是声明差异，不是依赖求解或平台兼容性证明。

未纳入对象分析的产品工作树变化：[]

<details><summary>两侧独有提交（分叉时均需审阅）</summary>

#### baseline_only_commits

| 提交 | 主题 |
|---|---|
| [4dfe3723def34df3fadb245981081ccf8c94c257](https://github.com/NVIDIA/nvalchemi-toolkit/commit/4dfe3723def34df3fadb245981081ccf8c94c257) | Omnibus Domain Decomposition Bug Fixes (#160) |
| [7c2a4f07178e810a8c06deae5723acfa68d5dca1](https://github.com/NVIDIA/nvalchemi-toolkit/commit/7c2a4f07178e810a8c06deae5723acfa68d5dca1) | docs: update README for 0.2.0 (#157) |
| [1bdc2e773ae6242d6aec8375e499d4fec38b7d1d](https://github.com/NVIDIA/nvalchemi-toolkit/commit/1bdc2e773ae6242d6aec8375e499d4fec38b7d1d) | Exp docs 0.2.0 preview (#146) |
| [3f7e3d7cd0d0059f624e0a597175aef2f36a007c](https://github.com/NVIDIA/nvalchemi-toolkit/commit/3f7e3d7cd0d0059f624e0a597175aef2f36a007c) | NVE energy conservation in neighbor rebuilds (#153) |
| [c2a8eaa0bb411cb73fbd1cfa20f0cec44a689323](https://github.com/NVIDIA/nvalchemi-toolkit/commit/c2a8eaa0bb411cb73fbd1cfa20f0cec44a689323) | fix(distributed): Fix halo routing for unwrapped periodic migrants (#158) |
| [d943abe819d9eec109073b3d6a53cfde300419c1](https://github.com/NVIDIA/nvalchemi-toolkit/commit/d943abe819d9eec109073b3d6a53cfde300419c1) | fix(distributed): all-reduce graph-parallel stress (#156) |
| [e1d049193a8942c1032690446a9e0492d4ff4238](https://github.com/NVIDIA/nvalchemi-toolkit/commit/e1d049193a8942c1032690446a9e0492d4ff4238) | fix: scope DD communication to domain groups (#155) |
| [93438f37f90dff0d04dadd4a8cd5f3ab8688c97c](https://github.com/NVIDIA/nvalchemi-toolkit/commit/93438f37f90dff0d04dadd4a8cd5f3ab8688c97c) | fix(distributed): synchronize convergence across ranks (#150) |
| [48a15c1a4373ff082a149d0131502ecce3a387a6](https://github.com/NVIDIA/nvalchemi-toolkit/commit/48a15c1a4373ff082a149d0131502ecce3a387a6) | ci: applying best practice again |
| [67d359ce43d03f93223088cb05e454a14a0255c4](https://github.com/NVIDIA/nvalchemi-toolkit/commit/67d359ce43d03f93223088cb05e454a14a0255c4) | ci: specifying commit hash for ci step |
| [c13a9fbc23e9b4a32345aea4338e8a33d6a3111b](https://github.com/NVIDIA/nvalchemi-toolkit/commit/c13a9fbc23e9b4a32345aea4338e8a33d6a3111b) | ci: switching hashing algorithms for uv pull |
| [85df02307352206d3fc78ac4bb5d24d66469ea3e](https://github.com/NVIDIA/nvalchemi-toolkit/commit/85df02307352206d3fc78ac4bb5d24d66469ea3e) | fix: reject undersized domain ghost widths (#152) |
| [b827e5d4277878cf684f988f3796369b6f55499b](https://github.com/NVIDIA/nvalchemi-toolkit/commit/b827e5d4277878cf684f988f3796369b6f55499b) | fix(distributed): ghost atoms retained across domain boundaries (#148) |
| [7b3cca7d437cca992e6348d62f1f0b81396445bc](https://github.com/NVIDIA/nvalchemi-toolkit/commit/7b3cca7d437cca992e6348d62f1f0b81396445bc) | fix(DD): Refresh NPT halo geometry (#147) |
| [3b6e3f7a84465fedaa55f2c3a421d7c0cb12ab0a](https://github.com/NVIDIA/nvalchemi-toolkit/commit/3b6e3f7a84465fedaa55f2c3a421d7c0cb12ab0a) | test: align distributed correctness tests with current API (#145) |
| [c47301d3028a6cf72f391813f7a8fe413db4f904](https://github.com/NVIDIA/nvalchemi-toolkit/commit/c47301d3028a6cf72f391813f7a8fe413db4f904) | Fix bug with Stage Timing Hook (#144) |
| [331d6b2a17d7aabe64a3c77bc9b0cfdbc0e85409](https://github.com/NVIDIA/nvalchemi-toolkit/commit/331d6b2a17d7aabe64a3c77bc9b0cfdbc0e85409) | Fix/add doc to training ex (#142) |
| [72c91b676075ad68bfd7297ea11859b0d1c215b3](https://github.com/NVIDIA/nvalchemi-toolkit/commit/72c91b676075ad68bfd7297ea11859b0d1c215b3) | fix distributed deadlock that comes up in npt (#143) |
| [d29be3bd63c28ed1b37bdabc5f2b00ac8918ebd6](https://github.com/NVIDIA/nvalchemi-toolkit/commit/d29be3bd63c28ed1b37bdabc5f2b00ac8918ebd6) | Dist pipeline example fix (#141) |
| [cf8f20cdb343517ad18b066a85939b3e56a40527](https://github.com/NVIDIA/nvalchemi-toolkit/commit/cf8f20cdb343517ad18b066a85939b3e56a40527) | Aimnet ewald example fix (#139) |
| [cf081e6b148ef0b319e8360af81c0b6f63b5c59a](https://github.com/NVIDIA/nvalchemi-toolkit/commit/cf081e6b148ef0b319e8360af81c0b6f63b5c59a) | chore: adding alias for __version__ per general practice (#137) |
| [182a76826714a3b3e121e99521acd4b798d657ca](https://github.com/NVIDIA/nvalchemi-toolkit/commit/182a76826714a3b3e121e99521acd4b798d657ca) | docs(skills): expand and correct nvalchemi agent skills (#134) |
| [c4fc20ea4f3269cf64ab71254ca389d7c6cbdde1](https://github.com/NVIDIA/nvalchemi-toolkit/commit/c4fc20ea4f3269cf64ab71254ca389d7c6cbdde1) | EMA MACE fix (#138) |
| [80aab5ccd9525bc7cc0a0e9d5f1773ecb04cea4e](https://github.com/NVIDIA/nvalchemi-toolkit/commit/80aab5ccd9525bc7cc0a0e9d5f1773ecb04cea4e) | bump version |

#### candidate_only_commits

| 提交 | 主题 |
|---|---|
| [f59e28dc6701583a13bb294e536076c2a7379e98](https://github.com/NVIDIA/nvalchemi-toolkit/commit/f59e28dc6701583a13bb294e536076c2a7379e98) | Add opt-in hook to exempt fields from the fp-dtype downcast (#127) |
| [f8fe70d1bb9b3a9a4221905fd0ad973523637ca0](https://github.com/NVIDIA/nvalchemi-toolkit/commit/f8fe70d1bb9b3a9a4221905fd0ad973523637ca0) | Prevent Gloo test rendezvous port races (#221) |
| [38d178a1c3d6eba0bf384f96eb8b4f7edf940f15](https://github.com/NVIDIA/nvalchemi-toolkit/commit/38d178a1c3d6eba0bf384f96eb8b4f7edf940f15) | Enable periodic OMol stress and preserve dynamic UMA compilation (#220) |
| [3b512c7758a74bf97637b89865d1eefb0786af47](https://github.com/NVIDIA/nvalchemi-toolkit/commit/3b512c7758a74bf97637b89865d1eefb0786af47) | deprecate fire (#218) |
| [f0a440a2202ef46bf0724500c04cbe9f0570b580](https://github.com/NVIDIA/nvalchemi-toolkit/commit/f0a440a2202ef46bf0724500c04cbe9f0570b580) | feat(csp): add molecular csp module (#207) |
| [2cc02a64ddc37bb0d18f677d71d21f1666e154c8](https://github.com/NVIDIA/nvalchemi-toolkit/commit/2cc02a64ddc37bb0d18f677d71d21f1666e154c8) | fix(data): keep lazy batch-pointer initialization outside compilation (#219) |
| [4fe95603572831ba2f6adad4cfe0102bf7f8cc08](https://github.com/NVIDIA/nvalchemi-toolkit/commit/4fe95603572831ba2f6adad4cfe0102bf7f8cc08) | docs(distillation): add the user guide, on-policy example, and series changelog (#211) |
| [fee711c58760d0f55accbd60483dab2d652dacd4](https://github.com/NVIDIA/nvalchemi-toolkit/commit/fee711c58760d0f55accbd60483dab2d652dacd4) | feat(gen): direct pydantic serialization for generators and pipelines (#213) |
| [492c59b6b114db22bf337da9bbe38cf84bca804d](https://github.com/NVIDIA/nvalchemi-toolkit/commit/492c59b6b114db22bf337da9bbe38cf84bca804d) | feat(distillation): add the evaluation and acceptance suite (#209) |
| [bf605f9191b2978fc865f4286d11e4481b679999](https://github.com/NVIDIA/nvalchemi-toolkit/commit/bf605f9191b2978fc865f4286d11e4481b679999) | ES/06 Adaptive Biasing Force (#179) |
| [17990b80594138951762e0955b2a2afe8c53e4c4](https://github.com/NVIDIA/nvalchemi-toolkit/commit/17990b80594138951762e0955b2a2afe8c53e4c4) | feat(distillation): add representation, curvature, and Boltzmann objectives (#208) |
| [8e5ccd7078971cba3cbf878ce1b553a646e8ac85](https://github.com/NVIDIA/nvalchemi-toolkit/commit/8e5ccd7078971cba3cbf878ce1b553a646e8ac85) | feat: add grouped LBFGS updates (#214) |
| [26bad7b5db83b0bd5c365612980dfa38bdd31cc6](https://github.com/NVIDIA/nvalchemi-toolkit/commit/26bad7b5db83b0bd5c365612980dfa38bdd31cc6) | docs: generative models user guide and API reference (#212) |
| [9862eccb68cc544753d0f7075c309acd460e3002](https://github.com/NVIDIA/nvalchemi-toolkit/commit/9862eccb68cc544753d0f7075c309acd460e3002) | LBFGS API (#206) |
| [9d6147089571e842e95d4d5c62f96a6d501b03f0](https://github.com/NVIDIA/nvalchemi-toolkit/commit/9d6147089571e842e95d4d5c62f96a6d501b03f0) | NEB: API and benchmarks (3&4/4) (#204) |
| [ff7441b68ad572c8d62a163496708729e4ccd6d1](https://github.com/NVIDIA/nvalchemi-toolkit/commit/ff7441b68ad572c8d62a163496708729e4ccd6d1) | feat(models): add position Hessian APIs (#198) |
| [dde0dcbf26413fae491657e70d5038d2deb7006a](https://github.com/NVIDIA/nvalchemi-toolkit/commit/dde0dcbf26413fae491657e70d5038d2deb7006a) | feat(distillation): run the on-policy segment loop data-parallel (#202) |
| [b81fcf7e5d556977c854d6288886e925cc524b02](https://github.com/NVIDIA/nvalchemi-toolkit/commit/b81fcf7e5d556977c854d6288886e925cc524b02) | feat(distillation): add relaxation-trajectory on-policy generation (#201) |
| [b4fec855cd06f6ed69a11447c916ef7da31a0641](https://github.com/NVIDIA/nvalchemi-toolkit/commit/b4fec855cd06f6ed69a11447c916ef7da31a0641) | NEB: kernels (1/4) (#196) |
| [f813081a1d3f4c3e718b665ea7bcf32689230143](https://github.com/NVIDIA/nvalchemi-toolkit/commit/f813081a1d3f4c3e718b665ea7bcf32689230143) | feat(gen): the AtomisticGenerator driver and generative model contract (1/3) (#184) |
| [dd1f60bfc8fba66042dcef94ad1b1a8b537f889b](https://github.com/NVIDIA/nvalchemi-toolkit/commit/dd1f60bfc8fba66042dcef94ad1b1a8b537f889b) | feat(distillation): add the on-policy loop, replay mixing, and teacher label hook (#200) |
| [701055f787e8f79e2e1f633f80e34f4f767797f8](https://github.com/NVIDIA/nvalchemi-toolkit/commit/701055f787e8f79e2e1f633f80e34f4f767797f8) | feat(distillation): add offline DistillationStrategy and atomic-energy matching (#199) |
| [8b58e405152b1a038338e00d884e3bdad40f5361](https://github.com/NVIDIA/nvalchemi-toolkit/commit/8b58e405152b1a038338e00d884e3bdad40f5361) | build(deps): pin nvalchemi-toolkit-ops to 0.5.0-rc GitHub branch (#203) |
| [a26efb4873780128c728acfb013bbe642452c46d](https://github.com/NVIDIA/nvalchemi-toolkit/commit/a26efb4873780128c728acfb013bbe642452c46d) | feat(distillation): add teacher scoring and offline dataset labeling (#195) |
| [72d01d6dc97aef3a35eb7d6c9c69157b9bcc36a3](https://github.com/NVIDIA/nvalchemi-toolkit/commit/72d01d6dc97aef3a35eb7d6c9c69157b9bcc36a3) | update test (#205) |
| [a4215f450d68aaf7402391e3384a165af3f38175](https://github.com/NVIDIA/nvalchemi-toolkit/commit/a4215f450d68aaf7402391e3384a165af3f38175) | Fix/dynamics hooks and order (#182) |
| [7138132f2d75dcfaf41f2ce635620c5edd97d3b6](https://github.com/NVIDIA/nvalchemi-toolkit/commit/7138132f2d75dcfaf41f2ce635620c5edd97d3b6) | fix: device- and dtype-robust segment storage, fp32 loss accumulation, and resumable device moves (#193) |
| [bcc43f55748b6d1bbd928c660c1eaeb109fa604c](https://github.com/NVIDIA/nvalchemi-toolkit/commit/bcc43f55748b6d1bbd928c660c1eaeb109fa604c) | feat(data): add extensible batch level schemas (#194) |
| [d13c973f64f53656c7ab4040df98d92fee97b33d](https://github.com/NVIDIA/nvalchemi-toolkit/commit/d13c973f64f53656c7ab4040df98d92fee97b33d) | build(deps): extend Python support to 3.14 and relax numpy to >=2,<3 (#189) |
| [79b0a78aa1169933c4ce4812176d9a732245fca3](https://github.com/NVIDIA/nvalchemi-toolkit/commit/79b0a78aa1169933c4ce4812176d9a732245fca3) | refactor(dynamics): remove redundant GPUBuffer batch_ptr capacity workaround (#140) |
| [a51139165780efda51f361d76a8b3da48c6e05d1](https://github.com/NVIDIA/nvalchemi-toolkit/commit/a51139165780efda51f361d76a8b3da48c6e05d1) | fix: make dynamics hook lifecycle status-aware (#181) |
| [8ee519088a90b3f5fffa705e31c58a126d7aea5f](https://github.com/NVIDIA/nvalchemi-toolkit/commit/8ee519088a90b3f5fffa705e31c58a126d7aea5f) | fix: update vulnerable dependencies and isolate UMA setuptools (#170) |
| [e772d7732956664387a568b99ddf06adc44757f7](https://github.com/NVIDIA/nvalchemi-toolkit/commit/e772d7732956664387a568b99ddf06adc44757f7) | fix(dynamics): protect LoggingHook snapshots from allocator reuse (#190) |
| [0abf15aa4bfb6b7070693ac81023039d92d78694](https://github.com/NVIDIA/nvalchemi-toolkit/commit/0abf15aa4bfb6b7070693ac81023039d92d78694) | docs(models): correct stale stress pre-allocation note in lj wrapper (#188) |
| [ab0b4adb94db8b98322d0d3f398e186e3a182855](https://github.com/NVIDIA/nvalchemi-toolkit/commit/ab0b4adb94db8b98322d0d3f398e186e3a182855) | feat(training): add LoRA fine-tuning strategy (#149) |
| [9040bed92d300500a4ccf2552d384477fe5f7d8b](https://github.com/NVIDIA/nvalchemi-toolkit/commit/9040bed92d300500a4ccf2552d384477fe5f7d8b) | feat(dynamics): mask-based branchless convergence evaluation (#169) |
| [7f4311765bcdd46b586d88a3416bf58591f4fb47](https://github.com/NVIDIA/nvalchemi-toolkit/commit/7f4311765bcdd46b586d88a3416bf58591f4fb47) | fix(models): compose DFT-D3 virial/forces in pipeline autograd groups (#166) |
| [fa7e8cd09c2cebb48c131eb820b76a1a3bc83634](https://github.com/NVIDIA/nvalchemi-toolkit/commit/fa7e8cd09c2cebb48c131eb820b76a1a3bc83634) | feat(dynamics): support compiled fused steps with CUDA graphs (#168) |
| [7aad0ef37aaaa50147b790ff71cea592e8f25e4b](https://github.com/NVIDIA/nvalchemi-toolkit/commit/7aad0ef37aaaa50147b790ff71cea592e8f25e4b) | ci: run multi-GPU tests on the 2-GPU H100 runner (#167) |
| [a4cc17f05a269c79d65894bd9cc4b97315a505c0](https://github.com/NVIDIA/nvalchemi-toolkit/commit/a4cc17f05a269c79d65894bd9cc4b97315a505c0) | Greptile refactor (#164) |
| [8c2c307c1c0c76baee6f7a68eb75a45da83ffd18](https://github.com/NVIDIA/nvalchemi-toolkit/commit/8c2c307c1c0c76baee6f7a68eb75a45da83ffd18) | 0.2.0 (#161) |

</details>

## ops

| 版本 | 完整 SHA | 声明版本 |
|---|---|---|
| Baseline | 26dbceb61e30cca80e1a5805eebeb51d7dc68fd1 | 0.4.1 |
| Candidate (0.5.0-rc) | 5fffc0726b1e71bb8b68e10b5f87942064759f5b | 0.5.0 |

官方源码：[基线](https://github.com/NVIDIA/nvalchemi-toolkit-ops/tree/26dbceb61e30cca80e1a5805eebeb51d7dc68fd1)、[候选](https://github.com/NVIDIA/nvalchemi-toolkit-ops/tree/5fffc0726b1e71bb8b68e10b5f87942064759f5b)。

历史关系：**diverged**；基线独有 4 个提交，候选独有 42 个提交。
merge-base：200452c75797a5230c55cbbc88fef6c99f137500

### 文件与主要模块

上游文件变化 322：{'A': 77, 'D': 1, 'M': 244}；本地文件变化 83；潜在交集 8。

| 主要模块/路径前缀 | 变化文件 |
|---|---|
| .github/workflows | 2 |
| .gitignore | 1 |
| CHANGELOG.md | 1 |
| Makefile | 1 |
| README.md | 1 |
| benchmarks/dynamics | 3 |
| benchmarks/interactions | 5 |
| benchmarks/neighborlist | 1 |
| benchmarks/suite_systems.py | 1 |
| benchmarks/suite_utils.py | 1 |
| docs/Makefile | 1 |
| docs/_gallery_worker.py | 1 |
| docs/benchmarks | 18 |
| docs/changes.md | 1 |
| docs/conf.py | 1 |
| docs/index.md | 1 |
| docs/modules | 12 |
| docs/sphinxext.py | 1 |
| docs/userguide | 12 |
| examples/dispersion | 4 |
| examples/dynamics | 3 |
| examples/neighbors | 3 |
| nvalchemiops/__init__.py | 1 |
| nvalchemiops/dynamics | 4 |
| nvalchemiops/interactions | 10 |
| nvalchemiops/jax | 36 |
| nvalchemiops/math | 4 |
| nvalchemiops/neighborlist | 1 |
| nvalchemiops/neighbors | 11 |
| nvalchemiops/segment_ops.py | 1 |
| nvalchemiops/segment_ops_backward.py | 1 |
| nvalchemiops/torch | 60 |
| pyproject.toml | 1 |
| test/_license | 1 |
| test/benchmarks | 4 |
| test/conftest.py | 1 |
| test/dynamics | 10 |
| test/interactions | 38 |
| test/math | 5 |
| test/neighbors | 50 |
| test/test-requires.txt | 1 |
| test/test_benchmark_pme_mesh_integration.py | 1 |
| test/test_segment_ops_jax.py | 1 |
| test/test_segment_ops_torch.py | 1 |
| test/torch | 1 |
| tools/group_coverage.py | 1 |
| tools/measure_test_coverage.sh | 1 |
| uv.lock | 1 |

<details><summary>完整上游文件变化（A/M/D/R/T 等 Git 状态）</summary>

| 状态 | 之前 | 之后 |
|---|---|---|
| M | .github/workflows/ci.yml | .github/workflows/ci.yml |
| M | .github/workflows/docs.yml | .github/workflows/docs.yml |
| M | .gitignore | .gitignore |
| M | CHANGELOG.md | CHANGELOG.md |
| M | Makefile | Makefile |
| M | README.md | README.md |
| M | benchmarks/dynamics/README.md | benchmarks/dynamics/README.md |
| M | benchmarks/dynamics/benchmark_config.yaml | benchmarks/dynamics/benchmark_config.yaml |
| A | — | benchmarks/dynamics/benchmark_lbfgs.py |
| A | — | benchmarks/interactions/dispersion/README_fourier.rst |
| A | — | benchmarks/interactions/dispersion/benchmark_fourier_config.yaml |
| A | — | benchmarks/interactions/dispersion/benchmark_fourier_dftd3.py |
| M | benchmarks/interactions/electrostatics/benchmark_config.yaml | benchmarks/interactions/electrostatics/benchmark_config.yaml |
| M | benchmarks/interactions/electrostatics/benchmark_electrostatics_suite.py | benchmarks/interactions/electrostatics/benchmark_electrostatics_suite.py |
| M | benchmarks/neighborlist/benchmark_neighborlist.py | benchmarks/neighborlist/benchmark_neighborlist.py |
| M | benchmarks/suite_systems.py | benchmarks/suite_systems.py |
| M | benchmarks/suite_utils.py | benchmarks/suite_utils.py |
| M | docs/Makefile | docs/Makefile |
| A | — | docs/_gallery_worker.py |
| M | docs/benchmarks/benchmark_results/README.md | docs/benchmarks/benchmark_results/README.md |
| M | docs/benchmarks/benchmark_results/el-cscl-batch-scaling.csv | docs/benchmarks/benchmark_results/el-cscl-batch-scaling.csv |
| M | docs/benchmarks/benchmark_results/el-cscl-constant-workload-scaling.csv | docs/benchmarks/benchmark_results/el-cscl-constant-workload-scaling.csv |
| M | docs/benchmarks/benchmark_results/el-cscl-system-size-scaling.csv | docs/benchmarks/benchmark_results/el-cscl-system-size-scaling.csv |
| M | docs/benchmarks/benchmark_results/el-nh3-batch-scaling.csv | docs/benchmarks/benchmark_results/el-nh3-batch-scaling.csv |
| M | docs/benchmarks/benchmark_results/el-nh3-constant-workload-scaling.csv | docs/benchmarks/benchmark_results/el-nh3-constant-workload-scaling.csv |
| M | docs/benchmarks/benchmark_results/el-nh3-system-size-scaling.csv | docs/benchmarks/benchmark_results/el-nh3-system-size-scaling.csv |
| M | docs/benchmarks/benchmark_results/nl-cscl-batch-scaling.csv | docs/benchmarks/benchmark_results/nl-cscl-batch-scaling.csv |
| M | docs/benchmarks/benchmark_results/nl-cscl-constant-workload-scaling.csv | docs/benchmarks/benchmark_results/nl-cscl-constant-workload-scaling.csv |
| M | docs/benchmarks/benchmark_results/nl-cscl-system-size-scaling.csv | docs/benchmarks/benchmark_results/nl-cscl-system-size-scaling.csv |
| M | docs/benchmarks/benchmark_results/nl-nh3-batch-scaling.csv | docs/benchmarks/benchmark_results/nl-nh3-batch-scaling.csv |
| M | docs/benchmarks/benchmark_results/nl-nh3-constant-workload-scaling.csv | docs/benchmarks/benchmark_results/nl-nh3-constant-workload-scaling.csv |
| M | docs/benchmarks/benchmark_results/nl-nh3-system-size-scaling.csv | docs/benchmarks/benchmark_results/nl-nh3-system-size-scaling.csv |
| M | docs/benchmarks/dynamics.md | docs/benchmarks/dynamics.md |
| M | docs/benchmarks/electrostatics.md | docs/benchmarks/electrostatics.md |
| M | docs/benchmarks/generate_plots.py | docs/benchmarks/generate_plots.py |
| M | docs/benchmarks/index.md | docs/benchmarks/index.md |
| M | docs/benchmarks/neighborlist.md | docs/benchmarks/neighborlist.md |
| M | docs/changes.md | docs/changes.md |
| M | docs/conf.py | docs/conf.py |
| M | docs/index.md | docs/index.md |
| M | docs/modules/index.md | docs/modules/index.md |
| M | docs/modules/jax/dispersion.rst | docs/modules/jax/dispersion.rst |
| A | — | docs/modules/jax/dynamics.rst |
| M | docs/modules/jax/electrostatics.rst | docs/modules/jax/electrostatics.rst |
| M | docs/modules/jax/neighbors.rst | docs/modules/jax/neighbors.rst |
| M | docs/modules/torch/dispersion.rst | docs/modules/torch/dispersion.rst |
| M | docs/modules/torch/dynamics.rst | docs/modules/torch/dynamics.rst |
| M | docs/modules/torch/electrostatics.rst | docs/modules/torch/electrostatics.rst |
| M | docs/modules/torch/neighbors.rst | docs/modules/torch/neighbors.rst |
| M | docs/modules/warp/dispersion.rst | docs/modules/warp/dispersion.rst |
| M | docs/modules/warp/dynamics.rst | docs/modules/warp/dynamics.rst |
| M | docs/modules/warp/neighbors.rst | docs/modules/warp/neighbors.rst |
| M | docs/sphinxext.py | docs/sphinxext.py |
| M | docs/userguide/about/contributing.md | docs/userguide/about/contributing.md |
| M | docs/userguide/about/install.md | docs/userguide/about/install.md |
| M | docs/userguide/about/kernel-style-guide.md | docs/userguide/about/kernel-style-guide.md |
| M | docs/userguide/about/migration.md | docs/userguide/about/migration.md |
| A | — | docs/userguide/about/performance.md |
| M | docs/userguide/components/dispersion.md | docs/userguide/components/dispersion.md |
| M | docs/userguide/components/dynamics.md | docs/userguide/components/dynamics.md |
| M | docs/userguide/components/electrostatics.md | docs/userguide/components/electrostatics.md |
| M | docs/userguide/components/neighborlist.md | docs/userguide/components/neighborlist.md |
| A | — | docs/userguide/components/neighborlist_advanced.md |
| A | — | docs/userguide/components/neighborlist_repeated.md |
| M | docs/userguide/index.md | docs/userguide/index.md |
| M | examples/dispersion/03_jax_dftd3_molecule.py | examples/dispersion/03_jax_dftd3_molecule.py |
| A | — | examples/dispersion/04_fourier_dftd3_crystal.py |
| A | — | examples/dispersion/05_jax_fourier_dftd3_crystal.py |
| M | examples/dispersion/README.rst | examples/dispersion/README.rst |
| A | — | examples/dynamics/12_lbfgs_optimization.py |
| A | — | examples/dynamics/13_lbfgs_variable_cell.py |
| M | examples/dynamics/README.rst | examples/dynamics/README.rst |
| M | examples/neighbors/01_simple_neighbor_list.py | examples/neighbors/01_simple_neighbor_list.py |
| M | examples/neighbors/04_neighbors_list_torch_compile_performance.py | examples/neighbors/04_neighbors_list_torch_compile_performance.py |
| M | examples/neighbors/05_jax_neighbor_list.py | examples/neighbors/05_jax_neighbor_list.py |
| M | nvalchemiops/__init__.py | nvalchemiops/__init__.py |
| M | nvalchemiops/dynamics/optimizers/__init__.py | nvalchemiops/dynamics/optimizers/__init__.py |
| M | nvalchemiops/dynamics/optimizers/fire2.py | nvalchemiops/dynamics/optimizers/fire2.py |
| A | — | nvalchemiops/dynamics/optimizers/lbfgs.py |
| M | nvalchemiops/dynamics/utils/cell_filter.py | nvalchemiops/dynamics/utils/cell_filter.py |
| A | — | nvalchemiops/interactions/dispersion/_c6_decomposition.py |
| A | — | nvalchemiops/interactions/dispersion/_fourier_dftd3.py |
| M | nvalchemiops/interactions/electrostatics/__init__.py | nvalchemiops/interactions/electrostatics/__init__.py |
| M | nvalchemiops/interactions/electrostatics/_factory_common.py | nvalchemiops/interactions/electrostatics/_factory_common.py |
| A | — | nvalchemiops/interactions/electrostatics/_pme_mesh.py |
| M | nvalchemiops/interactions/electrostatics/ewald_kernels.py | nvalchemiops/interactions/electrostatics/ewald_kernels.py |
| M | nvalchemiops/interactions/electrostatics/ewald_real_factory.py | nvalchemiops/interactions/electrostatics/ewald_real_factory.py |
| M | nvalchemiops/interactions/electrostatics/ewald_recip_factory.py | nvalchemiops/interactions/electrostatics/ewald_recip_factory.py |
| M | nvalchemiops/interactions/electrostatics/multipole_direct_kspace_kernels.py | nvalchemiops/interactions/electrostatics/multipole_direct_kspace_kernels.py |
| M | nvalchemiops/interactions/electrostatics/pme_multipole_kernels.py | nvalchemiops/interactions/electrostatics/pme_multipole_kernels.py |
| M | nvalchemiops/jax/__init__.py | nvalchemiops/jax/__init__.py |
| A | — | nvalchemiops/jax/_lazy_jax_kernels.py |
| M | nvalchemiops/jax/interactions/dispersion/__init__.py | nvalchemiops/jax/interactions/dispersion/__init__.py |
| M | nvalchemiops/jax/interactions/dispersion/_dftd3.py | nvalchemiops/jax/interactions/dispersion/_dftd3.py |
| A | — | nvalchemiops/jax/interactions/dispersion/_fourier_dftd3.py |
| M | nvalchemiops/jax/interactions/electrostatics/__init__.py | nvalchemiops/jax/interactions/electrostatics/__init__.py |
| D | nvalchemiops/jax/interactions/electrostatics/_lazy_jax_kernels.py | — |
| M | nvalchemiops/jax/interactions/electrostatics/_utils.py | nvalchemiops/jax/interactions/electrostatics/_utils.py |
| M | nvalchemiops/jax/interactions/electrostatics/coulomb.py | nvalchemiops/jax/interactions/electrostatics/coulomb.py |
| M | nvalchemiops/jax/interactions/electrostatics/ewald.py | nvalchemiops/jax/interactions/electrostatics/ewald.py |
| M | nvalchemiops/jax/interactions/electrostatics/k_vectors.py | nvalchemiops/jax/interactions/electrostatics/k_vectors.py |
| M | nvalchemiops/jax/interactions/electrostatics/parameters.py | nvalchemiops/jax/interactions/electrostatics/parameters.py |
| M | nvalchemiops/jax/interactions/electrostatics/pme.py | nvalchemiops/jax/interactions/electrostatics/pme.py |
| M | nvalchemiops/jax/interactions/electrostatics/slab.py | nvalchemiops/jax/interactions/electrostatics/slab.py |
| A | — | nvalchemiops/jax/lbfgs.py |
| M | nvalchemiops/jax/neighbors/__init__.py | nvalchemiops/jax/neighbors/__init__.py |
| A | — | nvalchemiops/jax/neighbors/_cell_grid.py |
| M | nvalchemiops/jax/neighbors/_cluster_tile_preload.py | nvalchemiops/jax/neighbors/_cluster_tile_preload.py |
| M | nvalchemiops/jax/neighbors/_dispatch.py | nvalchemiops/jax/neighbors/_dispatch.py |
| M | nvalchemiops/jax/neighbors/_registration.py | nvalchemiops/jax/neighbors/_registration.py |
| A | — | nvalchemiops/jax/neighbors/_resolution.py |
| A | — | nvalchemiops/jax/neighbors/_status.py |
| M | nvalchemiops/jax/neighbors/batch_cell_list.py | nvalchemiops/jax/neighbors/batch_cell_list.py |
| M | nvalchemiops/jax/neighbors/batch_cluster_tile.py | nvalchemiops/jax/neighbors/batch_cluster_tile.py |
| M | nvalchemiops/jax/neighbors/batch_naive.py | nvalchemiops/jax/neighbors/batch_naive.py |
| M | nvalchemiops/jax/neighbors/batch_naive_dual_cutoff.py | nvalchemiops/jax/neighbors/batch_naive_dual_cutoff.py |
| M | nvalchemiops/jax/neighbors/cell_list.py | nvalchemiops/jax/neighbors/cell_list.py |
| M | nvalchemiops/jax/neighbors/cluster_tile.py | nvalchemiops/jax/neighbors/cluster_tile.py |
| M | nvalchemiops/jax/neighbors/naive.py | nvalchemiops/jax/neighbors/naive.py |
| M | nvalchemiops/jax/neighbors/naive_dual_cutoff.py | nvalchemiops/jax/neighbors/naive_dual_cutoff.py |
| M | nvalchemiops/jax/neighbors/neighbor_utils.py | nvalchemiops/jax/neighbors/neighbor_utils.py |
| A | — | nvalchemiops/jax/neighbors/prepared_neighbor_list.py |
| M | nvalchemiops/jax/neighbors/rebuild_detection.py | nvalchemiops/jax/neighbors/rebuild_detection.py |
| M | nvalchemiops/jax/segment_ops.py | nvalchemiops/jax/segment_ops.py |
| M | nvalchemiops/jax/spline.py | nvalchemiops/jax/spline.py |
| M | nvalchemiops/jax/types.py | nvalchemiops/jax/types.py |
| M | nvalchemiops/math/__init__.py | nvalchemiops/math/__init__.py |
| M | nvalchemiops/math/math.py | nvalchemiops/math/math.py |
| M | nvalchemiops/math/spherical_harmonics.py | nvalchemiops/math/spherical_harmonics.py |
| M | nvalchemiops/math/spline.py | nvalchemiops/math/spline.py |
| M | nvalchemiops/neighborlist/__init__.py | nvalchemiops/neighborlist/__init__.py |
| M | nvalchemiops/neighbors/__init__.py | nvalchemiops/neighbors/__init__.py |
| M | nvalchemiops/neighbors/base_dispatch.py | nvalchemiops/neighbors/base_dispatch.py |
| A | — | nvalchemiops/neighbors/cell_list/_grid_selection.py |
| M | nvalchemiops/neighbors/cell_list/kernels.py | nvalchemiops/neighbors/cell_list/kernels.py |
| M | nvalchemiops/neighbors/cell_list/launchers.py | nvalchemiops/neighbors/cell_list/launchers.py |
| M | nvalchemiops/neighbors/cluster_tile/kernels.py | nvalchemiops/neighbors/cluster_tile/kernels.py |
| M | nvalchemiops/neighbors/cluster_tile/launchers.py | nvalchemiops/neighbors/cluster_tile/launchers.py |
| M | nvalchemiops/neighbors/naive/dispatch.py | nvalchemiops/neighbors/naive/dispatch.py |
| M | nvalchemiops/neighbors/naive/kernels.py | nvalchemiops/neighbors/naive/kernels.py |
| M | nvalchemiops/neighbors/naive/launchers.py | nvalchemiops/neighbors/naive/launchers.py |
| M | nvalchemiops/neighbors/neighbor_utils.py | nvalchemiops/neighbors/neighbor_utils.py |
| M | nvalchemiops/segment_ops.py | nvalchemiops/segment_ops.py |
| M | nvalchemiops/segment_ops_backward.py | nvalchemiops/segment_ops_backward.py |
| M | nvalchemiops/torch/__init__.py | nvalchemiops/torch/__init__.py |
| A | — | nvalchemiops/torch/_warnings.py |
| M | nvalchemiops/torch/_warp_op_helpers.py | nvalchemiops/torch/_warp_op_helpers.py |
| M | nvalchemiops/torch/autograd.py | nvalchemiops/torch/autograd.py |
| M | nvalchemiops/torch/fire2.py | nvalchemiops/torch/fire2.py |
| M | nvalchemiops/torch/interactions/dispersion/__init__.py | nvalchemiops/torch/interactions/dispersion/__init__.py |
| M | nvalchemiops/torch/interactions/dispersion/_dftd3.py | nvalchemiops/torch/interactions/dispersion/_dftd3.py |
| A | — | nvalchemiops/torch/interactions/dispersion/_fourier_dftd3.py |
| M | nvalchemiops/torch/interactions/electrostatics/__init__.py | nvalchemiops/torch/interactions/electrostatics/__init__.py |
| M | nvalchemiops/torch/interactions/electrostatics/_ewald_corrections_chain.py | nvalchemiops/torch/interactions/electrostatics/_ewald_corrections_chain.py |
| M | nvalchemiops/torch/interactions/electrostatics/_ewald_direct.py | nvalchemiops/torch/interactions/electrostatics/_ewald_direct.py |
| M | nvalchemiops/torch/interactions/electrostatics/_ewald_real_chain.py | nvalchemiops/torch/interactions/electrostatics/_ewald_real_chain.py |
| M | nvalchemiops/torch/interactions/electrostatics/_ewald_recip_chain.py | nvalchemiops/torch/interactions/electrostatics/_ewald_recip_chain.py |
| M | nvalchemiops/torch/interactions/electrostatics/_multipole_moments.py | nvalchemiops/torch/interactions/electrostatics/_multipole_moments.py |
| M | nvalchemiops/torch/interactions/electrostatics/_slab_chain.py | nvalchemiops/torch/interactions/electrostatics/_slab_chain.py |
| M | nvalchemiops/torch/interactions/electrostatics/_util.py | nvalchemiops/torch/interactions/electrostatics/_util.py |
| M | nvalchemiops/torch/interactions/electrostatics/dsf.py | nvalchemiops/torch/interactions/electrostatics/dsf.py |
| M | nvalchemiops/torch/interactions/electrostatics/ewald.py | nvalchemiops/torch/interactions/electrostatics/ewald.py |
| M | nvalchemiops/torch/interactions/electrostatics/k_vectors.py | nvalchemiops/torch/interactions/electrostatics/k_vectors.py |
| M | nvalchemiops/torch/interactions/electrostatics/multipole_autograd.py | nvalchemiops/torch/interactions/electrostatics/multipole_autograd.py |
| M | nvalchemiops/torch/interactions/electrostatics/multipole_autograd_batch.py | nvalchemiops/torch/interactions/electrostatics/multipole_autograd_batch.py |
| M | nvalchemiops/torch/interactions/electrostatics/multipole_autograd_kernels.py | nvalchemiops/torch/interactions/electrostatics/multipole_autograd_kernels.py |
| M | nvalchemiops/torch/interactions/electrostatics/multipole_electrostatics.py | nvalchemiops/torch/interactions/electrostatics/multipole_electrostatics.py |
| M | nvalchemiops/torch/interactions/electrostatics/multipole_ewald.py | nvalchemiops/torch/interactions/electrostatics/multipole_ewald.py |
| M | nvalchemiops/torch/interactions/electrostatics/multipole_ewald_quadrupole.py | nvalchemiops/torch/interactions/electrostatics/multipole_ewald_quadrupole.py |
| M | nvalchemiops/torch/interactions/electrostatics/multipole_features.py | nvalchemiops/torch/interactions/electrostatics/multipole_features.py |
| M | nvalchemiops/torch/interactions/electrostatics/multipole_scf_cache.py | nvalchemiops/torch/interactions/electrostatics/multipole_scf_cache.py |
| M | nvalchemiops/torch/interactions/electrostatics/multipole_scf_step.py | nvalchemiops/torch/interactions/electrostatics/multipole_scf_step.py |
| M | nvalchemiops/torch/interactions/electrostatics/parameters.py | nvalchemiops/torch/interactions/electrostatics/parameters.py |
| M | nvalchemiops/torch/interactions/electrostatics/pme.py | nvalchemiops/torch/interactions/electrostatics/pme.py |
| M | nvalchemiops/torch/interactions/electrostatics/pme_multipole.py | nvalchemiops/torch/interactions/electrostatics/pme_multipole.py |
| M | nvalchemiops/torch/interactions/electrostatics/slab.py | nvalchemiops/torch/interactions/electrostatics/slab.py |
| A | — | nvalchemiops/torch/lbfgs.py |
| M | nvalchemiops/torch/math/__init__.py | nvalchemiops/torch/math/__init__.py |
| M | nvalchemiops/torch/math/gto.py | nvalchemiops/torch/math/gto.py |
| M | nvalchemiops/torch/math/gto_self_overlap.py | nvalchemiops/torch/math/gto_self_overlap.py |
| M | nvalchemiops/torch/math/solid_harmonics.py | nvalchemiops/torch/math/solid_harmonics.py |
| A | — | nvalchemiops/torch/math/spherical_harmonics.py |
| M | nvalchemiops/torch/neighbors/__init__.py | nvalchemiops/torch/neighbors/__init__.py |
| M | nvalchemiops/torch/neighbors/_autograd.py | nvalchemiops/torch/neighbors/_autograd.py |
| A | — | nvalchemiops/torch/neighbors/_cell_grid.py |
| A | — | nvalchemiops/torch/neighbors/_cluster_tile_state.py |
| M | nvalchemiops/torch/neighbors/_dispatch.py | nvalchemiops/torch/neighbors/_dispatch.py |
| A | — | nvalchemiops/torch/neighbors/_fixed_cell.py |
| A | — | nvalchemiops/torch/neighbors/_naive_partial.py |
| A | — | nvalchemiops/torch/neighbors/_naive_prepared_guard.py |
| A | — | nvalchemiops/torch/neighbors/_resolution.py |
| M | nvalchemiops/torch/neighbors/batch_cell_list.py | nvalchemiops/torch/neighbors/batch_cell_list.py |
| M | nvalchemiops/torch/neighbors/batch_cluster_tile.py | nvalchemiops/torch/neighbors/batch_cluster_tile.py |
| M | nvalchemiops/torch/neighbors/batch_naive.py | nvalchemiops/torch/neighbors/batch_naive.py |
| M | nvalchemiops/torch/neighbors/batch_naive_dual_cutoff.py | nvalchemiops/torch/neighbors/batch_naive_dual_cutoff.py |
| M | nvalchemiops/torch/neighbors/cell_list.py | nvalchemiops/torch/neighbors/cell_list.py |
| M | nvalchemiops/torch/neighbors/cluster_tile.py | nvalchemiops/torch/neighbors/cluster_tile.py |
| M | nvalchemiops/torch/neighbors/naive.py | nvalchemiops/torch/neighbors/naive.py |
| M | nvalchemiops/torch/neighbors/naive_dual_cutoff.py | nvalchemiops/torch/neighbors/naive_dual_cutoff.py |
| M | nvalchemiops/torch/neighbors/neighbor_utils.py | nvalchemiops/torch/neighbors/neighbor_utils.py |
| A | — | nvalchemiops/torch/neighbors/prepared_neighbor_list.py |
| M | nvalchemiops/torch/neighbors/rebuild_detection.py | nvalchemiops/torch/neighbors/rebuild_detection.py |
| M | nvalchemiops/torch/segment_ops.py | nvalchemiops/torch/segment_ops.py |
| M | nvalchemiops/torch/spline.py | nvalchemiops/torch/spline.py |
| M | pyproject.toml | pyproject.toml |
| M | test/_license/config.json | test/_license/config.json |
| A | — | test/benchmarks/test_el_benchmark_configuration.py |
| A | — | test/benchmarks/test_el_neighbor_conversion.py |
| A | — | test/benchmarks/test_el_shared_helpers.py |
| A | — | test/benchmarks/test_result_validation.py |
| M | test/conftest.py | test/conftest.py |
| A | — | test/dynamics/bindings/__init__.py |
| A | — | test/dynamics/bindings/jax/__init__.py |
| A | — | test/dynamics/bindings/jax/conftest.py |
| A | — | test/dynamics/bindings/jax/test_lbfgs.py |
| A | — | test/dynamics/bindings/torch/__init__.py |
| A | — | test/dynamics/bindings/torch/test_lbfgs.py |
| A | — | test/dynamics/conftest.py |
| M | test/dynamics/test_fire.py | test/dynamics/test_fire.py |
| M | test/dynamics/test_fire2.py | test/dynamics/test_fire2.py |
| A | — | test/dynamics/test_lbfgs.py |
| A | — | test/interactions/dispersion/_fourier_harness.py |
| A | — | test/interactions/dispersion/_fourier_reference.py |
| M | test/interactions/dispersion/bindings/jax/test_dftd3.py | test/interactions/dispersion/bindings/jax/test_dftd3.py |
| A | — | test/interactions/dispersion/bindings/jax/test_fourier_dftd3.py |
| A | — | test/interactions/dispersion/bindings/torch/test_fourier_dftd3.py |
| A | — | test/interactions/dispersion/test_benchmark_fourier_dftd3.py |
| A | — | test/interactions/dispersion/test_c6_decomposition.py |
| A | — | test/interactions/dispersion/test_fourier_dftd3.py |
| M | test/interactions/electrostatics/bindings/jax/test_ewald.py | test/interactions/electrostatics/bindings/jax/test_ewald.py |
| M | test/interactions/electrostatics/bindings/jax/test_gradient_contract.py | test/interactions/electrostatics/bindings/jax/test_gradient_contract.py |
| M | test/interactions/electrostatics/bindings/jax/test_kvectors.py | test/interactions/electrostatics/bindings/jax/test_kvectors.py |
| M | test/interactions/electrostatics/bindings/jax/test_parameters.py | test/interactions/electrostatics/bindings/jax/test_parameters.py |
| M | test/interactions/electrostatics/bindings/jax/test_pme.py | test/interactions/electrostatics/bindings/jax/test_pme.py |
| A | — | test/interactions/electrostatics/bindings/jax/test_pme_cached_cell_derivatives.py |
| M | test/interactions/electrostatics/bindings/jax/test_public_api.py | test/interactions/electrostatics/bindings/jax/test_public_api.py |
| A | — | test/interactions/electrostatics/bindings/test_pme_mesh_selection.py |
| M | test/interactions/electrostatics/bindings/torch/test_dsf.py | test/interactions/electrostatics/bindings/torch/test_dsf.py |
| M | test/interactions/electrostatics/bindings/torch/test_ewald.py | test/interactions/electrostatics/bindings/torch/test_ewald.py |
| M | test/interactions/electrostatics/bindings/torch/test_gradient_contract.py | test/interactions/electrostatics/bindings/torch/test_gradient_contract.py |
| M | test/interactions/electrostatics/bindings/torch/test_kvectors.py | test/interactions/electrostatics/bindings/torch/test_kvectors.py |
| M | test/interactions/electrostatics/bindings/torch/test_multipole_api.py | test/interactions/electrostatics/bindings/torch/test_multipole_api.py |
| M | test/interactions/electrostatics/bindings/torch/test_multipole_coverage_gaps.py | test/interactions/electrostatics/bindings/torch/test_multipole_coverage_gaps.py |
| M | test/interactions/electrostatics/bindings/torch/test_multipole_ewald.py | test/interactions/electrostatics/bindings/torch/test_multipole_ewald.py |
| A | — | test/interactions/electrostatics/bindings/torch/test_multipole_mesh_selection.py |
| M | test/interactions/electrostatics/bindings/torch/test_multipole_moments.py | test/interactions/electrostatics/bindings/torch/test_multipole_moments.py |
| M | test/interactions/electrostatics/bindings/torch/test_multipole_pme.py | test/interactions/electrostatics/bindings/torch/test_multipole_pme.py |
| M | test/interactions/electrostatics/bindings/torch/test_multipole_real_space.py | test/interactions/electrostatics/bindings/torch/test_multipole_real_space.py |
| M | test/interactions/electrostatics/bindings/torch/test_parameters.py | test/interactions/electrostatics/bindings/torch/test_parameters.py |
| M | test/interactions/electrostatics/bindings/torch/test_pme.py | test/interactions/electrostatics/bindings/torch/test_pme.py |
| A | — | test/interactions/electrostatics/bindings/torch/test_pme_cached_cell_derivatives.py |
| A | — | test/interactions/electrostatics/bindings/torch/test_pme_mesh_accuracy.py |
| M | test/interactions/electrostatics/bindings/torch/test_public_api.py | test/interactions/electrostatics/bindings/torch/test_public_api.py |
| M | test/interactions/electrostatics/bindings/torch/test_slab.py | test/interactions/electrostatics/bindings/torch/test_slab.py |
| A | — | test/interactions/electrostatics/test_ewald_recip_fp32_nostore.py |
| M | test/interactions/electrostatics/test_factory_ewald_recip.py | test/interactions/electrostatics/test_factory_ewald_recip.py |
| M | test/interactions/electrostatics/test_jax_autograd.py | test/interactions/electrostatics/test_jax_autograd.py |
| M | test/interactions/electrostatics/test_multipole_kernels.py | test/interactions/electrostatics/test_multipole_kernels.py |
| A | — | test/interactions/electrostatics/test_pme_mesh_dimensions.py |
| M | test/math/bindings/torch/test_gto.py | test/math/bindings/torch/test_gto.py |
| M | test/math/bindings/torch/test_solid_harmonics.py | test/math/bindings/torch/test_solid_harmonics.py |
| M | test/math/bindings/torch/test_spline.py | test/math/bindings/torch/test_spline.py |
| M | test/math/test_spherical_harmonics.py | test/math/test_spherical_harmonics.py |
| M | test/math/test_spline.py | test/math/test_spline.py |
| M | test/neighbors/bindings/jax/conftest.py | test/neighbors/bindings/jax/conftest.py |
| M | test/neighbors/bindings/jax/test_base_dispatch.py | test/neighbors/bindings/jax/test_base_dispatch.py |
| M | test/neighbors/bindings/jax/test_batch_cell_list.py | test/neighbors/bindings/jax/test_batch_cell_list.py |
| M | test/neighbors/bindings/jax/test_batch_cluster_tile.py | test/neighbors/bindings/jax/test_batch_cluster_tile.py |
| M | test/neighbors/bindings/jax/test_batch_naive.py | test/neighbors/bindings/jax/test_batch_naive.py |
| M | test/neighbors/bindings/jax/test_batch_naive_dual_cutoff.py | test/neighbors/bindings/jax/test_batch_naive_dual_cutoff.py |
| M | test/neighbors/bindings/jax/test_cell_list.py | test/neighbors/bindings/jax/test_cell_list.py |
| A | — | test/neighbors/bindings/jax/test_cell_list_eager_cache.py |
| M | test/neighbors/bindings/jax/test_cluster_tile.py | test/neighbors/bindings/jax/test_cluster_tile.py |
| A | — | test/neighbors/bindings/jax/test_dispatch.py |
| M | test/neighbors/bindings/jax/test_naive.py | test/neighbors/bindings/jax/test_naive.py |
| M | test/neighbors/bindings/jax/test_naive_dual_cutoff.py | test/neighbors/bindings/jax/test_naive_dual_cutoff.py |
| M | test/neighbors/bindings/jax/test_neighbor_utils.py | test/neighbors/bindings/jax/test_neighbor_utils.py |
| M | test/neighbors/bindings/jax/test_neighborlist.py | test/neighbors/bindings/jax/test_neighborlist.py |
| M | test/neighbors/bindings/jax/test_pair_fn.py | test/neighbors/bindings/jax/test_pair_fn.py |
| A | — | test/neighbors/bindings/jax/test_pair_grid_selection.py |
| A | — | test/neighbors/bindings/jax/test_prepared_neighbor_list.py |
| A | — | test/neighbors/bindings/jax/test_prepared_neighbor_list_status.py |
| M | test/neighbors/bindings/jax/test_rebuild_detection.py | test/neighbors/bindings/jax/test_rebuild_detection.py |
| M | test/neighbors/bindings/jax/test_registration.py | test/neighbors/bindings/jax/test_registration.py |
| A | — | test/neighbors/bindings/jax/test_tile_cache_signature.py |
| M | test/neighbors/bindings/torch/conftest.py | test/neighbors/bindings/torch/conftest.py |
| A | — | test/neighbors/bindings/torch/prepared_test_helpers.py |
| M | test/neighbors/bindings/torch/test_base_dispatch.py | test/neighbors/bindings/torch/test_base_dispatch.py |
| M | test/neighbors/bindings/torch/test_batch_cell_list.py | test/neighbors/bindings/torch/test_batch_cell_list.py |
| M | test/neighbors/bindings/torch/test_batch_cluster_tile.py | test/neighbors/bindings/torch/test_batch_cluster_tile.py |
| M | test/neighbors/bindings/torch/test_batch_naive.py | test/neighbors/bindings/torch/test_batch_naive.py |
| M | test/neighbors/bindings/torch/test_batch_naive_dual_cutoff.py | test/neighbors/bindings/torch/test_batch_naive_dual_cutoff.py |
| M | test/neighbors/bindings/torch/test_cell_list.py | test/neighbors/bindings/torch/test_cell_list.py |
| M | test/neighbors/bindings/torch/test_cluster_tile.py | test/neighbors/bindings/torch/test_cluster_tile.py |
| A | — | test/neighbors/bindings/torch/test_fixed_grid_rule.py |
| M | test/neighbors/bindings/torch/test_naive.py | test/neighbors/bindings/torch/test_naive.py |
| M | test/neighbors/bindings/torch/test_naive_dual_cutoff.py | test/neighbors/bindings/torch/test_naive_dual_cutoff.py |
| M | test/neighbors/bindings/torch/test_neighborlist.py | test/neighbors/bindings/torch/test_neighborlist.py |
| M | test/neighbors/bindings/torch/test_pair_fn.py | test/neighbors/bindings/torch/test_pair_fn.py |
| A | — | test/neighbors/bindings/torch/test_pair_grid_selection.py |
| A | — | test/neighbors/bindings/torch/test_prepared_neighbor_list.py |
| A | — | test/neighbors/bindings/torch/test_prepared_neighbor_list_compiler.py |
| M | test/neighbors/bindings/torch/test_rebuild_detection.py | test/neighbors/bindings/torch/test_rebuild_detection.py |
| A | — | test/neighbors/bindings/torch/test_target_grid_radius.py |
| M | test/neighbors/test_base_dispatch.py | test/neighbors/test_base_dispatch.py |
| M | test/neighbors/test_batch_cell_list_kernels.py | test/neighbors/test_batch_cell_list_kernels.py |
| M | test/neighbors/test_batch_naive_kernels.py | test/neighbors/test_batch_naive_kernels.py |
| M | test/neighbors/test_cell_list_kernels.py | test/neighbors/test_cell_list_kernels.py |
| M | test/neighbors/test_cluster_tile_kernel_getters.py | test/neighbors/test_cluster_tile_kernel_getters.py |
| A | — | test/neighbors/test_fixed_cell_core.py |
| A | — | test/neighbors/test_naive_dispatch.py |
| M | test/neighbors/test_naive_kernel_getters.py | test/neighbors/test_naive_kernel_getters.py |
| M | test/neighbors/test_rebuild_detection_kernels.py | test/neighbors/test_rebuild_detection_kernels.py |
| M | test/neighbors/test_utils.py | test/neighbors/test_utils.py |
| M | test/test-requires.txt | test/test-requires.txt |
| A | — | test/test_benchmark_pme_mesh_integration.py |
| M | test/test_segment_ops_jax.py | test/test_segment_ops_jax.py |
| M | test/test_segment_ops_torch.py | test/test_segment_ops_torch.py |
| M | test/torch/test_warp_boundary_audit.py | test/torch/test_warp_boundary_audit.py |
| M | tools/group_coverage.py | tools/group_coverage.py |
| M | tools/measure_test_coverage.sh | tools/measure_test_coverage.sh |
| M | uv.lock | uv.lock |

</details>

### 潜在冲突与本地补丁

| 交集路径 | 登记 LP |
|---|---|
| Makefile | 无路径匹配，人工复核 |
| nvalchemiops/__init__.py | 无路径匹配，人工复核 |
| nvalchemiops/dynamics/utils/cell_filter.py | LP-072, LP-074 |
| nvalchemiops/jax/__init__.py | 无路径匹配，人工复核 |
| nvalchemiops/math/__init__.py | 无路径匹配，人工复核 |
| nvalchemiops/neighbors/__init__.py | 无路径匹配，人工复核 |
| nvalchemiops/torch/__init__.py | 无路径匹配，人工复核 |
| pyproject.toml | 无路径匹配，人工复核 |

LP 按显式 packages/ 或 nvalchemi/nvalchemiops 路径匹配；简写、通配和语义关联需人工复核。

<details><summary>完整本地变更文件（相对导入基线）</summary>

| 状态 | 之前 | 之后 |
|---|---|---|
| M | Makefile | Makefile |
| M | nvalchemiops/__init__.py | nvalchemiops/__init__.py |
| A | — | nvalchemiops/_backend_catalog/__init__.py |
| A | — | nvalchemiops/_backend_catalog/dynamics.py |
| A | — | nvalchemiops/_backend_catalog/interactions.py |
| A | — | nvalchemiops/_backend_catalog/legacy.py |
| A | — | nvalchemiops/_backend_catalog/neighbors.py |
| A | — | nvalchemiops/_backend_catalog/observability.py |
| A | — | nvalchemiops/_backend_types.py |
| A | — | nvalchemiops/_cell_list_abi.py |
| A | — | nvalchemiops/_hip_batch_cell_build.py |
| A | — | nvalchemiops/_hip_batch_cell_key_count.py |
| A | — | nvalchemiops/_hip_batch_cell_query.py |
| A | — | nvalchemiops/_hip_batch_neighbor_runtime.py |
| A | — | nvalchemiops/_hip_batch_query_geometry.py |
| A | — | nvalchemiops/_hip_batch_query_materialize.py |
| A | — | nvalchemiops/_hip_cell_count.py |
| A | — | nvalchemiops/_hip_cell_fill.py |
| A | — | nvalchemiops/_hip_cell_key.py |
| A | — | nvalchemiops/_hip_cell_scan.py |
| A | — | nvalchemiops/_hip_extension_loader.py |
| A | — | nvalchemiops/_hip_neighbor_executor.py |
| A | — | nvalchemiops/_native/batch_cell_key_count.cpp |
| A | — | nvalchemiops/_native/batch_cell_key_count.cu |
| A | — | nvalchemiops/_native/batch_cell_query.cpp |
| A | — | nvalchemiops/_native/batch_cell_query.cu |
| A | — | nvalchemiops/_native/batch_query_geometry.cpp |
| A | — | nvalchemiops/_native/batch_query_geometry.cu |
| A | — | nvalchemiops/_native/batch_query_materialize.cpp |
| A | — | nvalchemiops/_native/batch_query_materialize.cu |
| A | — | nvalchemiops/_native/cell_atom_list_fill.cpp |
| A | — | nvalchemiops/_native/cell_atom_list_fill.cu |
| A | — | nvalchemiops/_native/cell_key_build.cpp |
| A | — | nvalchemiops/_native/cell_key_build.cu |
| A | — | nvalchemiops/_native/cell_key_count.cpp |
| A | — | nvalchemiops/_native/cell_key_count.cu |
| A | — | nvalchemiops/_native/cell_start_scan.cpp |
| A | — | nvalchemiops/_native/cell_start_scan.cu |
| A | — | nvalchemiops/_torch_batch_query_materialization.py |
| A | — | nvalchemiops/backend.py |
| A | — | nvalchemiops/cell_fire_reference.py |
| A | — | nvalchemiops/cell_reference.py |
| A | — | nvalchemiops/dispatch.py |
| M | nvalchemiops/dynamics/__init__.py | nvalchemiops/dynamics/__init__.py |
| M | nvalchemiops/dynamics/integrators/nose_hoover.py | nvalchemiops/dynamics/integrators/nose_hoover.py |
| M | nvalchemiops/dynamics/optimizers/fire.py | nvalchemiops/dynamics/optimizers/fire.py |
| M | nvalchemiops/dynamics/utils/cell_filter.py | nvalchemiops/dynamics/utils/cell_filter.py |
| A | — | nvalchemiops/executor.py |
| M | nvalchemiops/interactions/__init__.py | nvalchemiops/interactions/__init__.py |
| M | nvalchemiops/jax/__init__.py | nvalchemiops/jax/__init__.py |
| M | nvalchemiops/math/__init__.py | nvalchemiops/math/__init__.py |
| M | nvalchemiops/neighbors/__init__.py | nvalchemiops/neighbors/__init__.py |
| A | — | nvalchemiops/stress_reference.py |
| M | nvalchemiops/torch/__init__.py | nvalchemiops/torch/__init__.py |
| A | — | nvalchemiops/torch_backend.py |
| A | — | nvalchemiops/torch_nhc.py |
| A | — | nvalchemiops/torch_reference.py |
| A | — | nvalchemiops/torch_reference_cell_list.py |
| M | pyproject.toml | pyproject.toml |
| A | — | test/torch/test_backend_registry.py |
| A | — | test/torch/test_cell_list_abi.py |
| A | — | test/torch/test_hip_batch_cell_build_boundary.py |
| A | — | test/torch/test_hip_batch_cell_key_count_boundary.py |
| A | — | test/torch/test_hip_batch_cell_query_boundary.py |
| A | — | test/torch/test_hip_batch_metadata.py |
| A | — | test/torch/test_hip_batch_neighbor_runtime.py |
| A | — | test/torch/test_hip_batch_query_geometry.py |
| A | — | test/torch/test_hip_batch_query_materialize.py |
| A | — | test/torch/test_hip_batch_query_materialize_composite.py |
| A | — | test/torch/test_hip_cell_count_boundary.py |
| A | — | test/torch/test_hip_cell_fill_boundary.py |
| A | — | test/torch/test_hip_cell_key_boundary.py |
| A | — | test/torch/test_hip_cell_scan_boundary.py |
| A | — | test/torch/test_hip_cooperative_query.py |
| A | — | test/torch/test_hip_current_cell_neighbors.py |
| A | — | test/torch/test_hip_neighbor_executor.py |
| A | — | test/torch/test_torch_batch_query_materialization.py |
| A | — | test/torch/test_torch_cell_tools.py |
| A | — | test/torch/test_torch_nhc.py |
| A | — | test/torch/test_torch_reference_backend.py |
| A | — | test/torch/test_torch_reference_cell_list.py |
| A | — | test/torch/test_torch_stress.py |
| A | — | test/torch/test_variable_cell_fire.py |

</details>

### 静态 API 变化

357 个 AST 接口/导出条目变化；82 个签名相同的实现变化，实际返回值/行为 unknown，需复核。

<details><summary>完整静态接口变化（含参数、默认值和返回注解）</summary>

| 状态 | 路径:符号 | 基线 | 候选 |
|---|---|---|---|
| changed | nvalchemiops/dynamics/optimizers/__init__.py:exports | __all__ = ['fire_step', 'fire_update', 'fire_compute_vf_vv_ff', 'fire2_step', 'fire2_update', 'fire2_apply_step', 'fire2_reduce', '_fire_step_no_downhill_ptr_kernel', '_fire_step_downhill_ptr_kernel', '_fire_update_params_no_downhill_ptr_kernel', '_fire_update_params_downhill_ptr_kernel'] ; from nvalchemiops.dynamics.optimizers.fire import _fire_step_downhill_ptr_kernel, _fire_step_no_downhill_ptr_kernel, _fire_update_params_downhill_ptr_kernel, _fire_update_params_no_downhill_ptr_kernel, fire_compute_vf_vv_ff, fire_step, fire_update ; from nvalchemiops.dynamics.optimizers.fire2 import fire2_apply_step, fire2_reduce, fire2_step, fire2_update | __all__ = ['fire_step', 'fire_update', 'fire_compute_vf_vv_ff', 'fire2_step', 'fire2_update', 'fire2_apply_step', 'fire2_reduce', 'LBFGSState', 'LBFGSCellState', 'lbfgs_prepare_state', 'lbfgs_prepare_cell_state', 'lbfgs_step', 'lbfgs_step_coord_cell', '_fire_step_no_downhill_ptr_kernel', '_fire_step_downhill_ptr_kernel', '_fire_update_params_no_downhill_ptr_kernel', '_fire_update_params_downhill_ptr_kernel'] ; from nvalchemiops.dynamics.optimizers.fire import _fire_step_downhill_ptr_kernel, _fire_step_no_downhill_ptr_kernel, _fire_update_params_downhill_ptr_kernel, _fire_update_params_no_downhill_ptr_kernel, fire_compute_vf_vv_ff, fire_step, fire_update ; from nvalchemiops.dynamics.optimizers.fire2 import fire2_apply_step, fire2_reduce, fire2_step, fire2_update ; from nvalchemiops.dynamics.optimizers.lbfgs import LBFGSCellState, LBFGSState, lbfgs_prepare_cell_state, lbfgs_prepare_state, lbfgs_step, lbfgs_step_coord_cell |
| added | nvalchemiops/dynamics/optimizers/lbfgs.py:LBFGSCellState | — | class() @ dataclasses.dataclass |
| added | nvalchemiops/dynamics/optimizers/lbfgs.py:LBFGSCellState.cell_dof_a | — | Any = required |
| added | nvalchemiops/dynamics/optimizers/lbfgs.py:LBFGSCellState.cell_dof_b | — | Any = required |
| added | nvalchemiops/dynamics/optimizers/lbfgs.py:LBFGSCellState.cell_force_a | — | Any = required |
| added | nvalchemiops/dynamics/optimizers/lbfgs.py:LBFGSCellState.cell_force_b | — | Any = required |
| added | nvalchemiops/dynamics/optimizers/lbfgs.py:LBFGSCellState.d_phi | — | Any = required |
| added | nvalchemiops/dynamics/optimizers/lbfgs.py:LBFGSCellState.ext_atom_ptr | — | Any = required |
| added | nvalchemiops/dynamics/optimizers/lbfgs.py:LBFGSCellState.ext_batch_idx | — | Any = required |
| added | nvalchemiops/dynamics/optimizers/lbfgs.py:LBFGSCellState.ext_forces | — | Any = required |
| added | nvalchemiops/dynamics/optimizers/lbfgs.py:LBFGSCellState.ext_positions | — | Any = required |
| added | nvalchemiops/dynamics/optimizers/lbfgs.py:LBFGSCellState.kappa | — | Any = required |
| added | nvalchemiops/dynamics/optimizers/lbfgs.py:LBFGSCellState.num_packed_dofs | — | (self) -> int @ property |
| added | nvalchemiops/dynamics/optimizers/lbfgs.py:LBFGSCellState.num_systems | — | (self) -> int @ property |
| added | nvalchemiops/dynamics/optimizers/lbfgs.py:LBFGSCellState.phi | — | Any = required |
| added | nvalchemiops/dynamics/optimizers/lbfgs.py:LBFGSCellState.phi_inv | — | Any = required |
| added | nvalchemiops/dynamics/optimizers/lbfgs.py:LBFGSCellState.ref_cell | — | Any = required |
| added | nvalchemiops/dynamics/optimizers/lbfgs.py:LBFGSCellState.ref_cell_inv | — | Any = required |
| added | nvalchemiops/dynamics/optimizers/lbfgs.py:LBFGSCellState.validate | — | (self, num_atoms: int \| None=None) -> None |
| added | nvalchemiops/dynamics/optimizers/lbfgs.py:LBFGSState | — | class() @ dataclasses.dataclass |
| added | nvalchemiops/dynamics/optimizers/lbfgs.py:LBFGSState.alpha_hist | — | Any = required |
| added | nvalchemiops/dynamics/optimizers/lbfgs.py:LBFGSState.alpha_step | — | Any = required |
| added | nvalchemiops/dynamics/optimizers/lbfgs.py:LBFGSState.beta_hist | — | Any = required |
| added | nvalchemiops/dynamics/optimizers/lbfgs.py:LBFGSState.d0 | — | Any = required |
| added | nvalchemiops/dynamics/optimizers/lbfgs.py:LBFGSState.direction | — | Any = required |
| added | nvalchemiops/dynamics/optimizers/lbfgs.py:LBFGSState.dmax | — | Any = required |
| added | nvalchemiops/dynamics/optimizers/lbfgs.py:LBFGSState.dquad | — | Any = required |
| added | nvalchemiops/dynamics/optimizers/lbfgs.py:LBFGSState.end | — | Any = required |
| added | nvalchemiops/dynamics/optimizers/lbfgs.py:LBFGSState.force_base | — | Any = required |
| added | nvalchemiops/dynamics/optimizers/lbfgs.py:LBFGSState.gg | — | Any = required |
| added | nvalchemiops/dynamics/optimizers/lbfgs.py:LBFGSState.history_count | — | Any = required |
| added | nvalchemiops/dynamics/optimizers/lbfgs.py:LBFGSState.history_size | — | (self) -> int @ property |
| added | nvalchemiops/dynamics/optimizers/lbfgs.py:LBFGSState.iteration | — | Any = required |
| added | nvalchemiops/dynamics/optimizers/lbfgs.py:LBFGSState.n_loop | — | Any = required |
| added | nvalchemiops/dynamics/optimizers/lbfgs.py:LBFGSState.num_dofs | — | (self) -> int @ property |
| added | nvalchemiops/dynamics/optimizers/lbfgs.py:LBFGSState.num_systems | — | (self) -> int @ property |
| added | nvalchemiops/dynamics/optimizers/lbfgs.py:LBFGSState.s_history | — | Any = required |
| added | nvalchemiops/dynamics/optimizers/lbfgs.py:LBFGSState.ss | — | Any = required |
| added | nvalchemiops/dynamics/optimizers/lbfgs.py:LBFGSState.validate | — | (self) -> None |
| added | nvalchemiops/dynamics/optimizers/lbfgs.py:LBFGSState.x_base | — | Any = required |
| added | nvalchemiops/dynamics/optimizers/lbfgs.py:LBFGSState.y_history | — | Any = required |
| added | nvalchemiops/dynamics/optimizers/lbfgs.py:LBFGSState.ys | — | Any = required |
| added | nvalchemiops/dynamics/optimizers/lbfgs.py:LBFGSState.yy | — | Any = required |
| added | nvalchemiops/dynamics/optimizers/lbfgs.py:lbfgs_prepare_cell_state | — | (atom_ptr, cell, *, cell_force_scale: float=1.0, dtype=wp.vec3d, device=None) -> LBFGSCellState |
| added | nvalchemiops/dynamics/optimizers/lbfgs.py:lbfgs_prepare_state | — | (num_dofs: int, num_systems: int, *, dtype=wp.vec3d, history_size: int=6, device=None) -> LBFGSState |
| added | nvalchemiops/dynamics/optimizers/lbfgs.py:lbfgs_step | — | (positions, forces, state: LBFGSState, batch_idx, **kwargs) -> None |
| added | nvalchemiops/dynamics/optimizers/lbfgs.py:lbfgs_step_coord_cell | — | (positions, forces, cell, stress, state: LBFGSState, cell_state: LBFGSCellState, batch_idx, **kwargs) -> None |
| added | nvalchemiops/interactions/dispersion/_c6_decomposition.py:C6Decomposition | — | class() @ dataclass(frozen=True) |
| added | nvalchemiops/interactions/dispersion/_c6_decomposition.py:C6Decomposition.cn_ref | — | np.ndarray = required |
| added | nvalchemiops/interactions/dispersion/_c6_decomposition.py:C6Decomposition.eigs | — | np.ndarray = required |
| added | nvalchemiops/interactions/dispersion/_c6_decomposition.py:C6Decomposition.max_relative_error | — | float = required |
| added | nvalchemiops/interactions/dispersion/_c6_decomposition.py:C6Decomposition.n_ref | — | (self) -> int @ property |
| added | nvalchemiops/interactions/dispersion/_c6_decomposition.py:C6Decomposition.n_species | — | (self) -> int @ property |
| added | nvalchemiops/interactions/dispersion/_c6_decomposition.py:C6Decomposition.num_channels | — | (self) -> int @ property |
| added | nvalchemiops/interactions/dispersion/_c6_decomposition.py:C6Decomposition.rank | — | (self) -> int @ property |
| added | nvalchemiops/interactions/dispersion/_c6_decomposition.py:C6Decomposition.species | — | np.ndarray = required |
| added | nvalchemiops/interactions/dispersion/_c6_decomposition.py:C6Decomposition.species_map | — | np.ndarray = required |
| added | nvalchemiops/interactions/dispersion/_c6_decomposition.py:C6Decomposition.v_q | — | np.ndarray = required |
| added | nvalchemiops/interactions/dispersion/_c6_decomposition.py:C6Decomposition.valid | — | np.ndarray = required |
| added | nvalchemiops/interactions/dispersion/_c6_decomposition.py:clear_decomposition_cache | — | () -> None |
| added | nvalchemiops/interactions/dispersion/_c6_decomposition.py:decompose_c6_reference | — | (c6ab: np.ndarray, cn_ref: np.ndarray, species: Sequence[int], tol: float=0.0001, max_rank: int \| None=None) -> C6Decomposition |
| added | nvalchemiops/interactions/dispersion/_c6_decomposition.py:extract_species_reference_cn | — | (c6ab: np.ndarray, cn_ref: np.ndarray, species: Sequence[int]) -> tuple[np.ndarray, np.ndarray] |
| added | nvalchemiops/interactions/dispersion/_fourier_dftd3.py:check_neighbour_format | — | (neighbor_matrix, neighbor_matrix_shifts, neighbor_list, neighbor_ptr, unit_shifts) -> unknown |
| added | nvalchemiops/interactions/dispersion/_fourier_dftd3.py:fd3_cn_chain | — | (d_energy_d_c6: wp.array, dc6_dcn: wp.array, positions: wp.array, numbers: wp.array, idx_j: wp.array, neighbor_ptr: wp.array, cartesian_shifts: wp.array, covalent_radii: wp.array, cutoff: float, batch_idx: wp.array, d_energy_d_cn: wp.array, forces: wp.array, virial: wp.array, wp_dtype: type, device: str \| None=None, compute_virial: bool=False) -> None |
| added | nvalchemiops/interactions/dispersion/_fourier_dftd3.py:fd3_cn_chain_matrix | — | (d_energy_d_c6: wp.array, dc6_dcn: wp.array, positions: wp.array, numbers: wp.array, neighbor_matrix: wp.array, cartesian_shifts: wp.array, covalent_radii: wp.array, cutoff: float, batch_idx: wp.array, d_energy_d_cn: wp.array, forces: wp.array, virial: wp.array, wp_dtype: type, device: str \| None=None, compute_virial: bool=False, fill_value: int \| None=None) -> None |
| added | nvalchemiops/interactions/dispersion/_fourier_dftd3.py:fd3_coefficients | — | (coord_num: wp.array, species_index: wp.array, cnref: wp.array, v_q: wp.array, c6: wp.array, dc6_dcn: wp.array, wp_dtype: type, device: str \| None=None) -> None |
| added | nvalchemiops/interactions/dispersion/_fourier_dftd3.py:fd3_coordination_numbers | — | (positions: wp.array, numbers: wp.array, idx_j: wp.array, neighbor_ptr: wp.array, cartesian_shifts: wp.array, covalent_radii: wp.array, cutoff: float, coord_num: wp.array, wp_dtype: type, device: str \| None=None) -> None |
| added | nvalchemiops/interactions/dispersion/_fourier_dftd3.py:fd3_coordination_numbers_matrix | — | (positions: wp.array, numbers: wp.array, neighbor_matrix: wp.array, cartesian_shifts: wp.array, covalent_radii: wp.array, cutoff: float, coord_num: wp.array, wp_dtype: type, device: str \| None=None, fill_value: int \| None=None) -> None |
| added | nvalchemiops/interactions/dispersion/_fourier_dftd3.py:fd3_gather_and_force | — | (potential: wp.array, positions: wp.array, c6: wp.array, group_idx: wp.array, cell_inv_t: wp.array, spline_order: int, rank: int, d_energy_d_c6: wp.array, forces: wp.array, wp_dtype: type, device: str \| None=None) -> None |
| added | nvalchemiops/interactions/dispersion/_fourier_dftd3.py:fd3_kspace | — | (mesh_fft: wp.array, k_matrix: wp.array, moduli_x: wp.array, moduli_y: wp.array, moduli_z: wp.array, volumes: wp.array, sqrt_q: wp.array, eigs: wp.array, s6: float, s8: float, a1: float, a2: float, mesh_dimensions: tuple[int, int, int], n_species: int, rank: int, energy: wp.array, cotangent: wp.array, virial: wp.array, wp_dtype: type, device: str \| None=None, compute_virial: bool=False) -> None |
| added | nvalchemiops/interactions/dispersion/_fourier_dftd3.py:fd3_self_energy | — | (c6: wp.array, species_index: wp.array, batch_idx: wp.array, sqrt_q: wp.array, eigs: wp.array, s6: float, s8: float, a1: float, a2: float, energy: wp.array, d_energy_d_c6: wp.array, wp_dtype: type, device: str \| None=None) -> None |
| added | nvalchemiops/interactions/dispersion/_fourier_dftd3.py:fd3_spread | — | (positions: wp.array, c6: wp.array, group_idx: wp.array, cell_inv_t: wp.array, spline_order: int, rank: int, mesh: wp.array, wp_dtype: type, device: str \| None=None) -> None |
| added | nvalchemiops/interactions/dispersion/_fourier_dftd3.py:resolve_rank_slots | — | (rank_chunk_size, rank) -> unknown |
| changed | nvalchemiops/interactions/electrostatics/__init__.py:exports | __all__ = ['dsf_csr', 'dsf_matrix', 'coulomb_energy', 'coulomb_energy_forces', 'coulomb_energy_matrix', 'coulomb_energy_forces_matrix', 'batch_coulomb_energy', 'batch_coulomb_energy_forces', 'batch_coulomb_energy_matrix', 'batch_coulomb_energy_forces_matrix', 'ewald_real_space_energy', 'ewald_real_space_energy_forces', 'ewald_real_space_energy_matrix', 'ewald_real_space_energy_forces_matrix', 'ewald_real_space_energy_forces_charge_grad', 'ewald_real_space_energy_forces_charge_grad_matrix', 'batch_ewald_real_space_energy', 'batch_ewald_real_space_energy_forces', 'batch_ewald_real_space_energy_matrix', 'batch_ewald_real_space_energy_forces_matrix', 'batch_ewald_real_space_energy_forces_charge_grad', 'batch_ewald_real_space_energy_forces_charge_grad_matrix', 'ewald_reciprocal_space_fill_structure_factors', 'ewald_reciprocal_space_compute_energy', 'ewald_subtract_self_energy', 'ewald_reciprocal_space_energy_forces', 'ewald_reciprocal_space_energy_forces_charge_grad', 'batch_ewald_reciprocal_space_fill_structure_factors', 'batch_ewald_reciprocal_space_compute_energy', 'batch_ewald_subtract_self_energy', 'batch_ewald_reciprocal_space_energy_forces', 'batch_ewald_reciprocal_space_energy_forces_charge_grad', 'slab_reduce_moments', 'slab_precompute_geometry', 'slab_correction', 'pme_green_structure_factor', 'batch_pme_green_structure_factor', 'pme_energy_corrections', 'batch_pme_energy_corrections', 'pme_energy_corrections_with_charge_grad', 'batch_pme_energy_corrections_with_charge_grad', 'ewald_real_space', 'ewald_reciprocal_space', 'ewald_summation', 'compute_slab_correction', 'particle_mesh_ewald', 'pme_reciprocal_space'] ; from .dsf import dsf_csr, dsf_matrix ; from nvalchemiops.interactions.electrostatics.coulomb import batch_coulomb_energy, batch_coulomb_energy_forces, batch_coulomb_energy_forces_matrix, batch_coulomb_energy_matrix, coulomb_energy, coulomb_energy_forces, coulomb_energy_forces_matrix, coulomb_energy_matrix ; from nvalchemiops.interactions.electrostatics.ewald_kernels import batch_ewald_real_space_energy, batch_ewald_real_space_energy_forces, batch_ewald_real_space_energy_forces_charge_grad, batch_ewald_real_space_energy_forces_charge_grad_matrix, batch_ewald_real_space_energy_forces_matrix, batch_ewald_real_space_energy_matrix, batch_ewald_reciprocal_space_compute_energy, batch_ewald_reciprocal_space_energy_forces, batch_ewald_reciprocal_space_energy_forces_charge_grad, batch_ewald_reciprocal_space_fill_structure_factors, batch_ewald_subtract_self_energy, ewald_real_space_energy, ewald_real_space_energy_forces, ewald_real_space_energy_forces_charge_grad, ewald_real_space_energy_forces_charge_grad_matrix, ewald_real_space_energy_forces_matrix, ewald_real_space_energy_matrix, ewald_reciprocal_space_compute_energy, ewald_reciprocal_space_energy_forces, ewald_reciprocal_space_energy_forces_charge_grad, ewald_reciprocal_space_fill_structure_factors, ewald_subtract_self_energy ; from nvalchemiops.interactions.electrostatics.multipole_direct_kspace_kernels import batch_apply_per_k_factor, batch_assemble_rho_k_dipole, batch_build_structure_factor_table, batch_compute_energy_product_per_k, batch_eval_gto_fourier_dipole, batch_eval_receiver_gto_fourier_dipole, batch_position_gradient_from_feature_grad, batch_position_gradient_from_rhok, batch_project_features_dipole, batch_v_gradient_from_feature_grad ; from nvalchemiops.interactions.electrostatics.pme_kernels import batch_pme_energy_corrections, batch_pme_energy_corrections_with_charge_grad, batch_pme_green_structure_factor, pme_energy_corrections, pme_energy_corrections_with_charge_grad, pme_green_structure_factor ; from nvalchemiops.interactions.electrostatics.slab_kernels import slab_correction, slab_precompute_geometry, slab_reduce_moments | __all__ = ['dsf_csr', 'dsf_matrix', 'coulomb_energy', 'coulomb_energy_forces', 'coulomb_energy_matrix', 'coulomb_energy_forces_matrix', 'batch_coulomb_energy', 'batch_coulomb_energy_forces', 'batch_coulomb_energy_matrix', 'batch_coulomb_energy_forces_matrix', 'ewald_real_space_energy', 'ewald_real_space_energy_forces', 'ewald_real_space_energy_matrix', 'ewald_real_space_energy_forces_matrix', 'ewald_real_space_energy_forces_charge_grad', 'ewald_real_space_energy_forces_charge_grad_matrix', 'batch_ewald_real_space_energy', 'batch_ewald_real_space_energy_forces', 'batch_ewald_real_space_energy_matrix', 'batch_ewald_real_space_energy_forces_matrix', 'batch_ewald_real_space_energy_forces_charge_grad', 'batch_ewald_real_space_energy_forces_charge_grad_matrix', 'ewald_reciprocal_space_fill_structure_factors', 'ewald_reciprocal_space_compute_energy', 'ewald_subtract_self_energy', 'ewald_reciprocal_space_energy_forces', 'ewald_reciprocal_space_energy_forces_charge_grad', 'batch_ewald_reciprocal_space_fill_structure_factors', 'batch_ewald_reciprocal_space_compute_energy', 'batch_ewald_subtract_self_energy', 'batch_ewald_reciprocal_space_energy_forces', 'batch_ewald_reciprocal_space_energy_forces_charge_grad', 'slab_reduce_moments', 'slab_precompute_geometry', 'slab_correction', 'pme_green_structure_factor', 'batch_pme_green_structure_factor', 'pme_energy_corrections', 'batch_pme_energy_corrections', 'pme_energy_corrections_with_charge_grad', 'batch_pme_energy_corrections_with_charge_grad', 'PositionGradientFromRhokTiledScratch', 'ProjectFeaturesDipoleTiledScratch', 'PositionGradientFromFeatureGradTiledScratch', 'RhokPositionGradBackwardMomentsTiledScratch', 'RhokPositionGradBackwardPositionsTiledScratch', 'FeatPositionGradBackwardGradRawTiledScratch', 'FeatPositionGradBackwardPositionsTiledScratch', 'VGradFromFeatGradBackwardPositionsTiledScratch', 'ewald_real_space', 'ewald_reciprocal_space', 'ewald_summation', 'compute_slab_correction', 'particle_mesh_ewald', 'pme_reciprocal_space'] ; from .dsf import dsf_csr, dsf_matrix ; from nvalchemiops.interactions.electrostatics.coulomb import batch_coulomb_energy, batch_coulomb_energy_forces, batch_coulomb_energy_forces_matrix, batch_coulomb_energy_matrix, coulomb_energy, coulomb_energy_forces, coulomb_energy_forces_matrix, coulomb_energy_matrix ; from nvalchemiops.interactions.electrostatics.ewald_kernels import batch_ewald_real_space_energy, batch_ewald_real_space_energy_forces, batch_ewald_real_space_energy_forces_charge_grad, batch_ewald_real_space_energy_forces_charge_grad_matrix, batch_ewald_real_space_energy_forces_matrix, batch_ewald_real_space_energy_matrix, batch_ewald_reciprocal_space_compute_energy, batch_ewald_reciprocal_space_energy_forces, batch_ewald_reciprocal_space_energy_forces_charge_grad, batch_ewald_reciprocal_space_fill_structure_factors, batch_ewald_subtract_self_energy, ewald_real_space_energy, ewald_real_space_energy_forces, ewald_real_space_energy_forces_charge_grad, ewald_real_space_energy_forces_charge_grad_matrix, ewald_real_space_energy_forces_matrix, ewald_real_space_energy_matrix, ewald_reciprocal_space_compute_energy, ewald_reciprocal_space_energy_forces, ewald_reciprocal_space_energy_forces_charge_grad, ewald_reciprocal_space_fill_structure_factors, ewald_subtract_self_energy ; from nvalchemiops.interactions.electrostatics.multipole_direct_kspace_kernels import FeatPositionGradBackwardGradRawTiledScratch, FeatPositionGradBackwardPositionsTiledScratch, PositionGradientFromFeatureGradTiledScratch, PositionGradientFromRhokTiledScratch, ProjectFeaturesDipoleTiledScratch, RhokPositionGradBackwardMomentsTiledScratch, RhokPositionGradBackwardPositionsTiledScratch, VGradFromFeatGradBackwardPositionsTiledScratch, batch_apply_per_k_factor, batch_assemble_rho_k_dipole, batch_build_structure_factor_table, batch_compute_energy_product_per_k, batch_eval_gto_fourier_dipole, batch_eval_receiver_gto_fourier_dipole, batch_position_gradient_from_feature_grad, batch_position_gradient_from_rhok, batch_project_features_dipole, batch_v_gradient_from_feature_grad ; from nvalchemiops.interactions.electrostatics.pme_kernels import batch_pme_energy_corrections, batch_pme_energy_corrections_with_charge_grad, batch_pme_green_structure_factor, pme_energy_corrections, pme_energy_corrections_with_charge_grad, pme_green_structure_factor ; from nvalchemiops.interactions.electrostatics.slab_kernels import slab_correction, slab_precompute_geometry, slab_reduce_moments |
| added | nvalchemiops/interactions/electrostatics/_factory_common.py:electrostatics_uses_legacy_fp32 | — | () -> bool |
| changed | nvalchemiops/interactions/electrostatics/ewald_real_factory.py:get_ewald_real_kernel | (wp_dtype: type, *, batched: bool=False, neighbor_input: str='list', deriv_state: _DerivState=_DerivState.E, cell_grad: bool=False, order: str='forward', tiled: bool=False, cell_literal: bool=False, energy_layout: Literal['atom', 'system']='atom', component: str='ewald_real') -> wp.Kernel | (wp_dtype: type, *, batched: bool=False, neighbor_input: str='list', deriv_state: _DerivState=_DerivState.E, cell_grad: bool=False, order: str='forward', tiled: bool=False, cell_literal: bool=False, energy_layout: Literal['atom', 'system']='atom', component: str='ewald_real', core_scalar: type \| None=None) -> wp.Kernel |
| changed | nvalchemiops/interactions/electrostatics/ewald_real_factory.py:make_ewald_real_kernel | (wp_dtype: type, *, batched: bool=False, neighbor_input: str='list', deriv_state: _DerivState=_DerivState.E, cell_grad: bool=False, order: str='forward', tiled: bool=False, cell_literal: bool=False, energy_layout: Literal['atom', 'system']='atom') -> wp.Kernel @ lru_cache(maxsize=None) | (wp_dtype: type, *, batched: bool=False, neighbor_input: str='list', deriv_state: _DerivState=_DerivState.E, cell_grad: bool=False, order: str='forward', tiled: bool=False, cell_literal: bool=False, energy_layout: Literal['atom', 'system']='atom', core_scalar: type \| None=None) -> wp.Kernel @ lru_cache(maxsize=None) |
| changed | nvalchemiops/interactions/electrostatics/ewald_recip_factory.py:get_ewald_recip_kernel | (wp_dtype: type, *, batched: bool=False, neighbor_input: str='none', deriv_state: _DerivState=_DerivState.E, cell_grad: bool=False, order: str='forward', component: str='ewald_recip', tiled: bool=False) -> _RecipKernels | (wp_dtype: type, *, batched: bool=False, neighbor_input: str='none', deriv_state: _DerivState=_DerivState.E, cell_grad: bool=False, order: str='forward', component: str='ewald_recip', tiled: bool=False, phase_scalar: type=wp.float64) -> _RecipKernels |
| changed | nvalchemiops/interactions/electrostatics/ewald_recip_factory.py:make_ewald_recip_kernel | (wp_dtype: type, *, batched: bool=False, neighbor_input: str='none', deriv_state: _DerivState=_DerivState.E, cell_grad: bool=False, order: str='forward', tiled: bool=False) -> _RecipKernels @ lru_cache(maxsize=None) | (wp_dtype: type, *, batched: bool=False, neighbor_input: str='none', deriv_state: _DerivState=_DerivState.E, cell_grad: bool=False, order: str='forward', tiled: bool=False, phase_scalar: type=wp.float64) -> _RecipKernels @ lru_cache(maxsize=None) |
| added | nvalchemiops/interactions/electrostatics/multipole_direct_kspace_kernels.py:FeatPositionGradBackwardGradRawTiledScratch | — | class(NamedTuple) |
| added | nvalchemiops/interactions/electrostatics/multipole_direct_kspace_kernels.py:FeatPositionGradBackwardGradRawTiledScratch.contribs | — | wp.array = required |
| added | nvalchemiops/interactions/electrostatics/multipole_direct_kspace_kernels.py:FeatPositionGradBackwardGradRawTiledScratch.expected_shapes | — | (cls, n_k: int, n_atoms: int, n_sigma: int) -> tuple[tuple[int, int], ...] @ classmethod |
| added | nvalchemiops/interactions/electrostatics/multipole_direct_kspace_kernels.py:FeatPositionGradBackwardGradRawTiledScratch.m_cos | — | wp.array = required |
| added | nvalchemiops/interactions/electrostatics/multipole_direct_kspace_kernels.py:FeatPositionGradBackwardGradRawTiledScratch.m_sin | — | wp.array = required |
| added | nvalchemiops/interactions/electrostatics/multipole_direct_kspace_kernels.py:FeatPositionGradBackwardPositionsTiledScratch | — | class(NamedTuple) |
| added | nvalchemiops/interactions/electrostatics/multipole_direct_kspace_kernels.py:FeatPositionGradBackwardPositionsTiledScratch.contribs | — | wp.array = required |
| added | nvalchemiops/interactions/electrostatics/multipole_direct_kspace_kernels.py:FeatPositionGradBackwardPositionsTiledScratch.expected_shapes | — | (cls, n_k: int, n_atoms: int, n_sigma: int) -> tuple[tuple[int, int], ...] @ classmethod |
| added | nvalchemiops/interactions/electrostatics/multipole_direct_kspace_kernels.py:FeatPositionGradBackwardPositionsTiledScratch.m_cos | — | wp.array = required |
| added | nvalchemiops/interactions/electrostatics/multipole_direct_kspace_kernels.py:FeatPositionGradBackwardPositionsTiledScratch.m_sin | — | wp.array = required |
| added | nvalchemiops/interactions/electrostatics/multipole_direct_kspace_kernels.py:PositionGradientFromFeatureGradTiledScratch | — | class(NamedTuple) |
| added | nvalchemiops/interactions/electrostatics/multipole_direct_kspace_kernels.py:PositionGradientFromFeatureGradTiledScratch.beta_cos | — | wp.array = required |
| added | nvalchemiops/interactions/electrostatics/multipole_direct_kspace_kernels.py:PositionGradientFromFeatureGradTiledScratch.beta_sin | — | wp.array = required |
| added | nvalchemiops/interactions/electrostatics/multipole_direct_kspace_kernels.py:PositionGradientFromFeatureGradTiledScratch.contribs | — | wp.array = required |
| added | nvalchemiops/interactions/electrostatics/multipole_direct_kspace_kernels.py:PositionGradientFromFeatureGradTiledScratch.expected_shapes | — | (cls, n_k: int, n_atoms: int, n_sigma: int) -> tuple[tuple[int, int], ...] @ classmethod |
| added | nvalchemiops/interactions/electrostatics/multipole_direct_kspace_kernels.py:PositionGradientFromRhokTiledScratch | — | class(NamedTuple) |
| added | nvalchemiops/interactions/electrostatics/multipole_direct_kspace_kernels.py:PositionGradientFromRhokTiledScratch.big_cos | — | wp.array = required |
| added | nvalchemiops/interactions/electrostatics/multipole_direct_kspace_kernels.py:PositionGradientFromRhokTiledScratch.big_sin | — | wp.array = required |
| added | nvalchemiops/interactions/electrostatics/multipole_direct_kspace_kernels.py:PositionGradientFromRhokTiledScratch.contribs | — | wp.array = required |
| added | nvalchemiops/interactions/electrostatics/multipole_direct_kspace_kernels.py:PositionGradientFromRhokTiledScratch.expected_shapes | — | (cls, n_k: int, n_atoms: int) -> tuple[tuple[int, int], ...] @ classmethod |
| added | nvalchemiops/interactions/electrostatics/multipole_direct_kspace_kernels.py:ProjectFeaturesDipoleTiledScratch | — | class(NamedTuple) |
| added | nvalchemiops/interactions/electrostatics/multipole_direct_kspace_kernels.py:ProjectFeaturesDipoleTiledScratch.a_flat | — | wp.array = required |
| added | nvalchemiops/interactions/electrostatics/multipole_direct_kspace_kernels.py:ProjectFeaturesDipoleTiledScratch.b_flat | — | wp.array = required |
| added | nvalchemiops/interactions/electrostatics/multipole_direct_kspace_kernels.py:ProjectFeaturesDipoleTiledScratch.contribs | — | wp.array = required |
| added | nvalchemiops/interactions/electrostatics/multipole_direct_kspace_kernels.py:ProjectFeaturesDipoleTiledScratch.expected_shapes | — | (cls, n_k: int, n_atoms: int, n_sigma: int) -> tuple[tuple[int, int], ...] @ classmethod |
| added | nvalchemiops/interactions/electrostatics/multipole_direct_kspace_kernels.py:RhokPositionGradBackwardMomentsTiledScratch | — | class(NamedTuple) |
| added | nvalchemiops/interactions/electrostatics/multipole_direct_kspace_kernels.py:RhokPositionGradBackwardMomentsTiledScratch.big_cos | — | wp.array = required |
| added | nvalchemiops/interactions/electrostatics/multipole_direct_kspace_kernels.py:RhokPositionGradBackwardMomentsTiledScratch.big_sin | — | wp.array = required |
| added | nvalchemiops/interactions/electrostatics/multipole_direct_kspace_kernels.py:RhokPositionGradBackwardMomentsTiledScratch.contribs | — | wp.array = required |
| added | nvalchemiops/interactions/electrostatics/multipole_direct_kspace_kernels.py:RhokPositionGradBackwardMomentsTiledScratch.expected_shapes | — | (cls, n_k: int, n_atoms: int) -> tuple[tuple[int, int], ...] @ classmethod |
| added | nvalchemiops/interactions/electrostatics/multipole_direct_kspace_kernels.py:RhokPositionGradBackwardPositionsTiledScratch | — | class(NamedTuple) |
| added | nvalchemiops/interactions/electrostatics/multipole_direct_kspace_kernels.py:RhokPositionGradBackwardPositionsTiledScratch.beta_cos | — | wp.array = required |
| added | nvalchemiops/interactions/electrostatics/multipole_direct_kspace_kernels.py:RhokPositionGradBackwardPositionsTiledScratch.beta_sin | — | wp.array = required |
| added | nvalchemiops/interactions/electrostatics/multipole_direct_kspace_kernels.py:RhokPositionGradBackwardPositionsTiledScratch.contribs | — | wp.array = required |
| added | nvalchemiops/interactions/electrostatics/multipole_direct_kspace_kernels.py:RhokPositionGradBackwardPositionsTiledScratch.expected_shapes | — | (cls, n_k: int, n_atoms: int) -> tuple[tuple[int, int], ...] @ classmethod |
| added | nvalchemiops/interactions/electrostatics/multipole_direct_kspace_kernels.py:VGradFromFeatGradBackwardPositionsTiledScratch | — | class(NamedTuple) |
| added | nvalchemiops/interactions/electrostatics/multipole_direct_kspace_kernels.py:VGradFromFeatGradBackwardPositionsTiledScratch.contribs | — | wp.array = required |
| added | nvalchemiops/interactions/electrostatics/multipole_direct_kspace_kernels.py:VGradFromFeatGradBackwardPositionsTiledScratch.expected_shapes | — | (cls, n_k: int, n_atoms: int, n_sigma: int) -> tuple[tuple[int, int], ...] @ classmethod |
| added | nvalchemiops/interactions/electrostatics/multipole_direct_kspace_kernels.py:VGradFromFeatGradBackwardPositionsTiledScratch.m_cos | — | wp.array = required |
| added | nvalchemiops/interactions/electrostatics/multipole_direct_kspace_kernels.py:VGradFromFeatGradBackwardPositionsTiledScratch.m_sin | — | wp.array = required |
| changed | nvalchemiops/interactions/electrostatics/multipole_direct_kspace_kernels.py:feat_position_grad_backward_grad_raw | (receiver_phi_hat: wp.array, cosines: wp.array, sines: wp.array, k_factor_proj: wp.array, potential: wp.array, gg_positions: wp.array, k_vectors: wp.array, ggrad_grad_raw: wp.array, device: str \| None=None) -> None | (receiver_phi_hat: wp.array, cosines: wp.array, sines: wp.array, k_factor_proj: wp.array, potential: wp.array, gg_positions: wp.array, k_vectors: wp.array, ggrad_grad_raw: wp.array, device: str \| None=None, scratch: FeatPositionGradBackwardGradRawTiledScratch \| None=None) -> None |
| changed | nvalchemiops/interactions/electrostatics/multipole_direct_kspace_kernels.py:feat_position_grad_backward_positions | (grad_raw: wp.array, receiver_phi_hat: wp.array, cosines: wp.array, sines: wp.array, k_factor_proj: wp.array, potential: wp.array, gg_positions: wp.array, k_vectors: wp.array, ggrad_positions: wp.array, device: str \| None=None) -> None | (grad_raw: wp.array, receiver_phi_hat: wp.array, cosines: wp.array, sines: wp.array, k_factor_proj: wp.array, potential: wp.array, gg_positions: wp.array, k_vectors: wp.array, ggrad_positions: wp.array, device: str \| None=None, scratch: FeatPositionGradBackwardPositionsTiledScratch \| None=None) -> None |
| changed | nvalchemiops/interactions/electrostatics/multipole_direct_kspace_kernels.py:position_gradient_from_feature_grad | (grad_raw: wp.array, receiver_phi_hat: wp.array, cosines: wp.array, sines: wp.array, k_factor_proj: wp.array, potential: wp.array, k_vectors: wp.array, grad_positions: wp.array, device: str \| None=None) -> None | (grad_raw: wp.array, receiver_phi_hat: wp.array, cosines: wp.array, sines: wp.array, k_factor_proj: wp.array, potential: wp.array, k_vectors: wp.array, grad_positions: wp.array, device: str \| None=None, scratch: PositionGradientFromFeatureGradTiledScratch \| None=None) -> None |
| changed | nvalchemiops/interactions/electrostatics/multipole_direct_kspace_kernels.py:position_gradient_from_rhok | (charges: wp.array, dipoles: wp.array, cosines: wp.array, sines: wp.array, source_phi_hat: wp.array, grad_rho: wp.array, k_vectors: wp.array, scale: float, grad_positions: wp.array, wp_dtype: type, device: str \| None=None) -> None | (charges: wp.array, dipoles: wp.array, cosines: wp.array, sines: wp.array, source_phi_hat: wp.array, grad_rho: wp.array, k_vectors: wp.array, scale: float, grad_positions: wp.array, wp_dtype: type, device: str \| None=None, scratch: PositionGradientFromRhokTiledScratch \| None=None) -> None |
| changed | nvalchemiops/interactions/electrostatics/multipole_direct_kspace_kernels.py:project_features_dipole | (potential: wp.array, receiver_phi_hat: wp.array, cosines: wp.array, sines: wp.array, k_factor_proj: wp.array, source_feats_lm: wp.array, overlap_constants: wp.array, subtract_self: bool, out_col_lut: wp.array, features: wp.array, device: str \| None=None) -> None | (potential: wp.array, receiver_phi_hat: wp.array, cosines: wp.array, sines: wp.array, k_factor_proj: wp.array, source_feats_lm: wp.array, overlap_constants: wp.array, subtract_self: bool, out_col_lut: wp.array, features: wp.array, device: str \| None=None, scratch: ProjectFeaturesDipoleTiledScratch \| None=None) -> None |
| changed | nvalchemiops/interactions/electrostatics/multipole_direct_kspace_kernels.py:rhok_position_grad_backward_moments | (cosines: wp.array, sines: wp.array, source_phi_hat: wp.array, grad_rho: wp.array, gg_positions: wp.array, k_vectors: wp.array, scale: float, ggrad_moments: wp.array, device: str \| None=None) -> None | (cosines: wp.array, sines: wp.array, source_phi_hat: wp.array, grad_rho: wp.array, gg_positions: wp.array, k_vectors: wp.array, scale: float, ggrad_moments: wp.array, device: str \| None=None, scratch: RhokPositionGradBackwardMomentsTiledScratch \| None=None) -> None |
| changed | nvalchemiops/interactions/electrostatics/multipole_direct_kspace_kernels.py:rhok_position_grad_backward_positions | (charges: wp.array, dipoles: wp.array, cosines: wp.array, sines: wp.array, source_phi_hat: wp.array, grad_rho: wp.array, gg_positions: wp.array, k_vectors: wp.array, scale: float, ggrad_positions: wp.array, wp_dtype: type, device: str \| None=None) -> None | (charges: wp.array, dipoles: wp.array, cosines: wp.array, sines: wp.array, source_phi_hat: wp.array, grad_rho: wp.array, gg_positions: wp.array, k_vectors: wp.array, scale: float, ggrad_positions: wp.array, wp_dtype: type, device: str \| None=None, scratch: RhokPositionGradBackwardPositionsTiledScratch \| None=None) -> None |
| changed | nvalchemiops/interactions/electrostatics/multipole_direct_kspace_kernels.py:v_grad_from_feat_grad_backward_positions | (grad_raw: wp.array, receiver_phi_hat: wp.array, cosines: wp.array, sines: wp.array, k_factor_proj: wp.array, gg_v: wp.array, k_vectors: wp.array, ggrad_positions: wp.array, device: str \| None=None) -> None | (grad_raw: wp.array, receiver_phi_hat: wp.array, cosines: wp.array, sines: wp.array, k_factor_proj: wp.array, gg_v: wp.array, k_vectors: wp.array, ggrad_positions: wp.array, device: str \| None=None, scratch: VGradFromFeatGradBackwardPositionsTiledScratch \| None=None) -> None |
| added | nvalchemiops/interactions/electrostatics/pme_multipole_kernels.py:batch_pme_k_squared_backward_launch | — | (miller_x: wp.array, miller_y: wp.array, miller_z: wp.array, cell_inv_T: wp.array, grad_k_squared: wp.array, grad_cell_inv_T: wp.array, wp_dtype: type, device: str \| None=None) -> None |
| added | nvalchemiops/jax/_lazy_jax_kernels.py:make_jax_kernel_factory | — | (wp_kernel_factory: Callable[[Any], Any], num_outputs: int, in_out_argnames: Sequence[str] \| None=None, block_dim: int \| None=None) -> _LazyJaxKernels |
| added | nvalchemiops/jax/_lazy_jax_kernels.py:make_jax_kernels | — | (wp_overload_dict: dict, num_outputs: int, in_out_argnames: Sequence[str] \| None=None, block_dim: int \| None=None) -> _LazyJaxKernels |
| changed | nvalchemiops/jax/interactions/dispersion/__init__.py:exports | __all__ = ['dftd3', 'D3Parameters'] ; from __future__ import annotations ; from nvalchemiops.jax.interactions.dispersion._dftd3 import D3Parameters, dftd3 | __all__ = ['dftd3', 'D3Parameters', 'fourier_dftd3', 'FourierD3Parameters'] ; from __future__ import annotations ; from nvalchemiops.jax.interactions.dispersion._dftd3 import D3Parameters, dftd3 ; from nvalchemiops.jax.interactions.dispersion._fourier_dftd3 import FourierD3Parameters, fourier_dftd3 |
| changed | nvalchemiops/jax/interactions/dispersion/_dftd3.py:D3Parameters | class() @ dataclass | class() @ partial(jax.tree_util.register_dataclass, data_fields=['rcov', 'r4r2', 'c6ab', 'cn_ref'], meta_fields=['interp_mesh']), dataclass |
| added | nvalchemiops/jax/interactions/dispersion/_fourier_dftd3.py:FourierD3Parameters | — | class() @ partial(jax.tree_util.register_dataclass, data_fields=['rcov', 'sqrt_q', 'cn_ref', 'v_q', 'eigs', 'species_map'], meta_fields=['max_relative_error']), dataclass |
| added | nvalchemiops/jax/interactions/dispersion/_fourier_dftd3.py:FourierD3Parameters.cn_ref | — | jax.Array = required |
| added | nvalchemiops/jax/interactions/dispersion/_fourier_dftd3.py:FourierD3Parameters.eigs | — | jax.Array = required |
| added | nvalchemiops/jax/interactions/dispersion/_fourier_dftd3.py:FourierD3Parameters.from_tables | — | (cls, rcov, r4r2, c6ab, cn_ref, species, tol: float=0.0001, max_rank: int \| None=None, dtype=jnp.float64) -> FourierD3Parameters @ classmethod |
| added | nvalchemiops/jax/interactions/dispersion/_fourier_dftd3.py:FourierD3Parameters.max_relative_error | — | float = required |
| added | nvalchemiops/jax/interactions/dispersion/_fourier_dftd3.py:FourierD3Parameters.n_species | — | (self) -> int @ property |
| added | nvalchemiops/jax/interactions/dispersion/_fourier_dftd3.py:FourierD3Parameters.rank | — | (self) -> int @ property |
| added | nvalchemiops/jax/interactions/dispersion/_fourier_dftd3.py:FourierD3Parameters.rcov | — | jax.Array = required |
| added | nvalchemiops/jax/interactions/dispersion/_fourier_dftd3.py:FourierD3Parameters.species_map | — | jax.Array = required |
| added | nvalchemiops/jax/interactions/dispersion/_fourier_dftd3.py:FourierD3Parameters.sqrt_q | — | jax.Array = required |
| added | nvalchemiops/jax/interactions/dispersion/_fourier_dftd3.py:FourierD3Parameters.uncovered_species | — | (self, numbers) -> list[int] |
| added | nvalchemiops/jax/interactions/dispersion/_fourier_dftd3.py:FourierD3Parameters.v_q | — | jax.Array = required |
| added | nvalchemiops/jax/interactions/dispersion/_fourier_dftd3.py:fourier_dftd3 | — | (positions, numbers, a1: float, a2: float, s8: float, *, fourier_d3_params: FourierD3Parameters, cell, cutoff: float, mesh_dimensions: tuple[int, int, int] \| None=None, mesh_spacing: float \| None=None, neighbor_matrix=None, neighbor_matrix_shifts=None, neighbor_list=None, neighbor_ptr=None, unit_shifts=None, fill_value: int \| None=None, s6: float=1.0, spline_order: int=4, rank_chunk_size: int \| None=None, batch_idx=None, compute_virial: bool=False, num_systems: int \| None=None) -> unknown |
| changed | nvalchemiops/jax/interactions/electrostatics/__init__.py:exports | __all__ = ['coulomb_energy', 'coulomb_forces', 'coulomb_energy_forces', 'compute_slab_correction', 'ewald_real_space', 'ewald_reciprocal_space', 'ewald_summation', 'particle_mesh_ewald', 'pme_reciprocal_space', 'pme_green_structure_factor', 'pme_energy_corrections', 'pme_energy_corrections_with_charge_grad', 'compute_bspline_moduli_1d', 'generate_k_vectors_ewald_summation', 'generate_k_vectors_pme', 'generate_miller_indices', 'EwaldParameters', 'PMEParameters', 'estimate_ewald_parameters', 'estimate_pme_parameters', 'estimate_pme_mesh_dimensions', 'mesh_spacing_to_dimensions'] ; from __future__ import annotations ; from nvalchemiops.jax.interactions.electrostatics.coulomb import coulomb_energy, coulomb_energy_forces, coulomb_forces ; from nvalchemiops.jax.interactions.electrostatics.ewald import ewald_real_space, ewald_reciprocal_space, ewald_summation ; from nvalchemiops.jax.interactions.electrostatics.k_vectors import generate_k_vectors_ewald_summation, generate_k_vectors_pme, generate_miller_indices ; from nvalchemiops.jax.interactions.electrostatics.parameters import EwaldParameters, PMEParameters, estimate_ewald_parameters, estimate_pme_mesh_dimensions, estimate_pme_parameters, mesh_spacing_to_dimensions ; from nvalchemiops.jax.interactions.electrostatics.pme import compute_bspline_moduli_1d, particle_mesh_ewald, pme_reciprocal_space ; from nvalchemiops.jax.interactions.electrostatics.slab import compute_slab_correction ; from warnings import warn ; import inspect ; import jax ; import warnings | __all__ = ['coulomb_energy', 'coulomb_forces', 'coulomb_energy_forces', 'compute_slab_correction', 'ewald_real_space', 'ewald_reciprocal_space', 'ewald_reciprocal_space_from_miller_indices', 'ewald_summation', 'particle_mesh_ewald', 'pme_reciprocal_space', 'pme_green_structure_factor', 'pme_energy_corrections', 'pme_energy_corrections_with_charge_grad', 'compute_bspline_moduli_1d', 'generate_ewald_miller_indices', 'generate_k_squared_pme', 'generate_k_vectors_ewald_summation', 'generate_k_vectors_pme', 'generate_miller_indices', 'k_vectors_from_miller_indices', 'EwaldParameters', 'PMEParameters', 'estimate_ewald_parameters', 'estimate_pme_parameters', 'estimate_pme_mesh_dimensions', 'mesh_spacing_to_dimensions'] ; from __future__ import annotations ; from nvalchemiops.jax.interactions.electrostatics.coulomb import coulomb_energy, coulomb_energy_forces, coulomb_forces ; from nvalchemiops.jax.interactions.electrostatics.ewald import ewald_real_space, ewald_reciprocal_space, ewald_reciprocal_space_from_miller_indices, ewald_summation ; from nvalchemiops.jax.interactions.electrostatics.k_vectors import generate_ewald_miller_indices, generate_k_squared_pme, generate_k_vectors_ewald_summation, generate_k_vectors_pme, generate_miller_indices, k_vectors_from_miller_indices ; from nvalchemiops.jax.interactions.electrostatics.parameters import EwaldParameters, PMEParameters, estimate_ewald_parameters, estimate_pme_mesh_dimensions, estimate_pme_parameters, mesh_spacing_to_dimensions ; from nvalchemiops.jax.interactions.electrostatics.pme import compute_bspline_moduli_1d, particle_mesh_ewald, pme_reciprocal_space ; from nvalchemiops.jax.interactions.electrostatics.slab import compute_slab_correction ; import inspect ; import jax ; import warnings |
| added | nvalchemiops/jax/interactions/electrostatics/ewald.py:ewald_reciprocal_space_from_miller_indices | — | (positions: jax.Array, charges: jax.Array, cell: jax.Array, miller_indices: jax.Array, alpha: float \| jax.Array, batch_idx: jax.Array \| None=None, max_atoms_per_system: int \| None=None, compute_forces: bool=False, compute_charge_gradients: bool=False, compute_virial: bool=False, *, energy_reduction: Literal['atom', 'system']='atom') -> jax.Array \| tuple[jax.Array, ...] |
| changed | nvalchemiops/jax/interactions/electrostatics/ewald.py:ewald_summation | (positions: jax.Array, charges: jax.Array, cell: jax.Array, alpha: float \| jax.Array \| None=None, k_vectors: jax.Array \| None=None, k_cutoff: float \| jax.Array \| None=None, batch_idx: jax.Array \| None=None, max_atoms_per_system: int \| None=None, neighbor_list: jax.Array \| None=None, neighbor_ptr: jax.Array \| None=None, neighbor_shifts: jax.Array \| None=None, neighbor_matrix: jax.Array \| None=None, neighbor_matrix_shifts: jax.Array \| None=None, mask_value: int \| None=None, compute_forces: bool=False, compute_charge_gradients: bool=False, compute_virial: bool=False, accuracy: float=1e-06, hybrid_forces: bool=False, pbc: jax.Array \| None=None, slab_correction: bool=False, *, miller_bounds: tuple[int, int, int] \| None=None, energy_reduction: Literal['atom', 'system']='atom') -> jax.Array \| tuple[jax.Array, ...] | (positions: jax.Array, charges: jax.Array, cell: jax.Array, alpha: float \| jax.Array \| None=None, k_vectors: jax.Array \| None=None, k_cutoff: float \| jax.Array \| None=None, batch_idx: jax.Array \| None=None, max_atoms_per_system: int \| None=None, neighbor_list: jax.Array \| None=None, neighbor_ptr: jax.Array \| None=None, neighbor_shifts: jax.Array \| None=None, neighbor_matrix: jax.Array \| None=None, neighbor_matrix_shifts: jax.Array \| None=None, mask_value: int \| None=None, compute_forces: bool=False, compute_charge_gradients: bool=False, compute_virial: bool=False, accuracy: float=1e-06, hybrid_forces: bool=False, pbc: jax.Array \| None=None, slab_correction: bool=False, *, miller_bounds: tuple[int, int, int] \| None=None, miller_indices: jax.Array \| None=None, energy_reduction: Literal['atom', 'system']='atom') -> jax.Array \| tuple[jax.Array, ...] |
| added | nvalchemiops/jax/interactions/electrostatics/k_vectors.py:generate_ewald_miller_indices | — | (cell: jax.Array, k_cutoff: float \| jax.Array, miller_bounds: tuple[int, int, int] \| None=None) -> jax.Array |
| added | nvalchemiops/jax/interactions/electrostatics/k_vectors.py:generate_k_squared_pme | — | (cell: jax.Array, mesh_dimensions: tuple[int, int, int], reciprocal_cell: jax.Array \| None=None) -> jax.Array |
| added | nvalchemiops/jax/interactions/electrostatics/k_vectors.py:k_vectors_from_miller_indices | — | (cell: jax.Array, miller_indices: jax.Array) -> jax.Array |
| changed | nvalchemiops/jax/interactions/electrostatics/parameters.py:estimate_pme_mesh_dimensions | (cell: jax.Array, alpha: jax.Array, accuracy: float=1e-06, mesh_safety_factor: float=1.0) -> tuple[int, int, int] | (cell: jax.Array, alpha: jax.Array, accuracy: float=1e-06, mesh_safety_factor: float=1.0, *, spline_order: int=_DEFAULT_PME_SPLINE_ORDER, fft_padding_fraction: float=_DEFAULT_PME_FFT_PADDING_FRACTION) -> tuple[int, int, int] |
| changed | nvalchemiops/jax/interactions/electrostatics/parameters.py:estimate_pme_parameters | (positions: jax.Array, cell: jax.Array, batch_idx: jax.Array \| None=None, accuracy: float=1e-06, real_space_cutoff: float \| None=None, mesh_safety_factor: float=1.0) -> PMEParameters | (positions: jax.Array, cell: jax.Array, batch_idx: jax.Array \| None=None, accuracy: float=1e-06, real_space_cutoff: float \| None=None, mesh_safety_factor: float=1.0, *, spline_order: int=_DEFAULT_PME_SPLINE_ORDER, fft_padding_fraction: float=_DEFAULT_PME_FFT_PADDING_FRACTION) -> PMEParameters |
| changed | nvalchemiops/jax/interactions/electrostatics/parameters.py:mesh_spacing_to_dimensions | (cell: jax.Array, mesh_spacing: float \| jax.Array) -> tuple[int, int, int] | (cell: jax.Array, mesh_spacing: float \| jax.Array, *, spline_order: int=_DEFAULT_PME_SPLINE_ORDER, fft_padding_fraction: float=_DEFAULT_PME_FFT_PADDING_FRACTION) -> tuple[int, int, int] |
| changed | nvalchemiops/jax/interactions/electrostatics/pme.py:particle_mesh_ewald | (positions: jax.Array, charges: jax.Array, cell: jax.Array, alpha: float \| jax.Array \| None=None, mesh_spacing: float \| None=None, mesh_dimensions: tuple[int, int, int] \| None=None, spline_order: int=4, batch_idx: jax.Array \| None=None, k_vectors: jax.Array \| None=None, k_squared: jax.Array \| None=None, neighbor_list: jax.Array \| None=None, neighbor_ptr: jax.Array \| None=None, neighbor_shifts: jax.Array \| None=None, neighbor_matrix: jax.Array \| None=None, neighbor_matrix_shifts: jax.Array \| None=None, mask_value: int \| None=None, compute_forces: bool=False, compute_charge_gradients: bool=False, compute_virial: bool=False, accuracy: float=1e-06, hybrid_forces: bool=False, pbc: jax.Array \| None=None, slab_correction: bool=False, *, energy_reduction: Literal['atom', 'system']='atom', cell_inv_t: jax.Array \| None=None, volume: jax.Array \| None=None, moduli_x: jax.Array \| None=None, moduli_y: jax.Array \| None=None, moduli_z: jax.Array \| None=None) -> jax.Array \| tuple[jax.Array, jax.Array] \| tuple[jax.Array, jax.Array, jax.Array] \| tuple[jax.Array, jax.Array, jax.Array, jax.Array] | (positions: jax.Array, charges: jax.Array, cell: jax.Array, alpha: float \| jax.Array \| None=None, mesh_spacing: float \| None=None, mesh_dimensions: tuple[int, int, int] \| None=None, spline_order: int=_DEFAULT_PME_SPLINE_ORDER, batch_idx: jax.Array \| None=None, k_vectors: jax.Array \| None=None, k_squared: jax.Array \| None=None, neighbor_list: jax.Array \| None=None, neighbor_ptr: jax.Array \| None=None, neighbor_shifts: jax.Array \| None=None, neighbor_matrix: jax.Array \| None=None, neighbor_matrix_shifts: jax.Array \| None=None, mask_value: int \| None=None, compute_forces: bool=False, compute_charge_gradients: bool=False, compute_virial: bool=False, accuracy: float=1e-06, hybrid_forces: bool=False, pbc: jax.Array \| None=None, slab_correction: bool=False, *, energy_reduction: Literal['atom', 'system']='atom', cell_inv_t: jax.Array \| None=None, volume: jax.Array \| None=None, moduli_x: jax.Array \| None=None, moduli_y: jax.Array \| None=None, moduli_z: jax.Array \| None=None) -> jax.Array \| tuple[jax.Array, jax.Array] \| tuple[jax.Array, jax.Array, jax.Array] \| tuple[jax.Array, jax.Array, jax.Array, jax.Array] |
| changed | nvalchemiops/jax/interactions/electrostatics/pme.py:pme_green_structure_factor | (k_squared: jax.Array, mesh_dimensions: tuple[int, int, int], alpha: jax.Array, cell: jax.Array, spline_order: int=4, batch_idx: jax.Array \| None=None) -> tuple[jax.Array, jax.Array] | (k_squared: jax.Array, mesh_dimensions: tuple[int, int, int], alpha: jax.Array, cell: jax.Array, spline_order: int=_DEFAULT_PME_SPLINE_ORDER, batch_idx: jax.Array \| None=None) -> tuple[jax.Array, jax.Array] |
| changed | nvalchemiops/jax/interactions/electrostatics/pme.py:pme_reciprocal_space | (positions: jax.Array, charges: jax.Array, cell: jax.Array, alpha: jax.Array, mesh_dimensions: tuple[int, int, int] \| None=None, mesh_spacing: float \| None=None, spline_order: int=4, batch_idx: jax.Array \| None=None, k_vectors: jax.Array \| None=None, k_squared: jax.Array \| None=None, compute_forces: bool=False, compute_charge_gradients: bool=False, compute_virial: bool=False, hybrid_forces: bool=False, *, energy_reduction: Literal['atom', 'system']='atom', cell_inv_t: jax.Array \| None=None, volume: jax.Array \| None=None, moduli_x: jax.Array \| None=None, moduli_y: jax.Array \| None=None, moduli_z: jax.Array \| None=None) -> jax.Array \| tuple[jax.Array, jax.Array] \| tuple[jax.Array, jax.Array, jax.Array] \| tuple[jax.Array, jax.Array, jax.Array, jax.Array] | (positions: jax.Array, charges: jax.Array, cell: jax.Array, alpha: jax.Array, mesh_dimensions: tuple[int, int, int] \| None=None, mesh_spacing: float \| None=None, spline_order: int=_DEFAULT_PME_SPLINE_ORDER, batch_idx: jax.Array \| None=None, k_vectors: jax.Array \| None=None, k_squared: jax.Array \| None=None, compute_forces: bool=False, compute_charge_gradients: bool=False, compute_virial: bool=False, hybrid_forces: bool=False, *, energy_reduction: Literal['atom', 'system']='atom', cell_inv_t: jax.Array \| None=None, volume: jax.Array \| None=None, moduli_x: jax.Array \| None=None, moduli_y: jax.Array \| None=None, moduli_z: jax.Array \| None=None) -> jax.Array \| tuple[jax.Array, jax.Array] \| tuple[jax.Array, jax.Array, jax.Array] \| tuple[jax.Array, jax.Array, jax.Array, jax.Array] |
| added | nvalchemiops/jax/lbfgs.py:lbfgs_prepare_cell_state | — | (atom_ptr: jax.Array, cell: jax.Array, *, cell_force_scale: float=1.0, dtype=jnp.float64, alignment_atol: float \| None=None) -> LBFGSCellState |
| added | nvalchemiops/jax/lbfgs.py:lbfgs_prepare_state | — | (num_dofs: int, num_systems: int, *, dtype=jnp.float64, history_size: int=6) -> LBFGSState |
| added | nvalchemiops/jax/lbfgs.py:lbfgs_step_coord | — | (positions: jax.Array, forces: jax.Array, state: LBFGSState, batch_idx: jax.Array, *, maxstep: float=0.2, curvature_eps: float \| None=None, graph_mode: str='warp') -> tuple[jax.Array, LBFGSState] |
| added | nvalchemiops/jax/lbfgs.py:lbfgs_step_coord_cell | — | (positions: jax.Array, cell: jax.Array, forces: jax.Array, stress: jax.Array, state: LBFGSState, cell_state: LBFGSCellState, batch_idx: jax.Array, *, maxstep: float=0.2, curvature_eps: float \| None=None, graph_mode: str='warp') -> tuple[jax.Array, jax.Array, LBFGSState, LBFGSCellState] |
| changed | nvalchemiops/jax/neighbors/__init__.py:exports | __all__ = ['neighbor_list', 'estimate_neighbor_list_costs', 'suggest_neighbor_list_method', 'naive_neighbor_list', 'naive_neighbor_list_dual_cutoff', 'estimate_cell_list_sizes', 'build_cell_list', 'query_cell_list', 'cell_list', 'estimate_cluster_tile_list_sizes', 'build_cluster_tile_list', 'query_cluster_tile', 'query_cluster_tile_coo', 'cluster_tile_neighbor_list', 'batch_naive_neighbor_list', 'batch_naive_neighbor_list_dual_cutoff', 'estimate_batch_cell_list_sizes', 'batch_build_cell_list', 'batch_query_cell_list', 'batch_cell_list', 'estimate_batch_cluster_tile_list_sizes', 'estimate_batch_max_tiles_per_group', 'estimate_batch_cluster_tile_segments', 'allocate_batch_cluster_tile_list', 'batch_build_cluster_tile_list', 'batch_query_cluster_tile', 'batch_query_cluster_tile_coo', 'batch_cluster_tile_neighbor_list', 'cell_list_needs_rebuild', 'neighbor_list_needs_rebuild', 'check_cell_list_rebuild_needed', 'check_neighbor_list_rebuild_needed', 'compute_naive_num_shifts', 'get_neighbor_list_from_neighbor_matrix', 'prepare_batch_idx_ptr', 'allocate_cell_list', 'estimate_max_neighbors', 'NeighborOverflowError'] ; from __future__ import annotations ; from nvalchemiops.jax.neighbors._dispatch import _auto_method_from_geometry, _reject_unsupported_cluster_tile_combo, estimate_neighbor_list_costs, suggest_neighbor_list_method, synthesize_cell_for_cell_list ; from nvalchemiops.jax.neighbors.batch_cell_list import batch_build_cell_list, batch_cell_list, batch_query_cell_list, estimate_batch_cell_list_sizes ; from nvalchemiops.jax.neighbors.batch_cluster_tile import allocate_batch_cluster_tile_list, batch_build_cluster_tile_list, batch_cluster_tile_neighbor_list, batch_query_cluster_tile, batch_query_cluster_tile_coo, estimate_batch_cluster_tile_list_sizes, estimate_batch_cluster_tile_segments, estimate_batch_max_tiles_per_group ; from nvalchemiops.jax.neighbors.batch_naive import batch_naive_neighbor_list ; from nvalchemiops.jax.neighbors.batch_naive_dual_cutoff import batch_naive_neighbor_list_dual_cutoff ; from nvalchemiops.jax.neighbors.cell_list import build_cell_list, cell_list, estimate_cell_list_sizes, query_cell_list ; from nvalchemiops.jax.neighbors.cluster_tile import build_cluster_tile_list, cluster_tile_neighbor_list, estimate_cluster_tile_list_sizes, query_cluster_tile, query_cluster_tile_coo ; from nvalchemiops.jax.neighbors.naive import naive_neighbor_list ; from nvalchemiops.jax.neighbors.naive_dual_cutoff import naive_neighbor_list_dual_cutoff ; from nvalchemiops.jax.neighbors.neighbor_utils import NeighborOverflowError, allocate_cell_list, compute_naive_num_shifts, estimate_max_neighbors, get_neighbor_list_from_neighbor_matrix, prepare_batch_idx_ptr ; from nvalchemiops.jax.neighbors.rebuild_detection import batch_cell_list_needs_rebuild, batch_neighbor_list_needs_rebuild, cell_list_needs_rebuild, check_batch_cell_list_rebuild_needed, check_batch_neighbor_list_rebuild_needed, check_cell_list_rebuild_needed, check_neighbor_list_rebuild_needed, neighbor_list_needs_rebuild ; from nvalchemiops.neighbors.base_dispatch import NEIGHBOR_LIST_STRATEGIES, neighbor_list_strategy_run_args ; import jax ; import jax.numpy as jnp | __all__ = ['neighbor_list', 'NeighborListState', 'prepare_neighbor_list', 'check_neighbor_list_state', 'estimate_neighbor_list_costs', 'suggest_neighbor_list_method', 'naive_neighbor_list', 'naive_neighbor_list_dual_cutoff', 'estimate_cell_list_sizes', 'build_cell_list', 'query_cell_list', 'cell_list', 'estimate_cluster_tile_list_sizes', 'build_cluster_tile_list', 'query_cluster_tile', 'query_cluster_tile_coo', 'cluster_tile_neighbor_list', 'batch_naive_neighbor_list', 'batch_naive_neighbor_list_dual_cutoff', 'estimate_batch_cell_list_sizes', 'batch_build_cell_list', 'batch_query_cell_list', 'batch_cell_list', 'estimate_batch_cluster_tile_list_sizes', 'estimate_batch_max_tiles_per_group', 'estimate_batch_cluster_tile_segments', 'allocate_batch_cluster_tile_list', 'batch_build_cluster_tile_list', 'batch_query_cluster_tile', 'batch_query_cluster_tile_coo', 'batch_cluster_tile_neighbor_list', 'cell_list_needs_rebuild', 'neighbor_list_needs_rebuild', 'check_cell_list_rebuild_needed', 'check_neighbor_list_rebuild_needed', 'compute_naive_num_shifts', 'get_fixed_capacity_neighbor_list_from_neighbor_matrix', 'get_neighbor_list_from_neighbor_matrix', 'prepare_batch_idx_ptr', 'allocate_cell_list', 'estimate_max_neighbors', 'compute_batch_pair_centric_n_outer', 'NeighborOverflowError', 'TileBufferOverflow'] ; from __future__ import annotations ; from nvalchemiops.jax.neighbors._dispatch import _auto_method_from_geometry, _reject_unsupported_cluster_tile_combo, estimate_neighbor_list_costs, suggest_neighbor_list_method, synthesize_cell_for_cell_list ; from nvalchemiops.jax.neighbors.batch_cell_list import batch_build_cell_list, batch_cell_list, batch_query_cell_list, estimate_batch_cell_list_sizes ; from nvalchemiops.jax.neighbors.batch_cluster_tile import allocate_batch_cluster_tile_list, batch_build_cluster_tile_list, batch_cluster_tile_neighbor_list, batch_query_cluster_tile, batch_query_cluster_tile_coo, estimate_batch_cluster_tile_list_sizes, estimate_batch_cluster_tile_segments, estimate_batch_max_tiles_per_group ; from nvalchemiops.jax.neighbors.batch_naive import batch_naive_neighbor_list ; from nvalchemiops.jax.neighbors.batch_naive_dual_cutoff import batch_naive_neighbor_list_dual_cutoff ; from nvalchemiops.jax.neighbors.cell_list import build_cell_list, cell_list, estimate_cell_list_sizes, query_cell_list ; from nvalchemiops.jax.neighbors.cluster_tile import build_cluster_tile_list, cluster_tile_neighbor_list, estimate_cluster_tile_list_sizes, query_cluster_tile, query_cluster_tile_coo ; from nvalchemiops.jax.neighbors.naive import naive_neighbor_list ; from nvalchemiops.jax.neighbors.naive_dual_cutoff import naive_neighbor_list_dual_cutoff ; from nvalchemiops.jax.neighbors.neighbor_utils import NeighborOverflowError, TileBufferOverflow, _validate_dual_cutoff_order, allocate_cell_list, compute_naive_num_shifts, estimate_max_neighbors, get_fixed_capacity_neighbor_list_from_neighbor_matrix, get_neighbor_list_from_neighbor_matrix, prepare_batch_idx_ptr ; from nvalchemiops.jax.neighbors.prepared_neighbor_list import NeighborListState, _execute_prepared_neighbor_list, check_neighbor_list_state, prepare_neighbor_list ; from nvalchemiops.jax.neighbors.rebuild_detection import batch_cell_list_needs_rebuild, batch_neighbor_list_needs_rebuild, cell_list_needs_rebuild, check_batch_cell_list_rebuild_needed, check_batch_neighbor_list_rebuild_needed, check_cell_list_rebuild_needed, check_neighbor_list_rebuild_needed, neighbor_list_needs_rebuild ; from nvalchemiops.neighbors.base_dispatch import NEIGHBOR_LIST_STRATEGIES, neighbor_list_strategy_run_args ; from nvalchemiops.neighbors.cell_list import compute_batch_pair_centric_n_outer ; from typing import Any ; import jax ; import jax.numpy as jnp |
| changed | nvalchemiops/jax/neighbors/__init__.py:neighbor_list | (positions: jax.Array, cutoff: float, cell: jax.Array \| None=None, pbc: jax.Array \| None=None, batch_idx: jax.Array \| None=None, batch_ptr: jax.Array \| None=None, cutoff2: float \| None=None, half_fill: bool=False, fill_value: int \| None=None, return_neighbor_list: bool=False, method: str \| None=None, wrap_positions: bool=True, **kwargs: dict) -> unknown | (positions: jax.Array, cutoff: float \| None=None, cell: jax.Array \| None=None, pbc: jax.Array \| None=None, batch_idx: jax.Array \| None=None, batch_ptr: jax.Array \| None=None, cutoff2: float \| None=None, half_fill: bool=False, fill_value: int \| None=None, return_neighbor_list: bool=False, method: str \| None=None, wrap_positions: bool=True, *, grid_policy: str='configured', state: NeighborListState \| None=None, **kwargs: Any) -> unknown |
| changed | nvalchemiops/jax/neighbors/_dispatch.py:estimate_neighbor_list_costs | (batch_ptr: jax.Array, cell: jax.Array, pbc: jax.Array, cutoff: float, *, batch_idx: jax.Array \| None=None, max_nbins: int \| None=None, optional_outputs: Iterable[str] \| None=None, cutoff2: float \| None=None, half_fill: bool=False, return_neighbor_list: bool=False, target_indices: jax.Array \| None=None, return_vectors: bool=False, return_distances: bool=False, use_pair_fn: bool=False, rebuild_flags: jax.Array \| None=None, wrap_positions: bool=True, positions_dtype=None) -> list[tuple[str, float]] | (batch_ptr: jax.Array, cell: jax.Array \| None=None, pbc: jax.Array \| None=None, cutoff: float=0.0, *, positions: jax.Array \| None=None, batch_idx: jax.Array \| None=None, max_nbins: int \| None=None, optional_outputs: Iterable[str] \| None=None, cutoff2: float \| None=None, half_fill: bool=False, return_neighbor_list: bool=False, target_indices: jax.Array \| None=None, return_vectors: bool=False, return_distances: bool=False, use_pair_fn: bool=False, rebuild_flags: jax.Array \| None=None, wrap_positions: bool=True, positions_dtype=None) -> list[tuple[str, float]] |
| added | nvalchemiops/jax/neighbors/_resolution.py:resolve_cell_strategy | — | (strategy: str, *, total_atoms: int, cutoff: float, device_is_cpu: bool, half_fill: bool=False) -> str |
| changed | nvalchemiops/jax/neighbors/batch_cell_list.py:batch_build_cell_list | (positions: jax.Array, batch_idx: jax.Array \| None=None, batch_ptr: jax.Array \| None=None, cell: jax.Array \| None=None, pbc: jax.Array \| None=None, cutoff: float=5.0, max_total_cells: int \| None=None, target_indices: jax.Array \| None=None, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| None=None, pair_params: jax.Array \| None=None, neighbor_vectors: jax.Array \| None=None, neighbor_distances: jax.Array \| None=None, pair_energies: jax.Array \| None=None, pair_forces: jax.Array \| None=None) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array] | (positions: jax.Array, batch_idx: jax.Array \| None=None, batch_ptr: jax.Array \| None=None, cell: jax.Array \| None=None, pbc: jax.Array \| None=None, cutoff: float=5.0, max_total_cells: int \| None=None, target_indices: jax.Array \| None=None, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| None=None, pair_params: jax.Array \| None=None, neighbor_vectors: jax.Array \| None=None, neighbor_distances: jax.Array \| None=None, pair_energies: jax.Array \| None=None, pair_forces: jax.Array \| None=None, _fixed_cell_geometry: tuple[jax.Array, ...] \| None=None) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array] |
| changed | nvalchemiops/jax/neighbors/batch_cell_list.py:batch_cell_list | (positions: jax.Array, cutoff: float, cell: jax.Array \| None=None, pbc: jax.Array \| None=None, batch_idx: jax.Array \| None=None, batch_ptr: jax.Array \| None=None, max_neighbors: int \| None=None, max_total_cells: int \| None=None, neighbor_matrix_shifts: jax.Array \| None=None, return_neighbor_list: bool=False, half_fill: bool=False, fill_value: int \| None=None, strategy: str='auto', atom_centric_path: str='auto', target_indices: jax.Array \| None=None, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| None=None, pair_params: jax.Array \| None=None, neighbor_vectors: jax.Array \| None=None, neighbor_distances: jax.Array \| None=None, pair_energies: jax.Array \| None=None, pair_forces: jax.Array \| None=None) -> tuple[jax.Array, ...] | (positions: jax.Array, cutoff: float, cell: jax.Array \| None=None, pbc: jax.Array \| None=None, batch_idx: jax.Array \| None=None, batch_ptr: jax.Array \| None=None, max_neighbors: int \| None=None, max_total_cells: int \| None=None, neighbor_matrix_shifts: jax.Array \| None=None, return_neighbor_list: bool=False, half_fill: bool=False, fill_value: int \| None=None, strategy: str='auto', atom_centric_path: str='auto', target_indices: jax.Array \| None=None, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| None=None, pair_params: jax.Array \| None=None, neighbor_vectors: jax.Array \| None=None, neighbor_distances: jax.Array \| None=None, pair_energies: jax.Array \| None=None, pair_forces: jax.Array \| None=None, coo_capacity: int \| None=None, pair_centric_total_cells: int \| None=None, pair_centric_n_outer: int \| None=None, pair_centric_r_max: tuple[int, int, int] \| None=None, _return_status: bool=False, _fixed_cell_geometry: tuple[jax.Array, ...] \| None=None, *, grid_policy: str='configured') -> tuple[jax.Array, ...] |
| changed | nvalchemiops/jax/neighbors/batch_cell_list.py:batch_query_cell_list | (positions: jax.Array, batch_idx: jax.Array \| None=None, batch_ptr: jax.Array \| None=None, cutoff: float=5.0, cell: jax.Array \| None=None, pbc: jax.Array \| None=None, cells_per_dimension: jax.Array \| None=None, atom_periodic_shifts: jax.Array \| None=None, atom_to_cell_mapping: jax.Array \| None=None, cell_atom_start_indices: jax.Array \| None=None, cell_atom_list: jax.Array \| None=None, atoms_per_cell_count: jax.Array \| None=None, neighbor_search_radius: jax.Array \| None=None, max_neighbors: int \| None=None, neighbor_matrix: jax.Array \| None=None, num_neighbors: jax.Array \| None=None, neighbor_matrix_shifts: jax.Array \| None=None, rebuild_flags: jax.Array \| None=None, half_fill: bool=False, strategy: str='auto', atom_centric_path: str='auto', target_indices: jax.Array \| None=None, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| None=None, pair_params: jax.Array \| None=None, neighbor_vectors: jax.Array \| None=None, neighbor_distances: jax.Array \| None=None, pair_energies: jax.Array \| None=None, pair_forces: jax.Array \| None=None) -> tuple[jax.Array, ...] | (positions: jax.Array, batch_idx: jax.Array \| None=None, batch_ptr: jax.Array \| None=None, cutoff: float=5.0, cell: jax.Array \| None=None, pbc: jax.Array \| None=None, cells_per_dimension: jax.Array \| None=None, atom_periodic_shifts: jax.Array \| None=None, atom_to_cell_mapping: jax.Array \| None=None, cell_atom_start_indices: jax.Array \| None=None, cell_atom_list: jax.Array \| None=None, atoms_per_cell_count: jax.Array \| None=None, neighbor_search_radius: jax.Array \| None=None, max_neighbors: int \| None=None, neighbor_matrix: jax.Array \| None=None, num_neighbors: jax.Array \| None=None, neighbor_matrix_shifts: jax.Array \| None=None, rebuild_flags: jax.Array \| None=None, half_fill: bool=False, strategy: str='auto', atom_centric_path: str='auto', target_indices: jax.Array \| None=None, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| None=None, pair_params: jax.Array \| None=None, neighbor_vectors: jax.Array \| None=None, neighbor_distances: jax.Array \| None=None, pair_energies: jax.Array \| None=None, pair_forces: jax.Array \| None=None, pair_centric_total_cells: int \| None=None, pair_centric_n_outer: int \| None=None, pair_centric_r_max: tuple[int, int, int] \| None=None) -> tuple[jax.Array, ...] |
| changed | nvalchemiops/jax/neighbors/batch_cluster_tile.py:batch_build_cluster_tile_list | (positions: jax.Array, cutoff: float, cell_batch: jax.Array, batch_ptr: jax.Array, *, rebuild_flags: jax.Array \| None=None, tile_offsets: jax.Array \| None=None, tile_counts: jax.Array \| None=None, num_tiles: jax.Array \| None=None, tile_row_group: jax.Array \| None=None, tile_col_group: jax.Array \| None=None, tile_system: jax.Array \| None=None) -> tuple[jax.Array, ...] | (positions: jax.Array, cutoff: float, cell_batch: jax.Array, batch_ptr: jax.Array, *, max_tiles_per_group: int \| None=None, rebuild_flags: jax.Array \| None=None, tile_offsets: jax.Array \| None=None, tile_counts: jax.Array \| None=None, num_tiles: jax.Array \| None=None, tile_row_group: jax.Array \| None=None, tile_col_group: jax.Array \| None=None, tile_system: jax.Array \| None=None, _fixed_cell_geometry: tuple[jax.Array \| None, ...] \| None=None) -> tuple[jax.Array, ...] |
| changed | nvalchemiops/jax/neighbors/batch_cluster_tile.py:batch_cluster_tile_neighbor_list | (positions: jax.Array, cutoff: float, cell_batch: jax.Array, batch_ptr: jax.Array, max_neighbors: int \| None=None, fill_value: int \| None=None, format: str='matrix', max_pairs: int \| None=None, *, cutoff2: float \| None=None, rebuild_flags: jax.Array \| None=None, tile_offsets: jax.Array \| None=None, pair_offsets: jax.Array \| None=None, previous_tile_counts: jax.Array \| None=None, previous_pair_counts: jax.Array \| None=None, previous_neighbor_list: jax.Array \| None=None, previous_neighbor_list_shifts: jax.Array \| None=None, previous_num_tiles: jax.Array \| None=None, previous_tile_row_group: jax.Array \| None=None, previous_tile_col_group: jax.Array \| None=None, previous_tile_system: jax.Array \| None=None, previous_neighbor_matrix: jax.Array \| None=None, previous_num_neighbors: jax.Array \| None=None, previous_neighbor_matrix_shifts: jax.Array \| None=None, previous_neighbor_matrix2: jax.Array \| None=None, previous_num_neighbors2: jax.Array \| None=None, previous_neighbor_matrix_shifts2: jax.Array \| None=None, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| None=None, pair_params: jax.Array \| None=None, neighbor_vectors: jax.Array \| None=None, neighbor_distances: jax.Array \| None=None, pair_energies: jax.Array \| None=None, pair_forces: jax.Array \| None=None) -> tuple[jax.Array, ...] | (positions: jax.Array, cutoff: float, cell_batch: jax.Array, batch_ptr: jax.Array, max_neighbors: int \| None=None, fill_value: int \| None=None, format: str='matrix', max_pairs: int \| None=None, *, cutoff2: float \| None=None, rebuild_flags: jax.Array \| None=None, tile_offsets: jax.Array \| None=None, pair_offsets: jax.Array \| None=None, previous_tile_counts: jax.Array \| None=None, previous_pair_counts: jax.Array \| None=None, previous_neighbor_list: jax.Array \| None=None, previous_neighbor_list_shifts: jax.Array \| None=None, previous_num_tiles: jax.Array \| None=None, previous_tile_row_group: jax.Array \| None=None, previous_tile_col_group: jax.Array \| None=None, previous_tile_system: jax.Array \| None=None, previous_neighbor_matrix: jax.Array \| None=None, previous_num_neighbors: jax.Array \| None=None, previous_neighbor_matrix_shifts: jax.Array \| None=None, previous_neighbor_matrix2: jax.Array \| None=None, previous_num_neighbors2: jax.Array \| None=None, previous_neighbor_matrix_shifts2: jax.Array \| None=None, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| None=None, pair_params: jax.Array \| None=None, neighbor_vectors: jax.Array \| None=None, neighbor_distances: jax.Array \| None=None, pair_energies: jax.Array \| None=None, pair_forces: jax.Array \| None=None, max_tiles_per_group: int \| None=None, _return_status: bool=False, _fixed_cell_geometry: tuple[jax.Array \| None, ...] \| None=None) -> tuple[jax.Array, ...] |
| changed | nvalchemiops/jax/neighbors/batch_cluster_tile.py:batch_query_cluster_tile | (sorted_atom_index: jax.Array, sorted_pos_x: jax.Array, sorted_pos_y: jax.Array, sorted_pos_z: jax.Array, cell_batch: jax.Array, num_tiles: jax.Array, tile_row_group: jax.Array, tile_col_group: jax.Array, tile_system: jax.Array, cutoff: float, natom: int, max_neighbors: int, *, fill_value: int \| None=None, cutoff2: float \| None=None, rebuild_flags: jax.Array \| None=None, tile_offsets: jax.Array \| None=None, tile_counts: jax.Array \| None=None, batch_idx: jax.Array \| None=None, neighbor_matrix: jax.Array \| None=None, num_neighbors: jax.Array \| None=None, neighbor_matrix_shifts: jax.Array \| None=None, neighbor_matrix2: jax.Array \| None=None, num_neighbors2: jax.Array \| None=None, neighbor_matrix_shifts2: jax.Array \| None=None, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| None=None, pair_params: jax.Array \| None=None, neighbor_vectors: jax.Array \| None=None, neighbor_distances: jax.Array \| None=None, pair_energies: jax.Array \| None=None, pair_forces: jax.Array \| None=None) -> tuple[jax.Array, ...] | (sorted_atom_index: jax.Array, sorted_pos_x: jax.Array, sorted_pos_y: jax.Array, sorted_pos_z: jax.Array, cell_batch: jax.Array, num_tiles: jax.Array, tile_row_group: jax.Array, tile_col_group: jax.Array, tile_system: jax.Array, cutoff: float, natom: int, max_neighbors: int, *, fill_value: int \| None=None, cutoff2: float \| None=None, rebuild_flags: jax.Array \| None=None, tile_offsets: jax.Array \| None=None, tile_counts: jax.Array \| None=None, batch_idx: jax.Array \| None=None, neighbor_matrix: jax.Array \| None=None, num_neighbors: jax.Array \| None=None, neighbor_matrix_shifts: jax.Array \| None=None, neighbor_matrix2: jax.Array \| None=None, num_neighbors2: jax.Array \| None=None, neighbor_matrix_shifts2: jax.Array \| None=None, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| None=None, pair_params: jax.Array \| None=None, neighbor_vectors: jax.Array \| None=None, neighbor_distances: jax.Array \| None=None, pair_energies: jax.Array \| None=None, pair_forces: jax.Array \| None=None, _fixed_cell_geometry: tuple[jax.Array \| None, ...] \| None=None) -> tuple[jax.Array, ...] |
| changed | nvalchemiops/jax/neighbors/batch_cluster_tile.py:batch_query_cluster_tile_coo | (sorted_atom_index: jax.Array, sorted_pos_x: jax.Array, sorted_pos_y: jax.Array, sorted_pos_z: jax.Array, cell_batch: jax.Array, num_tiles: jax.Array, tile_row_group: jax.Array, tile_col_group: jax.Array, tile_system: jax.Array, cutoff: float, natom: int, max_pairs: int, *, rebuild_flags: jax.Array \| None=None, tile_offsets: jax.Array \| None=None, tile_counts: jax.Array \| None=None, pair_offsets: jax.Array \| None=None, pair_counts: jax.Array \| None=None, neighbor_list: jax.Array \| None=None, neighbor_list_shifts: jax.Array \| None=None) -> tuple[jax.Array, ...] | (sorted_atom_index: jax.Array, sorted_pos_x: jax.Array, sorted_pos_y: jax.Array, sorted_pos_z: jax.Array, cell_batch: jax.Array, num_tiles: jax.Array, tile_row_group: jax.Array, tile_col_group: jax.Array, tile_system: jax.Array, cutoff: float, natom: int, max_pairs: int, *, rebuild_flags: jax.Array \| None=None, tile_offsets: jax.Array \| None=None, tile_counts: jax.Array \| None=None, pair_offsets: jax.Array \| None=None, pair_counts: jax.Array \| None=None, neighbor_list: jax.Array \| None=None, neighbor_list_shifts: jax.Array \| None=None, _return_pair_counter: bool=False, _fixed_cell_geometry: tuple[jax.Array \| None, ...] \| None=None) -> tuple[jax.Array, ...] |
| changed | nvalchemiops/jax/neighbors/batch_naive.py:batch_naive_neighbor_list | (positions: jax.Array, cutoff: float, batch_idx: jax.Array \| None=None, batch_ptr: jax.Array \| None=None, pbc: jax.Array \| None=None, cell: jax.Array \| None=None, max_neighbors: int \| None=None, half_fill: bool=False, fill_value: int \| None=None, return_neighbor_list: bool=False, neighbor_matrix: jax.Array \| None=None, neighbor_matrix_shifts: jax.Array \| None=None, num_neighbors: jax.Array \| None=None, shift_range_per_dimension: jax.Array \| None=None, num_shifts_per_system: jax.Array \| None=None, max_shifts_per_system: int \| None=None, max_atoms_per_system: int \| None=None, rebuild_flags: jax.Array \| None=None, wrap_positions: bool=True, positions_wrapped_buffer: jax.Array \| None=None, per_atom_cell_offsets_buffer: jax.Array \| None=None, inv_cell_buffer: jax.Array \| None=None, strategy: str='auto', *, return_distances: bool=False, return_vectors: bool=False, neighbor_vectors: jax.Array \| None=None, neighbor_distances: jax.Array \| None=None, target_indices: jax.Array \| None=None, pair_fn=None, pair_params: jax.Array \| None=None, pair_energies: jax.Array \| None=None, pair_forces: jax.Array \| None=None) -> tuple[jax.Array, ...] | (positions: jax.Array, cutoff: float, batch_idx: jax.Array \| None=None, batch_ptr: jax.Array \| None=None, pbc: jax.Array \| None=None, cell: jax.Array \| None=None, max_neighbors: int \| None=None, half_fill: bool=False, fill_value: int \| None=None, return_neighbor_list: bool=False, neighbor_matrix: jax.Array \| None=None, neighbor_matrix_shifts: jax.Array \| None=None, num_neighbors: jax.Array \| None=None, shift_range_per_dimension: jax.Array \| None=None, num_shifts_per_system: jax.Array \| None=None, max_shifts_per_system: int \| None=None, max_atoms_per_system: int \| None=None, rebuild_flags: jax.Array \| None=None, wrap_positions: bool=True, positions_wrapped_buffer: jax.Array \| None=None, per_atom_cell_offsets_buffer: jax.Array \| None=None, inv_cell_buffer: jax.Array \| None=None, strategy: str='auto', *, coo_capacity: int \| None=None, return_distances: bool=False, return_vectors: bool=False, neighbor_vectors: jax.Array \| None=None, neighbor_distances: jax.Array \| None=None, target_indices: jax.Array \| None=None, pair_fn=None, pair_params: jax.Array \| None=None, pair_energies: jax.Array \| None=None, pair_forces: jax.Array \| None=None, _fixed_cell_geometry: tuple[jax.Array \| None, ...] \| None=None) -> tuple[jax.Array, ...] |
| changed | nvalchemiops/jax/neighbors/batch_naive_dual_cutoff.py:batch_naive_neighbor_list_dual_cutoff | (positions: jax.Array, cutoff1: float, cutoff2: float, batch_idx: jax.Array \| None=None, batch_ptr: jax.Array \| None=None, pbc: jax.Array \| None=None, cell: jax.Array \| None=None, max_neighbors1: int \| None=None, max_neighbors2: int \| None=None, half_fill: bool=False, fill_value: int \| None=None, return_neighbor_list: bool=False, neighbor_matrix1: jax.Array \| None=None, neighbor_matrix2: jax.Array \| None=None, neighbor_matrix_shifts1: jax.Array \| None=None, neighbor_matrix_shifts2: jax.Array \| None=None, num_neighbors1: jax.Array \| None=None, num_neighbors2: jax.Array \| None=None, shift_range_per_dimension: jax.Array \| None=None, num_shifts_per_system: jax.Array \| None=None, max_shifts_per_system: int \| None=None, max_atoms_per_system: int \| None=None, rebuild_flags: jax.Array \| None=None, wrap_positions: bool=True, positions_wrapped_buffer: jax.Array \| None=None, per_atom_cell_offsets_buffer: jax.Array \| None=None, inv_cell_buffer: jax.Array \| None=None) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array] \| tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array] \| tuple[jax.Array, jax.Array, jax.Array, jax.Array] | (positions: jax.Array, cutoff1: float, cutoff2: float, batch_idx: jax.Array \| None=None, batch_ptr: jax.Array \| None=None, pbc: jax.Array \| None=None, cell: jax.Array \| None=None, max_neighbors1: int \| None=None, max_neighbors2: int \| None=None, half_fill: bool=False, fill_value: int \| None=None, return_neighbor_list: bool=False, neighbor_matrix1: jax.Array \| None=None, neighbor_matrix2: jax.Array \| None=None, neighbor_matrix_shifts1: jax.Array \| None=None, neighbor_matrix_shifts2: jax.Array \| None=None, num_neighbors1: jax.Array \| None=None, num_neighbors2: jax.Array \| None=None, shift_range_per_dimension: jax.Array \| None=None, num_shifts_per_system: jax.Array \| None=None, max_shifts_per_system: int \| None=None, max_atoms_per_system: int \| None=None, rebuild_flags: jax.Array \| None=None, wrap_positions: bool=True, positions_wrapped_buffer: jax.Array \| None=None, per_atom_cell_offsets_buffer: jax.Array \| None=None, inv_cell_buffer: jax.Array \| None=None, coo_capacity: int \| tuple[int, int] \| None=None, *, _fixed_cell_geometry: tuple[jax.Array \| None, ...] \| None=None) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array] \| tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array] \| tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array] \| tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array] |
| changed | nvalchemiops/jax/neighbors/cell_list.py:build_cell_list | (positions: jax.Array, cutoff: float, cell: jax.Array, pbc: jax.Array, cells_per_dimension: jax.Array \| None=None, neighbor_search_radius: jax.Array \| None=None, atom_periodic_shifts: jax.Array \| None=None, atom_to_cell_mapping: jax.Array \| None=None, atoms_per_cell_count: jax.Array \| None=None, cell_atom_start_indices: jax.Array \| None=None, cell_atom_list: jax.Array \| None=None, max_total_cells: int \| None=None, graph_mode: Literal['none', 'warp']='none', target_indices: jax.Array \| None=None, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| None=None, pair_params: jax.Array \| None=None, neighbor_vectors: jax.Array \| None=None, neighbor_distances: jax.Array \| None=None, pair_energies: jax.Array \| None=None, pair_forces: jax.Array \| None=None) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array] | (positions: jax.Array, cutoff: float, cell: jax.Array, pbc: jax.Array, cells_per_dimension: jax.Array \| None=None, neighbor_search_radius: jax.Array \| None=None, atom_periodic_shifts: jax.Array \| None=None, atom_to_cell_mapping: jax.Array \| None=None, atoms_per_cell_count: jax.Array \| None=None, cell_atom_start_indices: jax.Array \| None=None, cell_atom_list: jax.Array \| None=None, max_total_cells: int \| None=None, graph_mode: Literal['none', 'warp']='none', target_indices: jax.Array \| None=None, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| None=None, pair_params: jax.Array \| None=None, neighbor_vectors: jax.Array \| None=None, neighbor_distances: jax.Array \| None=None, pair_energies: jax.Array \| None=None, pair_forces: jax.Array \| None=None, _fixed_cell_geometry: tuple[jax.Array, ...] \| None=None) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array] |
| changed | nvalchemiops/jax/neighbors/cell_list.py:cell_list | (positions: jax.Array, cutoff: float, cell: jax.Array \| None=None, pbc: jax.Array \| None=None, max_neighbors: int \| None=None, max_total_cells: int \| None=None, return_neighbor_list: bool=False, half_fill: bool=False, fill_value: int \| None=None, strategy: str='auto', atom_centric_path: str='auto', cells_per_dimension: jax.Array \| None=None, neighbor_search_radius: jax.Array \| None=None, atom_periodic_shifts: jax.Array \| None=None, atom_to_cell_mapping: jax.Array \| None=None, atoms_per_cell_count: jax.Array \| None=None, cell_atom_start_indices: jax.Array \| None=None, cell_atom_list: jax.Array \| None=None, sorted_positions: jax.Array \| None=None, sorted_atom_periodic_shifts: jax.Array \| None=None, neighbor_matrix: jax.Array \| None=None, neighbor_matrix_shifts: jax.Array \| None=None, num_neighbors: jax.Array \| None=None, graph_mode: Literal['none', 'warp']='none', target_indices: jax.Array \| None=None, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| None=None, pair_params: jax.Array \| None=None, neighbor_vectors: jax.Array \| None=None, neighbor_distances: jax.Array \| None=None, pair_energies: jax.Array \| None=None, pair_forces: jax.Array \| None=None) -> tuple[jax.Array, ...] | (positions: jax.Array, cutoff: float, cell: jax.Array \| None=None, pbc: jax.Array \| None=None, max_neighbors: int \| None=None, max_total_cells: int \| None=None, return_neighbor_list: bool=False, half_fill: bool=False, fill_value: int \| None=None, strategy: str='auto', atom_centric_path: str='auto', cells_per_dimension: jax.Array \| None=None, neighbor_search_radius: jax.Array \| None=None, atom_periodic_shifts: jax.Array \| None=None, atom_to_cell_mapping: jax.Array \| None=None, atoms_per_cell_count: jax.Array \| None=None, cell_atom_start_indices: jax.Array \| None=None, cell_atom_list: jax.Array \| None=None, sorted_positions: jax.Array \| None=None, sorted_atom_periodic_shifts: jax.Array \| None=None, neighbor_matrix: jax.Array \| None=None, neighbor_matrix_shifts: jax.Array \| None=None, num_neighbors: jax.Array \| None=None, graph_mode: Literal['none', 'warp']='none', target_indices: jax.Array \| None=None, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| None=None, pair_params: jax.Array \| None=None, neighbor_vectors: jax.Array \| None=None, neighbor_distances: jax.Array \| None=None, pair_energies: jax.Array \| None=None, pair_forces: jax.Array \| None=None, coo_capacity: int \| None=None, pair_centric_n_outer: int \| None=None, _return_status: bool=False, _fixed_cell_geometry: tuple[jax.Array, ...] \| None=None, *, grid_policy: str='configured') -> tuple[jax.Array, ...] |
| changed | nvalchemiops/jax/neighbors/cell_list.py:query_cell_list | (positions: jax.Array, cutoff: float, cell: jax.Array, pbc: jax.Array, cells_per_dimension: jax.Array, atom_periodic_shifts: jax.Array, atom_to_cell_mapping: jax.Array, atoms_per_cell_count: jax.Array, cell_atom_start_indices: jax.Array, cell_atom_list: jax.Array, neighbor_search_radius: jax.Array, max_neighbors: int \| None=None, neighbor_matrix: jax.Array \| None=None, neighbor_matrix_shifts: jax.Array \| None=None, num_neighbors: jax.Array \| None=None, rebuild_flags: jax.Array \| None=None, graph_mode: Literal['none', 'warp']='none', half_fill: bool=False, strategy: str='auto', atom_centric_path: str='auto', sorted_positions: jax.Array \| None=None, sorted_atom_periodic_shifts: jax.Array \| None=None, target_indices: jax.Array \| None=None, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| None=None, pair_params: jax.Array \| None=None, neighbor_vectors: jax.Array \| None=None, neighbor_distances: jax.Array \| None=None, pair_energies: jax.Array \| None=None, pair_forces: jax.Array \| None=None) -> tuple[jax.Array, ...] | (positions: jax.Array, cutoff: float, cell: jax.Array, pbc: jax.Array, cells_per_dimension: jax.Array, atom_periodic_shifts: jax.Array, atom_to_cell_mapping: jax.Array, atoms_per_cell_count: jax.Array, cell_atom_start_indices: jax.Array, cell_atom_list: jax.Array, neighbor_search_radius: jax.Array, max_neighbors: int \| None=None, neighbor_matrix: jax.Array \| None=None, neighbor_matrix_shifts: jax.Array \| None=None, num_neighbors: jax.Array \| None=None, rebuild_flags: jax.Array \| None=None, graph_mode: Literal['none', 'warp']='none', half_fill: bool=False, strategy: str='auto', atom_centric_path: str='auto', sorted_positions: jax.Array \| None=None, sorted_atom_periodic_shifts: jax.Array \| None=None, target_indices: jax.Array \| None=None, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| None=None, pair_params: jax.Array \| None=None, neighbor_vectors: jax.Array \| None=None, neighbor_distances: jax.Array \| None=None, pair_energies: jax.Array \| None=None, pair_forces: jax.Array \| None=None, pair_centric_n_outer: int \| None=None) -> tuple[jax.Array, ...] |
| changed | nvalchemiops/jax/neighbors/cluster_tile.py:build_cluster_tile_list | (positions: jax.Array, cutoff: float, cell: jax.Array, *, rebuild_flags: jax.Array \| None=None, num_tiles: jax.Array \| None=None, tile_row_group: jax.Array \| None=None, tile_col_group: jax.Array \| None=None) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array] | (positions: jax.Array, cutoff: float, cell: jax.Array, *, max_tiles_per_group: int \| None=None, rebuild_flags: jax.Array \| None=None, num_tiles: jax.Array \| None=None, tile_row_group: jax.Array \| None=None, tile_col_group: jax.Array \| None=None, _fixed_cell_geometry: tuple[jax.Array \| None, ...] \| None=None) -> tuple[jax.Array, ...] |
| changed | nvalchemiops/jax/neighbors/cluster_tile.py:cluster_tile_neighbor_list | (positions: jax.Array, cutoff: float, cell: jax.Array, max_neighbors: int \| None=None, fill_value: int \| None=None, format: str='matrix', max_pairs: int \| None=None, *, cutoff2: float \| None=None, rebuild_flags: jax.Array \| None=None, pair_offsets: jax.Array \| None=None, previous_pair_counts: jax.Array \| None=None, previous_neighbor_list: jax.Array \| None=None, previous_neighbor_list_shifts: jax.Array \| None=None, previous_num_tiles: jax.Array \| None=None, previous_tile_row_group: jax.Array \| None=None, previous_tile_col_group: jax.Array \| None=None, previous_neighbor_matrix: jax.Array \| None=None, previous_num_neighbors: jax.Array \| None=None, previous_neighbor_matrix_shifts: jax.Array \| None=None, previous_neighbor_matrix2: jax.Array \| None=None, previous_num_neighbors2: jax.Array \| None=None, previous_neighbor_matrix_shifts2: jax.Array \| None=None, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| None=None, pair_params: jax.Array \| None=None, neighbor_vectors: jax.Array \| None=None, neighbor_distances: jax.Array \| None=None, pair_energies: jax.Array \| None=None, pair_forces: jax.Array \| None=None) -> tuple[jax.Array, ...] | (positions: jax.Array, cutoff: float, cell: jax.Array, max_neighbors: int \| None=None, fill_value: int \| None=None, format: str='matrix', max_pairs: int \| None=None, *, max_tiles_per_group: int \| None=None, cutoff2: float \| None=None, rebuild_flags: jax.Array \| None=None, pair_offsets: jax.Array \| None=None, previous_pair_counts: jax.Array \| None=None, previous_neighbor_list: jax.Array \| None=None, previous_neighbor_list_shifts: jax.Array \| None=None, previous_num_tiles: jax.Array \| None=None, previous_tile_row_group: jax.Array \| None=None, previous_tile_col_group: jax.Array \| None=None, previous_neighbor_matrix: jax.Array \| None=None, previous_num_neighbors: jax.Array \| None=None, previous_neighbor_matrix_shifts: jax.Array \| None=None, previous_neighbor_matrix2: jax.Array \| None=None, previous_num_neighbors2: jax.Array \| None=None, previous_neighbor_matrix_shifts2: jax.Array \| None=None, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| None=None, pair_params: jax.Array \| None=None, neighbor_vectors: jax.Array \| None=None, neighbor_distances: jax.Array \| None=None, pair_energies: jax.Array \| None=None, pair_forces: jax.Array \| None=None, _return_status: bool=False, _fixed_cell_geometry: tuple[jax.Array \| None, ...] \| None=None) -> tuple[jax.Array, ...] |
| changed | nvalchemiops/jax/neighbors/cluster_tile.py:query_cluster_tile | (sorted_atom_index: jax.Array, sorted_pos_x: jax.Array, sorted_pos_y: jax.Array, sorted_pos_z: jax.Array, num_tiles: jax.Array, tile_row_group: jax.Array, tile_col_group: jax.Array, cell: jax.Array, cutoff: float, natom: int, max_neighbors: int, *, fill_value: int \| None=None, cutoff2: float \| None=None, neighbor_matrix: jax.Array \| None=None, num_neighbors: jax.Array \| None=None, neighbor_matrix_shifts: jax.Array \| None=None, neighbor_matrix2: jax.Array \| None=None, num_neighbors2: jax.Array \| None=None, neighbor_matrix_shifts2: jax.Array \| None=None, rebuild_flags: jax.Array \| None=None, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| None=None, pair_params: jax.Array \| None=None, neighbor_vectors: jax.Array \| None=None, neighbor_distances: jax.Array \| None=None, pair_energies: jax.Array \| None=None, pair_forces: jax.Array \| None=None) -> tuple[jax.Array, ...] | (sorted_atom_index: jax.Array, sorted_pos_x: jax.Array, sorted_pos_y: jax.Array, sorted_pos_z: jax.Array, num_tiles: jax.Array, tile_row_group: jax.Array, tile_col_group: jax.Array, cell: jax.Array, cutoff: float, natom: int, max_neighbors: int, *, fill_value: int \| None=None, cutoff2: float \| None=None, neighbor_matrix: jax.Array \| None=None, num_neighbors: jax.Array \| None=None, neighbor_matrix_shifts: jax.Array \| None=None, neighbor_matrix2: jax.Array \| None=None, num_neighbors2: jax.Array \| None=None, neighbor_matrix_shifts2: jax.Array \| None=None, rebuild_flags: jax.Array \| None=None, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| None=None, pair_params: jax.Array \| None=None, neighbor_vectors: jax.Array \| None=None, neighbor_distances: jax.Array \| None=None, pair_energies: jax.Array \| None=None, pair_forces: jax.Array \| None=None, _fixed_cell_geometry: tuple[jax.Array \| None, ...] \| None=None) -> tuple[jax.Array, ...] |
| changed | nvalchemiops/jax/neighbors/cluster_tile.py:query_cluster_tile_coo | (sorted_atom_index: jax.Array, sorted_pos_x: jax.Array, sorted_pos_y: jax.Array, sorted_pos_z: jax.Array, num_tiles: jax.Array, tile_row_group: jax.Array, tile_col_group: jax.Array, cell: jax.Array, cutoff: float, natom: int, max_pairs: int, *, rebuild_flags: jax.Array \| None=None, pair_offsets: jax.Array \| None=None, pair_counts: jax.Array \| None=None, neighbor_list: jax.Array \| None=None, neighbor_list_shifts: jax.Array \| None=None) -> tuple[jax.Array, ...] | (sorted_atom_index: jax.Array, sorted_pos_x: jax.Array, sorted_pos_y: jax.Array, sorted_pos_z: jax.Array, num_tiles: jax.Array, tile_row_group: jax.Array, tile_col_group: jax.Array, cell: jax.Array, cutoff: float, natom: int, max_pairs: int, *, rebuild_flags: jax.Array \| None=None, pair_offsets: jax.Array \| None=None, pair_counts: jax.Array \| None=None, neighbor_list: jax.Array \| None=None, neighbor_list_shifts: jax.Array \| None=None, _return_pair_counter: bool=False, _fixed_cell_geometry: tuple[jax.Array \| None, ...] \| None=None) -> tuple[jax.Array, ...] |
| changed | nvalchemiops/jax/neighbors/naive.py:naive_neighbor_list | (positions: jax.Array, cutoff: float, cell: jax.Array \| None=None, pbc: jax.Array \| None=None, max_neighbors: int \| None=None, half_fill: bool=False, fill_value: int \| None=None, return_neighbor_list: bool=False, neighbor_matrix: jax.Array \| None=None, neighbor_matrix_shifts: jax.Array \| None=None, num_neighbors: jax.Array \| None=None, shift_range_per_dimension: jax.Array \| None=None, num_shifts_per_system: jax.Array \| None=None, max_shifts_per_system: int \| None=None, rebuild_flags: jax.Array \| None=None, wrap_positions: bool=True, inv_cell_buffer: jax.Array \| None=None, positions_wrapped_buffer: jax.Array \| None=None, per_atom_cell_offsets_buffer: jax.Array \| None=None, strategy: str='auto', *, return_distances: bool=False, return_vectors: bool=False, neighbor_vectors: jax.Array \| None=None, neighbor_distances: jax.Array \| None=None, target_indices: jax.Array \| None=None, pair_fn=None, pair_params: jax.Array \| None=None, pair_energies: jax.Array \| None=None, pair_forces: jax.Array \| None=None, inv_cell: jax.Array \| None=None, positions_wrapped: jax.Array \| None=None, per_atom_cell_offsets: jax.Array \| None=None, graph_mode: Literal['none', 'warp']='none') -> tuple[jax.Array, jax.Array, jax.Array, jax.Array] \| tuple[jax.Array, jax.Array, jax.Array] \| tuple[jax.Array, jax.Array] | (positions: jax.Array, cutoff: float, cell: jax.Array \| None=None, pbc: jax.Array \| None=None, max_neighbors: int \| None=None, half_fill: bool=False, fill_value: int \| None=None, return_neighbor_list: bool=False, neighbor_matrix: jax.Array \| None=None, neighbor_matrix_shifts: jax.Array \| None=None, num_neighbors: jax.Array \| None=None, shift_range_per_dimension: jax.Array \| None=None, num_shifts_per_system: jax.Array \| None=None, max_shifts_per_system: int \| None=None, rebuild_flags: jax.Array \| None=None, wrap_positions: bool=True, inv_cell_buffer: jax.Array \| None=None, positions_wrapped_buffer: jax.Array \| None=None, per_atom_cell_offsets_buffer: jax.Array \| None=None, strategy: str='auto', *, coo_capacity: int \| None=None, return_distances: bool=False, return_vectors: bool=False, neighbor_vectors: jax.Array \| None=None, neighbor_distances: jax.Array \| None=None, target_indices: jax.Array \| None=None, pair_fn=None, pair_params: jax.Array \| None=None, pair_energies: jax.Array \| None=None, pair_forces: jax.Array \| None=None, _fixed_cell_geometry: tuple[jax.Array \| None, ...] \| None=None, inv_cell: jax.Array \| None=None, positions_wrapped: jax.Array \| None=None, per_atom_cell_offsets: jax.Array \| None=None, graph_mode: Literal['none', 'warp']='none') -> tuple[jax.Array, jax.Array, jax.Array, jax.Array] \| tuple[jax.Array, jax.Array, jax.Array] \| tuple[jax.Array, jax.Array] |
| changed | nvalchemiops/jax/neighbors/naive_dual_cutoff.py:naive_neighbor_list_dual_cutoff | (positions: jax.Array, cutoff1: float, cutoff2: float, pbc: jax.Array \| None=None, cell: jax.Array \| None=None, max_neighbors1: int \| None=None, max_neighbors2: int \| None=None, half_fill: bool=False, fill_value: int \| None=None, return_neighbor_list: bool=False, neighbor_matrix1: jax.Array \| None=None, neighbor_matrix2: jax.Array \| None=None, neighbor_matrix_shifts1: jax.Array \| None=None, neighbor_matrix_shifts2: jax.Array \| None=None, num_neighbors1: jax.Array \| None=None, num_neighbors2: jax.Array \| None=None, shift_range_per_dimension: jax.Array \| None=None, num_shifts_per_system: jax.Array \| None=None, max_shifts_per_system: int \| None=None, rebuild_flags: jax.Array \| None=None, wrap_positions: bool=True, positions_wrapped_buffer: jax.Array \| None=None, per_atom_cell_offsets_buffer: jax.Array \| None=None, inv_cell_buffer: jax.Array \| None=None) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array] \| tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array] \| tuple[jax.Array, jax.Array, jax.Array, jax.Array] | (positions: jax.Array, cutoff1: float, cutoff2: float, pbc: jax.Array \| None=None, cell: jax.Array \| None=None, max_neighbors1: int \| None=None, max_neighbors2: int \| None=None, half_fill: bool=False, fill_value: int \| None=None, return_neighbor_list: bool=False, neighbor_matrix1: jax.Array \| None=None, neighbor_matrix2: jax.Array \| None=None, neighbor_matrix_shifts1: jax.Array \| None=None, neighbor_matrix_shifts2: jax.Array \| None=None, num_neighbors1: jax.Array \| None=None, num_neighbors2: jax.Array \| None=None, shift_range_per_dimension: jax.Array \| None=None, num_shifts_per_system: jax.Array \| None=None, max_shifts_per_system: int \| None=None, rebuild_flags: jax.Array \| None=None, wrap_positions: bool=True, positions_wrapped_buffer: jax.Array \| None=None, per_atom_cell_offsets_buffer: jax.Array \| None=None, inv_cell_buffer: jax.Array \| None=None, coo_capacity: int \| tuple[int, int] \| None=None, *, _fixed_cell_geometry: tuple[jax.Array \| None, ...] \| None=None) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array] \| tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array] \| tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array] \| tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array] |
| changed | nvalchemiops/jax/neighbors/neighbor_utils.py:coo_pack_pair_geometry | (active_mask: jax.Array, distances: jax.Array \| None=None, vectors: jax.Array \| None=None) -> tuple[jax.Array \| None, jax.Array \| None] | (active_mask: jax.Array, distances: jax.Array \| None=None, vectors: jax.Array \| None=None, capacity: int \| None=None, plan: tuple[jax.Array, jax.Array, jax.Array] \| None=None) -> tuple[jax.Array \| None, jax.Array \| None] |
| added | nvalchemiops/jax/neighbors/neighbor_utils.py:get_fixed_capacity_neighbor_list_from_neighbor_matrix | — | (neighbor_matrix: jax.Array, num_neighbors: jax.Array, capacity: int, neighbor_shift_matrix: jax.Array \| None=None, fill_value: int=-1) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array] \| tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array] |
| added | nvalchemiops/jax/neighbors/prepared_neighbor_list.py:NeighborListState | — | class() @ jax.tree_util.register_pytree_node_class |
| added | nvalchemiops/jax/neighbors/prepared_neighbor_list.py:NeighborListState.__init__ | — | (self, spec: _Spec \| None=None, leaves: tuple[Any, ...] \| None=None, *, token: object \| None=None) -> None |
| added | nvalchemiops/jax/neighbors/prepared_neighbor_list.py:NeighborListState.compilation_blocker | — | (self) -> str \| None @ property |
| added | nvalchemiops/jax/neighbors/prepared_neighbor_list.py:NeighborListState.coo_layout | — | (self) -> str \| None @ property |
| added | nvalchemiops/jax/neighbors/prepared_neighbor_list.py:NeighborListState.cutoff | — | (self) -> float @ property |
| added | nvalchemiops/jax/neighbors/prepared_neighbor_list.py:NeighborListState.cutoff2 | — | (self) -> float \| None @ property |
| added | nvalchemiops/jax/neighbors/prepared_neighbor_list.py:NeighborListState.fill_value | — | (self) -> int @ property |
| added | nvalchemiops/jax/neighbors/prepared_neighbor_list.py:NeighborListState.fixed_cell | — | (self) -> bool @ property |
| added | nvalchemiops/jax/neighbors/prepared_neighbor_list.py:NeighborListState.format | — | (self) -> str @ property |
| added | nvalchemiops/jax/neighbors/prepared_neighbor_list.py:NeighborListState.half_fill | — | (self) -> bool @ property |
| added | nvalchemiops/jax/neighbors/prepared_neighbor_list.py:NeighborListState.initialized | — | (self) -> jax.Array @ property |
| added | nvalchemiops/jax/neighbors/prepared_neighbor_list.py:NeighborListState.is_batched | — | (self) -> bool @ property |
| added | nvalchemiops/jax/neighbors/prepared_neighbor_list.py:NeighborListState.method | — | (self) -> str @ property |
| added | nvalchemiops/jax/neighbors/prepared_neighbor_list.py:NeighborListState.num_atoms | — | (self) -> int @ property |
| added | nvalchemiops/jax/neighbors/prepared_neighbor_list.py:NeighborListState.num_systems | — | (self) -> int @ property |
| added | nvalchemiops/jax/neighbors/prepared_neighbor_list.py:NeighborListState.return_distances | — | (self) -> bool @ property |
| added | nvalchemiops/jax/neighbors/prepared_neighbor_list.py:NeighborListState.return_vectors | — | (self) -> bool @ property |
| added | nvalchemiops/jax/neighbors/prepared_neighbor_list.py:NeighborListState.selective | — | (self) -> bool @ property |
| added | nvalchemiops/jax/neighbors/prepared_neighbor_list.py:NeighborListState.span_margin | — | (self) -> float @ property |
| added | nvalchemiops/jax/neighbors/prepared_neighbor_list.py:NeighborListState.strategy | — | (self) -> str @ property |
| added | nvalchemiops/jax/neighbors/prepared_neighbor_list.py:NeighborListState.supports_compilation | — | (self) -> bool @ property |
| added | nvalchemiops/jax/neighbors/prepared_neighbor_list.py:NeighborListState.tree_flatten | — | (self) -> tuple[tuple[Any, ...], _Spec] |
| added | nvalchemiops/jax/neighbors/prepared_neighbor_list.py:NeighborListState.tree_unflatten | — | (cls, spec: _Spec, leaves: tuple[Any, ...]) -> NeighborListState @ classmethod |
| added | nvalchemiops/jax/neighbors/prepared_neighbor_list.py:NeighborListState.valid | — | (self) -> jax.Array @ property |
| added | nvalchemiops/jax/neighbors/prepared_neighbor_list.py:NeighborListState.wrap_positions | — | (self) -> bool @ property |
| added | nvalchemiops/jax/neighbors/prepared_neighbor_list.py:check_neighbor_list_state | — | (state: NeighborListState) -> None |
| added | nvalchemiops/jax/neighbors/prepared_neighbor_list.py:prepare_neighbor_list | — | (positions: jax.Array, cutoff: float, cell: jax.Array \| None=None, pbc: jax.Array \| None=None, batch_idx: jax.Array \| None=None, batch_ptr: jax.Array \| None=None, cutoff2: float \| None=None, half_fill: bool=False, fill_value: int \| None=None, return_neighbor_list: bool=False, method: str \| None=None, format: str \| None=None, wrap_positions: bool=True, *, max_neighbors: int \| None=None, max_neighbors1: int \| None=None, max_neighbors2: int \| None=None, atomic_density: float \| None=None, coo_capacity: int \| tuple[int, int] \| None=None, max_tiles_per_group: int \| None=None, return_vectors: bool=False, return_distances: bool=False, selective: bool=False, pair_fn: Any=None, span_margin: float=0.0, target_indices: jax.Array \| None=None, strategy: str='auto', atom_centric_path: str='auto', fixed_cell: bool=False, **kwargs: Any) -> NeighborListState |
| added | nvalchemiops/jax/types.py:normalize_float_dtype | — | (dtype, origin: str='input') -> unknown |
| changed | nvalchemiops/math/__init__.py:exports | __all__ = ['wp_safe_divide', 'wp_exp_kernel', 'wpdivmod', 'wp_erfc', 'spherical_harmonic_00', 'spherical_harmonic_1m1', 'spherical_harmonic_10', 'spherical_harmonic_1p1', 'spherical_harmonic_2m2', 'spherical_harmonic_2m1', 'spherical_harmonic_20', 'spherical_harmonic_2p1', 'spherical_harmonic_2p2', 'eval_spherical_harmonics_l0', 'eval_spherical_harmonics_l1', 'eval_spherical_harmonics_l2', 'eval_all_spherical_harmonics', 'spherical_harmonic_00_gradient', 'spherical_harmonic_1m1_gradient', 'spherical_harmonic_10_gradient', 'spherical_harmonic_1p1_gradient', 'spherical_harmonic_2m2_gradient', 'spherical_harmonic_2m1_gradient', 'spherical_harmonic_20_gradient', 'spherical_harmonic_2p1_gradient', 'spherical_harmonic_2p2_gradient', 'gto_normalization', 'gto_gaussian_factor', 'gto_density_l0', 'gto_density_l1', 'gto_density_l2', 'gto_density_all', 'gto_density_l0_gradient', 'gto_fourier_l0', 'gto_fourier_l1_real', 'gto_fourier_l1_imag', 'gto_fourier_l2_real', 'gto_integral_l0', 'gto_self_overlap', 'regular_solid_harmonic_l0', 'regular_solid_harmonic_l1', 'irregular_solid_harmonic_l0', 'irregular_solid_harmonic_l1', 'bspline_weight', 'bspline_derivative', 'bspline_weight_3d', 'bspline_weight_gradient_3d', 'compute_fractional_coords', 'bspline_grid_offset', 'wrap_grid_index', 'spline_spread', 'spline_gather', 'spline_gather_vec3', 'spline_gather_gradient', 'batch_spline_spread', 'batch_spline_gather', 'batch_spline_gather_vec3', 'batch_spline_gather_gradient'] ; from nvalchemiops.math.gto import gto_density_all, gto_density_l0, gto_density_l0_gradient, gto_density_l1, gto_density_l2, gto_fourier_l0, gto_fourier_l1_imag, gto_fourier_l1_real, gto_fourier_l2_real, gto_gaussian_factor, gto_integral_l0, gto_normalization, gto_self_overlap ; from nvalchemiops.math.math import wp_erfc, wp_exp_kernel, wp_safe_divide, wpdivmod ; from nvalchemiops.math.solid_harmonics import irregular_solid_harmonic_l0, irregular_solid_harmonic_l1, regular_solid_harmonic_l0, regular_solid_harmonic_l1 ; from nvalchemiops.math.spherical_harmonics import eval_all_spherical_harmonics, eval_spherical_harmonics_l0, eval_spherical_harmonics_l1, eval_spherical_harmonics_l2, spherical_harmonic_00, spherical_harmonic_00_gradient, spherical_harmonic_1m1, spherical_harmonic_1m1_gradient, spherical_harmonic_1p1, spherical_harmonic_1p1_gradient, spherical_harmonic_2m1, spherical_harmonic_2m1_gradient, spherical_harmonic_2m2, spherical_harmonic_2m2_gradient, spherical_harmonic_2p1, spherical_harmonic_2p1_gradient, spherical_harmonic_2p2, spherical_harmonic_2p2_gradient, spherical_harmonic_10, spherical_harmonic_10_gradient, spherical_harmonic_20, spherical_harmonic_20_gradient ; from nvalchemiops.math.spline import batch_spline_gather, batch_spline_gather_gradient, batch_spline_gather_vec3, batch_spline_spread, bspline_derivative, bspline_grid_offset, bspline_weight, bspline_weight_3d, bspline_weight_gradient_3d, compute_fractional_coords, spline_gather, spline_gather_gradient, spline_gather_vec3, spline_spread, wrap_grid_index | __all__ = ['wp_safe_divide', 'wp_exp_kernel', 'wpdivmod', 'wp_erfc', 'wp_erfc_f32_fast', 'wp_erfc_input_precision', 'wp_logistic', 'spherical_harmonic_00', 'spherical_harmonic_1m1', 'spherical_harmonic_10', 'spherical_harmonic_1p1', 'spherical_harmonic_2m2', 'spherical_harmonic_2m1', 'spherical_harmonic_20', 'spherical_harmonic_2p1', 'spherical_harmonic_2p2', 'eval_spherical_harmonics_l0', 'eval_spherical_harmonics_l1', 'eval_spherical_harmonics_l2', 'eval_all_spherical_harmonics', 'spherical_harmonic_00_gradient', 'spherical_harmonic_1m1_gradient', 'spherical_harmonic_10_gradient', 'spherical_harmonic_1p1_gradient', 'spherical_harmonic_2m2_gradient', 'spherical_harmonic_2m1_gradient', 'spherical_harmonic_20_gradient', 'spherical_harmonic_2p1_gradient', 'spherical_harmonic_2p2_gradient', 'gto_normalization', 'gto_gaussian_factor', 'gto_density_l0', 'gto_density_l1', 'gto_density_l2', 'gto_density_all', 'gto_density_l0_gradient', 'gto_fourier_l0', 'gto_fourier_l1_real', 'gto_fourier_l1_imag', 'gto_fourier_l2_real', 'gto_integral_l0', 'gto_self_overlap', 'regular_solid_harmonic_l0', 'regular_solid_harmonic_l1', 'irregular_solid_harmonic_l0', 'irregular_solid_harmonic_l1', 'bspline_weight', 'bspline_derivative', 'bspline_weight_3d', 'bspline_weight_gradient_3d', 'compute_fractional_coords', 'bspline_grid_offset', 'wrap_grid_index', 'spline_spread', 'spline_gather', 'spline_gather_vec3', 'spline_gather_gradient', 'batch_spline_spread', 'batch_spline_gather', 'batch_spline_gather_vec3', 'batch_spline_gather_gradient'] ; from nvalchemiops.math.gto import gto_density_all, gto_density_l0, gto_density_l0_gradient, gto_density_l1, gto_density_l2, gto_fourier_l0, gto_fourier_l1_imag, gto_fourier_l1_real, gto_fourier_l2_real, gto_gaussian_factor, gto_integral_l0, gto_normalization, gto_self_overlap ; from nvalchemiops.math.math import wp_erfc, wp_erfc_f32_fast, wp_erfc_input_precision, wp_exp_kernel, wp_logistic, wp_safe_divide, wpdivmod ; from nvalchemiops.math.solid_harmonics import irregular_solid_harmonic_l0, irregular_solid_harmonic_l1, regular_solid_harmonic_l0, regular_solid_harmonic_l1 ; from nvalchemiops.math.spherical_harmonics import eval_all_spherical_harmonics, eval_spherical_harmonics_l0, eval_spherical_harmonics_l1, eval_spherical_harmonics_l2, spherical_harmonic_00, spherical_harmonic_00_gradient, spherical_harmonic_1m1, spherical_harmonic_1m1_gradient, spherical_harmonic_1p1, spherical_harmonic_1p1_gradient, spherical_harmonic_2m1, spherical_harmonic_2m1_gradient, spherical_harmonic_2m2, spherical_harmonic_2m2_gradient, spherical_harmonic_2p1, spherical_harmonic_2p1_gradient, spherical_harmonic_2p2, spherical_harmonic_2p2_gradient, spherical_harmonic_10, spherical_harmonic_10_gradient, spherical_harmonic_20, spherical_harmonic_20_gradient ; from nvalchemiops.math.spline import batch_spline_gather, batch_spline_gather_gradient, batch_spline_gather_vec3, batch_spline_spread, bspline_derivative, bspline_grid_offset, bspline_weight, bspline_weight_3d, bspline_weight_gradient_3d, compute_fractional_coords, spline_gather, spline_gather_gradient, spline_gather_vec3, spline_spread, wrap_grid_index |
| added | nvalchemiops/math/math.py:wp_erfc_f32_fast | — | (x: Any) -> Any @ wp.func |
| added | nvalchemiops/math/math.py:wp_erfc_input_precision | — | (x: wp.float64) -> wp.float64 @ wp.func ; (x: wp.float32) -> wp.float32 @ wp.func |
| added | nvalchemiops/math/math.py:wp_logistic | — | (x: Any) -> Any @ wp.func |
| removed | nvalchemiops/math/spherical_harmonics.py:eval_spherical_harmonics_gradient_pytorch | (positions, L_max: int=2, device=None) -> unknown | — |
| removed | nvalchemiops/math/spherical_harmonics.py:eval_spherical_harmonics_pytorch | (positions, L_max: int=2, device=None) -> unknown | — |
| added | nvalchemiops/math/spline.py:batch_spline_gather_channels | — | (positions: wp.array, group_idx: wp.array, cell_inv_t: wp.array, order: int, num_channels: int, mesh: wp.array, output: wp.array, wp_dtype: type, device: str \| None=None) -> None |
| added | nvalchemiops/math/spline.py:batch_spline_spread_channels | — | (positions: wp.array, values: wp.array, group_idx: wp.array, cell_inv_t: wp.array, order: int, num_channels: int, mesh: wp.array, wp_dtype: type, device: str \| None=None) -> None |
| changed | nvalchemiops/neighborlist/__init__.py:exports | __all__ = ['NeighborOverflowError', 'wp_naive_neighbor_matrix', 'wp_naive_neighbor_matrix_pbc', 'wp_build_cell_list', 'wp_query_cell_list', 'wp_batch_naive_neighbor_matrix', 'wp_batch_naive_neighbor_matrix_pbc', 'wp_batch_build_cell_list', 'wp_batch_query_cell_list', 'wp_naive_neighbor_matrix_dual_cutoff', 'wp_naive_neighbor_matrix_pbc_dual_cutoff', 'wp_batch_naive_neighbor_matrix_dual_cutoff', 'wp_batch_naive_neighbor_matrix_pbc_dual_cutoff', 'wp_check_cell_list_rebuild', 'wp_check_neighbor_list_rebuild', 'wp_compute_naive_num_shifts', 'estimate_max_neighbors', 'neighbor_list', 'naive', 'naive_dual_cutoff', 'cell_list', 'batch_cell_list', 'batch_naive', 'batch_naive_dual_cutoff', 'neighbor_utils', 'rebuild_detection'] ; from __future__ import annotations ; from nvalchemiops.neighbors import NeighborOverflowError, estimate_max_neighbors ; from nvalchemiops.neighbors import batch_build_cell_list as wp_batch_build_cell_list ; from nvalchemiops.neighbors import batch_naive_neighbor_matrix as wp_batch_naive_neighbor_matrix ; from nvalchemiops.neighbors import batch_naive_neighbor_matrix_dual_cutoff as wp_batch_naive_neighbor_matrix_dual_cutoff ; from nvalchemiops.neighbors import batch_naive_neighbor_matrix_pbc as wp_batch_naive_neighbor_matrix_pbc ; from nvalchemiops.neighbors import batch_naive_neighbor_matrix_pbc_dual_cutoff as wp_batch_naive_neighbor_matrix_pbc_dual_cutoff ; from nvalchemiops.neighbors import batch_query_cell_list as wp_batch_query_cell_list ; from nvalchemiops.neighbors import build_cell_list as wp_build_cell_list ; from nvalchemiops.neighbors import check_cell_list_rebuild as wp_check_cell_list_rebuild ; from nvalchemiops.neighbors import check_neighbor_list_rebuild as wp_check_neighbor_list_rebuild ; from nvalchemiops.neighbors import compute_naive_num_shifts as wp_compute_naive_num_shifts ; from nvalchemiops.neighbors import naive_neighbor_matrix as wp_naive_neighbor_matrix ; from nvalchemiops.neighbors import naive_neighbor_matrix_dual_cutoff as wp_naive_neighbor_matrix_dual_cutoff ; from nvalchemiops.neighbors import naive_neighbor_matrix_pbc as wp_naive_neighbor_matrix_pbc ; from nvalchemiops.neighbors import naive_neighbor_matrix_pbc_dual_cutoff as wp_naive_neighbor_matrix_pbc_dual_cutoff ; from nvalchemiops.neighbors import query_cell_list as wp_query_cell_list ; import sys ; import warnings | __all__ = ['NeighborOverflowError', 'TileBufferOverflow', 'wp_naive_neighbor_matrix', 'wp_naive_neighbor_matrix_pbc', 'wp_build_cell_list', 'wp_query_cell_list', 'wp_batch_naive_neighbor_matrix', 'wp_batch_naive_neighbor_matrix_pbc', 'wp_batch_build_cell_list', 'wp_batch_query_cell_list', 'wp_naive_neighbor_matrix_dual_cutoff', 'wp_naive_neighbor_matrix_pbc_dual_cutoff', 'wp_batch_naive_neighbor_matrix_dual_cutoff', 'wp_batch_naive_neighbor_matrix_pbc_dual_cutoff', 'wp_check_cell_list_rebuild', 'wp_check_neighbor_list_rebuild', 'wp_compute_naive_num_shifts', 'estimate_max_neighbors', 'neighbor_list', 'naive', 'naive_dual_cutoff', 'cell_list', 'batch_cell_list', 'batch_naive', 'batch_naive_dual_cutoff', 'neighbor_utils', 'rebuild_detection'] ; from __future__ import annotations ; from nvalchemiops.neighbors import NeighborOverflowError, TileBufferOverflow, estimate_max_neighbors ; from nvalchemiops.neighbors import batch_build_cell_list as wp_batch_build_cell_list ; from nvalchemiops.neighbors import batch_naive_neighbor_matrix as wp_batch_naive_neighbor_matrix ; from nvalchemiops.neighbors import batch_naive_neighbor_matrix_dual_cutoff as wp_batch_naive_neighbor_matrix_dual_cutoff ; from nvalchemiops.neighbors import batch_naive_neighbor_matrix_pbc as wp_batch_naive_neighbor_matrix_pbc ; from nvalchemiops.neighbors import batch_naive_neighbor_matrix_pbc_dual_cutoff as wp_batch_naive_neighbor_matrix_pbc_dual_cutoff ; from nvalchemiops.neighbors import batch_query_cell_list as wp_batch_query_cell_list ; from nvalchemiops.neighbors import build_cell_list as wp_build_cell_list ; from nvalchemiops.neighbors import check_cell_list_rebuild as wp_check_cell_list_rebuild ; from nvalchemiops.neighbors import check_neighbor_list_rebuild as wp_check_neighbor_list_rebuild ; from nvalchemiops.neighbors import compute_naive_num_shifts as wp_compute_naive_num_shifts ; from nvalchemiops.neighbors import naive_neighbor_matrix as wp_naive_neighbor_matrix ; from nvalchemiops.neighbors import naive_neighbor_matrix_dual_cutoff as wp_naive_neighbor_matrix_dual_cutoff ; from nvalchemiops.neighbors import naive_neighbor_matrix_pbc as wp_naive_neighbor_matrix_pbc ; from nvalchemiops.neighbors import naive_neighbor_matrix_pbc_dual_cutoff as wp_naive_neighbor_matrix_pbc_dual_cutoff ; from nvalchemiops.neighbors import query_cell_list as wp_query_cell_list ; import sys ; import warnings |
| changed | nvalchemiops/neighbors/__init__.py:exports | __all__ = ['NeighborOverflowError', 'TILE_GROUP_SIZE', 'batch_build_cell_list', 'batch_build_cluster_tile_list', 'batch_naive_neighbor_matrix', 'batch_naive_neighbor_matrix_dual_cutoff', 'batch_naive_neighbor_matrix_pbc', 'batch_naive_neighbor_matrix_pbc_dual_cutoff', 'batch_query_cell_list', 'batch_query_cluster_tile', 'batch_query_cluster_tile_coo', 'build_cell_list', 'build_cluster_tile_list', 'check_batch_cell_list_rebuild', 'check_batch_neighbor_list_rebuild', 'check_cell_list_rebuild', 'check_neighbor_list_rebuild', 'compute_naive_num_shifts', 'estimate_max_neighbors', 'naive_neighbor_matrix', 'naive_neighbor_matrix_dual_cutoff', 'naive_neighbor_matrix_pbc', 'naive_neighbor_matrix_pbc_dual_cutoff', 'neighbor_list', 'query_cell_list', 'query_cluster_tile', 'query_cluster_tile_coo', 'estimate_neighbor_list_costs', 'suggest_neighbor_list_method', 'zero_array'] ; from __future__ import annotations ; from nvalchemiops.neighbors.base_dispatch import estimate_neighbor_list_costs, suggest_neighbor_list_method ; from nvalchemiops.neighbors.cell_list import batch_build_cell_list, batch_query_cell_list, build_cell_list, query_cell_list ; from nvalchemiops.neighbors.cluster_tile import TILE_GROUP_SIZE, batch_build_cluster_tile_list, batch_query_cluster_tile, batch_query_cluster_tile_coo, build_cluster_tile_list, query_cluster_tile, query_cluster_tile_coo ; from nvalchemiops.neighbors.naive import batch_naive_neighbor_matrix, batch_naive_neighbor_matrix_dual_cutoff, batch_naive_neighbor_matrix_pbc, batch_naive_neighbor_matrix_pbc_dual_cutoff, naive_neighbor_matrix, naive_neighbor_matrix_dual_cutoff, naive_neighbor_matrix_pbc, naive_neighbor_matrix_pbc_dual_cutoff ; from nvalchemiops.neighbors.neighbor_utils import NeighborOverflowError, compute_naive_num_shifts, estimate_max_neighbors, zero_array ; from nvalchemiops.neighbors.rebuild import check_batch_cell_list_rebuild, check_batch_neighbor_list_rebuild, check_cell_list_rebuild, check_neighbor_list_rebuild ; import importlib ; import warnings | __all__ = ['NeighborOverflowError', 'TileBufferOverflow', 'TILE_GROUP_SIZE', 'batch_build_cell_list', 'batch_build_cluster_tile_list', 'batch_naive_neighbor_matrix', 'batch_naive_neighbor_matrix_dual_cutoff', 'batch_naive_neighbor_matrix_pbc', 'batch_naive_neighbor_matrix_pbc_dual_cutoff', 'batch_query_cell_list', 'batch_query_cluster_tile', 'batch_query_cluster_tile_coo', 'build_cell_list', 'build_cluster_tile_list', 'check_batch_cell_list_rebuild', 'check_batch_neighbor_list_rebuild', 'check_cell_list_rebuild', 'check_neighbor_list_rebuild', 'compute_naive_num_shifts', 'estimate_max_neighbors', 'naive_neighbor_matrix', 'naive_neighbor_matrix_dual_cutoff', 'naive_neighbor_matrix_pbc', 'naive_neighbor_matrix_pbc_dual_cutoff', 'neighbor_list', 'query_cell_list', 'query_cluster_tile', 'query_cluster_tile_coo', 'estimate_neighbor_list_costs', 'suggest_neighbor_list_method', 'zero_array'] ; from __future__ import annotations ; from nvalchemiops.neighbors.base_dispatch import estimate_neighbor_list_costs, suggest_neighbor_list_method ; from nvalchemiops.neighbors.cell_list import batch_build_cell_list, batch_query_cell_list, build_cell_list, query_cell_list ; from nvalchemiops.neighbors.cluster_tile import TILE_GROUP_SIZE, batch_build_cluster_tile_list, batch_query_cluster_tile, batch_query_cluster_tile_coo, build_cluster_tile_list, query_cluster_tile, query_cluster_tile_coo ; from nvalchemiops.neighbors.naive import batch_naive_neighbor_matrix, batch_naive_neighbor_matrix_dual_cutoff, batch_naive_neighbor_matrix_pbc, batch_naive_neighbor_matrix_pbc_dual_cutoff, naive_neighbor_matrix, naive_neighbor_matrix_dual_cutoff, naive_neighbor_matrix_pbc, naive_neighbor_matrix_pbc_dual_cutoff ; from nvalchemiops.neighbors.neighbor_utils import NeighborOverflowError, TileBufferOverflow, compute_naive_num_shifts, estimate_max_neighbors, zero_array ; from nvalchemiops.neighbors.rebuild import check_batch_cell_list_rebuild, check_batch_neighbor_list_rebuild, check_cell_list_rebuild, check_neighbor_list_rebuild ; import importlib ; import warnings |
| added | nvalchemiops/neighbors/base_dispatch.py:fp64_naive_penalty | — | (device: wp.Device) -> float |
| changed | nvalchemiops/neighbors/cell_list/kernels.py:get_build_cell_list_kernel | (stage: Literal['estimate_sizes', 'construct_bin_size', 'count_atoms', 'bin_atoms'], wp_dtype: type, *, batched: bool=False, min_cells_per_dimension: int=4) -> wp.Kernel | (stage: Literal['estimate_sizes', 'construct_bin_size', 'count_atoms', 'bin_atoms'], wp_dtype: type, *, batched: bool=False, min_cells_per_dimension: int=_DEFAULT_MIN_CELLS_PER_DIMENSION, fixed_cell: bool=False) -> wp.Kernel |
| changed | nvalchemiops/neighbors/cell_list/launchers.py:batch_build_cell_list | (positions: wp.array, cell: wp.array, pbc: wp.array, cutoff: float, batch_idx: wp.array, cells_per_dimension: wp.array, cell_offsets: wp.array, cells_per_system: wp.array, atom_periodic_shifts: wp.array, atom_to_cell_mapping: wp.array, atoms_per_cell_count: wp.array, cell_atom_start_indices: wp.array, cell_atom_list: wp.array, wp_dtype: type, device: str, min_cells_per_dimension: int=4) -> None | (positions: wp.array, cell: wp.array, pbc: wp.array, cutoff: float, batch_idx: wp.array, cells_per_dimension: wp.array, cell_offsets: wp.array, cells_per_system: wp.array, atom_periodic_shifts: wp.array, atom_to_cell_mapping: wp.array, atoms_per_cell_count: wp.array, cell_atom_start_indices: wp.array, cell_atom_list: wp.array, wp_dtype: type, device: str, min_cells_per_dimension: int=4, grid_is_prepared: bool=False, cells_per_system_is_prepared: bool=False, *, fixed_cell: bool=False, inv_cell: wp.array \| None=None) -> None |
| changed | nvalchemiops/neighbors/cell_list/launchers.py:batch_query_cell_list_pair_centric_sorted | (positions: wp.array, cell: wp.array, pbc: wp.array, cutoff: float, cells_per_dimension: wp.array, neighbor_search_radius: wp.array, cell_offsets: wp.array, cells_per_system: wp.array, atom_periodic_shifts: wp.array, atoms_per_cell_count: wp.array, cell_atom_start_indices: wp.array, cell_atom_list: wp.array, sorted_positions: wp.array, sorted_atom_periodic_shifts: wp.array, cell_to_system: wp.array, neighbor_matrix: wp.array, neighbor_matrix_shifts: wp.array, num_neighbors: wp.array, wp_dtype: type, device: str, total_cells: int, n_outer: int, R_max: tuple[int, int, int], half_fill: bool=True, block_dim: int=64, *, target_indices: wp.array \| None=None, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| None=None, pair_params: wp.array \| None=None, neighbor_vectors: wp.array \| None=None, neighbor_distances: wp.array \| None=None, pair_energies: wp.array \| None=None, pair_forces: wp.array \| None=None, rebuild_flags: wp.array \| None=None, target_row_lookup: wp.array \| None=None, max_launch_size: int=PAIR_CENTRIC_MAX_LINEAR_LAUNCH) -> None | (positions: wp.array, cell: wp.array, pbc: wp.array, cutoff: float, cells_per_dimension: wp.array, neighbor_search_radius: wp.array, cell_offsets: wp.array, cells_per_system: wp.array, atom_periodic_shifts: wp.array, atoms_per_cell_count: wp.array, cell_atom_start_indices: wp.array, cell_atom_list: wp.array, sorted_positions: wp.array, sorted_atom_periodic_shifts: wp.array, cell_to_system: wp.array, neighbor_matrix: wp.array, neighbor_matrix_shifts: wp.array, num_neighbors: wp.array, wp_dtype: type, device: str, total_cells: int, n_outer: int, R_max: tuple[int, int, int], half_fill: bool=True, block_dim: int=_PAIR_CENTRIC_BLOCK_DIM, *, target_indices: wp.array \| None=None, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| None=None, pair_params: wp.array \| None=None, neighbor_vectors: wp.array \| None=None, neighbor_distances: wp.array \| None=None, pair_energies: wp.array \| None=None, pair_forces: wp.array \| None=None, rebuild_flags: wp.array \| None=None, target_row_lookup: wp.array \| None=None, max_launch_size: int=PAIR_CENTRIC_MAX_LINEAR_LAUNCH) -> None |
| changed | nvalchemiops/neighbors/cell_list/launchers.py:build_cell_list | (positions: wp.array, cell: wp.array, pbc: wp.array, cutoff: float, cells_per_dimension: wp.array, atom_periodic_shifts: wp.array, atom_to_cell_mapping: wp.array, atoms_per_cell_count: wp.array, cell_atom_start_indices: wp.array, cell_atom_list: wp.array, wp_dtype: type, device: str, min_cells_per_dimension: int=4) -> None | (positions: wp.array, cell: wp.array, pbc: wp.array, cutoff: float, cells_per_dimension: wp.array, atom_periodic_shifts: wp.array, atom_to_cell_mapping: wp.array, atoms_per_cell_count: wp.array, cell_atom_start_indices: wp.array, cell_atom_list: wp.array, wp_dtype: type, device: str, min_cells_per_dimension: int=4, grid_is_prepared: bool=False, *, fixed_cell: bool=False, inv_cell: wp.array \| None=None) -> None |
| changed | nvalchemiops/neighbors/cluster_tile/kernels.py:get_batch_query_cluster_tile_coo_kernel | (*, tile_segmented: bool=False, coo_segmented: bool=False, selective: bool=False, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| None=None) -> wp.Kernel | (*, tile_segmented: bool=False, coo_segmented: bool=False, selective: bool=False, fixed_cell: bool=False, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| None=None) -> wp.Kernel |
| changed | nvalchemiops/neighbors/cluster_tile/kernels.py:get_batch_query_cluster_tile_kernel | (*, tile_segmented: bool=False, selective: bool=False, dual_cutoff: bool=False, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| None=None) -> wp.Kernel | (*, tile_segmented: bool=False, selective: bool=False, fixed_cell: bool=False, dual_cutoff: bool=False, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| None=None) -> wp.Kernel |
| added | nvalchemiops/neighbors/cluster_tile/kernels.py:get_prepare_cluster_tile_geometry_kernel | — | () -> wp.Kernel |
| changed | nvalchemiops/neighbors/cluster_tile/kernels.py:get_query_cluster_tile_coo_kernel | (*, tile_segmented: bool=False, coo_segmented: bool=False, selective: bool=False, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| None=None) -> wp.Kernel | (*, tile_segmented: bool=False, coo_segmented: bool=False, selective: bool=False, fixed_cell: bool=False, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| None=None) -> wp.Kernel |
| changed | nvalchemiops/neighbors/cluster_tile/kernels.py:get_query_cluster_tile_kernel | (*, tile_segmented: bool=False, selective: bool=False, dual_cutoff: bool=False, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| None=None) -> wp.Kernel | (*, tile_segmented: bool=False, selective: bool=False, fixed_cell: bool=False, dual_cutoff: bool=False, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| None=None) -> wp.Kernel |
| changed | nvalchemiops/neighbors/cluster_tile/launchers.py:batch_build_cluster_tile_list | (sorted_pos_x: wp.array, sorted_pos_y: wp.array, sorted_pos_z: wp.array, group_system: wp.array, group_ptr: wp.array, cell_batch: wp.array, inv_cell_batch: wp.array, cutoff: float, num_tiles: wp.array, tile_row_group: wp.array, tile_col_group: wp.array, tile_system: wp.array, wp_dtype: type, device: str, *, group_ctr_x_buffer: wp.array \| None=None, group_ctr_y_buffer: wp.array \| None=None, group_ctr_z_buffer: wp.array \| None=None, group_ext_x_buffer: wp.array \| None=None, group_ext_y_buffer: wp.array \| None=None, group_ext_z_buffer: wp.array \| None=None, rebuild_flags: wp.array \| None=None, tile_offsets: wp.array \| None=None, tile_counts: wp.array \| None=None) -> None | (sorted_pos_x: wp.array, sorted_pos_y: wp.array, sorted_pos_z: wp.array, group_system: wp.array, group_ptr: wp.array, cell_batch: wp.array, inv_cell_batch: wp.array, cutoff: float, num_tiles: wp.array, tile_row_group: wp.array, tile_col_group: wp.array, tile_system: wp.array, wp_dtype: type, device: str, *, group_ctr_x_buffer: wp.array \| None=None, group_ctr_y_buffer: wp.array \| None=None, group_ctr_z_buffer: wp.array \| None=None, group_ext_x_buffer: wp.array \| None=None, group_ext_y_buffer: wp.array \| None=None, group_ext_z_buffer: wp.array \| None=None, rebuild_flags: wp.array \| None=None, tile_offsets: wp.array \| None=None, tile_counts: wp.array \| None=None, fixed_cell: bool=False, qr: wp.array \| None=None, axis_aligned: wp.array \| None=None, fractional_rounding_certified: wp.array \| None=None, qr_height_certified: wp.array \| None=None, bbox_cutoff_bounds: wp.array \| None=None) -> None |
| changed | nvalchemiops/neighbors/cluster_tile/launchers.py:batch_query_cluster_tile | (sorted_atom_index: wp.array, sorted_pos_x: wp.array, sorted_pos_y: wp.array, sorted_pos_z: wp.array, cell_batch: wp.array, inv_cell_batch: wp.array, num_tiles: wp.array, tile_row_group: wp.array, tile_col_group: wp.array, tile_system: wp.array, cutoff: float, natom: int, neighbor_matrix: wp.array, num_neighbors: wp.array, neighbor_matrix_shifts: wp.array, wp_dtype: type, device: str, *, n_tiles: int \| None=None, cutoff2: float \| None=None, neighbor_matrix2: wp.array \| None=None, num_neighbors2: wp.array \| None=None, neighbor_matrix_shifts2: wp.array \| None=None, rebuild_flags: wp.array \| None=None, tile_offsets: wp.array \| None=None, tile_counts: wp.array \| None=None, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| None=None, pair_params: wp.array \| None=None, neighbor_vectors: wp.array \| None=None, neighbor_distances: wp.array \| None=None, pair_energies: wp.array \| None=None, pair_forces: wp.array \| None=None) -> None | (sorted_atom_index: wp.array, sorted_pos_x: wp.array, sorted_pos_y: wp.array, sorted_pos_z: wp.array, cell_batch: wp.array, inv_cell_batch: wp.array, num_tiles: wp.array, tile_row_group: wp.array, tile_col_group: wp.array, tile_system: wp.array, cutoff: float, natom: int, neighbor_matrix: wp.array, num_neighbors: wp.array, neighbor_matrix_shifts: wp.array, wp_dtype: type, device: str, *, n_tiles: int \| None=None, cutoff2: float \| None=None, neighbor_matrix2: wp.array \| None=None, num_neighbors2: wp.array \| None=None, neighbor_matrix_shifts2: wp.array \| None=None, rebuild_flags: wp.array \| None=None, tile_offsets: wp.array \| None=None, tile_counts: wp.array \| None=None, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| None=None, pair_params: wp.array \| None=None, neighbor_vectors: wp.array \| None=None, neighbor_distances: wp.array \| None=None, pair_energies: wp.array \| None=None, pair_forces: wp.array \| None=None, fixed_cell: bool=False, qr: wp.array \| None=None, axis_aligned: wp.array \| None=None, fractional_rounding_certified: wp.array \| None=None, qr_height_certified: wp.array \| None=None, bbox_cutoff_bounds: wp.array \| None=None) -> None |
| changed | nvalchemiops/neighbors/cluster_tile/launchers.py:batch_query_cluster_tile_coo | (sorted_atom_index: wp.array, sorted_pos_x: wp.array, sorted_pos_y: wp.array, sorted_pos_z: wp.array, cell_batch: wp.array, inv_cell_batch: wp.array, num_tiles: wp.array, tile_row_group: wp.array, tile_col_group: wp.array, tile_system: wp.array, cutoff: float, natom: int, max_pairs: int, pair_counter: wp.array, coo_list: wp.array, coo_shifts: wp.array, wp_dtype: type, device: str, *, n_tiles: int \| None=None, rebuild_flags: wp.array \| None=None, tile_offsets: wp.array \| None=None, tile_counts: wp.array \| None=None, pair_offsets: wp.array \| None=None, pair_counts: wp.array \| None=None, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| None=None, pair_params: wp.array \| None=None, neighbor_vectors: wp.array \| None=None, neighbor_distances: wp.array \| None=None, pair_energies: wp.array \| None=None, pair_forces: wp.array \| None=None) -> None | (sorted_atom_index: wp.array, sorted_pos_x: wp.array, sorted_pos_y: wp.array, sorted_pos_z: wp.array, cell_batch: wp.array, inv_cell_batch: wp.array, num_tiles: wp.array, tile_row_group: wp.array, tile_col_group: wp.array, tile_system: wp.array, cutoff: float, natom: int, max_pairs: int, pair_counter: wp.array, coo_list: wp.array, coo_shifts: wp.array, wp_dtype: type, device: str, *, n_tiles: int \| None=None, rebuild_flags: wp.array \| None=None, tile_offsets: wp.array \| None=None, tile_counts: wp.array \| None=None, pair_offsets: wp.array \| None=None, pair_counts: wp.array \| None=None, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| None=None, pair_params: wp.array \| None=None, neighbor_vectors: wp.array \| None=None, neighbor_distances: wp.array \| None=None, pair_energies: wp.array \| None=None, pair_forces: wp.array \| None=None, fixed_cell: bool=False, qr: wp.array \| None=None, axis_aligned: wp.array \| None=None, fractional_rounding_certified: wp.array \| None=None, qr_height_certified: wp.array \| None=None, bbox_cutoff_bounds: wp.array \| None=None) -> None |
| changed | nvalchemiops/neighbors/cluster_tile/launchers.py:build_cluster_tile_list | (sorted_pos_x: wp.array, sorted_pos_y: wp.array, sorted_pos_z: wp.array, cell: wp.array, inv_cell: wp.array, cutoff: float, num_tiles: wp.array, tile_row_group: wp.array, tile_col_group: wp.array, wp_dtype: type, device: str, *, group_ctr_x_buffer: wp.array \| None=None, group_ctr_y_buffer: wp.array \| None=None, group_ctr_z_buffer: wp.array \| None=None, group_ext_x_buffer: wp.array \| None=None, group_ext_y_buffer: wp.array \| None=None, group_ext_z_buffer: wp.array \| None=None, rebuild_flags: wp.array \| None=None) -> None | (sorted_pos_x: wp.array, sorted_pos_y: wp.array, sorted_pos_z: wp.array, cell: wp.array, inv_cell: wp.array, cutoff: float, num_tiles: wp.array, tile_row_group: wp.array, tile_col_group: wp.array, wp_dtype: type, device: str, *, group_ctr_x_buffer: wp.array \| None=None, group_ctr_y_buffer: wp.array \| None=None, group_ctr_z_buffer: wp.array \| None=None, group_ext_x_buffer: wp.array \| None=None, group_ext_y_buffer: wp.array \| None=None, group_ext_z_buffer: wp.array \| None=None, rebuild_flags: wp.array \| None=None, fixed_cell: bool=False, qr: wp.array \| None=None, axis_aligned: wp.array \| None=None, fractional_rounding_certified: wp.array \| None=None, qr_height_certified: wp.array \| None=None, bbox_cutoff_bounds: wp.array \| None=None) -> None |
| added | nvalchemiops/neighbors/cluster_tile/launchers.py:prepare_cluster_tile_geometry | — | (cell: wp.array, inv_cell: wp.array, outer_cutoff_sq: float, qr: wp.array, axis_aligned: wp.array, fractional_rounding_certified: wp.array, qr_height_certified: wp.array, bbox_cutoff_bounds: wp.array, device: str) -> None |
| changed | nvalchemiops/neighbors/cluster_tile/launchers.py:query_cluster_tile | (sorted_atom_index: wp.array, sorted_pos_x: wp.array, sorted_pos_y: wp.array, sorted_pos_z: wp.array, num_tiles: wp.array, tile_row_group: wp.array, tile_col_group: wp.array, cell: wp.array, inv_cell: wp.array, cutoff: float, natom: int, neighbor_matrix: wp.array, num_neighbors: wp.array, neighbor_matrix_shifts: wp.array, wp_dtype: type, device: str, *, n_tiles: int \| None=None, cutoff2: float \| None=None, neighbor_matrix2: wp.array \| None=None, num_neighbors2: wp.array \| None=None, neighbor_matrix_shifts2: wp.array \| None=None, rebuild_flags: wp.array \| None=None, tile_offsets: wp.array \| None=None, tile_counts: wp.array \| None=None, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| None=None, pair_params: wp.array \| None=None, neighbor_vectors: wp.array \| None=None, neighbor_distances: wp.array \| None=None, pair_energies: wp.array \| None=None, pair_forces: wp.array \| None=None) -> None | (sorted_atom_index: wp.array, sorted_pos_x: wp.array, sorted_pos_y: wp.array, sorted_pos_z: wp.array, num_tiles: wp.array, tile_row_group: wp.array, tile_col_group: wp.array, cell: wp.array, inv_cell: wp.array, cutoff: float, natom: int, neighbor_matrix: wp.array, num_neighbors: wp.array, neighbor_matrix_shifts: wp.array, wp_dtype: type, device: str, *, n_tiles: int \| None=None, cutoff2: float \| None=None, neighbor_matrix2: wp.array \| None=None, num_neighbors2: wp.array \| None=None, neighbor_matrix_shifts2: wp.array \| None=None, rebuild_flags: wp.array \| None=None, tile_offsets: wp.array \| None=None, tile_counts: wp.array \| None=None, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| None=None, pair_params: wp.array \| None=None, neighbor_vectors: wp.array \| None=None, neighbor_distances: wp.array \| None=None, pair_energies: wp.array \| None=None, pair_forces: wp.array \| None=None, fixed_cell: bool=False, qr: wp.array \| None=None, axis_aligned: wp.array \| None=None, fractional_rounding_certified: wp.array \| None=None, qr_height_certified: wp.array \| None=None, bbox_cutoff_bounds: wp.array \| None=None) -> None |
| changed | nvalchemiops/neighbors/cluster_tile/launchers.py:query_cluster_tile_coo | (sorted_atom_index: wp.array, sorted_pos_x: wp.array, sorted_pos_y: wp.array, sorted_pos_z: wp.array, num_tiles: wp.array, tile_row_group: wp.array, tile_col_group: wp.array, cell: wp.array, inv_cell: wp.array, cutoff: float, natom: int, max_pairs: int, pair_counter: wp.array, coo_list: wp.array, coo_shifts: wp.array, wp_dtype: type, device: str, *, n_tiles: int \| None=None, rebuild_flags: wp.array \| None=None, tile_offsets: wp.array \| None=None, tile_counts: wp.array \| None=None, pair_offsets: wp.array \| None=None, pair_counts: wp.array \| None=None, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| None=None, pair_params: wp.array \| None=None, neighbor_vectors: wp.array \| None=None, neighbor_distances: wp.array \| None=None, pair_energies: wp.array \| None=None, pair_forces: wp.array \| None=None) -> None | (sorted_atom_index: wp.array, sorted_pos_x: wp.array, sorted_pos_y: wp.array, sorted_pos_z: wp.array, num_tiles: wp.array, tile_row_group: wp.array, tile_col_group: wp.array, cell: wp.array, inv_cell: wp.array, cutoff: float, natom: int, max_pairs: int, pair_counter: wp.array, coo_list: wp.array, coo_shifts: wp.array, wp_dtype: type, device: str, *, n_tiles: int \| None=None, rebuild_flags: wp.array \| None=None, tile_offsets: wp.array \| None=None, tile_counts: wp.array \| None=None, pair_offsets: wp.array \| None=None, pair_counts: wp.array \| None=None, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| None=None, pair_params: wp.array \| None=None, neighbor_vectors: wp.array \| None=None, neighbor_distances: wp.array \| None=None, pair_energies: wp.array \| None=None, pair_forces: wp.array \| None=None, fixed_cell: bool=False, qr: wp.array \| None=None, axis_aligned: wp.array \| None=None, fractional_rounding_certified: wp.array \| None=None, qr_height_certified: wp.array \| None=None, bbox_cutoff_bounds: wp.array \| None=None) -> None |
| changed | nvalchemiops/neighbors/naive/launchers.py:batch_naive_neighbor_matrix_pbc | (positions: wp.array, cell: wp.array, cutoff: float, batch_ptr: wp.array, batch_idx: wp.array, shift_range: wp.array, num_shifts_arr: wp.array, max_shifts_per_system: int, neighbor_matrix: wp.array, neighbor_matrix_shifts: wp.array, num_neighbors: wp.array, wp_dtype: type, device: str, max_atoms_per_system: int, half_fill: bool=False, rebuild_flags: wp.array \| None=None, wrap_positions: bool=True, target_indices: wp.array \| None=None, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| None=None, pair_params: wp.array \| None=None, neighbor_vectors: wp.array \| None=None, neighbor_distances: wp.array \| None=None, pair_energies: wp.array \| None=None, pair_forces: wp.array \| None=None, positions_wrapped_buffer: wp.array \| None=None, per_atom_cell_offsets_buffer: wp.array \| None=None, inv_cell_buffer: wp.array \| None=None, strategy: str='auto', positions_wrapped: wp.array \| None=None, per_atom_cell_offsets: wp.array \| None=None, inv_cell: wp.array \| None=None, pbc: wp.array \| None=None) -> None | (positions: wp.array, cell: wp.array, cutoff: float, batch_ptr: wp.array, batch_idx: wp.array, shift_range: wp.array, num_shifts_arr: wp.array, max_shifts_per_system: int, neighbor_matrix: wp.array, neighbor_matrix_shifts: wp.array, num_neighbors: wp.array, wp_dtype: type, device: str, max_atoms_per_system: int \| None=None, half_fill: bool=False, rebuild_flags: wp.array \| None=None, wrap_positions: bool=True, target_indices: wp.array \| None=None, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| None=None, pair_params: wp.array \| None=None, neighbor_vectors: wp.array \| None=None, neighbor_distances: wp.array \| None=None, pair_energies: wp.array \| None=None, pair_forces: wp.array \| None=None, positions_wrapped_buffer: wp.array \| None=None, per_atom_cell_offsets_buffer: wp.array \| None=None, inv_cell_buffer: wp.array \| None=None, strategy: str='auto', positions_wrapped: wp.array \| None=None, per_atom_cell_offsets: wp.array \| None=None, inv_cell: wp.array \| None=None, pbc: wp.array \| None=None, reuse_inv_cell: bool=False) -> None |
| changed | nvalchemiops/neighbors/naive/launchers.py:batch_naive_neighbor_matrix_pbc_dual_cutoff | (positions: wp.array, cell: wp.array, cutoff1: float, cutoff2: float, batch_ptr: wp.array, batch_idx: wp.array, shift_range: wp.array, num_shifts_arr: wp.array, max_shifts_per_system: int, neighbor_matrix1: wp.array, neighbor_matrix2: wp.array, neighbor_matrix_shifts1: wp.array, neighbor_matrix_shifts2: wp.array, num_neighbors1: wp.array, num_neighbors2: wp.array, wp_dtype: type, device: str, max_atoms_per_system: int, half_fill: bool=False, rebuild_flags: wp.array \| None=None, wrap_positions: bool=True, pair_fn: wp.Function \| None=None, pair_params: wp.array \| None=None, positions_wrapped_buffer: wp.array \| None=None, per_atom_cell_offsets_buffer: wp.array \| None=None, inv_cell_buffer: wp.array \| None=None, positions_wrapped: wp.array \| None=None, per_atom_cell_offsets: wp.array \| None=None, inv_cell: wp.array \| None=None, pbc: wp.array \| None=None) -> None | (positions: wp.array, cell: wp.array, cutoff1: float, cutoff2: float, batch_ptr: wp.array, batch_idx: wp.array, shift_range: wp.array, num_shifts_arr: wp.array, max_shifts_per_system: int, neighbor_matrix1: wp.array, neighbor_matrix2: wp.array, neighbor_matrix_shifts1: wp.array, neighbor_matrix_shifts2: wp.array, num_neighbors1: wp.array, num_neighbors2: wp.array, wp_dtype: type, device: str, max_atoms_per_system: int, half_fill: bool=False, rebuild_flags: wp.array \| None=None, wrap_positions: bool=True, pair_fn: wp.Function \| None=None, pair_params: wp.array \| None=None, positions_wrapped_buffer: wp.array \| None=None, per_atom_cell_offsets_buffer: wp.array \| None=None, inv_cell_buffer: wp.array \| None=None, positions_wrapped: wp.array \| None=None, per_atom_cell_offsets: wp.array \| None=None, inv_cell: wp.array \| None=None, pbc: wp.array \| None=None, reuse_inv_cell: bool=False) -> None |
| changed | nvalchemiops/neighbors/naive/launchers.py:naive_neighbor_matrix_pbc | (positions: wp.array, cutoff: float, cell: wp.array, shift_range: wp.array, num_shifts: int, neighbor_matrix: wp.array, neighbor_matrix_shifts: wp.array, num_neighbors: wp.array, wp_dtype: type, device: str, half_fill: bool=False, rebuild_flags: wp.array \| None=None, wrap_positions: bool=True, target_indices: wp.array \| None=None, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| None=None, pair_params: wp.array \| None=None, neighbor_vectors: wp.array \| None=None, neighbor_distances: wp.array \| None=None, pair_energies: wp.array \| None=None, pair_forces: wp.array \| None=None, positions_wrapped_buffer: wp.array \| None=None, per_atom_cell_offsets_buffer: wp.array \| None=None, inv_cell_buffer: wp.array \| None=None, strategy: str='auto', positions_wrapped: wp.array \| None=None, per_atom_cell_offsets: wp.array \| None=None, inv_cell: wp.array \| None=None, pbc: wp.array \| None=None) -> None | (positions: wp.array, cutoff: float, cell: wp.array, shift_range: wp.array, num_shifts: int, neighbor_matrix: wp.array, neighbor_matrix_shifts: wp.array, num_neighbors: wp.array, wp_dtype: type, device: str, half_fill: bool=False, rebuild_flags: wp.array \| None=None, wrap_positions: bool=True, target_indices: wp.array \| None=None, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| None=None, pair_params: wp.array \| None=None, neighbor_vectors: wp.array \| None=None, neighbor_distances: wp.array \| None=None, pair_energies: wp.array \| None=None, pair_forces: wp.array \| None=None, positions_wrapped_buffer: wp.array \| None=None, per_atom_cell_offsets_buffer: wp.array \| None=None, inv_cell_buffer: wp.array \| None=None, strategy: str='auto', positions_wrapped: wp.array \| None=None, per_atom_cell_offsets: wp.array \| None=None, inv_cell: wp.array \| None=None, pbc: wp.array \| None=None, reuse_inv_cell: bool=False) -> None |
| changed | nvalchemiops/neighbors/naive/launchers.py:naive_neighbor_matrix_pbc_dual_cutoff | (positions: wp.array, cutoff1: float, cutoff2: float, cell: wp.array, shift_range: wp.array, num_shifts: int, neighbor_matrix1: wp.array, neighbor_matrix2: wp.array, neighbor_matrix_shifts1: wp.array, neighbor_matrix_shifts2: wp.array, num_neighbors1: wp.array, num_neighbors2: wp.array, wp_dtype: type, device: str, half_fill: bool=False, rebuild_flags: wp.array \| None=None, wrap_positions: bool=True, pair_fn: wp.Function \| None=None, pair_params: wp.array \| None=None, positions_wrapped_buffer: wp.array \| None=None, per_atom_cell_offsets_buffer: wp.array \| None=None, inv_cell_buffer: wp.array \| None=None, positions_wrapped: wp.array \| None=None, per_atom_cell_offsets: wp.array \| None=None, inv_cell: wp.array \| None=None, pbc: wp.array \| None=None) -> None | (positions: wp.array, cutoff1: float, cutoff2: float, cell: wp.array, shift_range: wp.array, num_shifts: int, neighbor_matrix1: wp.array, neighbor_matrix2: wp.array, neighbor_matrix_shifts1: wp.array, neighbor_matrix_shifts2: wp.array, num_neighbors1: wp.array, num_neighbors2: wp.array, wp_dtype: type, device: str, half_fill: bool=False, rebuild_flags: wp.array \| None=None, wrap_positions: bool=True, pair_fn: wp.Function \| None=None, pair_params: wp.array \| None=None, positions_wrapped_buffer: wp.array \| None=None, per_atom_cell_offsets_buffer: wp.array \| None=None, inv_cell_buffer: wp.array \| None=None, positions_wrapped: wp.array \| None=None, per_atom_cell_offsets: wp.array \| None=None, inv_cell: wp.array \| None=None, pbc: wp.array \| None=None, reuse_inv_cell: bool=False) -> None |
| added | nvalchemiops/neighbors/neighbor_utils.py:TileBufferOverflow | — | class(Exception) |
| added | nvalchemiops/neighbors/neighbor_utils.py:TileBufferOverflow.__init__ | — | (self, max_tiles: int, num_tiles: int, system_index: int \| None=None) -> unknown |
| added | nvalchemiops/neighbors/neighbor_utils.py:check_naive_shift_coverage | — | (cell: wp.array, cutoff: float, pbc: wp.array, prepared_range: wp.array, rebuild_flags: wp.array, status: wp.array, wp_dtype: type, device: str) -> None |
| added | nvalchemiops/neighbors/neighbor_utils.py:get_check_naive_shift_coverage_kernel | — | (wp_dtype: type) -> wp.Kernel @ lru_cache(maxsize=None) |
| changed | nvalchemiops/torch/__init__.py:exports | __all__ = ['torch_custom_op', 'get_wp_dtype', 'get_wp_mat_dtype', 'get_wp_vec_dtype', 'fire2_step_coord', 'fire2_step_coord_cell', 'fire2_step_coord_cell_mix', 'fire2_step_coord_cell_couple', 'fire2_step_coord_cell_apply', 'fire2_compute_extended_reductions', 'fire2_step_extended', 'segmented_dot', 'segmented_matvec', 'segmented_mean', 'segmented_mul', 'segmented_rms_norm', 'segmented_sum'] ; from nvalchemiops.torch._warp_op_helpers import torch_custom_op ; from nvalchemiops.torch.fire2 import fire2_compute_extended_reductions, fire2_step_coord, fire2_step_coord_cell, fire2_step_coord_cell_apply, fire2_step_coord_cell_couple, fire2_step_coord_cell_mix, fire2_step_extended ; from nvalchemiops.torch.segment_ops import segmented_dot, segmented_matvec, segmented_mean, segmented_mul, segmented_rms_norm, segmented_sum ; from nvalchemiops.torch.types import get_wp_dtype, get_wp_mat_dtype, get_wp_vec_dtype ; import importlib | __all__ = ['torch_custom_op', 'get_wp_dtype', 'get_wp_mat_dtype', 'get_wp_vec_dtype', 'fire2_step_coord', 'fire2_step_coord_cell', 'fire2_step_coord_cell_mix', 'fire2_step_coord_cell_couple', 'fire2_step_coord_cell_apply', 'fire2_compute_extended_reductions', 'fire2_step_extended', 'segmented_dot', 'segmented_matvec', 'segmented_mean', 'segmented_mul', 'segmented_rms_norm', 'segmented_sum', 'LBFGSState', 'LBFGSCellState', 'lbfgs_prepare_state', 'lbfgs_prepare_cell_state', 'lbfgs_step_coord', 'lbfgs_step_coord_cell'] ; from nvalchemiops.torch._warp_op_helpers import torch_custom_op ; from nvalchemiops.torch.fire2 import fire2_compute_extended_reductions, fire2_step_coord, fire2_step_coord_cell, fire2_step_coord_cell_apply, fire2_step_coord_cell_couple, fire2_step_coord_cell_mix, fire2_step_extended ; from nvalchemiops.torch.lbfgs import LBFGSCellState, LBFGSState, lbfgs_prepare_cell_state, lbfgs_prepare_state, lbfgs_step_coord, lbfgs_step_coord_cell ; from nvalchemiops.torch.segment_ops import segmented_dot, segmented_matvec, segmented_mean, segmented_mul, segmented_rms_norm, segmented_sum ; from nvalchemiops.torch.types import get_wp_dtype, get_wp_mat_dtype, get_wp_vec_dtype ; import importlib |
| added | nvalchemiops/torch/_warp_op_helpers.py:scoped_torch_warp_stream | — | (function: Callable[..., Any]) -> Callable[..., Any] |
| changed | nvalchemiops/torch/_warp_op_helpers.py:scoped_warp_stream | (device: torch.device \| str) -> unknown | (device: torch.device \| str, *, torch_stream: torch.cuda.Stream \| None=None) -> unknown |
| changed | nvalchemiops/torch/fire2.py:fire2_compute_extended_reductions | (positions: torch.Tensor, velocities: torch.Tensor, forces: torch.Tensor, cell: torch.Tensor, cell_velocities: torch.Tensor, cell_force: torch.Tensor, batch_idx: torch.Tensor, dt: torch.Tensor, *, atom_ptr: torch.Tensor \| None=None, ext_atom_ptr: torch.Tensor \| None=None, ext_velocities: torch.Tensor \| None=None, ext_forces: torch.Tensor \| None=None, ext_batch_idx: torch.Tensor \| None=None, cell_force_scale: float=1.0) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | (positions: torch.Tensor, velocities: torch.Tensor, forces: torch.Tensor, cell: torch.Tensor, cell_velocities: torch.Tensor, cell_force: torch.Tensor, batch_idx: torch.Tensor, dt: torch.Tensor, *, atom_ptr: torch.Tensor \| None=None, ext_atom_ptr: torch.Tensor \| None=None, ext_velocities: torch.Tensor \| None=None, ext_forces: torch.Tensor \| None=None, ext_batch_idx: torch.Tensor \| None=None, cell_force_scale: float=1.0) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] @ scoped_torch_warp_stream |
| changed | nvalchemiops/torch/fire2.py:fire2_step_coord_cell_apply | (positions: torch.Tensor, velocities: torch.Tensor, cell: torch.Tensor, cell_velocities: torch.Tensor, dt: torch.Tensor, vf: torch.Tensor, batch_idx: torch.Tensor, max_norm: torch.Tensor, *, maxstep: float=0.1, atom_ptr: torch.Tensor \| None=None, ext_atom_ptr: torch.Tensor \| None=None, ext_batch_idx: torch.Tensor \| None=None) -> None | (positions: torch.Tensor, velocities: torch.Tensor, cell: torch.Tensor, cell_velocities: torch.Tensor, dt: torch.Tensor, vf: torch.Tensor, batch_idx: torch.Tensor, max_norm: torch.Tensor, *, maxstep: float=0.1, atom_ptr: torch.Tensor \| None=None, ext_atom_ptr: torch.Tensor \| None=None, ext_batch_idx: torch.Tensor \| None=None) -> None @ scoped_torch_warp_stream |
| changed | nvalchemiops/torch/fire2.py:fire2_step_coord_cell_couple | (positions: torch.Tensor, velocities: torch.Tensor, cell: torch.Tensor, cell_velocities: torch.Tensor, dt: torch.Tensor, vf: torch.Tensor, batch_idx: torch.Tensor, max_norm: torch.Tensor, *, atom_ptr: torch.Tensor \| None=None, ext_atom_ptr: torch.Tensor \| None=None, ext_batch_idx: torch.Tensor \| None=None) -> None | (positions: torch.Tensor, velocities: torch.Tensor, cell: torch.Tensor, cell_velocities: torch.Tensor, dt: torch.Tensor, vf: torch.Tensor, batch_idx: torch.Tensor, max_norm: torch.Tensor, *, atom_ptr: torch.Tensor \| None=None, ext_atom_ptr: torch.Tensor \| None=None, ext_batch_idx: torch.Tensor \| None=None) -> None @ scoped_torch_warp_stream |
| changed | nvalchemiops/torch/fire2.py:fire2_step_extended | (ext_positions: torch.Tensor, ext_velocities: torch.Tensor, ext_forces: torch.Tensor, ext_batch_idx: torch.Tensor, alpha: torch.Tensor, dt: torch.Tensor, nsteps_inc: torch.Tensor, *, vf: torch.Tensor \| None=None, v_sumsq: torch.Tensor \| None=None, f_sumsq: torch.Tensor \| None=None, max_norm: torch.Tensor \| None=None, delaystep: int=60, dtgrow: float=1.05, dtshrink: float=0.75, alphashrink: float=0.985, alpha0: float=0.09, tmax: float=0.08, tmin: float=0.005, maxstep: float=0.1) -> None | (ext_positions: torch.Tensor, ext_velocities: torch.Tensor, ext_forces: torch.Tensor, ext_batch_idx: torch.Tensor, alpha: torch.Tensor, dt: torch.Tensor, nsteps_inc: torch.Tensor, *, vf: torch.Tensor \| None=None, v_sumsq: torch.Tensor \| None=None, f_sumsq: torch.Tensor \| None=None, max_norm: torch.Tensor \| None=None, delaystep: int=60, dtgrow: float=1.05, dtshrink: float=0.75, alphashrink: float=0.985, alpha0: float=0.09, tmax: float=0.08, tmin: float=0.005, maxstep: float=0.1) -> None @ scoped_torch_warp_stream |
| changed | nvalchemiops/torch/interactions/dispersion/__init__.py:exports | __all__ = ['dftd3', 'D3Parameters'] ; from nvalchemiops.torch.interactions.dispersion._dftd3 import D3Parameters, dftd3 | __all__ = ['dftd3', 'D3Parameters', 'fourier_dftd3', 'FourierD3Parameters', 'FourierD3Setup'] ; from nvalchemiops.torch.interactions.dispersion._dftd3 import D3Parameters, dftd3 ; from nvalchemiops.torch.interactions.dispersion._fourier_dftd3 import FourierD3Parameters, FourierD3Setup, fourier_dftd3 |
| added | nvalchemiops/torch/interactions/dispersion/_fourier_dftd3.py:FourierD3Parameters | — | class() @ dataclass |
| added | nvalchemiops/torch/interactions/dispersion/_fourier_dftd3.py:FourierD3Parameters.cn_ref | — | torch.Tensor = required |
| added | nvalchemiops/torch/interactions/dispersion/_fourier_dftd3.py:FourierD3Parameters.device | — | (self) -> torch.device @ property |
| added | nvalchemiops/torch/interactions/dispersion/_fourier_dftd3.py:FourierD3Parameters.eigs | — | torch.Tensor = required |
| added | nvalchemiops/torch/interactions/dispersion/_fourier_dftd3.py:FourierD3Parameters.from_tables | — | (cls, rcov: torch.Tensor, r4r2: torch.Tensor, c6ab: torch.Tensor, cn_ref: torch.Tensor, species: Sequence[int], tol: float=0.0001, max_rank: int \| None=None, device=None, dtype: torch.dtype=torch.float64) -> FourierD3Parameters @ classmethod |
| added | nvalchemiops/torch/interactions/dispersion/_fourier_dftd3.py:FourierD3Parameters.max_relative_error | — | float = required |
| added | nvalchemiops/torch/interactions/dispersion/_fourier_dftd3.py:FourierD3Parameters.n_species | — | (self) -> int @ property |
| added | nvalchemiops/torch/interactions/dispersion/_fourier_dftd3.py:FourierD3Parameters.rank | — | (self) -> int @ property |
| added | nvalchemiops/torch/interactions/dispersion/_fourier_dftd3.py:FourierD3Parameters.rcov | — | torch.Tensor = required |
| added | nvalchemiops/torch/interactions/dispersion/_fourier_dftd3.py:FourierD3Parameters.species_map | — | torch.Tensor = required |
| added | nvalchemiops/torch/interactions/dispersion/_fourier_dftd3.py:FourierD3Parameters.sqrt_q | — | torch.Tensor = required |
| added | nvalchemiops/torch/interactions/dispersion/_fourier_dftd3.py:FourierD3Parameters.to | — | (self, device=None, dtype=None) -> FourierD3Parameters |
| added | nvalchemiops/torch/interactions/dispersion/_fourier_dftd3.py:FourierD3Parameters.uncovered_species | — | (self, numbers: torch.Tensor) -> list[int] |
| added | nvalchemiops/torch/interactions/dispersion/_fourier_dftd3.py:FourierD3Parameters.v_q | — | torch.Tensor = required |
| added | nvalchemiops/torch/interactions/dispersion/_fourier_dftd3.py:FourierD3Setup | — | class() @ dataclass |
| added | nvalchemiops/torch/interactions/dispersion/_fourier_dftd3.py:FourierD3Setup.build | — | (cls, cell: torch.Tensor, n_species: int, mesh_dimensions: tuple[int, int, int], spline_order: int=4) -> FourierD3Setup @ classmethod |
| added | nvalchemiops/torch/interactions/dispersion/_fourier_dftd3.py:FourierD3Setup.cell_inv_grouped | — | torch.Tensor = required |
| added | nvalchemiops/torch/interactions/dispersion/_fourier_dftd3.py:FourierD3Setup.k_matrix | — | torch.Tensor = required |
| added | nvalchemiops/torch/interactions/dispersion/_fourier_dftd3.py:FourierD3Setup.mesh_dimensions | — | tuple[int, int, int] = required |
| added | nvalchemiops/torch/interactions/dispersion/_fourier_dftd3.py:FourierD3Setup.moduli_x | — | torch.Tensor = required |
| added | nvalchemiops/torch/interactions/dispersion/_fourier_dftd3.py:FourierD3Setup.moduli_y | — | torch.Tensor = required |
| added | nvalchemiops/torch/interactions/dispersion/_fourier_dftd3.py:FourierD3Setup.moduli_z | — | torch.Tensor = required |
| added | nvalchemiops/torch/interactions/dispersion/_fourier_dftd3.py:FourierD3Setup.spline_order | — | int = required |
| added | nvalchemiops/torch/interactions/dispersion/_fourier_dftd3.py:FourierD3Setup.validate_for | — | (self, cells, n_species, mesh_dimensions, mesh_spacing=None) -> unknown |
| added | nvalchemiops/torch/interactions/dispersion/_fourier_dftd3.py:FourierD3Setup.volumes | — | torch.Tensor = required |
| added | nvalchemiops/torch/interactions/dispersion/_fourier_dftd3.py:fourier_dftd3 | — | (positions: torch.Tensor, numbers: torch.Tensor, a1: float, a2: float, s8: float, *, fourier_d3_params: FourierD3Parameters, cell: torch.Tensor, cutoff: float, mesh_dimensions: tuple[int, int, int] \| None=None, mesh_spacing: float \| None=None, neighbor_matrix: torch.Tensor \| None=None, neighbor_matrix_shifts: torch.Tensor \| None=None, neighbor_list: torch.Tensor \| None=None, neighbor_ptr: torch.Tensor \| None=None, unit_shifts: torch.Tensor \| None=None, fill_value: int \| None=None, s6: float=1.0, spline_order: int=4, batch_idx: torch.Tensor \| None=None, compute_virial: bool=False, num_systems: int \| None=None, rank_chunk_size: int \| None=None, setup: FourierD3Setup \| None=None, device: str \| None=None) -> tuple[torch.Tensor, ...] |
| changed | nvalchemiops/torch/interactions/electrostatics/__init__.py:exports | __all__ = ['coulomb_energy', 'coulomb_forces', 'coulomb_energy_forces', 'dsf_coulomb', 'ewald_real_space', 'ewald_reciprocal_space', 'ewald_summation', 'compute_slab_correction', 'particle_mesh_ewald', 'pme_reciprocal_space', 'pme_energy_corrections', 'pme_energy_corrections_with_charge_grad', 'pme_green_structure_factor', 'compute_bspline_moduli_1d', 'generate_k_vectors_ewald_summation', 'generate_k_vectors_pme', 'pack_multipole_moments', 'infer_l_max', 'multipole_electrostatic_energy', 'multipole_electrostatic_features', 'multipole_reciprocal_space_energy', 'MultipoleSCFCache', 'prepare_multipole_scf_cache', 'multipole_scf_step_energy', 'multipole_scf_step_features', 'multipole_real_space_energy', 'multipole_real_space_quadrupole_energy', 'multipole_ewald_summation', 'multipole_particle_mesh_ewald', 'multipole_ewald_scf_step_energy', 'EwaldParameters', 'PMEParameters', 'MultipoleEwaldParameters', 'MultipolePMEParameters', 'estimate_ewald_parameters', 'estimate_pme_parameters', 'estimate_pme_mesh_dimensions', 'estimate_multipole_ewald_parameters', 'estimate_multipole_pme_parameters', 'mesh_spacing_to_dimensions'] ; from __future__ import annotations ; from nvalchemiops.torch.interactions.electrostatics._multipole_moments import infer_l_max, pack_multipole_moments ; from nvalchemiops.torch.interactions.electrostatics.coulomb import coulomb_energy, coulomb_energy_forces, coulomb_forces ; from nvalchemiops.torch.interactions.electrostatics.dsf import dsf_coulomb ; from nvalchemiops.torch.interactions.electrostatics.ewald import ewald_real_space, ewald_reciprocal_space, ewald_summation ; from nvalchemiops.torch.interactions.electrostatics.k_vectors import generate_k_vectors_ewald_summation, generate_k_vectors_pme ; from nvalchemiops.torch.interactions.electrostatics.multipole_electrostatics import multipole_electrostatic_energy, multipole_reciprocal_space_energy ; from nvalchemiops.torch.interactions.electrostatics.multipole_ewald import multipole_ewald_summation, multipole_real_space_energy ; from nvalchemiops.torch.interactions.electrostatics.multipole_ewald_quadrupole import multipole_real_space_quadrupole_energy ; from nvalchemiops.torch.interactions.electrostatics.multipole_features import multipole_electrostatic_features ; from nvalchemiops.torch.interactions.electrostatics.multipole_scf_cache import MultipoleSCFCache, prepare_multipole_scf_cache ; from nvalchemiops.torch.interactions.electrostatics.multipole_scf_step import multipole_ewald_scf_step_energy, multipole_scf_step_energy, multipole_scf_step_features ; from nvalchemiops.torch.interactions.electrostatics.parameters import EwaldParameters, MultipoleEwaldParameters, MultipolePMEParameters, PMEParameters, estimate_ewald_parameters, estimate_multipole_ewald_parameters, estimate_multipole_pme_parameters, estimate_pme_mesh_dimensions, estimate_pme_parameters, mesh_spacing_to_dimensions ; from nvalchemiops.torch.interactions.electrostatics.pme import compute_bspline_moduli_1d, particle_mesh_ewald, pme_reciprocal_space ; from nvalchemiops.torch.interactions.electrostatics.pme_multipole import multipole_particle_mesh_ewald ; from nvalchemiops.torch.interactions.electrostatics.slab import compute_slab_correction ; import inspect ; import warnings | __all__ = ['electrostatics_uses_legacy_fp32', 'coulomb_energy', 'coulomb_forces', 'coulomb_energy_forces', 'dsf_coulomb', 'ewald_real_space', 'ewald_reciprocal_space', 'ewald_reciprocal_space_from_miller_indices', 'ewald_summation', 'compute_slab_correction', 'particle_mesh_ewald', 'pme_reciprocal_space', 'pme_energy_corrections', 'pme_energy_corrections_with_charge_grad', 'pme_green_structure_factor', 'compute_bspline_moduli_1d', 'generate_ewald_miller_indices', 'generate_k_squared_pme', 'generate_k_vectors_ewald_summation', 'generate_k_vectors_pme', 'k_vectors_from_miller_indices', 'pack_multipole_moments', 'infer_l_max', 'multipole_electrostatic_energy', 'multipole_electrostatic_features', 'multipole_reciprocal_space_energy', 'MultipoleSCFCache', 'prepare_multipole_scf_cache', 'multipole_scf_step_energy', 'multipole_scf_step_features', 'multipole_real_space_energy', 'multipole_real_space_quadrupole_energy', 'multipole_ewald_summation', 'multipole_particle_mesh_ewald', 'multipole_ewald_scf_step_energy', 'EwaldParameters', 'PMEParameters', 'MultipoleEwaldParameters', 'MultipolePMEParameters', 'estimate_ewald_parameters', 'estimate_pme_parameters', 'estimate_pme_mesh_dimensions', 'estimate_multipole_ewald_parameters', 'estimate_multipole_pme_parameters', 'mesh_spacing_to_dimensions'] ; from __future__ import annotations ; from nvalchemiops.interactions.electrostatics._factory_common import electrostatics_uses_legacy_fp32 ; from nvalchemiops.torch.interactions.electrostatics._multipole_moments import infer_l_max, pack_multipole_moments ; from nvalchemiops.torch.interactions.electrostatics.coulomb import coulomb_energy, coulomb_energy_forces, coulomb_forces ; from nvalchemiops.torch.interactions.electrostatics.dsf import dsf_coulomb ; from nvalchemiops.torch.interactions.electrostatics.ewald import ewald_real_space, ewald_reciprocal_space, ewald_reciprocal_space_from_miller_indices, ewald_summation ; from nvalchemiops.torch.interactions.electrostatics.k_vectors import generate_ewald_miller_indices, generate_k_squared_pme, generate_k_vectors_ewald_summation, generate_k_vectors_pme, k_vectors_from_miller_indices ; from nvalchemiops.torch.interactions.electrostatics.multipole_electrostatics import multipole_electrostatic_energy, multipole_reciprocal_space_energy ; from nvalchemiops.torch.interactions.electrostatics.multipole_ewald import multipole_ewald_summation, multipole_real_space_energy ; from nvalchemiops.torch.interactions.electrostatics.multipole_ewald_quadrupole import multipole_real_space_quadrupole_energy ; from nvalchemiops.torch.interactions.electrostatics.multipole_features import multipole_electrostatic_features ; from nvalchemiops.torch.interactions.electrostatics.multipole_scf_cache import MultipoleSCFCache, prepare_multipole_scf_cache ; from nvalchemiops.torch.interactions.electrostatics.multipole_scf_step import multipole_ewald_scf_step_energy, multipole_scf_step_energy, multipole_scf_step_features ; from nvalchemiops.torch.interactions.electrostatics.parameters import EwaldParameters, MultipoleEwaldParameters, MultipolePMEParameters, PMEParameters, estimate_ewald_parameters, estimate_multipole_ewald_parameters, estimate_multipole_pme_parameters, estimate_pme_mesh_dimensions, estimate_pme_parameters, mesh_spacing_to_dimensions ; from nvalchemiops.torch.interactions.electrostatics.pme import compute_bspline_moduli_1d, particle_mesh_ewald, pme_reciprocal_space ; from nvalchemiops.torch.interactions.electrostatics.pme_multipole import multipole_particle_mesh_ewald ; from nvalchemiops.torch.interactions.electrostatics.slab import compute_slab_correction ; import inspect ; import warnings |
| added | nvalchemiops/torch/interactions/electrostatics/ewald.py:ewald_reciprocal_space_from_miller_indices | — | (positions: torch.Tensor, charges: torch.Tensor, cell: torch.Tensor, miller_indices: torch.Tensor, alpha: torch.Tensor, batch_idx: torch.Tensor \| None=None, compute_forces: bool=False, compute_charge_gradients: bool=False, compute_virial: bool=False, hybrid_forces: bool=False, *, max_atoms_per_system: int \| None=None, energy_reduction: Literal['atom', 'system']='atom') -> torch.Tensor \| tuple[torch.Tensor, ...] |
| changed | nvalchemiops/torch/interactions/electrostatics/ewald.py:ewald_summation | (positions: torch.Tensor, charges: torch.Tensor, cell: torch.Tensor, alpha: float \| torch.Tensor \| None=None, k_vectors: torch.Tensor \| None=None, k_cutoff: float \| None=None, batch_idx: torch.Tensor \| None=None, neighbor_list: torch.Tensor \| None=None, neighbor_ptr: torch.Tensor \| None=None, neighbor_shifts: torch.Tensor \| None=None, neighbor_matrix: torch.Tensor \| None=None, neighbor_matrix_shifts: torch.Tensor \| None=None, mask_value: int \| None=None, compute_forces: bool=False, compute_charge_gradients: bool=False, compute_virial: bool=False, accuracy: float=1e-06, hybrid_forces: bool=False, pbc: torch.Tensor \| None=None, slab_correction: bool=False, *, miller_bounds: tuple[int, int, int] \| torch.Tensor \| None=None, max_atoms_per_system: int \| None=None, energy_reduction: Literal['atom', 'system']='atom') -> tuple[torch.Tensor, ...] \| torch.Tensor | (positions: torch.Tensor, charges: torch.Tensor, cell: torch.Tensor, alpha: float \| torch.Tensor \| None=None, k_vectors: torch.Tensor \| None=None, k_cutoff: float \| None=None, batch_idx: torch.Tensor \| None=None, neighbor_list: torch.Tensor \| None=None, neighbor_ptr: torch.Tensor \| None=None, neighbor_shifts: torch.Tensor \| None=None, neighbor_matrix: torch.Tensor \| None=None, neighbor_matrix_shifts: torch.Tensor \| None=None, mask_value: int \| None=None, compute_forces: bool=False, compute_charge_gradients: bool=False, compute_virial: bool=False, accuracy: float=1e-06, hybrid_forces: bool=False, pbc: torch.Tensor \| None=None, slab_correction: bool=False, *, miller_bounds: tuple[int, int, int] \| torch.Tensor \| None=None, miller_indices: torch.Tensor \| None=None, max_atoms_per_system: int \| None=None, energy_reduction: Literal['atom', 'system']='atom') -> tuple[torch.Tensor, ...] \| torch.Tensor |
| added | nvalchemiops/torch/interactions/electrostatics/k_vectors.py:generate_ewald_miller_indices | — | (cell: torch.Tensor, k_cutoff: float \| torch.Tensor, miller_bounds: tuple[int, int, int] \| torch.Tensor \| None=None) -> torch.Tensor |
| added | nvalchemiops/torch/interactions/electrostatics/k_vectors.py:generate_k_squared_pme | — | (cell: torch.Tensor, mesh_dimensions: tuple[int, int, int], reciprocal_cell: torch.Tensor \| None=None) -> torch.Tensor |
| added | nvalchemiops/torch/interactions/electrostatics/k_vectors.py:k_vectors_from_miller_indices | — | (cell: torch.Tensor, miller_indices: torch.Tensor) -> torch.Tensor |
| changed | nvalchemiops/torch/interactions/electrostatics/multipole_autograd_kernels.py:ReceiverPhiHatFunction.backward | (ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor \| None, torch.Tensor \| None, None, None] @ staticmethod | (ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor \| None, torch.Tensor \| None, None, None] @ staticmethod, scoped_torch_warp_stream |
| changed | nvalchemiops/torch/interactions/electrostatics/multipole_autograd_kernels.py:ReceiverPhiHatFunction.forward | (ctx, k_vectors: torch.Tensor, k_norm2: torch.Tensor, sigmas: torch.Tensor, inv_cl_table: torch.Tensor) -> torch.Tensor @ staticmethod | (ctx, k_vectors: torch.Tensor, k_norm2: torch.Tensor, sigmas: torch.Tensor, inv_cl_table: torch.Tensor) -> torch.Tensor @ staticmethod, scoped_torch_warp_stream |
| changed | nvalchemiops/torch/interactions/electrostatics/multipole_autograd_kernels.py:ReceiverPhiHatQuadrupoleFunction.backward | (ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor \| None, torch.Tensor \| None, None, None] @ staticmethod | (ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor \| None, torch.Tensor \| None, None, None] @ staticmethod, scoped_torch_warp_stream |
| changed | nvalchemiops/torch/interactions/electrostatics/multipole_autograd_kernels.py:ReceiverPhiHatQuadrupoleFunction.forward | (ctx, k_vectors: torch.Tensor, k_norm2: torch.Tensor, sigmas: torch.Tensor, inv_cl_l2: torch.Tensor) -> torch.Tensor @ staticmethod | (ctx, k_vectors: torch.Tensor, k_norm2: torch.Tensor, sigmas: torch.Tensor, inv_cl_l2: torch.Tensor) -> torch.Tensor @ staticmethod, scoped_torch_warp_stream |
| changed | nvalchemiops/torch/interactions/electrostatics/multipole_electrostatics.py:multipole_electrostatic_energy | (positions: torch.Tensor, multipole_moments: torch.Tensor, cell: torch.Tensor, *, batch_idx: torch.Tensor \| None=None, sigma: float, k_cutoff: float \| None=None, k_vectors: torch.Tensor \| None=None, normalize: NormMode \| int \| str=NormMode.MULTIPOLES, include_self_interaction: bool=False) -> torch.Tensor | (positions: torch.Tensor, multipole_moments: torch.Tensor, cell: torch.Tensor, *, batch_idx: torch.Tensor \| None=None, sigma: float, k_cutoff: float \| None=None, k_vectors: torch.Tensor \| None=None, valid_k_counts: torch.Tensor \| None=None, source_overlap_constants: torch.Tensor \| None=None, normalize: NormMode \| int \| str=NormMode.MULTIPOLES, include_self_interaction: bool=False) -> torch.Tensor |
| changed | nvalchemiops/torch/interactions/electrostatics/multipole_ewald.py:BatchMultipoleRealSpaceBackwardFunction.backward | (ctx, gg_positions: torch.Tensor, gg_charges: torch.Tensor, gg_dipoles: torch.Tensor) -> unknown @ staticmethod | (ctx, gg_positions: torch.Tensor, gg_charges: torch.Tensor, gg_dipoles: torch.Tensor) -> unknown @ staticmethod, scoped_torch_warp_stream |
| changed | nvalchemiops/torch/interactions/electrostatics/multipole_ewald.py:BatchMultipoleRealSpaceBackwardFunction.forward | (ctx, grad_energies: torch.Tensor, positions: torch.Tensor, charges: torch.Tensor, dipoles: torch.Tensor, cells: torch.Tensor, sigmas: torch.Tensor, alphas: torch.Tensor, idx_j: torch.Tensor, neighbor_ptr: torch.Tensor, unit_shifts: torch.Tensor, batch_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] @ staticmethod | (ctx, grad_energies: torch.Tensor, positions: torch.Tensor, charges: torch.Tensor, dipoles: torch.Tensor, cells: torch.Tensor, sigmas: torch.Tensor, alphas: torch.Tensor, idx_j: torch.Tensor, neighbor_ptr: torch.Tensor, unit_shifts: torch.Tensor, batch_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] @ staticmethod, scoped_torch_warp_stream |
| changed | nvalchemiops/torch/interactions/electrostatics/multipole_ewald.py:BatchMultipoleRealSpaceDipoleFusedScalarFunction.forward | (ctx, positions: torch.Tensor, charges: torch.Tensor, dipoles: torch.Tensor, cells: torch.Tensor, sigmas: torch.Tensor, alphas: torch.Tensor, idx_j: torch.Tensor, neighbor_ptr: torch.Tensor, unit_shifts: torch.Tensor, batch_idx: torch.Tensor, half_neighbor_list: bool=False) -> torch.Tensor @ staticmethod | (ctx, positions: torch.Tensor, charges: torch.Tensor, dipoles: torch.Tensor, cells: torch.Tensor, sigmas: torch.Tensor, alphas: torch.Tensor, idx_j: torch.Tensor, neighbor_ptr: torch.Tensor, unit_shifts: torch.Tensor, batch_idx: torch.Tensor, half_neighbor_list: bool=False) -> torch.Tensor @ staticmethod, scoped_torch_warp_stream |
| changed | nvalchemiops/torch/interactions/electrostatics/multipole_ewald.py:BatchMultipoleRealSpaceFunction.forward | (ctx, positions: torch.Tensor, charges: torch.Tensor, dipoles: torch.Tensor, cells: torch.Tensor, sigmas: torch.Tensor, alphas: torch.Tensor, idx_j: torch.Tensor, neighbor_ptr: torch.Tensor, unit_shifts: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor @ staticmethod | (ctx, positions: torch.Tensor, charges: torch.Tensor, dipoles: torch.Tensor, cells: torch.Tensor, sigmas: torch.Tensor, alphas: torch.Tensor, idx_j: torch.Tensor, neighbor_ptr: torch.Tensor, unit_shifts: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor @ staticmethod, scoped_torch_warp_stream |
| changed | nvalchemiops/torch/interactions/electrostatics/multipole_ewald.py:BatchMultipoleRealSpaceMonopoleBackwardFunction.backward | (ctx, gg_positions: torch.Tensor, gg_charges: torch.Tensor) -> unknown @ staticmethod | (ctx, gg_positions: torch.Tensor, gg_charges: torch.Tensor) -> unknown @ staticmethod, scoped_torch_warp_stream |
| changed | nvalchemiops/torch/interactions/electrostatics/multipole_ewald.py:BatchMultipoleRealSpaceMonopoleBackwardFunction.forward | (ctx, grad_energies: torch.Tensor, positions: torch.Tensor, charges: torch.Tensor, cells: torch.Tensor, sigmas: torch.Tensor, alphas: torch.Tensor, idx_j: torch.Tensor, neighbor_ptr: torch.Tensor, unit_shifts: torch.Tensor, batch_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] @ staticmethod | (ctx, grad_energies: torch.Tensor, positions: torch.Tensor, charges: torch.Tensor, cells: torch.Tensor, sigmas: torch.Tensor, alphas: torch.Tensor, idx_j: torch.Tensor, neighbor_ptr: torch.Tensor, unit_shifts: torch.Tensor, batch_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] @ staticmethod, scoped_torch_warp_stream |
| changed | nvalchemiops/torch/interactions/electrostatics/multipole_ewald.py:BatchMultipoleRealSpaceMonopoleFunction.forward | (ctx, positions: torch.Tensor, charges: torch.Tensor, cells: torch.Tensor, sigmas: torch.Tensor, alphas: torch.Tensor, idx_j: torch.Tensor, neighbor_ptr: torch.Tensor, unit_shifts: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor @ staticmethod | (ctx, positions: torch.Tensor, charges: torch.Tensor, cells: torch.Tensor, sigmas: torch.Tensor, alphas: torch.Tensor, idx_j: torch.Tensor, neighbor_ptr: torch.Tensor, unit_shifts: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor @ staticmethod, scoped_torch_warp_stream |
| changed | nvalchemiops/torch/interactions/electrostatics/multipole_ewald.py:BatchMultipoleRealSpaceMonopoleFusedScalarFunction.forward | (ctx, positions: torch.Tensor, charges: torch.Tensor, cells: torch.Tensor, sigmas: torch.Tensor, alphas: torch.Tensor, idx_j: torch.Tensor, neighbor_ptr: torch.Tensor, unit_shifts: torch.Tensor, batch_idx: torch.Tensor, half_neighbor_list: bool=False) -> torch.Tensor @ staticmethod | (ctx, positions: torch.Tensor, charges: torch.Tensor, cells: torch.Tensor, sigmas: torch.Tensor, alphas: torch.Tensor, idx_j: torch.Tensor, neighbor_ptr: torch.Tensor, unit_shifts: torch.Tensor, batch_idx: torch.Tensor, half_neighbor_list: bool=False) -> torch.Tensor @ staticmethod, scoped_torch_warp_stream |
| changed | nvalchemiops/torch/interactions/electrostatics/multipole_ewald.py:MultipoleRealSpaceBackwardFunction.backward | (ctx, gg_positions: torch.Tensor, gg_charges: torch.Tensor, gg_dipoles: torch.Tensor) -> unknown @ staticmethod | (ctx, gg_positions: torch.Tensor, gg_charges: torch.Tensor, gg_dipoles: torch.Tensor) -> unknown @ staticmethod, scoped_torch_warp_stream |
| changed | nvalchemiops/torch/interactions/electrostatics/multipole_ewald.py:MultipoleRealSpaceBackwardFunction.forward | (ctx, grad_energies: torch.Tensor, positions: torch.Tensor, charges: torch.Tensor, dipoles: torch.Tensor, cell: torch.Tensor, sigma: torch.Tensor, alpha: torch.Tensor, idx_j: torch.Tensor, neighbor_ptr: torch.Tensor, unit_shifts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] @ staticmethod | (ctx, grad_energies: torch.Tensor, positions: torch.Tensor, charges: torch.Tensor, dipoles: torch.Tensor, cell: torch.Tensor, sigma: torch.Tensor, alpha: torch.Tensor, idx_j: torch.Tensor, neighbor_ptr: torch.Tensor, unit_shifts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] @ staticmethod, scoped_torch_warp_stream |
| changed | nvalchemiops/torch/interactions/electrostatics/multipole_ewald.py:MultipoleRealSpaceDipoleFusedScalarFunction.forward | (ctx, positions: torch.Tensor, charges: torch.Tensor, dipoles: torch.Tensor, cell: torch.Tensor, sigma: torch.Tensor, alpha: torch.Tensor, idx_j: torch.Tensor, neighbor_ptr: torch.Tensor, unit_shifts: torch.Tensor, half_neighbor_list: bool=False) -> torch.Tensor @ staticmethod | (ctx, positions: torch.Tensor, charges: torch.Tensor, dipoles: torch.Tensor, cell: torch.Tensor, sigma: torch.Tensor, alpha: torch.Tensor, idx_j: torch.Tensor, neighbor_ptr: torch.Tensor, unit_shifts: torch.Tensor, half_neighbor_list: bool=False) -> torch.Tensor @ staticmethod, scoped_torch_warp_stream |
| changed | nvalchemiops/torch/interactions/electrostatics/multipole_ewald.py:MultipoleRealSpaceFunction.forward | (ctx, positions: torch.Tensor, charges: torch.Tensor, dipoles: torch.Tensor, cell: torch.Tensor, sigma: torch.Tensor, alpha: torch.Tensor, idx_j: torch.Tensor, neighbor_ptr: torch.Tensor, unit_shifts: torch.Tensor) -> torch.Tensor @ staticmethod | (ctx, positions: torch.Tensor, charges: torch.Tensor, dipoles: torch.Tensor, cell: torch.Tensor, sigma: torch.Tensor, alpha: torch.Tensor, idx_j: torch.Tensor, neighbor_ptr: torch.Tensor, unit_shifts: torch.Tensor) -> torch.Tensor @ staticmethod, scoped_torch_warp_stream |
| changed | nvalchemiops/torch/interactions/electrostatics/multipole_ewald.py:MultipoleRealSpaceMonopoleBackwardFunction.backward | (ctx, gg_positions: torch.Tensor, gg_charges: torch.Tensor) -> unknown @ staticmethod | (ctx, gg_positions: torch.Tensor, gg_charges: torch.Tensor) -> unknown @ staticmethod, scoped_torch_warp_stream |
| changed | nvalchemiops/torch/interactions/electrostatics/multipole_ewald.py:MultipoleRealSpaceMonopoleBackwardFunction.forward | (ctx, grad_energies: torch.Tensor, positions: torch.Tensor, charges: torch.Tensor, cell: torch.Tensor, sigma: torch.Tensor, alpha: torch.Tensor, idx_j: torch.Tensor, neighbor_ptr: torch.Tensor, unit_shifts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] @ staticmethod | (ctx, grad_energies: torch.Tensor, positions: torch.Tensor, charges: torch.Tensor, cell: torch.Tensor, sigma: torch.Tensor, alpha: torch.Tensor, idx_j: torch.Tensor, neighbor_ptr: torch.Tensor, unit_shifts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] @ staticmethod, scoped_torch_warp_stream |
| changed | nvalchemiops/torch/interactions/electrostatics/multipole_ewald.py:MultipoleRealSpaceMonopoleFunction.forward | (ctx, positions: torch.Tensor, charges: torch.Tensor, cell: torch.Tensor, sigma: torch.Tensor, alpha: torch.Tensor, idx_j: torch.Tensor, neighbor_ptr: torch.Tensor, unit_shifts: torch.Tensor) -> torch.Tensor @ staticmethod | (ctx, positions: torch.Tensor, charges: torch.Tensor, cell: torch.Tensor, sigma: torch.Tensor, alpha: torch.Tensor, idx_j: torch.Tensor, neighbor_ptr: torch.Tensor, unit_shifts: torch.Tensor) -> torch.Tensor @ staticmethod, scoped_torch_warp_stream |
| changed | nvalchemiops/torch/interactions/electrostatics/multipole_ewald.py:MultipoleRealSpaceMonopoleFusedScalarFunction.forward | (ctx, positions: torch.Tensor, charges: torch.Tensor, cell: torch.Tensor, sigma: torch.Tensor, alpha: torch.Tensor, idx_j: torch.Tensor, neighbor_ptr: torch.Tensor, unit_shifts: torch.Tensor, half_neighbor_list: bool=False) -> torch.Tensor @ staticmethod | (ctx, positions: torch.Tensor, charges: torch.Tensor, cell: torch.Tensor, sigma: torch.Tensor, alpha: torch.Tensor, idx_j: torch.Tensor, neighbor_ptr: torch.Tensor, unit_shifts: torch.Tensor, half_neighbor_list: bool=False) -> torch.Tensor @ staticmethod, scoped_torch_warp_stream |
| changed | nvalchemiops/torch/interactions/electrostatics/multipole_features.py:multipole_electrostatic_features | (positions: torch.Tensor, multipole_moments: torch.Tensor, cell: torch.Tensor, *, batch_idx: torch.Tensor \| None=None, sigma: float, receiver_sigmas: list[float] \| tuple[float, ...] \| torch.Tensor, k_cutoff: float \| None=None, k_vectors: torch.Tensor \| None=None, feature_max_l: int=1, density_normalize: NormMode \| int \| str=NormMode.MULTIPOLES, feature_normalize: NormMode \| int \| str=NormMode.RECEIVER, include_self_interaction: bool=False) -> torch.Tensor | (positions: torch.Tensor, multipole_moments: torch.Tensor, cell: torch.Tensor, *, batch_idx: torch.Tensor \| None=None, sigma: float, receiver_sigmas: list[float] \| tuple[float, ...] \| torch.Tensor, k_cutoff: float \| None=None, k_vectors: torch.Tensor \| None=None, feature_max_l: int=1, density_normalize: NormMode \| int \| str=NormMode.MULTIPOLES, feature_normalize: NormMode \| int \| str=NormMode.RECEIVER, include_self_interaction: bool=False, cache: MultipoleSCFCache \| None=None) -> torch.Tensor |
| changed | nvalchemiops/torch/interactions/electrostatics/parameters.py:estimate_multipole_pme_parameters | (positions: torch.Tensor, cell: torch.Tensor, sigma: float \| torch.Tensor, batch_idx: torch.Tensor \| None=None, accuracy: float=1e-06, cost_ratio: float=1.0) -> MultipolePMEParameters | (positions: torch.Tensor, cell: torch.Tensor, sigma: float \| torch.Tensor, batch_idx: torch.Tensor \| None=None, accuracy: float=1e-06, cost_ratio: float=1.0, *, spline_order: int=_DEFAULT_PME_SPLINE_ORDER, fft_padding_fraction: float=_DEFAULT_PME_FFT_PADDING_FRACTION) -> MultipolePMEParameters |
| changed | nvalchemiops/torch/interactions/electrostatics/parameters.py:estimate_pme_mesh_dimensions | (cell: torch.Tensor, alpha: torch.Tensor, accuracy: float=1e-06, mesh_safety_factor: float=1.0) -> tuple[int, int, int] | (cell: torch.Tensor, alpha: torch.Tensor, accuracy: float=1e-06, mesh_safety_factor: float=1.0, *, spline_order: int=_DEFAULT_PME_SPLINE_ORDER, fft_padding_fraction: float=_DEFAULT_PME_FFT_PADDING_FRACTION) -> tuple[int, int, int] |
| changed | nvalchemiops/torch/interactions/electrostatics/parameters.py:estimate_pme_parameters | (positions: torch.Tensor, cell: torch.Tensor, batch_idx: torch.Tensor \| None=None, accuracy: float=1e-06, real_space_cutoff: float \| None=None, mesh_safety_factor: float=1.0) -> PMEParameters | (positions: torch.Tensor, cell: torch.Tensor, batch_idx: torch.Tensor \| None=None, accuracy: float=1e-06, real_space_cutoff: float \| None=None, mesh_safety_factor: float=1.0, *, spline_order: int=_DEFAULT_PME_SPLINE_ORDER, fft_padding_fraction: float=_DEFAULT_PME_FFT_PADDING_FRACTION) -> PMEParameters |
| changed | nvalchemiops/torch/interactions/electrostatics/parameters.py:mesh_spacing_to_dimensions | (cell: torch.Tensor, mesh_spacing: float \| torch.Tensor) -> tuple[int, int, int] | (cell: torch.Tensor, mesh_spacing: float \| torch.Tensor, *, spline_order: int=_DEFAULT_PME_SPLINE_ORDER, fft_padding_fraction: float=_DEFAULT_PME_FFT_PADDING_FRACTION) -> tuple[int, int, int] |
| changed | nvalchemiops/torch/interactions/electrostatics/pme.py:particle_mesh_ewald | (positions: torch.Tensor, charges: torch.Tensor, cell: torch.Tensor, alpha: float \| torch.Tensor \| None=None, mesh_spacing: float \| None=None, mesh_dimensions: tuple[int, int, int] \| None=None, spline_order: int=4, batch_idx: torch.Tensor \| None=None, k_vectors: torch.Tensor \| None=None, k_squared: torch.Tensor \| None=None, neighbor_list: torch.Tensor \| None=None, neighbor_ptr: torch.Tensor \| None=None, neighbor_shifts: torch.Tensor \| None=None, neighbor_matrix: torch.Tensor \| None=None, neighbor_matrix_shifts: torch.Tensor \| None=None, mask_value: int \| None=None, compute_forces: bool=False, compute_charge_gradients: bool=False, compute_virial: bool=False, accuracy: float=1e-06, hybrid_forces: bool=False, pbc: torch.Tensor \| None=None, slab_correction: bool=False, *, cell_inv_t: torch.Tensor \| None=None, volume: torch.Tensor \| None=None, moduli_x: torch.Tensor \| None=None, moduli_y: torch.Tensor \| None=None, moduli_z: torch.Tensor \| None=None, energy_reduction: Literal['atom', 'system']='atom') -> torch.Tensor \| tuple[torch.Tensor, ...] | (positions: torch.Tensor, charges: torch.Tensor, cell: torch.Tensor, alpha: float \| torch.Tensor \| None=None, mesh_spacing: float \| None=None, mesh_dimensions: tuple[int, int, int] \| None=None, spline_order: int=_DEFAULT_PME_SPLINE_ORDER, batch_idx: torch.Tensor \| None=None, k_vectors: torch.Tensor \| None=None, k_squared: torch.Tensor \| None=None, neighbor_list: torch.Tensor \| None=None, neighbor_ptr: torch.Tensor \| None=None, neighbor_shifts: torch.Tensor \| None=None, neighbor_matrix: torch.Tensor \| None=None, neighbor_matrix_shifts: torch.Tensor \| None=None, mask_value: int \| None=None, compute_forces: bool=False, compute_charge_gradients: bool=False, compute_virial: bool=False, accuracy: float=1e-06, hybrid_forces: bool=False, pbc: torch.Tensor \| None=None, slab_correction: bool=False, *, cell_inv_t: torch.Tensor \| None=None, volume: torch.Tensor \| None=None, moduli_x: torch.Tensor \| None=None, moduli_y: torch.Tensor \| None=None, moduli_z: torch.Tensor \| None=None, energy_reduction: Literal['atom', 'system']='atom') -> torch.Tensor \| tuple[torch.Tensor, ...] |
| changed | nvalchemiops/torch/interactions/electrostatics/pme.py:pme_green_structure_factor | (k_squared: torch.Tensor, mesh_dimensions: tuple[int, int, int], alpha: torch.Tensor, cell: torch.Tensor, spline_order: int=4, batch_idx: torch.Tensor \| None=None) -> tuple[torch.Tensor, torch.Tensor] | (k_squared: torch.Tensor, mesh_dimensions: tuple[int, int, int], alpha: torch.Tensor, cell: torch.Tensor, spline_order: int=_DEFAULT_PME_SPLINE_ORDER, batch_idx: torch.Tensor \| None=None) -> tuple[torch.Tensor, torch.Tensor] |
| changed | nvalchemiops/torch/interactions/electrostatics/pme.py:pme_reciprocal_space | (positions: torch.Tensor, charges: torch.Tensor, cell: torch.Tensor, alpha: float \| torch.Tensor, mesh_dimensions: tuple[int, int, int] \| None=None, mesh_spacing: float \| None=None, spline_order: int=4, batch_idx: torch.Tensor \| None=None, k_vectors: torch.Tensor \| None=None, k_squared: torch.Tensor \| None=None, compute_forces: bool=False, compute_charge_gradients: bool=False, compute_virial: bool=False, hybrid_forces: bool=False, *, cell_inv_t: torch.Tensor \| None=None, volume: torch.Tensor \| None=None, moduli_x: torch.Tensor \| None=None, moduli_y: torch.Tensor \| None=None, moduli_z: torch.Tensor \| None=None, energy_reduction: Literal['atom', 'system']='atom') -> torch.Tensor \| tuple[torch.Tensor, ...] | (positions: torch.Tensor, charges: torch.Tensor, cell: torch.Tensor, alpha: float \| torch.Tensor, mesh_dimensions: tuple[int, int, int] \| None=None, mesh_spacing: float \| None=None, spline_order: int=_DEFAULT_PME_SPLINE_ORDER, batch_idx: torch.Tensor \| None=None, k_vectors: torch.Tensor \| None=None, k_squared: torch.Tensor \| None=None, compute_forces: bool=False, compute_charge_gradients: bool=False, compute_virial: bool=False, hybrid_forces: bool=False, *, cell_inv_t: torch.Tensor \| None=None, volume: torch.Tensor \| None=None, moduli_x: torch.Tensor \| None=None, moduli_y: torch.Tensor \| None=None, moduli_z: torch.Tensor \| None=None, energy_reduction: Literal['atom', 'system']='atom') -> torch.Tensor \| tuple[torch.Tensor, ...] |
| changed | nvalchemiops/torch/interactions/electrostatics/pme_multipole.py:multipole_particle_mesh_ewald | (positions: torch.Tensor, multipole_moments: torch.Tensor, cell: torch.Tensor, idx_j: torch.Tensor, neighbor_ptr: torch.Tensor, unit_shifts: torch.Tensor, *, sigma: float, alpha: float \| None=None, mesh_dimensions: tuple[int, int, int] \| None=None, spline_order: int=4, cell_inv_t: torch.Tensor \| None=None, batch_idx: torch.Tensor \| None=None, accuracy: float=1e-06, cost_ratio: float=1.0, volume: torch.Tensor \| None=None, moduli: tuple[torch.Tensor, torch.Tensor, torch.Tensor] \| None=None, k_squared: torch.Tensor \| None=None) -> torch.Tensor | (positions: torch.Tensor, multipole_moments: torch.Tensor, cell: torch.Tensor, idx_j: torch.Tensor, neighbor_ptr: torch.Tensor, unit_shifts: torch.Tensor, *, sigma: float, alpha: float \| None=None, mesh_dimensions: tuple[int, int, int] \| None=None, spline_order: int=_DEFAULT_PME_SPLINE_ORDER, cell_inv_t: torch.Tensor \| None=None, batch_idx: torch.Tensor \| None=None, accuracy: float=1e-06, cost_ratio: float=1.0, volume: torch.Tensor \| None=None, moduli: tuple[torch.Tensor, torch.Tensor, torch.Tensor] \| None=None, k_squared: torch.Tensor \| None=None) -> torch.Tensor |
| changed | nvalchemiops/torch/interactions/electrostatics/pme_multipole.py:multipole_pme_gather_field | (mesh: torch.Tensor, positions: torch.Tensor, cell: torch.Tensor, *, spline_order: int=4, cell_inv_t: torch.Tensor \| None=None, batch_idx: torch.Tensor \| None=None) -> torch.Tensor | (mesh: torch.Tensor, positions: torch.Tensor, cell: torch.Tensor, *, spline_order: int=_DEFAULT_PME_SPLINE_ORDER, cell_inv_t: torch.Tensor \| None=None, batch_idx: torch.Tensor \| None=None) -> torch.Tensor |
| changed | nvalchemiops/torch/interactions/electrostatics/pme_multipole.py:multipole_pme_gather_hessian | (mesh: torch.Tensor, positions: torch.Tensor, cell: torch.Tensor, *, spline_order: int=4, cell_inv_t: torch.Tensor \| None=None) -> torch.Tensor | (mesh: torch.Tensor, positions: torch.Tensor, cell: torch.Tensor, *, spline_order: int=_DEFAULT_PME_SPLINE_ORDER, cell_inv_t: torch.Tensor \| None=None) -> torch.Tensor |
| changed | nvalchemiops/torch/interactions/electrostatics/pme_multipole.py:multipole_pme_gather_potential | (mesh: torch.Tensor, positions: torch.Tensor, cell: torch.Tensor, *, spline_order: int=4, cell_inv_t: torch.Tensor \| None=None, batch_idx: torch.Tensor \| None=None) -> torch.Tensor | (mesh: torch.Tensor, positions: torch.Tensor, cell: torch.Tensor, *, spline_order: int=_DEFAULT_PME_SPLINE_ORDER, cell_inv_t: torch.Tensor \| None=None, batch_idx: torch.Tensor \| None=None) -> torch.Tensor |
| changed | nvalchemiops/torch/interactions/electrostatics/pme_multipole.py:multipole_pme_green_structure_factor | (k_squared: torch.Tensor, miller_x: torch.Tensor, miller_y: torch.Tensor, miller_z: torch.Tensor, alpha: torch.Tensor, sigma: torch.Tensor, volume: torch.Tensor, *, mesh_dimensions: tuple[int, int, int], spline_order: int=4) -> tuple[torch.Tensor, torch.Tensor] | (k_squared: torch.Tensor, miller_x: torch.Tensor, miller_y: torch.Tensor, miller_z: torch.Tensor, alpha: torch.Tensor, sigma: torch.Tensor, volume: torch.Tensor, *, mesh_dimensions: tuple[int, int, int], spline_order: int=_DEFAULT_PME_SPLINE_ORDER) -> tuple[torch.Tensor, torch.Tensor] |
| changed | nvalchemiops/torch/interactions/electrostatics/pme_multipole.py:multipole_pme_reciprocal_space | (positions: torch.Tensor, multipole_moments: torch.Tensor, cell: torch.Tensor, *, sigma: float, alpha: float, mesh_dimensions: tuple[int, int, int], spline_order: int=4, cell_inv_t: torch.Tensor \| None=None, batch_idx: torch.Tensor \| None=None, volume: torch.Tensor \| None=None, moduli: tuple[torch.Tensor, torch.Tensor, torch.Tensor] \| None=None, k_squared: torch.Tensor \| None=None) -> torch.Tensor | (positions: torch.Tensor, multipole_moments: torch.Tensor, cell: torch.Tensor, *, sigma: float, alpha: float, mesh_dimensions: tuple[int, int, int], spline_order: int=_DEFAULT_PME_SPLINE_ORDER, cell_inv_t: torch.Tensor \| None=None, batch_idx: torch.Tensor \| None=None, volume: torch.Tensor \| None=None, moduli: tuple[torch.Tensor, torch.Tensor, torch.Tensor] \| None=None, k_squared: torch.Tensor \| None=None) -> torch.Tensor |
| added | nvalchemiops/torch/lbfgs.py:lbfgs_prepare_cell_state | — | (atom_ptr: torch.Tensor, cell: torch.Tensor, *, cell_force_scale: float=1.0, dtype: torch.dtype=torch.float64, device=None, alignment_atol: float \| None=None) -> LBFGSCellState |
| added | nvalchemiops/torch/lbfgs.py:lbfgs_prepare_state | — | (num_dofs: int, num_systems: int, *, dtype: torch.dtype=torch.float64, device=None, history_size: int=6) -> LBFGSState |
| added | nvalchemiops/torch/lbfgs.py:lbfgs_step_coord | — | (positions: torch.Tensor, forces: torch.Tensor, state: LBFGSState, batch_idx: torch.Tensor, *, maxstep: float=0.2, curvature_eps: float \| None=None, compute_reductions: bool=True) -> None |
| added | nvalchemiops/torch/lbfgs.py:lbfgs_step_coord_cell | — | (positions: torch.Tensor, cell: torch.Tensor, forces: torch.Tensor, stress: torch.Tensor, state: LBFGSState, cell_state: LBFGSCellState, batch_idx: torch.Tensor, *, maxstep: float=0.2, curvature_eps: float \| None=None) -> None |
| changed | nvalchemiops/torch/math/__init__.py:exports | __all__ = ['NormMode', 'inv_cl', 'inv_cl_table', 'FIELD_CONSTANT', 'compute_overlap_constants', 'flatten_to_reference_layout', 'eval_gto_density_pytorch', 'eval_gto_fourier_pytorch', 'eval_regular_solid_harmonics_pytorch', 'eval_irregular_solid_harmonics_pytorch'] ; from nvalchemiops.torch.math.gto import NormMode, eval_gto_density_pytorch, eval_gto_fourier_pytorch, inv_cl, inv_cl_table ; from nvalchemiops.torch.math.gto_self_overlap import FIELD_CONSTANT, compute_overlap_constants, flatten_to_reference_layout ; from nvalchemiops.torch.math.solid_harmonics import eval_irregular_solid_harmonics_pytorch, eval_regular_solid_harmonics_pytorch | __all__ = ['NormMode', 'inv_cl', 'inv_cl_table', 'FIELD_CONSTANT', 'compute_overlap_constants', 'flatten_to_reference_layout', 'eval_gto_density_pytorch', 'eval_gto_fourier_pytorch', 'eval_regular_solid_harmonics_pytorch', 'eval_irregular_solid_harmonics_pytorch', 'eval_spherical_harmonics_pytorch', 'eval_spherical_harmonics_gradient_pytorch'] ; from nvalchemiops.torch.math.gto import NormMode, eval_gto_density_pytorch, eval_gto_fourier_pytorch, inv_cl, inv_cl_table ; from nvalchemiops.torch.math.gto_self_overlap import FIELD_CONSTANT, compute_overlap_constants, flatten_to_reference_layout ; from nvalchemiops.torch.math.solid_harmonics import eval_irregular_solid_harmonics_pytorch, eval_regular_solid_harmonics_pytorch ; from nvalchemiops.torch.math.spherical_harmonics import eval_spherical_harmonics_gradient_pytorch, eval_spherical_harmonics_pytorch |
| added | nvalchemiops/torch/math/spherical_harmonics.py:eval_spherical_harmonics_gradient_pytorch | — | (positions: torch.Tensor, L_max: int=2, device: torch.device \| None=None) -> torch.Tensor |
| added | nvalchemiops/torch/math/spherical_harmonics.py:eval_spherical_harmonics_pytorch | — | (positions: torch.Tensor, L_max: int=2, device: torch.device \| None=None) -> torch.Tensor |
| changed | nvalchemiops/torch/neighbors/__init__.py:exports | __all__ = ['neighbor_list', 'estimate_neighbor_list_costs', 'suggest_neighbor_list_method', 'CompiledPairFn', 'compile_pair_fn', 'cell_list', 'naive_neighbor_list', 'naive_neighbor_list_dual_cutoff', 'cluster_tile_neighbor_list', 'estimate_cell_list_sizes', 'batch_cell_list', 'batch_naive_neighbor_list', 'batch_naive_neighbor_list_dual_cutoff', 'batch_cluster_tile_neighbor_list', 'estimate_batch_cell_list_sizes'] ; from __future__ import annotations ; from nvalchemiops.neighbors.base_dispatch import NEIGHBOR_LIST_STRATEGIES, neighbor_list_strategy_run_args ; from nvalchemiops.torch.neighbors._compiled_pair_fn import CompiledPairFn, compile_pair_fn ; from nvalchemiops.torch.neighbors._dispatch import _auto_method_from_geometry, _reject_unsupported_cluster_tile_combo, _squeeze_single_system_cell_pbc, broadcast_shared_cell_for_batch, estimate_neighbor_list_costs, suggest_neighbor_list_method ; from nvalchemiops.torch.neighbors.batch_cell_list import batch_cell_list, estimate_batch_cell_list_sizes ; from nvalchemiops.torch.neighbors.batch_cluster_tile import batch_cluster_tile_neighbor_list ; from nvalchemiops.torch.neighbors.batch_naive import batch_naive_neighbor_list ; from nvalchemiops.torch.neighbors.batch_naive_dual_cutoff import batch_naive_neighbor_list_dual_cutoff ; from nvalchemiops.torch.neighbors.cell_list import cell_list, estimate_cell_list_sizes ; from nvalchemiops.torch.neighbors.cluster_tile import cluster_tile_neighbor_list ; from nvalchemiops.torch.neighbors.naive import naive_neighbor_list ; from nvalchemiops.torch.neighbors.naive_dual_cutoff import naive_neighbor_list_dual_cutoff ; from nvalchemiops.torch.neighbors.neighbor_utils import prepare_batch_idx_ptr, synthesize_cell_for_batch, synthesize_cell_for_ss ; import torch | __all__ = ['neighbor_list', 'estimate_neighbor_list_costs', 'suggest_neighbor_list_method', 'CompiledPairFn', 'compile_pair_fn', 'NeighborOverflowError', 'TileBufferOverflow', 'NeighborListState', 'prepare_neighbor_list', 'cell_list', 'naive_neighbor_list', 'naive_neighbor_list_dual_cutoff', 'cluster_tile_neighbor_list', 'estimate_cell_list_sizes', 'batch_cell_list', 'batch_naive_neighbor_list', 'batch_naive_neighbor_list_dual_cutoff', 'batch_cluster_tile_neighbor_list', 'estimate_batch_cell_list_sizes'] ; from __future__ import annotations ; from nvalchemiops.neighbors.base_dispatch import NEIGHBOR_LIST_STRATEGIES, neighbor_list_strategy_run_args ; from nvalchemiops.torch.neighbors._compiled_pair_fn import CompiledPairFn, compile_pair_fn ; from nvalchemiops.torch.neighbors._dispatch import _auto_method_from_geometry, _reject_unsupported_cluster_tile_combo, _squeeze_single_system_cell_pbc, broadcast_shared_cell_for_batch, estimate_neighbor_list_costs, suggest_neighbor_list_method ; from nvalchemiops.torch.neighbors.batch_cell_list import batch_cell_list, estimate_batch_cell_list_sizes ; from nvalchemiops.torch.neighbors.batch_cluster_tile import batch_cluster_tile_neighbor_list ; from nvalchemiops.torch.neighbors.batch_naive import batch_naive_neighbor_list ; from nvalchemiops.torch.neighbors.batch_naive_dual_cutoff import batch_naive_neighbor_list_dual_cutoff ; from nvalchemiops.torch.neighbors.cell_list import cell_list, estimate_cell_list_sizes ; from nvalchemiops.torch.neighbors.cluster_tile import cluster_tile_neighbor_list ; from nvalchemiops.torch.neighbors.naive import naive_neighbor_list ; from nvalchemiops.torch.neighbors.naive_dual_cutoff import naive_neighbor_list_dual_cutoff ; from nvalchemiops.torch.neighbors.neighbor_utils import NeighborOverflowError, TileBufferOverflow, prepare_batch_idx_ptr, synthesize_cell_for_batch, synthesize_cell_for_ss ; from nvalchemiops.torch.neighbors.prepared_neighbor_list import NeighborListState, _execute_prepared_neighbor_list, prepare_neighbor_list ; from typing import Any ; import torch |
| changed | nvalchemiops/torch/neighbors/__init__.py:neighbor_list | (positions: torch.Tensor, cutoff: float, cell: torch.Tensor \| None=None, pbc: torch.Tensor \| None=None, batch_idx: torch.Tensor \| None=None, batch_ptr: torch.Tensor \| None=None, cutoff2: float \| None=None, half_fill: bool=False, fill_value: int \| None=None, return_neighbor_list: bool=False, method: str \| None=None, wrap_positions: bool=True, **kwargs: dict) -> unknown | (positions: torch.Tensor, cutoff: float \| None=None, cell: torch.Tensor \| None=None, pbc: torch.Tensor \| None=None, batch_idx: torch.Tensor \| None=None, batch_ptr: torch.Tensor \| None=None, cutoff2: float \| None=None, half_fill: bool=False, fill_value: int \| None=None, return_neighbor_list: bool=False, method: str \| None=None, wrap_positions: bool=True, *, grid_policy: str='configured', state: NeighborListState \| None=None, **kwargs: Any) -> unknown |
| changed | nvalchemiops/torch/neighbors/_dispatch.py:estimate_neighbor_list_costs | (batch_ptr: torch.Tensor, cell: torch.Tensor, pbc: torch.Tensor, cutoff: float, *, batch_idx: torch.Tensor \| None=None, max_nbins: int \| None=None, optional_outputs: Iterable[str] \| None=None, cutoff2: float \| None=None, half_fill: bool=False, return_neighbor_list: bool=False, target_indices: torch.Tensor \| None=None, return_vectors: bool=False, return_distances: bool=False, use_pair_fn: bool=False, rebuild_flags: torch.Tensor \| None=None, wrap_positions: bool=True, positions_dtype: torch.dtype \| None=None) -> list[tuple[str, float]] | (batch_ptr: torch.Tensor, cell: torch.Tensor \| None=None, pbc: torch.Tensor \| None=None, cutoff: float=0.0, *, positions: torch.Tensor \| None=None, batch_idx: torch.Tensor \| None=None, max_nbins: int \| None=None, optional_outputs: Iterable[str] \| None=None, cutoff2: float \| None=None, half_fill: bool=False, return_neighbor_list: bool=False, target_indices: torch.Tensor \| None=None, return_vectors: bool=False, return_distances: bool=False, use_pair_fn: bool=False, rebuild_flags: torch.Tensor \| None=None, wrap_positions: bool=True, positions_dtype: torch.dtype \| None=None) -> list[tuple[str, float]] @ scoped_torch_warp_stream |
| added | nvalchemiops/torch/neighbors/_naive_prepared_guard.py:check_prepared_naive_shift_coverage | — | (cell: torch.Tensor, cutoff: float, pbc: torch.Tensor, prepared_range: torch.Tensor, rebuild_flags: torch.Tensor, status: torch.Tensor) -> None |
| added | nvalchemiops/torch/neighbors/_resolution.py:cell_volume | — | (cell: torch.Tensor) -> float |
| changed | nvalchemiops/torch/neighbors/batch_cell_list.py:batch_build_cell_list | (positions: torch.Tensor, cutoff: float, cell: torch.Tensor, pbc: torch.Tensor, batch_idx: torch.Tensor, cells_per_dimension: torch.Tensor, neighbor_search_radius: torch.Tensor, atom_periodic_shifts: torch.Tensor, atom_to_cell_mapping: torch.Tensor, atoms_per_cell_count: torch.Tensor, cell_atom_start_indices: torch.Tensor, cell_atom_list: torch.Tensor, min_cells_per_dimension: int=4) -> None | (positions: torch.Tensor, cutoff: float, cell: torch.Tensor, pbc: torch.Tensor, batch_idx: torch.Tensor, cells_per_dimension: torch.Tensor, neighbor_search_radius: torch.Tensor, atom_periodic_shifts: torch.Tensor, atom_to_cell_mapping: torch.Tensor, atoms_per_cell_count: torch.Tensor, cell_atom_start_indices: torch.Tensor, cell_atom_list: torch.Tensor, min_cells_per_dimension: int=4, *, cell_offsets: torch.Tensor \| None=None, _fixed_cell_geometry: _FixedCellGeometry \| None=None) -> None |
| changed | nvalchemiops/torch/neighbors/batch_cell_list.py:batch_cell_list | (positions: torch.Tensor, cutoff: float, cell: torch.Tensor, pbc: torch.Tensor, batch_idx: torch.Tensor, max_neighbors: int \| None=None, half_fill: bool=False, fill_value: int \| None=None, return_neighbor_list: bool=False, neighbor_matrix: torch.Tensor \| None=None, neighbor_matrix_shifts: torch.Tensor \| None=None, num_neighbors: torch.Tensor \| None=None, cells_per_dimension: torch.Tensor \| None=None, neighbor_search_radius: torch.Tensor \| None=None, cell_offsets: torch.Tensor \| None=None, atom_periodic_shifts: torch.Tensor \| None=None, atom_to_cell_mapping: torch.Tensor \| None=None, atoms_per_cell_count: torch.Tensor \| None=None, cell_atom_start_indices: torch.Tensor \| None=None, cell_atom_list: torch.Tensor \| None=None, rebuild_flags: torch.Tensor \| None=None, strategy: str='auto', atom_centric_path: str='auto', target_indices: torch.Tensor \| None=None, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| CompiledPairFn \| None=None, pair_params: torch.Tensor \| None=None, neighbor_vectors: torch.Tensor \| None=None, neighbor_distances: torch.Tensor \| None=None, pair_energies: torch.Tensor \| None=None, pair_forces: torch.Tensor \| None=None) -> tuple[torch.Tensor, ...] | (positions: torch.Tensor, cutoff: float, cell: torch.Tensor, pbc: torch.Tensor, batch_idx: torch.Tensor, max_neighbors: int \| None=None, half_fill: bool=False, fill_value: int \| None=None, return_neighbor_list: bool=False, neighbor_matrix: torch.Tensor \| None=None, neighbor_matrix_shifts: torch.Tensor \| None=None, num_neighbors: torch.Tensor \| None=None, cells_per_dimension: torch.Tensor \| None=None, neighbor_search_radius: torch.Tensor \| None=None, cell_offsets: torch.Tensor \| None=None, atom_periodic_shifts: torch.Tensor \| None=None, atom_to_cell_mapping: torch.Tensor \| None=None, atoms_per_cell_count: torch.Tensor \| None=None, cell_atom_start_indices: torch.Tensor \| None=None, cell_atom_list: torch.Tensor \| None=None, rebuild_flags: torch.Tensor \| None=None, strategy: str='auto', atom_centric_path: str='auto', target_indices: torch.Tensor \| None=None, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| CompiledPairFn \| None=None, pair_params: torch.Tensor \| None=None, neighbor_vectors: torch.Tensor \| None=None, neighbor_distances: torch.Tensor \| None=None, pair_energies: torch.Tensor \| None=None, pair_forces: torch.Tensor \| None=None, batch_ptr: torch.Tensor \| None=None, *, grid_policy: str='configured', _fixed_cell_geometry: _FixedCellGeometry \| None=None) -> tuple[torch.Tensor, ...] |
| changed | nvalchemiops/torch/neighbors/batch_cell_list.py:batch_query_cell_list | (positions: torch.Tensor, cell: torch.Tensor, pbc: torch.Tensor, cutoff: float, batch_idx: torch.Tensor, cells_per_dimension: torch.Tensor, neighbor_search_radius: torch.Tensor, atom_periodic_shifts: torch.Tensor, atom_to_cell_mapping: torch.Tensor, atoms_per_cell_count: torch.Tensor, cell_atom_start_indices: torch.Tensor, cell_atom_list: torch.Tensor, neighbor_matrix: torch.Tensor, neighbor_matrix_shifts: torch.Tensor, num_neighbors: torch.Tensor, half_fill: bool=False, rebuild_flags: torch.Tensor \| None=None, fill_value: int \| None=None, strategy: str='auto', atom_centric_path: str='auto', target_indices: torch.Tensor \| None=None, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| CompiledPairFn \| None=None, pair_params: torch.Tensor \| None=None, neighbor_vectors: torch.Tensor \| None=None, neighbor_distances: torch.Tensor \| None=None, pair_energies: torch.Tensor \| None=None, pair_forces: torch.Tensor \| None=None) -> None | (positions: torch.Tensor, cell: torch.Tensor, pbc: torch.Tensor, cutoff: float, batch_idx: torch.Tensor, cells_per_dimension: torch.Tensor, neighbor_search_radius: torch.Tensor, atom_periodic_shifts: torch.Tensor, atom_to_cell_mapping: torch.Tensor, atoms_per_cell_count: torch.Tensor, cell_atom_start_indices: torch.Tensor, cell_atom_list: torch.Tensor, neighbor_matrix: torch.Tensor, neighbor_matrix_shifts: torch.Tensor, num_neighbors: torch.Tensor, half_fill: bool=False, rebuild_flags: torch.Tensor \| None=None, fill_value: int \| None=None, strategy: str='auto', atom_centric_path: str='auto', target_indices: torch.Tensor \| None=None, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| CompiledPairFn \| None=None, pair_params: torch.Tensor \| None=None, neighbor_vectors: torch.Tensor \| None=None, neighbor_distances: torch.Tensor \| None=None, pair_energies: torch.Tensor \| None=None, pair_forces: torch.Tensor \| None=None, *, cached_cell_offsets: torch.Tensor \| None=None, cached_cells_per_system: torch.Tensor \| None=None) -> None |
| changed | nvalchemiops/torch/neighbors/batch_cell_list.py:estimate_batch_cell_list_sizes | (cell: torch.Tensor, pbc: torch.Tensor, cutoff: float, max_nbins: int=8192, min_cells_per_dimension: int=4) -> tuple[int, torch.Tensor] | (cell: torch.Tensor, pbc: torch.Tensor, cutoff: float, max_nbins: int=DEFAULT_BATCH_MAX_NBINS, min_cells_per_dimension: int=4) -> tuple[int, torch.Tensor] @ scoped_torch_warp_stream |
| changed | nvalchemiops/torch/neighbors/batch_cluster_tile.py:batch_query_cluster_tile | (sorted_atom_index: torch.Tensor, sorted_pos_x: torch.Tensor, sorted_pos_y: torch.Tensor, sorted_pos_z: torch.Tensor, cell_batch: torch.Tensor, num_tiles: torch.Tensor, tile_row_group: torch.Tensor, tile_col_group: torch.Tensor, tile_system: torch.Tensor, cutoff: float, natom: int, neighbor_matrix: torch.Tensor, num_neighbors: torch.Tensor, neighbor_matrix_shifts: torch.Tensor, inv_cell_batch: torch.Tensor \| None=None, *, cutoff2: float \| None=None, neighbor_matrix2: torch.Tensor \| None=None, num_neighbors2: torch.Tensor \| None=None, neighbor_matrix_shifts2: torch.Tensor \| None=None, rebuild_flags: torch.Tensor \| None=None, tile_offsets: torch.Tensor \| None=None, tile_counts: torch.Tensor \| None=None, batch_idx: torch.Tensor \| None=None, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| None=None, pair_params: torch.Tensor \| None=None, neighbor_vectors: torch.Tensor \| None=None, neighbor_distances: torch.Tensor \| None=None, pair_energies: torch.Tensor \| None=None, pair_forces: torch.Tensor \| None=None) -> None | (sorted_atom_index: torch.Tensor, sorted_pos_x: torch.Tensor, sorted_pos_y: torch.Tensor, sorted_pos_z: torch.Tensor, cell_batch: torch.Tensor, num_tiles: torch.Tensor, tile_row_group: torch.Tensor, tile_col_group: torch.Tensor, tile_system: torch.Tensor, cutoff: float, natom: int, neighbor_matrix: torch.Tensor, num_neighbors: torch.Tensor, neighbor_matrix_shifts: torch.Tensor, inv_cell_batch: torch.Tensor \| None=None, *, cutoff2: float \| None=None, neighbor_matrix2: torch.Tensor \| None=None, num_neighbors2: torch.Tensor \| None=None, neighbor_matrix_shifts2: torch.Tensor \| None=None, rebuild_flags: torch.Tensor \| None=None, tile_offsets: torch.Tensor \| None=None, tile_counts: torch.Tensor \| None=None, batch_idx: torch.Tensor \| None=None, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| None=None, pair_params: torch.Tensor \| None=None, neighbor_vectors: torch.Tensor \| None=None, neighbor_distances: torch.Tensor \| None=None, pair_energies: torch.Tensor \| None=None, pair_forces: torch.Tensor \| None=None, _fixed_cell_geometry: _FixedCellGeometry \| None=None) -> None @ scoped_torch_warp_stream |
| changed | nvalchemiops/torch/neighbors/batch_cluster_tile.py:batch_query_cluster_tile_coo | (sorted_atom_index: torch.Tensor, sorted_pos_x: torch.Tensor, sorted_pos_y: torch.Tensor, sorted_pos_z: torch.Tensor, cell_batch: torch.Tensor, num_tiles: torch.Tensor, tile_row_group: torch.Tensor, tile_col_group: torch.Tensor, tile_system: torch.Tensor, cutoff: float, natom: int, max_pairs: int, pair_counter: torch.Tensor, coo_list: torch.Tensor, coo_shifts: torch.Tensor, inv_cell_batch: torch.Tensor \| None=None, *, rebuild_flags: torch.Tensor \| None=None, tile_offsets: torch.Tensor \| None=None, tile_counts: torch.Tensor \| None=None, pair_offsets: torch.Tensor \| None=None, pair_counts: torch.Tensor \| None=None, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| None=None, pair_params: torch.Tensor \| None=None, neighbor_vectors: torch.Tensor \| None=None, neighbor_distances: torch.Tensor \| None=None, pair_energies: torch.Tensor \| None=None, pair_forces: torch.Tensor \| None=None) -> None | (sorted_atom_index: torch.Tensor, sorted_pos_x: torch.Tensor, sorted_pos_y: torch.Tensor, sorted_pos_z: torch.Tensor, cell_batch: torch.Tensor, num_tiles: torch.Tensor, tile_row_group: torch.Tensor, tile_col_group: torch.Tensor, tile_system: torch.Tensor, cutoff: float, natom: int, max_pairs: int, pair_counter: torch.Tensor, coo_list: torch.Tensor, coo_shifts: torch.Tensor, inv_cell_batch: torch.Tensor \| None=None, *, rebuild_flags: torch.Tensor \| None=None, tile_offsets: torch.Tensor \| None=None, tile_counts: torch.Tensor \| None=None, pair_offsets: torch.Tensor \| None=None, pair_counts: torch.Tensor \| None=None, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| None=None, pair_params: torch.Tensor \| None=None, neighbor_vectors: torch.Tensor \| None=None, neighbor_distances: torch.Tensor \| None=None, pair_energies: torch.Tensor \| None=None, pair_forces: torch.Tensor \| None=None, _fixed_cell_geometry: _FixedCellGeometry \| None=None) -> None |
| changed | nvalchemiops/torch/neighbors/batch_naive.py:batch_naive_neighbor_list | (positions: torch.Tensor, cutoff: float, batch_idx: torch.Tensor \| None=None, batch_ptr: torch.Tensor \| None=None, pbc: torch.Tensor \| None=None, cell: torch.Tensor \| None=None, max_neighbors: int \| None=None, half_fill: bool=False, fill_value: int \| None=None, return_neighbor_list: bool=False, neighbor_matrix: torch.Tensor \| None=None, neighbor_matrix_shifts: torch.Tensor \| None=None, num_neighbors: torch.Tensor \| None=None, shift_range_per_dimension: torch.Tensor \| None=None, num_shifts_per_system: torch.Tensor \| None=None, max_shifts_per_system: int \| None=None, max_atoms_per_system: int \| None=None, rebuild_flags: torch.Tensor \| None=None, wrap_positions: bool=True, positions_wrapped_buffer: torch.Tensor \| None=None, per_atom_cell_offsets_buffer: torch.Tensor \| None=None, inv_cell_buffer: torch.Tensor \| None=None, *, return_distances: bool=False, return_vectors: bool=False, neighbor_vectors: torch.Tensor \| None=None, neighbor_distances: torch.Tensor \| None=None, target_indices: torch.Tensor \| None=None, pair_fn: wp.Function \| CompiledPairFn \| None=None, pair_params: torch.Tensor \| None=None, pair_energies: torch.Tensor \| None=None, pair_forces: torch.Tensor \| None=None, strategy: str='auto') -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] \| tuple[torch.Tensor, torch.Tensor, torch.Tensor] \| tuple[torch.Tensor, torch.Tensor] | (positions: torch.Tensor, cutoff: float, batch_idx: torch.Tensor \| None=None, batch_ptr: torch.Tensor \| None=None, pbc: torch.Tensor \| None=None, cell: torch.Tensor \| None=None, max_neighbors: int \| None=None, half_fill: bool=False, fill_value: int \| None=None, return_neighbor_list: bool=False, neighbor_matrix: torch.Tensor \| None=None, neighbor_matrix_shifts: torch.Tensor \| None=None, num_neighbors: torch.Tensor \| None=None, shift_range_per_dimension: torch.Tensor \| None=None, num_shifts_per_system: torch.Tensor \| None=None, max_shifts_per_system: int \| None=None, max_atoms_per_system: int \| None=None, rebuild_flags: torch.Tensor \| None=None, wrap_positions: bool=True, positions_wrapped_buffer: torch.Tensor \| None=None, per_atom_cell_offsets_buffer: torch.Tensor \| None=None, inv_cell_buffer: torch.Tensor \| None=None, *, return_distances: bool=False, return_vectors: bool=False, neighbor_vectors: torch.Tensor \| None=None, neighbor_distances: torch.Tensor \| None=None, target_indices: torch.Tensor \| None=None, pair_fn: wp.Function \| CompiledPairFn \| None=None, pair_params: torch.Tensor \| None=None, pair_energies: torch.Tensor \| None=None, pair_forces: torch.Tensor \| None=None, strategy: str='auto', _fixed_cell_geometry: _FixedCellGeometry \| None=None, _prepared_target_handle: _PreparedTargetValidationHandle \| None=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] \| tuple[torch.Tensor, torch.Tensor, torch.Tensor] \| tuple[torch.Tensor, torch.Tensor] |
| changed | nvalchemiops/torch/neighbors/batch_naive_dual_cutoff.py:batch_naive_neighbor_list_dual_cutoff | (positions: torch.Tensor, cutoff1: float, cutoff2: float, batch_idx: torch.Tensor \| None=None, batch_ptr: torch.Tensor \| None=None, pbc: torch.Tensor \| None=None, cell: torch.Tensor \| None=None, max_neighbors1: int \| None=None, max_neighbors2: int \| None=None, half_fill: bool=False, fill_value: int \| None=None, return_neighbor_list: bool=False, neighbor_matrix1: torch.Tensor \| None=None, neighbor_matrix2: torch.Tensor \| None=None, neighbor_matrix_shifts1: torch.Tensor \| None=None, neighbor_matrix_shifts2: torch.Tensor \| None=None, num_neighbors1: torch.Tensor \| None=None, num_neighbors2: torch.Tensor \| None=None, shift_range_per_dimension: torch.Tensor \| None=None, num_shifts_per_system: torch.Tensor \| None=None, max_shifts_per_system: int \| None=None, max_atoms_per_system: int \| None=None, rebuild_flags: torch.Tensor \| None=None, wrap_positions: bool=True, positions_wrapped_buffer: torch.Tensor \| None=None, per_atom_cell_offsets_buffer: torch.Tensor \| None=None, inv_cell_buffer: torch.Tensor \| None=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] \| tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] \| tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | (positions: torch.Tensor, cutoff1: float, cutoff2: float, batch_idx: torch.Tensor \| None=None, batch_ptr: torch.Tensor \| None=None, pbc: torch.Tensor \| None=None, cell: torch.Tensor \| None=None, max_neighbors1: int \| None=None, max_neighbors2: int \| None=None, half_fill: bool=False, fill_value: int \| None=None, return_neighbor_list: bool=False, neighbor_matrix1: torch.Tensor \| None=None, neighbor_matrix2: torch.Tensor \| None=None, neighbor_matrix_shifts1: torch.Tensor \| None=None, neighbor_matrix_shifts2: torch.Tensor \| None=None, num_neighbors1: torch.Tensor \| None=None, num_neighbors2: torch.Tensor \| None=None, shift_range_per_dimension: torch.Tensor \| None=None, num_shifts_per_system: torch.Tensor \| None=None, max_shifts_per_system: int \| None=None, max_atoms_per_system: int \| None=None, rebuild_flags: torch.Tensor \| None=None, wrap_positions: bool=True, positions_wrapped_buffer: torch.Tensor \| None=None, per_atom_cell_offsets_buffer: torch.Tensor \| None=None, inv_cell_buffer: torch.Tensor \| None=None, *, _fixed_cell_geometry: _FixedCellGeometry \| None=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] \| tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] \| tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] |
| changed | nvalchemiops/torch/neighbors/cell_list.py:build_cell_list | (positions: torch.Tensor, cutoff: float, cell: torch.Tensor, pbc: torch.Tensor, cells_per_dimension: torch.Tensor, neighbor_search_radius: torch.Tensor, atom_periodic_shifts: torch.Tensor, atom_to_cell_mapping: torch.Tensor, atoms_per_cell_count: torch.Tensor, cell_atom_start_indices: torch.Tensor, cell_atom_list: torch.Tensor, min_cells_per_dimension: int=4) -> None | (positions: torch.Tensor, cutoff: float, cell: torch.Tensor, pbc: torch.Tensor, cells_per_dimension: torch.Tensor, neighbor_search_radius: torch.Tensor, atom_periodic_shifts: torch.Tensor, atom_to_cell_mapping: torch.Tensor, atoms_per_cell_count: torch.Tensor, cell_atom_start_indices: torch.Tensor, cell_atom_list: torch.Tensor, min_cells_per_dimension: int=4, *, _fixed_cell_geometry: _FixedCellGeometry \| None=None) -> None |
| changed | nvalchemiops/torch/neighbors/cell_list.py:cell_list | (positions: torch.Tensor, cutoff: float, cell: torch.Tensor, pbc: torch.Tensor, max_neighbors: int \| None=None, half_fill: bool=False, fill_value: int \| None=None, return_neighbor_list: bool=False, neighbor_matrix: torch.Tensor \| None=None, neighbor_matrix_shifts: torch.Tensor \| None=None, num_neighbors: torch.Tensor \| None=None, cells_per_dimension: torch.Tensor \| None=None, neighbor_search_radius: torch.Tensor \| None=None, atom_periodic_shifts: torch.Tensor \| None=None, atom_to_cell_mapping: torch.Tensor \| None=None, atoms_per_cell_count: torch.Tensor \| None=None, cell_atom_start_indices: torch.Tensor \| None=None, cell_atom_list: torch.Tensor \| None=None, rebuild_flags: torch.Tensor \| None=None, strategy: str='auto', atom_centric_path: str='auto', sorted_positions: torch.Tensor \| None=None, sorted_shifts: torch.Tensor \| None=None, target_indices: torch.Tensor \| None=None, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| CompiledPairFn \| None=None, pair_params: torch.Tensor \| None=None, neighbor_vectors: torch.Tensor \| None=None, neighbor_distances: torch.Tensor \| None=None, pair_energies: torch.Tensor \| None=None, pair_forces: torch.Tensor \| None=None) -> tuple[torch.Tensor, ...] | (positions: torch.Tensor, cutoff: float, cell: torch.Tensor, pbc: torch.Tensor, max_neighbors: int \| None=None, half_fill: bool=False, fill_value: int \| None=None, return_neighbor_list: bool=False, neighbor_matrix: torch.Tensor \| None=None, neighbor_matrix_shifts: torch.Tensor \| None=None, num_neighbors: torch.Tensor \| None=None, cells_per_dimension: torch.Tensor \| None=None, neighbor_search_radius: torch.Tensor \| None=None, atom_periodic_shifts: torch.Tensor \| None=None, atom_to_cell_mapping: torch.Tensor \| None=None, atoms_per_cell_count: torch.Tensor \| None=None, cell_atom_start_indices: torch.Tensor \| None=None, cell_atom_list: torch.Tensor \| None=None, rebuild_flags: torch.Tensor \| None=None, strategy: str='auto', atom_centric_path: str='auto', sorted_positions: torch.Tensor \| None=None, sorted_shifts: torch.Tensor \| None=None, target_indices: torch.Tensor \| None=None, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| CompiledPairFn \| None=None, pair_params: torch.Tensor \| None=None, neighbor_vectors: torch.Tensor \| None=None, neighbor_distances: torch.Tensor \| None=None, pair_energies: torch.Tensor \| None=None, pair_forces: torch.Tensor \| None=None, *, grid_policy: str='configured', _fixed_cell_geometry: _FixedCellGeometry \| None=None) -> tuple[torch.Tensor, ...] |
| changed | nvalchemiops/torch/neighbors/cell_list.py:estimate_cell_list_sizes | (cell: torch.Tensor, pbc: torch.Tensor, cutoff: float, max_nbins: int=524288, min_cells_per_dimension: int=4) -> tuple[int, torch.Tensor] | (cell: torch.Tensor, pbc: torch.Tensor, cutoff: float, max_nbins: int=DEFAULT_SINGLE_MAX_NBINS, min_cells_per_dimension: int=4) -> tuple[int, torch.Tensor] @ scoped_torch_warp_stream |
| changed | nvalchemiops/torch/neighbors/cluster_tile.py:cluster_tile_neighbor_list | (positions: torch.Tensor, cutoff: float, cell: torch.Tensor, max_neighbors: int \| None=None, fill_value: int \| None=None, format: str='matrix', max_pairs: int \| None=None, cutoff2: float \| None=None, rebuild_flags: torch.Tensor \| None=None, return_state: bool=False, neighbor_matrix: torch.Tensor \| None=None, neighbor_matrix_shifts: torch.Tensor \| None=None, num_neighbors: torch.Tensor \| None=None, neighbor_matrix2: torch.Tensor \| None=None, neighbor_matrix_shifts2: torch.Tensor \| None=None, num_neighbors2: torch.Tensor \| None=None, neighbor_list: torch.Tensor \| None=None, neighbor_list_shifts: torch.Tensor \| None=None, pair_offsets: torch.Tensor \| None=None, pair_counts: torch.Tensor \| None=None, pair_counter: torch.Tensor \| None=None, sorted_atom_index: torch.Tensor \| None=None, morton_codes: torch.Tensor \| None=None, sorted_pos_x: torch.Tensor \| None=None, sorted_pos_y: torch.Tensor \| None=None, sorted_pos_z: torch.Tensor \| None=None, group_ctr_x: torch.Tensor \| None=None, group_ctr_y: torch.Tensor \| None=None, group_ctr_z: torch.Tensor \| None=None, group_ext_x: torch.Tensor \| None=None, group_ext_y: torch.Tensor \| None=None, group_ext_z: torch.Tensor \| None=None, num_tiles: torch.Tensor \| None=None, tile_row_group: torch.Tensor \| None=None, tile_col_group: torch.Tensor \| None=None, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| None=None, pair_params: torch.Tensor \| None=None, neighbor_vectors: torch.Tensor \| None=None, neighbor_distances: torch.Tensor \| None=None, pair_energies: torch.Tensor \| None=None, pair_forces: torch.Tensor \| None=None) -> tuple[torch.Tensor, ...] | (positions: torch.Tensor, cutoff: float, cell: torch.Tensor, max_neighbors: int \| None=None, fill_value: int \| None=None, format: str='matrix', max_pairs: int \| None=None, cutoff2: float \| None=None, rebuild_flags: torch.Tensor \| None=None, return_state: bool=False, neighbor_matrix: torch.Tensor \| None=None, neighbor_matrix_shifts: torch.Tensor \| None=None, num_neighbors: torch.Tensor \| None=None, neighbor_matrix2: torch.Tensor \| None=None, neighbor_matrix_shifts2: torch.Tensor \| None=None, num_neighbors2: torch.Tensor \| None=None, neighbor_list: torch.Tensor \| None=None, neighbor_list_shifts: torch.Tensor \| None=None, pair_offsets: torch.Tensor \| None=None, pair_counts: torch.Tensor \| None=None, pair_counter: torch.Tensor \| None=None, sorted_atom_index: torch.Tensor \| None=None, morton_codes: torch.Tensor \| None=None, sorted_pos_x: torch.Tensor \| None=None, sorted_pos_y: torch.Tensor \| None=None, sorted_pos_z: torch.Tensor \| None=None, group_ctr_x: torch.Tensor \| None=None, group_ctr_y: torch.Tensor \| None=None, group_ctr_z: torch.Tensor \| None=None, group_ext_x: torch.Tensor \| None=None, group_ext_y: torch.Tensor \| None=None, group_ext_z: torch.Tensor \| None=None, num_tiles: torch.Tensor \| None=None, tile_row_group: torch.Tensor \| None=None, tile_col_group: torch.Tensor \| None=None, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| None=None, pair_params: torch.Tensor \| None=None, neighbor_vectors: torch.Tensor \| None=None, neighbor_distances: torch.Tensor \| None=None, pair_energies: torch.Tensor \| None=None, pair_forces: torch.Tensor \| None=None, *, max_tiles_per_group: int \| None=None) -> tuple[torch.Tensor, ...] |
| changed | nvalchemiops/torch/neighbors/cluster_tile.py:query_cluster_tile | (sorted_atom_index: torch.Tensor, sorted_pos_x: torch.Tensor, sorted_pos_y: torch.Tensor, sorted_pos_z: torch.Tensor, num_tiles: torch.Tensor, tile_row_group: torch.Tensor, tile_col_group: torch.Tensor, cell: torch.Tensor, cutoff: float, natom: int, neighbor_matrix: torch.Tensor, num_neighbors: torch.Tensor, neighbor_matrix_shifts: torch.Tensor, *, cutoff2: float \| None=None, neighbor_matrix2: torch.Tensor \| None=None, num_neighbors2: torch.Tensor \| None=None, neighbor_matrix_shifts2: torch.Tensor \| None=None, rebuild_flags: torch.Tensor \| None=None, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| None=None, pair_params: torch.Tensor \| None=None, neighbor_vectors: torch.Tensor \| None=None, neighbor_distances: torch.Tensor \| None=None, pair_energies: torch.Tensor \| None=None, pair_forces: torch.Tensor \| None=None) -> None | (sorted_atom_index: torch.Tensor, sorted_pos_x: torch.Tensor, sorted_pos_y: torch.Tensor, sorted_pos_z: torch.Tensor, num_tiles: torch.Tensor, tile_row_group: torch.Tensor, tile_col_group: torch.Tensor, cell: torch.Tensor, cutoff: float, natom: int, neighbor_matrix: torch.Tensor, num_neighbors: torch.Tensor, neighbor_matrix_shifts: torch.Tensor, *, cutoff2: float \| None=None, neighbor_matrix2: torch.Tensor \| None=None, num_neighbors2: torch.Tensor \| None=None, neighbor_matrix_shifts2: torch.Tensor \| None=None, rebuild_flags: torch.Tensor \| None=None, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| None=None, pair_params: torch.Tensor \| None=None, neighbor_vectors: torch.Tensor \| None=None, neighbor_distances: torch.Tensor \| None=None, pair_energies: torch.Tensor \| None=None, pair_forces: torch.Tensor \| None=None, _fixed_cell_geometry: _FixedCellGeometry \| None=None) -> None |
| changed | nvalchemiops/torch/neighbors/cluster_tile.py:query_cluster_tile_coo | (sorted_atom_index: torch.Tensor, sorted_pos_x: torch.Tensor, sorted_pos_y: torch.Tensor, sorted_pos_z: torch.Tensor, num_tiles: torch.Tensor, tile_row_group: torch.Tensor, tile_col_group: torch.Tensor, cell: torch.Tensor, cutoff: float, natom: int, max_pairs: int, pair_counter: torch.Tensor, coo_list: torch.Tensor, coo_shifts: torch.Tensor, *, rebuild_flags: torch.Tensor \| None=None, pair_offsets: torch.Tensor \| None=None, pair_counts: torch.Tensor \| None=None, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| None=None, pair_params: torch.Tensor \| None=None, neighbor_vectors: torch.Tensor \| None=None, neighbor_distances: torch.Tensor \| None=None, pair_energies: torch.Tensor \| None=None, pair_forces: torch.Tensor \| None=None) -> None | (sorted_atom_index: torch.Tensor, sorted_pos_x: torch.Tensor, sorted_pos_y: torch.Tensor, sorted_pos_z: torch.Tensor, num_tiles: torch.Tensor, tile_row_group: torch.Tensor, tile_col_group: torch.Tensor, cell: torch.Tensor, cutoff: float, natom: int, max_pairs: int, pair_counter: torch.Tensor, coo_list: torch.Tensor, coo_shifts: torch.Tensor, *, rebuild_flags: torch.Tensor \| None=None, pair_offsets: torch.Tensor \| None=None, pair_counts: torch.Tensor \| None=None, return_vectors: bool=False, return_distances: bool=False, pair_fn: wp.Function \| None=None, pair_params: torch.Tensor \| None=None, neighbor_vectors: torch.Tensor \| None=None, neighbor_distances: torch.Tensor \| None=None, pair_energies: torch.Tensor \| None=None, pair_forces: torch.Tensor \| None=None, _fixed_cell_geometry: _FixedCellGeometry \| None=None) -> None |
| changed | nvalchemiops/torch/neighbors/naive.py:naive_neighbor_list | (positions: torch.Tensor, cutoff: float, cell: torch.Tensor \| None=None, pbc: torch.Tensor \| None=None, max_neighbors: int \| None=None, half_fill: bool=False, fill_value: int \| None=None, return_neighbor_list: bool=False, neighbor_matrix: torch.Tensor \| None=None, neighbor_matrix_shifts: torch.Tensor \| None=None, num_neighbors: torch.Tensor \| None=None, shift_range_per_dimension: torch.Tensor \| None=None, num_shifts_per_system: torch.Tensor \| None=None, max_shifts_per_system: int \| None=None, rebuild_flags: torch.Tensor \| None=None, wrap_positions: bool=True, positions_wrapped_buffer: torch.Tensor \| None=None, per_atom_cell_offsets_buffer: torch.Tensor \| None=None, inv_cell_buffer: torch.Tensor \| None=None, *, return_distances: bool=False, return_vectors: bool=False, neighbor_vectors: torch.Tensor \| None=None, neighbor_distances: torch.Tensor \| None=None, pair_fn: wp.Function \| CompiledPairFn \| None=None, pair_params: torch.Tensor \| None=None, pair_energies: torch.Tensor \| None=None, pair_forces: torch.Tensor \| None=None, strategy: str='auto', target_indices: torch.Tensor \| None=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] \| tuple[torch.Tensor, torch.Tensor, torch.Tensor] \| tuple[torch.Tensor, torch.Tensor] | (positions: torch.Tensor, cutoff: float, cell: torch.Tensor \| None=None, pbc: torch.Tensor \| None=None, max_neighbors: int \| None=None, half_fill: bool=False, fill_value: int \| None=None, return_neighbor_list: bool=False, neighbor_matrix: torch.Tensor \| None=None, neighbor_matrix_shifts: torch.Tensor \| None=None, num_neighbors: torch.Tensor \| None=None, shift_range_per_dimension: torch.Tensor \| None=None, num_shifts_per_system: torch.Tensor \| None=None, max_shifts_per_system: int \| None=None, rebuild_flags: torch.Tensor \| None=None, wrap_positions: bool=True, positions_wrapped_buffer: torch.Tensor \| None=None, per_atom_cell_offsets_buffer: torch.Tensor \| None=None, inv_cell_buffer: torch.Tensor \| None=None, *, return_distances: bool=False, return_vectors: bool=False, neighbor_vectors: torch.Tensor \| None=None, neighbor_distances: torch.Tensor \| None=None, pair_fn: wp.Function \| CompiledPairFn \| None=None, pair_params: torch.Tensor \| None=None, pair_energies: torch.Tensor \| None=None, pair_forces: torch.Tensor \| None=None, strategy: str='auto', target_indices: torch.Tensor \| None=None, _fixed_cell_geometry: _FixedCellGeometry \| None=None, _prepared_target_handle: _PreparedTargetValidationHandle \| None=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] \| tuple[torch.Tensor, torch.Tensor, torch.Tensor] \| tuple[torch.Tensor, torch.Tensor] |
| changed | nvalchemiops/torch/neighbors/naive_dual_cutoff.py:naive_neighbor_list_dual_cutoff | (positions: torch.Tensor, cutoff1: float, cutoff2: float, pbc: torch.Tensor \| None=None, cell: torch.Tensor \| None=None, max_neighbors1: int \| None=None, max_neighbors2: int \| None=None, half_fill: bool=False, fill_value: int \| None=None, return_neighbor_list: bool=False, neighbor_matrix1: torch.Tensor \| None=None, neighbor_matrix2: torch.Tensor \| None=None, neighbor_matrix_shifts1: torch.Tensor \| None=None, neighbor_matrix_shifts2: torch.Tensor \| None=None, num_neighbors1: torch.Tensor \| None=None, num_neighbors2: torch.Tensor \| None=None, shift_range_per_dimension: torch.Tensor \| None=None, num_shifts_per_system: torch.Tensor \| None=None, max_shifts_per_system: int \| None=None, rebuild_flags: torch.Tensor \| None=None, wrap_positions: bool=True, positions_wrapped_buffer: torch.Tensor \| None=None, per_atom_cell_offsets_buffer: torch.Tensor \| None=None, inv_cell_buffer: torch.Tensor \| None=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] \| tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] \| tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | (positions: torch.Tensor, cutoff1: float, cutoff2: float, pbc: torch.Tensor \| None=None, cell: torch.Tensor \| None=None, max_neighbors1: int \| None=None, max_neighbors2: int \| None=None, half_fill: bool=False, fill_value: int \| None=None, return_neighbor_list: bool=False, neighbor_matrix1: torch.Tensor \| None=None, neighbor_matrix2: torch.Tensor \| None=None, neighbor_matrix_shifts1: torch.Tensor \| None=None, neighbor_matrix_shifts2: torch.Tensor \| None=None, num_neighbors1: torch.Tensor \| None=None, num_neighbors2: torch.Tensor \| None=None, shift_range_per_dimension: torch.Tensor \| None=None, num_shifts_per_system: torch.Tensor \| None=None, max_shifts_per_system: int \| None=None, rebuild_flags: torch.Tensor \| None=None, wrap_positions: bool=True, positions_wrapped_buffer: torch.Tensor \| None=None, per_atom_cell_offsets_buffer: torch.Tensor \| None=None, inv_cell_buffer: torch.Tensor \| None=None, *, _fixed_cell_geometry: _FixedCellGeometry \| None=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] \| tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] \| tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] |
| changed | nvalchemiops/torch/neighbors/neighbor_utils.py:compute_naive_num_shifts | (cell: torch.Tensor, cutoff: float, pbc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int] | (cell: torch.Tensor, cutoff: float, pbc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int] @ scoped_torch_warp_stream |
| changed | nvalchemiops/torch/neighbors/neighbor_utils.py:prepare_batch_idx_ptr | (batch_idx: torch.Tensor \| None, batch_ptr: torch.Tensor \| None, num_atoms: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor] @ torch.compile | (batch_idx: torch.Tensor \| None, batch_ptr: torch.Tensor \| None, num_atoms: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor] |
| added | nvalchemiops/torch/neighbors/prepared_neighbor_list.py:NeighborListState | — | class() |
| added | nvalchemiops/torch/neighbors/prepared_neighbor_list.py:NeighborListState.__init__ | — | (self, *, method: str, strategy: str, format: str, coo_layout: str \| None, is_batched: bool, num_atoms: int, num_systems: int, cutoff: float, cutoff2: float \| None, max_neighbors: int, half_fill: bool, fill_value: int, wrap_positions: bool, return_neighbor_list: bool, selective: bool, return_vectors: bool, return_distances: bool, span_margin: float, fixed_cell: bool, supports_compilation: bool, compilation_blocker: str \| None, device: torch.device, dtype: torch.dtype, route: str, pbc: torch.Tensor \| None, cell: torch.Tensor \| None, cell_shape: tuple[int, ...] \| None, cell_is_shared: bool, batch_idx: torch.Tensor \| None, batch_ptr: torch.Tensor \| None, batch_ranges: tuple[tuple[int, int], ...] \| None, target_indices: torch.Tensor \| None, max_neighbors2: int \| None, max_pairs: int \| None, max_tiles_per_group: int \| None, max_atoms_per_system: int \| None, pair_fn: object \| None, buffers: dict[str, torch.Tensor], output_names: tuple[str \| None, ...], public_output_names: tuple[str \| None, ...], geometry_buffers: dict[str, torch.Tensor], span_capacity: torch.Tensor \| None, synthesized_cell: bool, fixed_cell_geometry: _FixedCellGeometry \| None, exemplar_positions: torch.Tensor, initialized: torch.Tensor, cluster_storage: _ClusterTileStorage \| None, _factory_token: object \| None=None) -> None |
| added | nvalchemiops/torch/neighbors/prepared_neighbor_list.py:NeighborListState.initialized | — | (self) -> torch.Tensor @ property |
| added | nvalchemiops/torch/neighbors/prepared_neighbor_list.py:prepare_neighbor_list | — | (positions: torch.Tensor, cutoff: float, cell: torch.Tensor \| None=None, pbc: torch.Tensor \| None=None, batch_idx: torch.Tensor \| None=None, batch_ptr: torch.Tensor \| None=None, cutoff2: float \| None=None, half_fill: bool=False, fill_value: int \| None=None, return_neighbor_list: bool=False, method: str \| None=None, format: str \| None=None, wrap_positions: bool=True, *, max_neighbors: int \| None=None, max_neighbors2: int \| None=None, max_pairs: int \| None=None, max_tiles_per_group: int \| None=None, return_vectors: bool=False, return_distances: bool=False, selective: bool=False, fixed_cell: bool=False, pair_fn: object \| None=None, span_margin: float=0.0, atomic_density: float \| None=None, target_indices: torch.Tensor \| None=None, strategy: str='auto', atom_centric_path: str='auto', **kwargs: object) -> NeighborListState |

</details>

解析失败：{"baseline": [], "candidate": []}

### 依赖变化

| 依赖范围（保留 marker） | 基线 | 候选 |
|---|---|---|
| requires-python | >=3.11,<3.15 | >=3.11,<3.15 |
| dependencies | ["warp-lang >= 1.13.0", "numpy"] | ["warp-lang >= 1.16.0", "numpy"] |
| extra:jax | ["jax[cuda13]"] | ["jax[cuda13]>=0.10.0,<0.10.2"] |
| extra:jax-cu12 | ["jax[cuda12]"] | ["jax[cuda12]>=0.10.0,<0.10.2"] |
| extra:jax-cu13 | ["jax[cuda13]"] | ["jax[cuda13]>=0.10.0,<0.10.2"] |
| extra:torch | ["torch>=2.8.0"] | ["torch>=2.10.0"] |
| extra:torch-cu12 | ["torch>=2.8.0"] | ["torch>=2.10.0"] |
| extra:torch-cu13 | ["torch>=2.8.0"] | ["torch>=2.10.0"] |

uv sources/metadata/override/conflicts、build-system 和开发依赖组完整保留于 JSON；
以上是声明差异，不是依赖求解或平台兼容性证明。

未纳入对象分析的产品工作树变化：[]

<details><summary>两侧独有提交（分叉时均需审阅）</summary>

#### baseline_only_commits

| 提交 | 主题 |
|---|---|
| [26dbceb61e30cca80e1a5805eebeb51d7dc68fd1](https://github.com/NVIDIA/nvalchemi-toolkit-ops/commit/26dbceb61e30cca80e1a5805eebeb51d7dc68fd1) | fix changelog |
| [c4eabf9e54777a3cc99cf7b01616e8bd668751fe](https://github.com/NVIDIA/nvalchemi-toolkit-ops/commit/c4eabf9e54777a3cc99cf7b01616e8bd668751fe) | updates from comments |
| [895e6c860159a4a433629c6bce6f93e82a846223](https://github.com/NVIDIA/nvalchemi-toolkit-ops/commit/895e6c860159a4a433629c6bce6f93e82a846223) | small bug fixes |
| [3f70e3026e50bede7e0fdb6e34a787064c351d8e](https://github.com/NVIDIA/nvalchemi-toolkit-ops/commit/3f70e3026e50bede7e0fdb6e34a787064c351d8e) | bump to 0.4.1 |

#### candidate_only_commits

| 提交 | 主题 |
|---|---|
| [5fffc0726b1e71bb8b68e10b5f87942064759f5b](https://github.com/NVIDIA/nvalchemi-toolkit-ops/commit/5fffc0726b1e71bb8b68e10b5f87942064759f5b) | docs: refresh neighbor list guides and landing pages (#196) |
| [955a9fea0f1e05ec73c7dc3f416352792ea04b67](https://github.com/NVIDIA/nvalchemi-toolkit-ops/commit/955a9fea0f1e05ec73c7dc3f416352792ea04b67) | docs: update README and release notes for 0.5 (#192) |
| [16e2361f651de078d8b6e997f177fb6d715bc0c5](https://github.com/NVIDIA/nvalchemi-toolkit-ops/commit/16e2361f651de078d8b6e997f177fb6d715bc0c5) | perf(neighbors): select and reuse pair-centric cell grids (#190) |
| [dfe1c96684129f5ccb9490d4ca161adc9e3c9c62](https://github.com/NVIDIA/nvalchemi-toolkit-ops/commit/dfe1c96684129f5ccb9490d4ca161adc9e3c9c62) | docs: 0.5.0-rc rendered-docs fixes (changelog, L-BFGS, benchmark and userguide pages) (#185) |
| [1c0f709a80c56a8b7004d44a20e1539b485178f0](https://github.com/NVIDIA/nvalchemi-toolkit-ops/commit/1c0f709a80c56a8b7004d44a20e1539b485178f0) | perf(neighbors): tile partial naive neighbor lists (#134) |
| [9caa51bd367be5d9a7e878960fdc19affdab5a12](https://github.com/NVIDIA/nvalchemi-toolkit-ops/commit/9caa51bd367be5d9a7e878960fdc19affdab5a12) | feat(neighbors): add shared prepared Torch and JAX neighbor state (#189) |
| [76d6c19fecb0e6f079259db2c7bdef1281b1f2c6](https://github.com/NVIDIA/nvalchemi-toolkit-ops/commit/76d6c19fecb0e6f079259db2c7bdef1281b1f2c6) | recalibrate the neighbor-list strategy selection cost model (#188) |
| [416256d20630ebcb62e67614ad7ae7faede41575](https://github.com/NVIDIA/nvalchemi-toolkit-ops/commit/416256d20630ebcb62e67614ad7ae7faede41575) | Ewald fp32 nostore sf (#157) |
| [2aa77647ffc189197535f68740dbc36310a7f59c](https://github.com/NVIDIA/nvalchemi-toolkit-ops/commit/2aa77647ffc189197535f68740dbc36310a7f59c) | Fix PME mesh sizing and reduce peak memory (#191) |
| [6967a46d39418cfeb0ea962fa7b85daaaa49a0d0](https://github.com/NVIDIA/nvalchemi-toolkit-ops/commit/6967a46d39418cfeb0ea962fa7b85daaaa49a0d0) | fix(dynamics): correct row-cell forces and FIRE2 motion (#194) |
| [470868373f92ae9d3e06aed5d99634d84b22092c](https://github.com/NVIDIA/nvalchemi-toolkit-ops/commit/470868373f92ae9d3e06aed5d99634d84b22092c) | fix(lbfgs): validate cell alignment in framework setup (#187) |
| [8cba8543d86fd4e6e79bd27685194eaaf1b730b7](https://github.com/NVIDIA/nvalchemi-toolkit-ops/commit/8cba8543d86fd4e6e79bd27685194eaaf1b730b7) | fix(torch): trace automatic multipole self-overlap (#186) |
| [5e5621fd76d0217710d9dda607db28a82212535b](https://github.com/NVIDIA/nvalchemi-toolkit-ops/commit/5e5621fd76d0217710d9dda607db28a82212535b) | fix(torch): reduce weighted Ewald reciprocal backward memory (#184) |
| [7db753be5b014476cd3ab0207173a55d6730fb14](https://github.com/NVIDIA/nvalchemi-toolkit-ops/commit/7db753be5b014476cd3ab0207173a55d6730fb14) | fix(torch): support symbolic multipole tracing (#183) |
| [61c7e060b981ef7ec87928931021c98b3582deff](https://github.com/NVIDIA/nvalchemi-toolkit-ops/commit/61c7e060b981ef7ec87928931021c98b3582deff) | fix(neighbors): correct cluster-tile images in skew triclinic cells (#181) |
| [da4107cec0b0959ef0bac1c9b6ed6ab09e1fbefd](https://github.com/NVIDIA/nvalchemi-toolkit-ops/commit/da4107cec0b0959ef0bac1c9b6ed6ab09e1fbefd) | chore: refresh compatible locked dependencies for 0.5.0-rc (#178) |
| [4aa870f713da02a06153bd0d8be66a0ec03f1378](https://github.com/NVIDIA/nvalchemi-toolkit-ops/commit/4aa870f713da02a06153bd0d8be66a0ec03f1378) | test(neighbors): fix Warp test stream ordering and audit entries (#177) |
| [0c6924bba7cae8344d52c69548a4d7bfdbc8c640](https://github.com/NVIDIA/nvalchemi-toolkit-ops/commit/0c6924bba7cae8344d52c69548a4d7bfdbc8c640) | fix(torch): avoid inverse sync during CUDA capture (#176) |
| [2229d76b3d4f9d28d6364d8f4202d1e066a5aa10](https://github.com/NVIDIA/nvalchemi-toolkit-ops/commit/2229d76b3d4f9d28d6364d8f4202d1e066a5aa10) | fix(torch): match reciprocal Ewald fake cache shape (#175) |
| [e56e7c2554efbad5c78fdd5560d7f0edbb4bdaa4](https://github.com/NVIDIA/nvalchemi-toolkit-ops/commit/e56e7c2554efbad5c78fdd5560d7f0edbb4bdaa4) | fix(jax): require 0.10 for CUDA graph capture (#174) |
| [6002cda69f6650f5c0cc2911c4ce3aac1516b79e](https://github.com/NVIDIA/nvalchemi-toolkit-ops/commit/6002cda69f6650f5c0cc2911c4ce3aac1516b79e) | update version number |
| [32445c5f00e83fccd475589f1a6f5ac9e81e6b02](https://github.com/NVIDIA/nvalchemi-toolkit-ops/commit/32445c5f00e83fccd475589f1a6f5ac9e81e6b02) | fourier dfdtd3 (#169) |
| [804d652f602f77f7c72548cd61c05a247a58bc4b](https://github.com/NVIDIA/nvalchemi-toolkit-ops/commit/804d652f602f77f7c72548cd61c05a247a58bc4b) | LBFGS optimizer (#166) |
| [c1fa79a6939d8e6d4553751e3815fc1904af15b8](https://github.com/NVIDIA/nvalchemi-toolkit-ops/commit/c1fa79a6939d8e6d4553751e3815fc1904af15b8) | feat(torch): add compiled and prepared cluster-tile execution (#170) |
| [9f74358a9f53acc80e347c5c48520875ea18d998](https://github.com/NVIDIA/nvalchemi-toolkit-ops/commit/9f74358a9f53acc80e347c5c48520875ea18d998) | fix(neighbors): annotate neighbor_list kwargs as Any (#173) |
| [d168d38045f6eb00bc00a2761d6956e2da5c5bcc](https://github.com/NVIDIA/nvalchemi-toolkit-ops/commit/d168d38045f6eb00bc00a2761d6956e2da5c5bcc) | fix(torch): support compiled matrix-to-COO conversion (#165) |
| [0313580e050b8d8122459bc99f197c163cc63729](https://github.com/NVIDIA/nvalchemi-toolkit-ops/commit/0313580e050b8d8122459bc99f197c163cc63729) | fix(torch): run Warp kernels on current CUDA streams (#171) |
| [fcf878f8d7e737b0a5b5b8009a39a4e20efe4776](https://github.com/NVIDIA/nvalchemi-toolkit-ops/commit/fcf878f8d7e737b0a5b5b8009a39a4e20efe4776) | feat(jax): define compiled neighbor-list boundaries (#160) |
| [ef426158b95783fa08706b752cc301926dc713fc](https://github.com/NVIDIA/nvalchemi-toolkit-ops/commit/ef426158b95783fa08706b752cc301926dc713fc) | fix(neighbors): detect cluster-tile buffer overflow (#164) |
| [7a894f572ad181921be8cc770d30b567aabefc4f](https://github.com/NVIDIA/nvalchemi-toolkit-ops/commit/7a894f572ad181921be8cc770d30b567aabefc4f) | fix(torch): restore license header check and enforce it in CI (#172) |
| [621ce29ade610209393a1a2a57c463c3fb96cbd7](https://github.com/NVIDIA/nvalchemi-toolkit-ops/commit/621ce29ade610209393a1a2a57c463c3fb96cbd7) | fix(segment-ops): support CUDA graph trees (#168) |
| [6a8140266b237ee63c0221857e96dd9334b284a4](https://github.com/NVIDIA/nvalchemi-toolkit-ops/commit/6a8140266b237ee63c0221857e96dd9334b284a4) | fix(electrostatics): correct multipole background term (#163) |
| [93e8a040044852d3ead733835db30999b1c9a0be](https://github.com/NVIDIA/nvalchemi-toolkit-ops/commit/93e8a040044852d3ead733835db30999b1c9a0be) | feat(jax): support D3 parameters under jit (#162) |
| [7d89f862429f1912cc2a07239d3dbe3b3881735e](https://github.com/NVIDIA/nvalchemi-toolkit-ops/commit/7d89f862429f1912cc2a07239d3dbe3b3881735e) | fix(jax): use public SymbolicZero sentinel (#161) |
| [0fe58d746c63e81875acd9f75925882b13fcbcfd](https://github.com/NVIDIA/nvalchemi-toolkit-ops/commit/0fe58d746c63e81875acd9f75925882b13fcbcfd) | fix(jax): support warp-lang>=1.15 (#158) |
| [d38b86d31b69a0f3ff8ab17263ef57de14744e9e](https://github.com/NVIDIA/nvalchemi-toolkit-ops/commit/d38b86d31b69a0f3ff8ab17263ef57de14744e9e) | fix(electrostatics): retain Ewald Miller topology (#152) |
| [139b0c69f8233c776ed14356dc953ceec0235941](https://github.com/NVIDIA/nvalchemi-toolkit-ops/commit/139b0c69f8233c776ed14356dc953ceec0235941) | fix(torch): preserve symbolic compiled shapes (#150) |
| [15ea37b4ff9a51fdb2b7c6580d9ac0b54dafbba4](https://github.com/NVIDIA/nvalchemi-toolkit-ops/commit/15ea37b4ff9a51fdb2b7c6580d9ac0b54dafbba4) | fix(neighbors): avoid forced compilation (#151) |
| [04eedc4fae3822e93b8f397b0eb718f37ab5f155](https://github.com/NVIDIA/nvalchemi-toolkit-ops/commit/04eedc4fae3822e93b8f397b0eb718f37ab5f155) | fix(torch): register FIRE2 custom ops (#149) |
| [c1e23460859a784e1d78043bcd1c8af0d1095fa2](https://github.com/NVIDIA/nvalchemi-toolkit-ops/commit/c1e23460859a784e1d78043bcd1c8af0d1095fa2) | CI examples dependencies fix (#146) |
| [a52bfe58223ce361b5d71d4fba21dffc561e1479](https://github.com/NVIDIA/nvalchemi-toolkit-ops/commit/a52bfe58223ce361b5d71d4fba21dffc561e1479) | Fixing documentation build chain (#144) |
| [3dc29988fda9ea93b4dfd3b4e35fa0add63d64aa](https://github.com/NVIDIA/nvalchemi-toolkit-ops/commit/3dc29988fda9ea93b4dfd3b4e35fa0add63d64aa) | 0.4.1 Bug Fixes Release (#143) |

</details>

## 本地 UBF 与官方修复线索

本节自动收集现有 UBF 编号，核验人工登记的官方 fix 是否包含于候选；
commit 包含性不证明数学/边界等价，未经 C 审阅的条目 equivalence=unknown、action=retain。

| UBF | 问题 | 官方修复线索 | 包含于候选 | 当前判断/行动 |
|---|---|---|---|---|
| UBF-001 | Yoshida 参数未传递 | unknown | unknown | unknown；retain，待 C 核验 |
| UBF-002 | 补位体系的 Q/ndof 计数错误 | unknown | unknown | unknown；retain，待 C 核验 |
| UBF-003 | 非活动体系的私有温控链仍被推进 | unknown | unknown | unknown；retain，待 C 核验 |
| UBF-004 | 链坐标 eta 的时间推进多一倍 | unknown | unknown | unknown；retain，待 C 核验 |
| UBF-005 | 混合状态 Batch 的输出未随坐标恢复 | unknown | unknown | unknown；retain，待 C 核验 |
| UBF-006 | HostMemory 快照与活跃数据共享存储 | unknown | unknown | unknown；retain，待 C 核验 |
| UBF-007 | MACE 本地 checkpoint 被交给下载器 | unknown | unknown | unknown；retain，待 C 核验 |
| UBF-008 | Uniform storage 多属性复制使用不同目标槽位 | unknown | unknown | unknown；retain，待 C 核验 |
| UBF-009 | 行晶格的 stress-to-cell-force 乘法顺序错误 | [PR](https://github.com/NVIDIA/nvalchemi-toolkit-ops/pull/194) 6967a46d39418cfeb0ea962fa7b85daaaa49a0d0 | True | 核心 stress-to-cell-force 公式对应；retain，本地额外边界和官方回归待 C 完整核验，未验证 Warp 数值 |
| UBF-010 | 晶胞对齐的左手坐标与非连续写回错误 | unknown | unknown | unknown；retain，待 C 核验 |
| UBF-011 | FIRE2 仿射运动使用了列晶格公式 | [PR](https://github.com/NVIDIA/nvalchemi-toolkit-ops/pull/194) 6967a46d39418cfeb0ea962fa7b85daaaa49a0d0 | True | 行晶格仿射运动核心数学对应；retain，官方位移限制等同时变化，整实现等价待 C 核验 |
| UBF-012 | 完整晶胞 FIRE 缺少应力驱动与联合控制 | unknown | unknown | unknown；retain，待 C 核验 |
| UBF-013 | FIRE 联合控制更新与混合存在竞态 | unknown | unknown | unknown；retain，待 C 核验 |
| UBF-014 | AlignCellHook 未保护体系子集及坐标系一致性 | unknown | unknown | unknown；retain，待 C 核验 |
| UBF-015 | 变胞退出及掩码更新遗漏私有优化器状态 | unknown | unknown | unknown；retain，待 C 核验 |
| UBF-016 | 补位后首步使用未初始化的力和应力 | unknown | unknown | unknown；retain，待 C 核验 |

## 人工审阅：依赖、特性及决策

下表来自候选记录的人工审阅输入；工具未从提交主题推断修复等价或 HCU 支持。

| 领域 | 源码依据 | 评估/未移植范围 | 行动 |
|---|---|---|---|
| 依赖组合与环境 | Framework f59e28dc:pyproject.toml ; Ops 5fffc072:pyproject.toml ; reports/g2-integration-a-2026-10-09.md | Framework Python 扩展到 <3.15，要求 Ops>=0.5.0、NumPy>=2,<3、PhysicsNeMo>=2.2；Ops Torch/torch-cu12/torch-cu13 extras 从>=2.8提高到>=2.10，Warp>=1.16。当前 Python3.12/Torch2.9/NumPy1.26.4：Python范围满足，NumPy及Torch路径声明有阻塞；不意味着不用Torch extra就证明Torch2.9兼容。Framework uv source指向0.5.0-rc分支；候选Ops源码声明__version__=0.5.0，分支名并非PEP440预发布版本。未验证PyPI当前发布状态或依赖求解。 候选Framework仍声明0.2.0，必须按SHA识别源码组合。 | defer；D 隔离评估，不升级生产 Torch/DTK/NumPy |
| Batch / Storage / 生命周期 | Framework nvalchemi/data/batch.py ; nvalchemi/data/level_storage.py ; nvalchemi/data/group_layout.py ; nvalchemi/dynamics/base.py ; nvalchemi/_checkpoint.py ; bcc43f55748b6d1bbd928c660c1eaeb109fa604c ; a51139165780efda51f361d76a8b3da48c6e05d1 ; a4215f450d68aaf7402391e3384a165af3f38175 | 可扩展层级schema、group layout、状态感知Hooks及checkpoint机制影响Batch边界、inactive恢复、私有状态和补位。UBF-014～016仅登记设计相关线索，尚未确认具体边界等价；需按依赖闭包评估，不能逐文件替换。 | retain 当前实现；进入 D 完整升级评估，C 先核对补丁边界 |
| 传统 FIRE / FIREVariableCell | Framework nvalchemi/dynamics/optimizers/fire.py ; 3b512c7758a74bf97637b89865d1eefb0786af47 | 候选保留源码但弃用FIRE和FIREVariableCell；弃用不等于采用UBF-012完整晶胞联合算法，不据此删除本地兼容实现。 | retain；记录差异，延期迁移默认优化器 |
| 行晶格科学修复 | Ops nvalchemiops/dynamics/utils/cell_filter.py ; Ops test/dynamics/test_fire.py ; Ops test/dynamics/test_fire2.py ; 6967a46d39418cfeb0ea962fa7b85daaaa49a0d0 | PR194的stress转换与FIRE2仿射运动核心对应UBF-009/011；同一提交还处理FIRE2 displacement limiting。只确认源码修复包含性和核心对应，不将其扩大成完整数值/边界等价。 | retain；C 列官方回归 backport 候选，不在 B 改源码 |
| 左手晶胞与非连续写回 | Ops nvalchemiops/dynamics/utils/cell_filter.py ; Framework nvalchemi/dynamics/_ops/cell_align.py ; docs/UPSTREAM_BUGFIXES.md UBF-010 | 候选仍需复核负determinant取负分支和positions.contiguous临时buffer写回；本轮没有UBF-010已完全解决的证据。 | retain；C 按两包具体调用和边界测试核验 |
| Neighbors / prepared / compile | Ops nvalchemiops/torch/neighbors/prepared_neighbor_list.py ; nvalchemiops/torch/neighbors/_cell_grid.py ; nvalchemiops/neighbors/base_dispatch.py ; 9caa51bd367be5d9a7e878960fdc19affdab5a12 ; 16e2361f651de078d8b6e997f177fb6d715bc0c5 | 候选新增共享prepared Torch/JAX邻居状态、pair-centric网格选择/复用和编译接口；Torch包装仍可能launch Warp，不是海光native后端。与本地HIP当前晶胞完整重建、Torch COO打包、registry明确选择须分别审阅。 | retain 本地HIP；defer prepared/compile全移植，D 评估API和依赖闭包 |
| Model Wrapper / 训练 | Framework nvalchemi/models ; nvalchemi/training ; ff7441b68ad572c8d62a163496708729e4ccd6d1 ; 38d178a1c3d6eba0bf384f96eb8b4f7edf940f15 ; ab0b4adb94db8b98322d0d3f398e186e3a182855 | position Hessian、OMol周期应力/UMA动态编译、LoRA与蒸馏扩展尚未在HCU验证；模型输出、stress约定、设备/dtype和训练梯度链需独立gate。 | defer 新特性；D 选择代表性Wrapper闭包，保留现有MACE接口 |
| 新功能队列 | Framework nvalchemi/dynamics/optimizers/lbfgs.py ; nvalchemi/dynamics/mep/neb.py ; nvalchemi/enhanced_sampling/biases ; nvalchemi/gen ; nvalchemi/csp ; nvalchemi/training/distillation ; Ops nvalchemiops/torch/lbfgs.py | 新增L-BFGS与grouped updates、NEB、ABF、生成模型、分子CSP、蒸馏，以及Ops Fourier DFT-D3/电静力相关扩展；均为候选新增范围，尚非HCU支持。是否优先移植由实际需求决定。 | defer；进入功能队列，不混入本轮Delta/UBF工作 |

## 风险与后续

- framework 历史关系为 diverged，须先审阅基线独有提交，不直接 subtree pull。
- framework 存在 40 个本地/上游修改交集，需三方源码审阅。
- framework 静态接口或实现发生变化；动态返回值、语义与平台兼容性尚未验证。
- framework 依赖声明/开发环境配置改变；未执行依赖求解。
- ops 历史关系为 diverged，须先审阅基线独有提交，不直接 subtree pull。
- ops 存在 8 个本地/上游修改交集，需三方源码审阅。
- ops 静态接口或实现发生变化；动态返回值、语义与平台兼容性尚未验证。
- ops 依赖声明/开发环境配置改变；未执行依赖求解。

## B 工具验证记录

| 字段 | 记录 |
|---|---|
| date | 2026-10-09 |
| command | .venv/bin/python -m unittest discover -s scripts -p test_upstream_delta.py -v |
| exit_code | 0 |
| tests_passed | 14 |
| log | artifacts/upstream-delta-v1/tool-tests.log |
| scope | Isolated CPU Git/AST fixtures: divergence/ancestry/unrelated history, rename/add/delete, API/defaults/returns/exports, markers, patch IDs, duplicate keys, missing SHA/shallow history, fix ancestry, protected outputs, deterministic rerun and read-only inputs/refs. No candidate runtime or HCU validation. |

验证记录由阶段执行后登记；生成报告时不自动运行测试。重复生成字节一致，
产品源码、版本锁、生产环境与缓存 refs 保持原值；未运行候选 CPU/HCU gate。


B 仅生成差异与风险；C 将建立完整 UBF 等价/额外边界表，D 再确认隔离验证范围。
保留当前稳定基线；backport、补丁退休、整版本升级均须另行验证与确认。
