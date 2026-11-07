# 参考版本候选交付检查

日期：2026-11-07。候选代码：`a365602da1ab86f3d2994ae310d6a5a7174eabd2`，
为九个示例全部合入后的本地 develop。本报告及相关说明属于后续文档改动；
以下包与源码归档均对应该候选代码，不包含后续文档提交。没有创建发布标签、推送或外部发布。

## 可审阅的交付物

均位于 dcu2 项目 `artifacts/final-delivery-2026-11-07/`，文件哈希见其中 `SHA256SUMS`。

| 文件 | 用途 |
| --- | --- |
| `alchemi-hygon-a365602.tar.gz` | 纯 Git 源码归档，含九个示例及当时文档，不含 external、本机环境、原始 artifacts 或用户外部模型/数据 |
| `dist/nvalchemi_toolkit-0.2.0-py3-none-any.whl` | framework 候选 wheel |
| `dist/nvalchemi_toolkit_ops-0.4.1-py3-none-any.whl` | ops 候选 wheel，携带原生源码，按目标环境 JIT |
| `dist/nvalchemi_toolkit-0.2.0.tar.gz` | framework sdist |
| `dist/nvalchemi_toolkit_ops-0.4.1.tar.gz` | ops sdist |

发行包版本沿用锁定基线 0.2.0/0.4.1，本地候选身份由代码 SHA 和文件哈希区分，不能
与官方同版本文件混淆。正式交付若要包含本报告，应在文档审阅合入后重新生成最终源码归档。
许可证保留；wheel 中全部 Python/原生源码与导出目录逐文件对照一致，ops wheel/sdist
均含 20 个 `.cpp`/`.cu` 文件。根目录示例不在 wheel 中，使用 wheel 仍需取得源码示例。

## 独立构建与安装环境

- 原项目 `.venv` 未安装或更新依赖。临时 `build-env` 只用于构建；
  `runtime-env` 是新建的 CPython 3.12.13 环境，按 `configs/hygon-reference-lock.txt` 安装。
- dcu2 连接北外及 PyPI 索引超时，离线缓存也不全。通过 WSL 从 PyPI 获取精确版本 wheel，
  核对传输前后 SHA-256 后在 dcu2 离线安装。唯一需从 sdist 构建的运行依赖是
  `python-hostlist==2.3.0`，在临时构建环境构建，版本未改变。
- 构建工具固定 `build==1.3.0`、`hatchling==1.27.0`；为 hostlist 另用
  `setuptools==75.8.0`、`wheel==0.45.1`。完整记录见构建工具安装日志。
- Torch/Triton 仍使用锁文件指定的本地海光 wheel，未安装 CUDA/PyPI Torch。
  DTK 26.04、系统 ninja 和 HCU4 沿用本机资源，不能据此声称另一台机器的驱动部署已通过。
- 两个项目 wheel 使用 `uv pip install --no-deps` 安装在新 reference 环境。
  运行时移除源码 PYTHONPATH 和原 `.venv/bin` 路径，模块来源断言为新环境 site-packages。

下面是本轮离线安装的重放方式，需先准备日志目录中的 wheelhouse、third-party-wheels、
项目构建产物，以及锁文件指定的本地海光 wheel；`uv` 需已在 PATH。它不替代目标主机的
DTK/ABI 核对。该环境是临时环境，不要把路径替换为已有项目 `.venv`。

```bash
cd /path/to/alchemi-hygon
delivery_dir=artifacts/final-delivery-2026-11-07
uv venv --python 3.12 "$delivery_dir/runtime-env"
uv pip sync --offline --no-index \
  --find-links "$delivery_dir/wheelhouse" \
  --find-links "$delivery_dir/third-party-wheels" \
  --python "$delivery_dir/runtime-env/bin/python" configs/hygon-reference-lock.txt
uv pip install --offline --no-deps --python "$delivery_dir/runtime-env/bin/python" \
  "$delivery_dir/dist/nvalchemi_toolkit_ops-0.4.1-py3-none-any.whl" \
  "$delivery_dir/dist/nvalchemi_toolkit-0.2.0-py3-none-any.whl"
```

## 运行检查

- 新建 runtime-env 的 CPU 检查覆盖全部九个程序，共 11 次命令执行（示例 09 含开始、恢复、连续对照），全部退出 0。
  真实模型示例以一至两步接线/单点为主，批量锂钠结构按广义力判据完成，保存状态采用 2+2 与连续 4 步对照。
  75 个锁定的索引依赖版本及 Torch/Triton 版本与 reference 锁一致，原环境同项检查也通过。

- 同一安装环境在空闲 HCU4 上运行含分子晶体的混合批量弛豫，92 个全局步收齐三个输入；
  CPU 重算末态能量和最大原子力与保存值对照通过，应力有限、晶胞体积为正。
  HCU 的独立进程 BFGS 开始、恢复与连续对照也通过。两端续算本轮检查几何和总步数；
  完整 Hessian/历史的数值对照继续引用示例 09 原验证，不扩大为逐位一致承诺。

输出与续算检查见 `cpu-installed-smoke.json`、`cuda-installed-smoke.json`、
`installed-imports-*.json` 及各次日志。此次是安装交付检查，原有数值对照、HCU 长短流程与
收敛边界继续以[项目示例说明](../examples/README.md)为准，本轮重跑必要的安装路径，没有重跑全套历史性能/长期统计验收。

## 交付边界与未完成项

1. `uv pip check` 报两项缺失：`click`、`nvidia-physicsnemo>=2.0.0`。
   它们在 framework 默认依赖元数据中，但不在精简 reference 锁文件中。
   本次只能验收指定 reference profile 的示例路径，不能宣称默认依赖组合或所有 CLI/
   训练/分布式入口可直接安装使用。完整 `test_mace.py` 还受 physicsnemo 导入限制。
2. 通用默认安装的依赖元数据/profile 整理，属于后续明确任务；本轮没有强行安装 NVIDIA
   依赖或修改上游依赖声明来消除检查结果。
3. 外部模型和结构需使用者提供，并按各自使用条件使用；海光 wheel、DTK 和 ABI 需要
   目标主机核实。状态文件仅验证同版本环境下的单结构 BFGS，未验收跨版本/完整流水线重启。
4. NPT/NPH、DomainParallel、完整训练、长期真实材料统计、生产吞吐等仍无相应完整验收。
5. 当前候选适合在上述条件下作为源码/reference 示例参考；广泛的默认 wheel 安装交付仍有
   第 1 项缺口。正式版本名、标签、推送、发布和模型/数据分发需另行确认。

原始证据集中在 `artifacts/final-delivery-2026-11-07/`，包括失败的网络安装尝试、离线
传输校验、构建/安装日志、依赖检查失败结果及运行记录，均保留而非改写为通过。
