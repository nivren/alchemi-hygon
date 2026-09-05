# 环境盘点（2026-09-05 UTC）

- Ubuntu 22.04.5 LTS，Linux 5.15.0-136-generic，x86_64。
- 沙箱外 hy-smi：8 张 BW200 / UBB BW1000，每卡 65520 MiB；驱动 6.3.30-V1.4.1a；任意两卡 Link Type 为 HSW。
- 13:11 UTC 首次资源盘点 8 卡均 100% HCU 利用率、43% VRAM；稍后型号盘点显存用量约 7.2–7.4 GiB，说明资源随时间变化，不能据此抢占。未取得专用卡分配，不启动计算。
- 沙箱内 /dev/dri、/dev/kfd 不可见，hy-smi 报无设备；沙箱外只读命令确认实际有卡。不是驱动缺失结论。
- PATH 中 hipcc 为 /opt/dtk-26.04/bin/hipcc；dcc 25.10.0-0、clang 17.0.0。/opt 还存在多套 DTK，不切换/升级系统。
- /opt/dtk-26.04/lib 存在 libamdhip64、librccl、libhipfft、librocfft、hipBLAS 等；文件存在不证明运行可用，RCCL 运行版本/性能待测。
- 系统 Python 3.10.12；uv 已安装 CPython 3.12.13，用该解释器创建项目 .venv，无系统 site-packages。
- Torch/Triton 使用用户提供的 cp312 海光 wheel，版本锁见 configs/probe-constraints.txt；导入和 CPU 结果见 reports/g0-validation.md。不安装产品两包、不启用 CUDA extras。
- 默认 uv cache 指向只读 /data/envs/uv-cache，探针显式使用 /tmp/alchemi-uv-cache；未修改用户全局配置。
- 默认清华 PyPI 镜像普通依赖下载返回 HTTP 403，安装改用 https://pypi.org/simple；本地 Torch/Triton 来源不变。
- 为解除 framework 导入的首个缺口，项目 `.venv` 已用 `/data/envs/uv-cache` 和 HUST 镜像补齐 `plum-dispatch==2.7.1`、`beartype`、`rich`；Torch/Triton 未被替换。该环境随后仍缺 `jaxtyping`，因此尚未视为完整 framework 环境。完整 Hook→LJ HCU 探针使用已有 `/home/wangleping/codes/nvalchemi-toolkit/.venv` 探索环境通过。

原始探针输出放 artifacts/g0，摘要放 reports。未知：可分配空闲卡、实际 Torch/HIP/Triton/RCCL 设备能力、MLIP checkpoint、NVIDIA 基线。设备型号、安装版本和编译成功不等于算子或产品支持。

用户随后明确 BW200/BW1000 适配目标为 gfx936；所有目标 HIP 编译使用 gfx936。此前 gfx928 编译仅为过程试探，不作为目标能力证据。

本轮已从 `/home/wangleping/codes/nvalchemi-toolkit/.venv` 复用已验证的 NumPy 1.26.4，安装到项目 `.venv`；未重新下载 Torch/Triton。当前 Torch 的 NumPy 互操作已通过，详见 G0 验证摘要。
