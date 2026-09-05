# G0 验证摘要（2026-09-05）

实现：新增只读 AST 清单、21 组功能兼容契约、环境/上游锁定和独立探针。尚未修改或安装产品代码。功能契约中的 A/B 分类不是验证状态。

已运行：
- 两个 external `git fsck --full --no-dangling`，退出 0；非浅克隆，工作树干净。
- `.venv/bin/python probes/audit_upstream.py`，退出 0。framework 191 个源码模块、178 个 test 目录文件、39 个 example 文件、20 个配置文件；ops 分别 159/155/47/26。这里是文件数量，不是测试用例数量。97 条显式跨包 from-import 静态解析成功；未验证动态导出/运行调用。
- `hipcc --offload-arch=gfx936 probes/hip_probe.cpp -o artifacts/g0/hip_probe`，退出 0。仅编译；gfx936 为用户明确指定适配目标。
- 从 `/home/wangleping/codes/nvalchemi-toolkit/.venv` 复用 NumPy 1.26.4 到项目 `.venv`；未重新安装 Torch/Triton。

CPU 数值：FP64 x=[1,2,3] 的立方和，一阶=[3,12,27]，二阶=[6,12,18]，容差 1e-12；segment=[4,2]，梯度与二阶解析对照严格一致；FFT roundtrip 最大绝对误差 0。仅显式 CPU/Torch reference，不是上游算子或 HCU 数值证据。

海光 wheel 导入的 torch.__version__=2.9.0、triton.__version__=3.3.0 不含发行 local suffix；安装 metadata 保留完整海光版本。torch.version.hip=6.3.26093，构建 USE_ROCM=ON / USE_CUDA=0 / USE_NCCL=1。依赖冻结记录以 metadata 为准。

未运行：全部设备计算（8 卡满载且无分配）、Triton JIT、HIP kernel 执行、RCCL collective/P2P、真实 MACE（缺环境依赖和指定可信 checkpoint）、上游两包测试、自定义算子 opcheck、PBC/LJ/NVE、域分解和 NVIDIA 基线。MACE 探针直接调用模型，不覆盖 toolkit wrapper；未用 mock 或旧探索结果充当证据。

原始输出：artifacts/g0/{audit.txt,hip_compile_gfx936.txt,torch_cpu.json,torch_cpu.stderr,torch_cpu_numpy1264.json,torch_cpu_numpy1264.stderr,build_metadata.txt}。完整可提交源码清单：reports/upstream_inventory.json。

导入验证：`.venv/bin/python probes/verify_import.py` 退出 0，见 reports/import-verification.json。完整源码/许可证/测试/示例树与指定 SHA 精确相同；两个 SHA 的历史在主仓库中可达，无嵌套 .git。产品代码尚未适配、安装或运行。

配置检查：两份 YAML 可解析，21 个特性 ID 唯一，均有真实源码和上游测试路径；vendor wheel SHA256 见 reports/vendor-wheels.sha256。

最终环境状态：NumPy 1.26.4；从探索环境复用该版本，避免重复下载。复跑 CPU 探针退出 0，NumPy↔Torch 转换、FP64 一阶/二阶、segment、FFT 均通过；stderr 仅有 `Could not open /var/log/hylog/.`，无 NumPy ABI 警告。最终 gfx936 HIP 编译退出 0、无警告。
