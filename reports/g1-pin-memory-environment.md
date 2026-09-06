# pin_memory 环境复核

日期：2026-09-05  
目标：区分 HCU 资源共享、设备节点可见性与海光 Torch 的 `pin_memory()` 行为。

## 受限探针环境

项目 `.venv` 中观察到：

```text
torch 2.9.0
torch.version.hip 6.3.26093
torch.cuda.is_available() False
torch.cuda.device_count() 0
tensor.pin_memory() -> RuntimeError: No HIP GPUs are available
```

该执行环境看不到 `/dev/kfd` 和 `/dev/dri`，所以这里的“无 HIP GPU”是进程可见性结果，不是主机没有 HCU。

## 主机设备可见环境

可重跑命令：

```bash
cd /home/wangleping/codes/alchemi-hygon
source /opt/dtk-26.04/env.sh
.venv/bin/python - <<'PY'
import torch
print(torch.__version__, torch.version.hip)
print(torch.cuda.is_available(), torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
x = torch.empty(1)
print(x.pin_memory().is_pinned())
PY
```

实测输出要点：

```text
2.9.0 6.3.26093
True 8
BW200, UBB BW1000
True
```

显式加载 `/opt/dtk-26.04/env.sh` 后，项目 `.venv` 可以访问 HCU；该结果说明 HCU 支持共享使用。由于设备当前有其他作业，后续探针仍需限制显存和运行时间。

在 2026-09-05 的复核时，项目 `.venv` 尚未安装 pytest，因此使用已有探索环境重跑上次失败的单测：

```text
packages/framework/test/data/test_batch.py::TestBatchIndexing::test_pin_memory PASSED
1 passed, 94 deselected in 16.02s
```

当前项目环境已补装 `pytest==8.4.2`；后续项目回归结果记录在
`reports/g1-project-reference-pytest.md`，本历史段落保留原始复核条件。
