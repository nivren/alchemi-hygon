# dcu-docker — 海光 DCU 容器操作

DCU 容器 Docker 操作 skill。定义带 DCU 设备的容器运行规范、容器内执行命令的注意点
（DTK 环境变量加载、numactl/SYS_NICE 权限）、调试挂载要求。

## 核心踩坑（实测确认）

1. **`.bashrc` 不自动加载**：`docker exec bash -c '...'`（非交互非登录）跳过 `.bashrc`，
   导致 rocblas/hipblaslt 未加载，推理性能掉 15-20%。容器内跑测评须用
   `docker exec -d bash -ilc '...'`（交互登录 shell 强制加载 DTK env）。
2. **DCU 设备挂载**：容器需 `--device=/dev/kfd --device=/dev/dri` 或 `--privileged`。
3. **6.3.x 调试挂载**：必须额外挂 `/sys/kernel/debug`，否则容器内 hy-smi `--showpids`
   等无法获取 DCU 进程信息。
4. **绑核权限**：容器内 numactl 需 `--cap-add SYS_NICE` 或 `--privileged`。

## 快速用法

```bash
# 正确跑测评（加载 DTK env）
docker exec -d <容器> bash -ilc 'python train.py'

# 带 DCU 设备 + 调试挂载起容器
docker run --privileged --device=/dev/kfd --device=/dev/dri \
  -v /sys/kernel/debug:/sys/kernel/debug <镜像>
```

详见 `SKILL.md`。
