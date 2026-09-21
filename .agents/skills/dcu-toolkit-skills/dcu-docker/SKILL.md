---
name: dcu-docker
description: "Use when operating DCU environments inside Docker — covers: (1) launching DCU images with correct device passthrough/driver mount (docker run template); (2) container-to-container passwordless SSH, ONLY for cross-node scenarios (multi-node collective-comm tests, multi-node distributed inference); (3) the docker exec launch-method performance pitfall (must use -ilc not -c or benchmark numbers lie). Portable across AI agent platforms."
license: MIT
metadata:
  version: 1.0.0
  author: DCU Engineer
  hermes:
    tags: [dcu, docker, ssh, passwordless, infrastructure, container]
    related_skills: [dcu-ssh, dcu-env, dcu-ontrack-experience]
---

# DCU Docker 操作 — 容器 SSH 免密配置

## Overview

海光DCU的软件环境基本都跑在 Docker 容器里（DTK 镜像、训练/推理镜像等）。
本 skill 覆盖三类 DCU 容器高频操作：
1. **DCU 镜像启动（docker run）** — 设备透传、驱动挂载、shm、模型/代码目录挂载的正确模板
2. **容器间 SSH 免密** — 仅限跨节点场景（多节点集合通信测试、多节点分布式推理）
3. **测评执行方式坑** — `docker exec` 启动方式直接影响推理性能，必须用 `-ilc`

核心参考：CSDN《两台服务器上的两个docker容器之间配置ssh免密登录》
https://blog.csdn.net/weixin_45973213/article/details/134891512

## When to Use

> 📌 **遇到具体报错先查排查经验库**：容器库缺失（`libxxx.so`）、编译失败、numpy/transformers 版本冲突等具体现象的处理经验（含 env/修复命令），统一收在 **`dcu-ontrack-experience`** 这个 skill 里。本 skill 只讲容器启动模板与免密配置，具体踩坑手法请去那里按"现象关键词"反查。

**仅在以下两类场景主动配置容器内 SSH 免密**（其余场景不主动配，用 `docker exec` 进容器即可）：
1. **多设备节点跑集合通信测试** — 多节点多卡，节点间需互信才能跑通集合通信（如 RCCL/通测工具）
2. **多节点一起跑模型推理** — 分布式推理跨节点，需节点间免密互信

Don't use for：
- 单节点 / 单机多卡 的容器操作（用 `docker exec` 进容器即可，无需 sshd）
- 宿主机本身的 SSH 配置（见 dcu-ssh）
- 纯本地 docker 构建/编排（非 DCU 相关）
- 临时进容器敲命令、看状态（不需要免密）

> 判定标准：只有"**跨节点**需要节点间互联通信"时才配容器免密；单机内多卡不需要。

## DCU 镜像启动（docker run）

DCU 容器必须用正确的设备透传、共享内存、驱动挂载参数才能跑起来。以下为实测可用的启动模板（来源：实际 vLLM on DCU 部署命令）。

### 标准启动命令模板
```bash
docker run -itd \
    --name <container> \
    --shm-size=16G \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --network=host \
    --ipc=host \
    --privileged \
    --device=/dev/kfd \
    --device=/dev/mkfd \
    --device=/dev/dri \
    -v /opt/hyhal:/opt/hyhal:ro \
    -v /root/.ssh:/root/.ssh \
    -v /public:/public -v /data:/data -v /home:/home \
    <internal_registry>/dcu/admin/base/vllm:<tag> \
    /bin/bash
```

### 关键参数说明
| 参数 | 作用 | 必要性 |
|---|---|---|
| `--device=/dev/kfd` | 透传 DCU 内核驱动设备（计算设备） | **必需** |
| `--device=/dev/mkfd` | 透传 DCU 管理设备 | **必需** |
| `--device=/dev/dri` | 透传 DRM 渲染设备 | **必需** |
| `-v /opt/hyhal:/opt/hyhal:ro` | 挂载 DCU 驱动/运行时（只读，宿主机已装 hyhal） | **必需**（容器靠它识别DCU） |
| `--shm-size=16G` | 扩大共享内存，避免多进程/多卡通信 shm 不足 | 推荐（默认64M太小） |
| `--group-add video` | 加入 video 组，访问 /dev/dri 等设备 | 必需 |
| `--cap-add=SYS_PTRACE` | 允许 ptrace（调试/性能分析、部分框架需要） | 推荐 |
| `--security-opt seccomp=unconfined` | 关闭 seccomp 限制，DCU 驱动需要 | 推荐 |
| `--network=host` | 主机网络，分布式训练/服务端口直通 | 多卡/服务必用 |
| `--ipc=host` | 共享 IPC 命名空间（多进程共享内存通信） | 多卡必用 |
| `--privileged` | 特权模式（驱动透传常需） | 常用，注意安全 |
| `-v /root/.ssh:/root/.ssh` | 挂载宿主机 ssh 密钥，便于容器内免密/拉代码 | 按需（配合免密章节） |

### ⚠️ 启动容器前先查宿主机剩余空间（重要）

拉取镜像（常 10~30 GB）+ 起容器可写层会持续吃盘，物理机上若已叠加多个业务容器，极易把盘写满导致 `docker pull` / 运行失败。**起容器前先确认目标盘剩余足够：**
```bash
df -h / /data /public                 # 按实际挂卷/ docker root 所在盘替换
docker system df                      # 看镜像/容器/卷已占用
docker images --format '{{.Repository}}:{{.Tag}} {{.Size}}' | sort -k2 -h
```
- 剩余 < 20 GB：先确认无更大写入；< 10 GB 先 `docker image prune -f` / 删不再用的旧容器再继续。
- docker 根目录所在盘以 `docker info | grep -i 'docker root'` 为准，不一定是 `/`。

### 镜像命名约定（建议）
`<user>-<框架><版本>-<日期>`，如 `<user>-vllm0110-20260204`：
- 含用户、框架版本、构建日期，方便多容器并存时区分
- 同名会冲突，重建前先 `docker rm` 旧容器

### `-v` 挂载模型与代码目录（重要补充）
上述模板挂载了 `/public /data /home` 等通用目录。实际使用时**必须额外挂载模型目录和代码目录**到容器内，否则容器内读不到本地模型/代码：
```bash
# 在上面的 -v 列表基础上追加（按实际路径替换）：
    -v /path/to/your/models:/models \
    -v /path/to/your/code:/workspace/code \
```
- **模型目录**：如 HuggingFace 权重、vLLM 模型缓存，挂载后容器内直接 `/models/xxx` 访问
- **代码目录**：训练/推理脚本、自研代码，挂载后容器内 `/workspace/code` 可直接改直接用
- 挂载点路径在容器内自定义，但建议语义清晰（`/models`、`/workspace/code`）
- 若模型在对象存储，也可容器内直接拉，但本地挂载更省带宽、可离线

### 启动后进入容器
```bash
docker exec -it <container> bash
# 验证 DCU 可见：
hy-smi          # 应列出所有 DCU 卡
```
> 若 `hy-smi` 报找不到命令，需 `source /opt/dtk/env.sh`（见 dcu-env）。

## 关键原则（务必先读）

0. **使用前提（最重要）**：容器免密**只**在"多设备节点跑集合通信测试"或"多节点一起跑模型推理"这两个跨节点场景才主动配置。单节点/单机多卡/临时进容器都**不配**，直接用 `docker exec` 即可。配免密=要跨节点互联通信时才做。
1. **容器内 SSH 端口必须与宿主机区分**：宿主机 sshd 通常占 22，容器内若也用 22 会导致 `ssh` 连到宿主机而非容器。→ 容器内 sshd 改用 **2233**（或其他非22端口）。
2. **先切 root 装/启 sshd，再切回业务用户生成密钥**：openssh 安装和 sshd 启动需要 root；密钥对用业务用户（如 `<user>`）身份生成，authorized_keys 也放在该用户家目录。
3. **互信 = 双方公钥交换进同一份 authorized_keys**：A、B 两容器各自的 `id_*.pub` 内容都写进对方的 `~/.ssh/authorized_keys`，且两边 authorized_keys 内容一致。
4. **所有步骤在容器内部执行**（不是宿主机），除非你明确是在配宿主机→容器的端口映射。

## Standard Procedure（容器内 SSH 免密，容器↔容器）

以下步骤均在**目标容器内部**执行：

### Step 1. 切 root + 安装 openssh
```bash
sudo -i            # 或 su -，切到 root
apt-get update && apt-get install -y openssh-client openssh-server
# 若基础镜像是 rpm 系（openEuler/Kylin）：yum install -y openssh-clients openssh-server
```

### Step 2. 改容器内 sshd 端口（避开宿主机22）
```bash
vim /etc/ssh/sshd_config
# 在文件末尾加：
Port 2233
```
> 目的：容器内 ssh 端口改为 2233，与宿主机默认 22 区分。不改的话，`ssh` 会默认连到容器外宿主机。

### Step 3. 启动/重启 sshd
```bash
service ssh restart          # Debian/Ubuntu 系
# 或：/usr/sbin/sshd          # 直接拉起守护进程（若无 service 命令）
```

### Step 4. 切回业务用户，生成密钥对
```bash
su - <user>                     # 切到业务用户（按需替换用户名）
ssh-keygen -t ed25519        # -t ed25519 生成短公钥；也可 rsa。一路回车不设口令
# 生成 ~/.ssh/id_ed25519 和 id_ed25519.pub
```

### Step 5. 交换公钥，建立互信
将 A 容器的 `~/.ssh/id_ed25519.pub` 内容，追加进 B 容器的 `~/.ssh/authorized_keys`；
同时将 B 的公钥追加进 A 的 authorized_keys。**最终两边 authorized_keys 内容相同**（各含两容器公钥）。
```bash
# 在 A 容器：
cat ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys
# 把 B 的公钥也写进来（可从 B scp 过来或手动粘贴）
chmod 600 ~/.ssh/authorized_keys
chmod 700 ~/.ssh
```
> 多容器/多机场景：N 个节点就把 N 个公钥全部汇进同一份 authorized_keys，再分发到每个节点。

### Step 6. 配置 SSH config 别名（可选但推荐）
在 `~/.ssh/config` 写入：
```
Host r1d
    HostName 10.0.0.1
    User <user>
    Port 2233
Host r4d
    HostName 10.0.0.4
    User <user>
    Port 2233
```
之后直接 `ssh r4d` 即可免密登录（首次 `yes` 接受指纹后不再提示）。

## 变体场景

### 场景 A：宿主机免密进容器（仅在确实需要 sshd 接入时才配）
- 多数情况用 `docker exec` 即可，不需 sshd；仅在需从跳板机/外部通过 ssh 接入容器时才配
- 容器内按 Step1–3 起 sshd（端口 2233），宿主机 `ssh -p 2233 <user>@<容器IP>`
- 把宿主机公钥加进容器 authorized_keys 即可单向免密（注意：这不属于"跨节点互信"主场景，按需处理）

### 场景 B：多机多卡分布式训练（最常见 DCU 需求）
- 每个节点的训练容器都按上述配好互信（所有 rank 节点的公钥汇进同一 authorized_keys 并分发）
- 确保各容器间 `ssh <其他容器IP> -p 2233` 能免密通
- 再跑 torchrun / horovod 等分布式框架

### 场景 C：容器重启后 sshd 不自动起
- 很多 DCU 镜像不把 sshd 设为自启。重启容器后需重新 `service ssh restart`
- 可在容器启动脚本（entrypoint / `~/.bashrc` 末尾）加 `service ssh start` 保活

## 测评执行方式（性能测试坑 — 重要）

在 DCU 容器里跑大模型推理/性能测评时，**`docker exec` 的启动方式会直接影响测得性能**，这是一个反复踩过的深坑。

### 三种启动方式对比（实测数据）

| 方式 | 命令 | 是否读 .bashrc/环境变量 | Req/s | Out tok/s |
|---|---|---|---|---|
| 方式1 ❌ | `docker exec -d bash -c '...'` | **不读**（非交互非登录 shell，跳过 .bashrc） | 43.59 | 87.17 |
| 方式2 ✅ | `docker exec -ti <容器> bash` 然后 `bash xxx.sh` | 读（交互 shell 会 source profile） | 51.36 | 102.72 |
| 方式3 ✅ | `docker exec -d bash -ilc '...'` | 读（`-i -l` 强制交互+登录，加载 .bashrc/.profile） | 52.26 | 104.53 |

> 方式1 比方式2/3 **慢约 15–20%**（Duration 11.75s vs 9.8–10s，吞吐掉一截）。

### 根因
- `docker exec -d bash -c 'cmd'`（方式1）是**非交互、非登录** shell，**不会执行 `/root/.bashrc` / `/etc/profile` / `~/.profile`**。
- DCU 的 **rocblas / hipblaslt** 等加速库路径依赖 `.bashrc` 里 `source /opt/dtk/env.sh`（或等价导出 `LD_LIBRARY_PATH`）来正确加载。
- 方式1 跳过了 `.bashrc` → **rocblas/hipblaslt 未正确加载**（可能 fallback 到慢路径或错误版本）→ 推理性能明显劣化。
- 方式2/3 都进入了会加载环境变量的 shell（方式2 是 `-ti` 交互 shell；方式3 是 `-i -l` 强制登录+交互），库正常加载，性能满血。

### 解决方案（强制规范）
**测评一律用方式3：**
```bash
docker exec -d bash -ilc 'bash /path/to/bench.sh'
# 或带完整命令：
docker exec -d bash -ilc 'source /opt/dtk/env.sh && python infer.py'
```
- `-i` 交互 shell + `-l` 登录 shell → 保证 `.bashrc`/`.profile` 被执行，rocblas/hipblaslt 正确加载
- `-d` 后台运行（测评通常跑批，不需要前台占用）
- 等价做法：方式2（`docker exec -ti` 进容器手动跑）也可，但不适合自动化/后台批跑，且易因终端断开中断

### 关联
- 环境变量加载细节见 dcu-env（DTK env）。若容器内 `.bashrc` 未包含 `source /opt/dtk/env.sh`，即使 `-ilc` 也可能缺库——先确认 `.bashrc` 已正确配置。
- 同理，**所有性能相关命令**（benchmark、集合通信测试）都应走 `-ilc`，否则测出的数据不可信。

## Common Pitfalls

1. **容器内 ssh 连到了宿主机** — 忘记改端口。容器内 sshd 必须用非22端口（如2233），连接时显式 `-p 2233` 或靠 config 的 `Port` 字段。
2. **root 生成的密钥无法用于业务用户登录** — 密钥和 authorized_keys 必须属于实际登录的业务用户家目录 `~/.ssh/`，权限 700/600。
3. **authorized_keys 权限过宽导致 ssh 拒绝** — 必须 `chmod 700 ~/.ssh` + `chmod 600 ~/.ssh/authorized_keys`，否则 sshd 安全策略会忽略。
4. **容器重启 sshd 没起来** — 镜像默认不自启 sshd，需手动 restart 或加到启动脚本。
5. **多节点 authorized_keys 内容不一致** — N 个节点要汇总全部公钥再统一分发，漏一个节点就会有一对无法互信。
6. **混淆"宿主机 ssh"与"容器内 ssh"** — DCU 操作绝大多数在容器内；配免密前先确认你进的是容器（`docker exec -it <cid> bash` 或 `hostname` 核对）。
7. **DTK 环境变量** — 配完免密后执行 DCU 命令，仍要 `source /opt/dtk/env.sh`（见 dcu-env）。
8. **docker run 漏挂 DCU 设备/驱动** — 缺 `--device=/dev/kfd`（或 mkfd/dri）或漏挂 `/opt/hyhal` 会导致容器内 `hy-smi` 看不到卡；必须按"DCU 镜像启动"章节参数全量带上。
9. **忘挂模型/代码目录** — 只用模板的 `/public /data /home` 不够，实际必须额外 `-v` 挂载模型权重与代码目录，否则容器内读不到本地资源。
10. **`--shm-size` 没设或太小** — 默认 64M 在多卡/多进程下会 shm 不足报错，DCU 容器务必 `--shm-size=16G`（或更大）。

## Verification Checklist

- [ ] 容器内 openssh 已安装（client+server）
- [ ] 容器内 sshd 端口已改为非22（如2233）且 `service ssh restart` 成功
- [ ] 业务用户已生成密钥对（~/.ssh/id_ed25519*）
- [ ] 互信双方 authorized_keys 含彼此公钥，权限 700/600 正确
- [ ] `ssh -p 2233 <对方> hostname` 能免密返回对方主机名（首次 yes 后不再提示）
- [ ] （多节点）所有 rank 节点两两免密通过
- [ ] 容器重启后 sshd 仍可用（或已加入自启）
- [ ] （docker run）DCU 设备（kfd/mkfd/dri）+ `/opt/hyhal` 已透传，`hy-smi` 能列出全部卡
- [ ] （docker run）模型目录与代码目录已 `-v` 挂载进容器
- [ ] （docker run）`--shm-size=16G` 已设置

## References

- CSDN 原文（步骤来源）：https://blog.csdn.net/weixin_45973213/article/details/134891512
