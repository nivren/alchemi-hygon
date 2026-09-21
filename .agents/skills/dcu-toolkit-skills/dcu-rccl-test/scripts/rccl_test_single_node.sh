#!/bin/bash
# rccl_test_single_node.sh — DCU 单节点 RCCL 通测一键流程（经验固化，非日志）
#
# 用法（在能直连目标机的跳板机/本地执行，依赖 sshpass + docker + 已配 credentials.json）：
#   TARGET=<device_ip>  PASS=$(python3 -c "...")  IMG=<用户指定镜像>  \
#     bash rccl_test_single_node.sh
#
# 本脚本固化 2026-07-22 实测踩坑后的正确流程，供下个会话直接复用，避免重复踩坑：
#   1. 容器必须挂 -v /opt/hyhal:/opt/hyhal:ro ，否则 hipGetDeviceCount=0（卡不可见）
#   2. gfx 用【物理机】rocminfo 查（带 ROCM_PATH/LD_LIBRARY_PATH），容器内 rocminfo 会把 DCU 认成 CPU
#   3. 编译与运行都在【容器内】做，物理机编译会因 /opt/dtk 软链错位失败
#   4. 跑 perf 前先查 NCCL_NCHANNELS_PER_PEER，有则 unset（镜像常预置 =2 污染基线）
#   5. 指定卡用 HIP_VISIBLE_DEVICES=0,1,...（编号从0），不要只靠 -g N 让 RCCL 自选
#   6. 日志重定向到容器内 /tmp/x.log，再 docker cp 到宿主机（容器/宿主机 /tmp 不共享）
#
# 注意：本脚本不内嵌任何性能数值；结果日志由用户自行查看，不写进 skill。

set -e
: "${TARGET:?需设置 TARGET=IP}"
: "${PASS:?需设置 PASS=root密码}"
: "${IMG:?需设置 IMG=用户指定的测试镜像全名}"
: "${SSHUSER:=root}"
CONTNAME="rccl-test-$(echo $TARGET | tr '.' '-')"
RCCL_COMMIT="5e838ad9df47079e0e586ed38049f4f579ea462d"   # BW1000/BW150 用这个

export SSHPASS="$PASS"
SSH="sshpass -e ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p 22 $SSHUSER@$TARGET"
SCP="sshpass -e scp -o StrictHostKeyChecking=no -P 22"

# 1) 物理机查 gfx（必带 ROCM_PATH/LD_LIBRARY_PATH，否则无输出）
echo "== [host] 查 gfx =="
$SSH "for d in /opt/dtk-*/bin/rocminfo; do echo ROCM=\$d; ROCM_PATH=\$(dirname \$(dirname \$d)); export ROCM_PATH; export LD_LIBRARY_PATH=\$ROCM_PATH/lib:\$LD_LIBRARY_PATH; \$d 2>&1 | grep -oiE 'gfx[0-9]{3}' | sort -u; done"

# 1.5) 空间检查（拉镜像/编译会占额外空间，盘满会静默失败）
echo "== [host] 检查磁盘剩余空间 =="
SPACE_GB=$($SSH "df -BG / /data /public 2>/dev/null | awk 'NR>1{print \$4+0}' | sort -n | head -1")
echo "目标盘最小剩余: ${SPACE_GB}G"
if [ "${SPACE_GB:-0}" -lt 10 ]; then
  echo "ERROR: 目标盘剩余 < 10G，拒绝继续；请先 docker image prune -f / 删旧测试容器释放空间" >&2
  exit 1
elif [ "${SPACE_GB:-0}" -lt 20 ]; then
  echo "WARN: 目标盘剩余 < 20G，确认无更大写入后再继续" >&2
fi

# 2) 起容器（必挂 /opt/hyhal）
echo "== [host] 起容器 $CONTNAME =="
$SSH "docker rm -f $CONTNAME 2>/dev/null; docker run -d --name $CONTNAME --network=host --ipc=host \
  -v /opt/dtk:/opt/dtk -v /opt/mpi:/opt/mpi -v /opt/hyhal:/opt/hyhal:ro \
  --device=/dev/kfd --device=/dev/dri --privileged $IMG sleep infinity"

# 3) 容器内：探环境 + 验 8 卡可见 + 查/清 NCCL_NCHANNELS_PER_PEER + 编译
echo "== [container] 探环境/验卡/清NCCL通道/编译 =="
$SSH "docker exec $CONTNAME bash -c '
source /opt/dtk/env.sh 2>/dev/null
echo DTK=\$(readlink -f /opt/dtk)
echo NCCL_NCHANNELS_PER_PEER=${NCCL_NCHANNELS_PER_PEER:-<unset>}
[ -n \"\$NCCL_NCHANNELS_PER_PEER\" ] && unset NCCL_NCHANNELS_PER_PEER
hipcc /dev/stdin -o /tmp/dc <<EOF
#include <hip/hip_runtime.h>
#include <stdio.h>
int main(){int n;hipGetDeviceCount(&n);printf(\"DEVCOUNT=%d\n\",n);for(int i=0;i<n;i++){hipDeviceProp_t p;hipGetDeviceProperties(&p,i);printf(\" gpu%d %s\n\",i,p.gcnArchName);}return 0;}
EOF
/tmp/dc
rm -rf /root/rccl-tests
git clone https://github.com/ROCm/rccl-tests.git -b master /root/rccl-tests
cd /root/rccl-tests && git reset --hard $RCCL_COMMIT
make ROCM_HOME=/opt/dtk NCCL_HOME=/opt/dtk/rccl CUSTOM_RCCL_LIB=/opt/dtk/rccl/lib/librccl.so MPI=1 MPI_HOME=/opt/mpi -j8
echo BUILD_DONE
'"

# 4) 跑测试：2/4/8 卡 all_reduce + 8 卡 alltoall，日志落容器内 /tmp 再 docker cp
echo "== [container] 跑 2/4/8 卡 all_reduce + 8 卡 alltoall =="
$SSH "docker exec $CONTNAME bash -c '
source /opt/dtk/env.sh 2>/dev/null
[ -n \"\$NCCL_NCHANNELS_PER_PEER\" ] && unset NCCL_NCHANNELS_PER_PEER
export HSA_FORCE_FINE_GRAIN_PCIE=1
export NCCL_TOPO_FILE=null
cd /root/rccl-tests
export HIP_VISIBLE_DEVICES=0,1;        ./build/all_reduce_perf -b 2 -e 2G -f 2 -g 2 > /tmp/ar2c.log 2>&1
export HIP_VISIBLE_DEVICES=0,1,2,3;    ./build/all_reduce_perf -b 2 -e 2G -f 2 -g 4 > /tmp/ar4c.log 2>&1
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; ./build/all_reduce_perf -b 2 -e 2G -f 2 -g 8 > /tmp/ar8c.log 2>&1
                                     ./build/alltoall_perf -b 2 -e 2G -f 2 -g 8 > /tmp/a2a8c.log 2>&1
echo TEST_DONE
'"
for f in ar2c ar4c ar8c a2a8c; do
  $SSH "docker cp $CONTNAME:/tmp/$f.log /tmp/$f.log"
  $SCP "$SSHUSER@$TARGET:/tmp/$f.log" "/tmp/$f.log"
  echo "== $f.log 已拉到本地 /tmp/$f.log（完整原始日志，请直接查阅，不进 skill）=="
done

echo "== 完成。容器 $CONTNAME 保留供查看；如需删除手动 docker rm -f $CONTNAME =="
