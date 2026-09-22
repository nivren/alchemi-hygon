#!/usr/bin/env bash
set -u

# Minimal two-process RCCL reproducer. It does not use PyTorch, torchrun, or MPI.
# Usage: ./min_rccl_repro.sh <physical_gpu0> <physical_gpu1>
# Example: ./min_rccl_repro.sh 6 7

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <physical_gpu0> <physical_gpu1>" >&2
  exit 2
fi

GPU0=$1
GPU1=$2
HIPCC=${HIPCC:-$(command -v hipcc || true)}
GPU_ARCH=${GPU_ARCH:-gfx936}
RCCL_INCLUDE=${RCCL_INCLUDE:-/opt/dtk-26.04/include}
RCCL_LIB=${RCCL_LIB:-/opt/dtk-26.04/lib}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-60}
OUT=${OUT:-/tmp/rccl-min-repro-$(date +%Y%m%d-%H%M%S)}

if [[ -z "$HIPCC" || ! -x "$HIPCC" ]]; then
  echo "hipcc not found; source the DTK environment or set HIPCC=/path/to/hipcc" >&2
  exit 3
fi
if [[ ! -f "$RCCL_INCLUDE/rccl/rccl.h" && ! -f "$RCCL_INCLUDE/rccl.h" ]]; then
  echo "RCCL header not found under $RCCL_INCLUDE; set RCCL_INCLUDE=..." >&2
  exit 3
fi
if [[ ! -e "$RCCL_LIB/librccl.so" && ! -e "$RCCL_LIB/librccl.so.1" ]]; then
  echo "RCCL library not found under $RCCL_LIB; set RCCL_LIB=..." >&2
  exit 3
fi
if [[ -e "$RCCL_LIB/librccl.so" ]]; then
  RCCL_LINK=(-lrccl)
else
  RCCL_LINK=("$RCCL_LIB/librccl.so.1")
fi

mkdir -p "$OUT"
SRC="$OUT/rccl_min.cpp"
BIN="$OUT/rccl_min"
ID_FILE="$OUT/unique_id"

cat > "$SRC" <<'CPP'
#include <hip/hip_runtime.h>
#if __has_include(<rccl/rccl.h>)
#include <rccl/rccl.h>
#else
#include <rccl.h>
#endif

#include <chrono>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <unistd.h>
#include <sys/stat.h>

static int hip_check(const char* op, hipError_t s, int rank) {
    if (s == hipSuccess) return 0;
    std::cerr << "rank=" << rank << " " << op << " failed: "
              << hipGetErrorName(s) << " (" << hipGetErrorString(s) << ")\n";
    return 1;
}

static int rccl_check(const char* op, ncclResult_t s, int rank) {
    if (s == ncclSuccess) return 0;
    std::cerr << "rank=" << rank << " " << op << " failed: "
              << ncclGetErrorString(s) << "\n";
    return 1;
}

static bool write_id(const std::string& path, const ncclUniqueId& id) {
    int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) return false;
    ssize_t n = write(fd, &id, sizeof(id));
    fsync(fd);
    close(fd);
    return n == static_cast<ssize_t>(sizeof(id));
}

static bool read_id(const std::string& path, ncclUniqueId& id) {
    struct stat st{};
    if (stat(path.c_str(), &st) != 0 ||
        st.st_size != static_cast<off_t>(sizeof(id))) return false;
    std::ifstream in(path, std::ios::binary);
    in.read(reinterpret_cast<char*>(&id), sizeof(id));
    return in.gcount() == static_cast<std::streamsize>(sizeof(id));
}

int main() {
    const int rank = std::atoi(std::getenv("RANK"));
    const std::string id_path = std::getenv("RCCL_UNIQUE_ID_FILE");
    int count = 0;
    if (hip_check("hipGetDeviceCount", hipGetDeviceCount(&count), rank)) return 1;
    if (rank < 0 || rank >= 2 || count < 2) {
        std::cerr << "rank=" << rank << " visible_device_count=" << count
                  << "; need two visible devices\n";
        return 2;
    }
    if (hip_check("hipSetDevice", hipSetDevice(rank), rank)) return 1;
    std::cout << "rank=" << rank << " device_count=" << count << std::endl;

    ncclUniqueId id{};
    if (rank == 0) {
        if (rccl_check("ncclGetUniqueId", ncclGetUniqueId(&id), rank) ||
            !write_id(id_path, id)) {
            std::cerr << "rank=0 failed to publish RCCL unique ID\n";
            return 3;
        }
    } else {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (!read_id(id_path, id) && std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (!read_id(id_path, id)) {
            std::cerr << "rank=1 timed out waiting for RCCL unique ID\n";
            return 4;
        }
    }

    ncclComm_t comm = nullptr;
    if (rccl_check("ncclCommInitRank", ncclCommInitRank(&comm, 2, id, rank), rank)) return 1;
    std::cout << "rank=" << rank << " communicator=initialized" << std::endl;

    hipStream_t stream = nullptr;
    float send = static_cast<float>(rank + 1), recv = 0.0f;
    float *send_dev = nullptr, *recv_dev = nullptr;
    if (hip_check("hipStreamCreate", hipStreamCreate(&stream), rank) ||
        hip_check("hipMalloc(send)", hipMalloc(&send_dev, sizeof(float)), rank) ||
        hip_check("hipMalloc(recv)", hipMalloc(&recv_dev, sizeof(float)), rank) ||
        hip_check("hipMemcpyH2D", hipMemcpy(send_dev, &send, sizeof(float), hipMemcpyHostToDevice), rank))
        return 1;

    std::cout << "rank=" << rank << " before_ncclAllReduce" << std::endl;
    if (rccl_check("ncclAllReduce", ncclAllReduce(send_dev, recv_dev, 1, ncclFloat,
                                                    ncclSum, comm, stream), rank)) return 1;
    if (hip_check("hipStreamSynchronize", hipStreamSynchronize(stream), rank)) return 1;
    if (hip_check("hipMemcpyD2H", hipMemcpy(&recv, recv_dev, sizeof(float), hipMemcpyDeviceToHost), rank)) return 1;
    std::cout << "rank=" << rank << " after_ncclAllReduce value=" << recv << std::endl;
    return 0;
}
CPP

echo "[build] $HIPCC -> $BIN"
"$HIPCC" -O0 -g -std=c++17 --offload-arch="$GPU_ARCH" \
  -I"$RCCL_INCLUDE" "$SRC" -L"$RCCL_LIB" \
  -Wl,-rpath,"$RCCL_LIB" "${RCCL_LINK[@]}" -lamdhip64 -o "$BIN" || exit 5

run_rank() {
  local rank=$1
  HIP_VISIBLE_DEVICES="$GPU0,$GPU1" RANK="$rank" RCCL_UNIQUE_ID_FILE="$ID_FILE" \
    NCCL_DEBUG="${NCCL_DEBUG:-INFO}" NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,GRAPH,NET}" \
    timeout --kill-after=5 "$TIMEOUT_SECONDS" "$BIN" >"$OUT/rank${rank}.log" 2>&1
  echo $? >"$OUT/rank${rank}.rc"
}

echo "[run] physical GPUs $GPU0,$GPU1; logs: $OUT"
run_rank 0 & p0=$!
run_rank 1 & p1=$!
wait "$p0" || true
wait "$p1" || true

echo "[result] rank0_rc=$(cat "$OUT/rank0.rc" 2>/dev/null || echo missing) rank1_rc=$(cat "$OUT/rank1.rc" 2>/dev/null || echo missing)"
echo "[result] expected failure marker: before_ncclAllReduce without after_ncclAllReduce"
cat "$OUT/rank0.log" "$OUT/rank1.log"
echo "[result] inspect device state now with: hy-smi"
