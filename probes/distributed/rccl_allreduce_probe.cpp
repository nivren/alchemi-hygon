// Minimal direct RCCL all-reduce probe for vendor escalation.
//
// A shared temporary file is used only to exchange ncclUniqueId and rank
// metadata. The device collective is a direct RCCL ncclAllReduce call;
// PyTorch and MPI are not involved.
// The program is intentionally one bounded, one-element collective.
//
// Expected launch shape on a single node is shown in probes/distributed/README.md.
// Wrap each process with timeout(1) and capture stdout/stderr.

#include <hip/hip_runtime.h>
#include <rccl/rccl.h>

#include <chrono>
#include <cstdlib>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <unistd.h>
#include <sys/stat.h>

namespace {

int fail_hip(const char* operation, hipError_t status, int rank) {
    std::cerr << "rank=" << rank << ' ' << operation << " failed: "
              << hipGetErrorName(status) << " (" << hipGetErrorString(status) << ")\n";
    return 1;
}

int fail_rccl(const char* operation, ncclResult_t status, int rank) {
    std::cerr << "rank=" << rank << ' ' << operation << " failed: "
              << ncclGetErrorString(status) << "\n";
    return 1;
}

#define HIP_CHECK_RANK(operation, rank)          \
    do {                                         \
        const hipError_t status = (operation);   \
        if (status != hipSuccess) {              \
            return fail_hip(#operation, status, rank); \
        }                                        \
    } while (false)

#define RCCL_CHECK_RANK(operation, rank)         \
    do {                                         \
        const ncclResult_t status = (operation); \
        if (status != ncclSuccess) {             \
            return fail_rccl(#operation, status, rank); \
        }                                        \
    } while (false)

bool read_unique_id(const std::string& path, ncclUniqueId* unique_id) {
    struct stat metadata{};
    if (stat(path.c_str(), &metadata) != 0 || metadata.st_size != static_cast<off_t>(sizeof(*unique_id))) {
        return false;
    }
    std::ifstream input(path, std::ios::binary);
    input.read(reinterpret_cast<char*>(unique_id), sizeof(*unique_id));
    return input.good() || input.gcount() == static_cast<std::streamsize>(sizeof(*unique_id));
}

bool write_unique_id(const std::string& path, const ncclUniqueId& unique_id) {
    const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) {
        return false;
    }
    const auto* bytes = reinterpret_cast<const char*>(&unique_id);
    const ssize_t written = write(fd, bytes, sizeof(unique_id));
    fsync(fd);
    close(fd);
    return written == static_cast<ssize_t>(sizeof(unique_id));
}

}  // namespace

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    const char* rank_text = std::getenv("RANK");
    const char* world_size_text = std::getenv("WORLD_SIZE");
    const char* id_file_text = std::getenv("RCCL_UNIQUE_ID_FILE");
    const int rank = rank_text == nullptr ? -1 : std::atoi(rank_text);
    const int world_size = world_size_text == nullptr ? -1 : std::atoi(world_size_text);
    const std::string id_file = id_file_text == nullptr ? "/tmp/rccl_unique_id.bin" : id_file_text;
    if (rank < 0 || world_size < 0) {
        std::cerr << "RANK and WORLD_SIZE must be set\n";
        return 2;
    }
    if (world_size != 2) {
        if (rank == 0) {
            std::cerr << "expected exactly 2 MPI ranks, got " << world_size << '\n';
        }
        return 2;
    }

    int device_count = -1;
    HIP_CHECK_RANK(hipGetDeviceCount(&device_count), rank);
    std::cout << "rank=" << rank << " world_size=" << world_size
              << " device_count=" << device_count << std::endl;
    if (rank >= device_count) {
        std::cerr << "rank=" << rank << " has no matching visible device\n";
        return 3;
    }
    HIP_CHECK_RANK(hipSetDevice(rank), rank);

    ncclUniqueId unique_id{};
    if (rank == 0) {
        RCCL_CHECK_RANK(ncclGetUniqueId(&unique_id), rank);
        if (!write_unique_id(id_file, unique_id)) {
            std::cerr << "rank=0 failed to write RCCL unique ID file: " << id_file << '\n';
            return 4;
        }
    } else {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        while (!read_unique_id(id_file, &unique_id) && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if (!read_unique_id(id_file, &unique_id)) {
            std::cerr << "rank=" << rank << " timed out waiting for RCCL unique ID file: " << id_file << '\n';
            return 5;
        }
    }

    ncclComm_t communicator = nullptr;
    RCCL_CHECK_RANK(ncclCommInitRank(&communicator, world_size, unique_id, rank), rank);
    std::cout << "rank=" << rank << " communicator=initialized" << std::endl;

    hipStream_t stream = nullptr;
    HIP_CHECK_RANK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), rank);
    float host_send = static_cast<float>(rank + 1);
    float host_recv = 0.0F;
    float* device_send = nullptr;
    float* device_recv = nullptr;
    HIP_CHECK_RANK(hipMalloc(&device_send, sizeof(float)), rank);
    HIP_CHECK_RANK(hipMalloc(&device_recv, sizeof(float)), rank);
    HIP_CHECK_RANK(hipMemcpyAsync(device_send, &host_send, sizeof(float), hipMemcpyHostToDevice, stream), rank);
    HIP_CHECK_RANK(hipStreamSynchronize(stream), rank);

    std::cout << "rank=" << rank << " before_ncclAllReduce" << std::endl;
    RCCL_CHECK_RANK(
        ncclAllReduce(device_send, device_recv, 1, ncclFloat, ncclSum, communicator, stream),
        rank);
    const hipError_t sync_status = hipStreamSynchronize(stream);
    if (sync_status != hipSuccess) {
        return fail_hip("hipStreamSynchronize(after ncclAllReduce)", sync_status, rank);
    }
    HIP_CHECK_RANK(hipMemcpy(&host_recv, device_recv, sizeof(float), hipMemcpyDeviceToHost), rank);
    std::cout << "rank=" << rank << " after_ncclAllReduce value=" << host_recv << std::endl;

    ncclResult_t asynchronous_error = ncclSuccess;
    RCCL_CHECK_RANK(ncclCommGetAsyncError(communicator, &asynchronous_error), rank);
    std::cout << "rank=" << rank << " async_error="
              << ncclGetErrorString(asynchronous_error) << std::endl;

    HIP_CHECK_RANK(hipFree(device_send), rank);
    HIP_CHECK_RANK(hipFree(device_recv), rank);
    HIP_CHECK_RANK(hipStreamDestroy(stream), rank);
    RCCL_CHECK_RANK(ncclCommDestroy(communicator), rank);
    std::cout << "rank=" << rank << " status=passed" << std::endl;
    return 0;
}
