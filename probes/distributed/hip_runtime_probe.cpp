// Minimal HIP/DTK runtime probe for vendor escalation.
//
// This intentionally does not use PyTorch, RCCL, MPI, or any project package.
// Run one process per selected device, for example:
//   HIP_VISIBLE_DEVICES=0 ./hip_runtime_probe 0
//
#include <hip/hip_runtime.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>

namespace {

int fail(const char* operation, hipError_t status) {
    std::cerr << operation << " failed: " << hipGetErrorName(status) << " ("
              << hipGetErrorString(status) << ")\n";
    return 1;
}

#define HIP_CHECK(operation)                     \
    do {                                         \
        const hipError_t status = (operation);   \
        if (status != hipSuccess) {              \
            return fail(#operation, status);     \
        }                                        \
    } while (false)

}  // namespace

int main(int argc, char** argv) {
    const int requested_device = argc > 1 ? std::atoi(argv[1]) : 0;
    int device_count = -1;
    HIP_CHECK(hipGetDeviceCount(&device_count));

    std::cout << "hip_device_count=" << device_count << '\n';
    std::cout << "requested_device=" << requested_device << '\n';
    if (requested_device < 0 || requested_device >= device_count) {
        std::cerr << "requested device is outside the visible range\n";
        return 2;
    }

    HIP_CHECK(hipSetDevice(requested_device));

    hipDeviceProp_t properties{};
    HIP_CHECK(hipGetDeviceProperties(&properties, requested_device));
    std::cout << "device_name=" << properties.name << '\n';
    std::cout << "total_global_mem=" << properties.totalGlobalMem << '\n';

    float* device_value = nullptr;
    const float host_value = 1.0F;
    float round_trip_value = 0.0F;
    HIP_CHECK(hipMalloc(&device_value, sizeof(host_value)));
    HIP_CHECK(hipMemcpy(device_value, &host_value, sizeof(host_value), hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(&round_trip_value, device_value, sizeof(round_trip_value), hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(device_value));

    std::cout << std::setprecision(8) << "round_trip_value=" << round_trip_value << '\n';
    std::cout << "status=passed\n";
    return 0;
}
