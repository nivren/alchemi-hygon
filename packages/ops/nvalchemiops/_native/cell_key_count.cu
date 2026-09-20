#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <hip/hip_runtime.h>

__global__ void cell_key_count_kernel(const int32_t* cell_keys,
                                      int32_t* cell_counts,
                                      int64_t num_atoms) {
  const int64_t atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom < num_atoms) {
    atomicAdd(cell_counts + cell_keys[atom], 1);
  }
}

void cell_key_count_hip(torch::Tensor cell_keys, torch::Tensor cell_counts) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (cell_counts.numel() > 0) {
    C10_CUDA_CHECK(hipMemsetAsync(cell_counts.data_ptr<int32_t>(), 0,
                                  cell_counts.nbytes(), stream));
  }
  const auto num_atoms = cell_keys.size(0);
  if (num_atoms == 0) {
    return;
  }
  constexpr int threads = 256;
  const int blocks = static_cast<int>((num_atoms + threads - 1) / threads);
  cell_key_count_kernel<<<blocks, threads, 0, stream>>>(
      cell_keys.data_ptr<int32_t>(), cell_counts.data_ptr<int32_t>(), num_atoms);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
