#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <hip/hip_runtime.h>

__global__ void cell_atom_list_fill_kernel(const int32_t* cell_keys,
                                           const int32_t* cell_starts,
                                           int32_t* cell_atom_list,
                                           int32_t* cell_cursor,
                                           int32_t global_atom_offset,
                                           int64_t num_atoms) {
  const int64_t atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= num_atoms) {
    return;
  }
  const int32_t key = cell_keys[atom];
  const int32_t slot = cell_starts[key] - global_atom_offset +
                       atomicAdd(cell_cursor + key, 1);
  cell_atom_list[slot] = static_cast<int32_t>(atom) + global_atom_offset;
}

void cell_atom_list_fill_hip(torch::Tensor cell_keys, torch::Tensor cell_counts,
                             torch::Tensor cell_starts, torch::Tensor cell_atom_list,
                             torch::Tensor cell_cursor, int64_t global_atom_offset) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (cell_cursor.numel() > 0) {
    C10_CUDA_CHECK(hipMemsetAsync(cell_cursor.data_ptr<int32_t>(), 0,
                                  cell_cursor.nbytes(), stream));
  }
  const auto num_atoms = cell_keys.size(0);
  if (num_atoms == 0) {
    return;
  }
  constexpr int threads = 256;
  const int blocks = static_cast<int>((num_atoms + threads - 1) / threads);
  cell_atom_list_fill_kernel<<<blocks, threads, 0, stream>>>(
      cell_keys.data_ptr<int32_t>(), cell_starts.data_ptr<int32_t>(),
      cell_atom_list.data_ptr<int32_t>(), cell_cursor.data_ptr<int32_t>(),
      static_cast<int32_t>(global_atom_offset), num_atoms);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
