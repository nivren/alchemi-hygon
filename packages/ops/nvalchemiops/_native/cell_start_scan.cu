#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <rocprim/device/device_scan.hpp>

namespace {

size_t scan_workspace_size(torch::Tensor cell_counts, torch::Tensor cell_starts,
                           int32_t global_atom_offset) {
  if (cell_counts.numel() == 0) {
    return 0;
  }
  size_t storage_size = 0;
  const auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(rocprim::exclusive_scan(
      nullptr, storage_size, cell_counts.data_ptr<int32_t>(),
      cell_starts.data_ptr<int32_t>(), global_atom_offset, cell_counts.numel(),
      rocprim::plus<int32_t>(), stream));
  return storage_size;
}

}  // namespace

size_t cell_start_scan_workspace_size_hip(torch::Tensor cell_counts,
                                          torch::Tensor cell_starts) {
  return scan_workspace_size(cell_counts, cell_starts, 0);
}

void cell_start_scan_hip(torch::Tensor cell_counts, torch::Tensor cell_starts,
                         torch::Tensor workspace, int64_t global_atom_offset) {
  if (cell_counts.numel() == 0) {
    return;
  }
  size_t storage_size = scan_workspace_size(
      cell_counts, cell_starts, static_cast<int32_t>(global_atom_offset));
  const auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(rocprim::exclusive_scan(
      workspace.data_ptr<uint8_t>(), storage_size, cell_counts.data_ptr<int32_t>(),
      cell_starts.data_ptr<int32_t>(), static_cast<int32_t>(global_atom_offset),
      cell_counts.numel(), rocprim::plus<int32_t>(), stream));
}
