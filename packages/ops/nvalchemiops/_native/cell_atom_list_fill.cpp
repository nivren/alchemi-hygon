#include <torch/extension.h>

#include <cstdint>
#include <limits>

void cell_atom_list_fill_hip(torch::Tensor cell_keys, torch::Tensor cell_counts,
                             torch::Tensor cell_starts, torch::Tensor cell_atom_list,
                             torch::Tensor cell_cursor, int64_t global_atom_offset);

void cell_atom_list_fill_into(torch::Tensor cell_keys, torch::Tensor cell_counts,
                              torch::Tensor cell_starts, torch::Tensor cell_atom_list,
                              torch::Tensor cell_cursor, int64_t global_atom_offset) {
  TORCH_CHECK(cell_keys.is_cuda() && cell_keys.dim() == 1 &&
                  cell_keys.scalar_type() == torch::kInt,
              "cell_keys must have shape (N,) and dtype int32 on a HIP device");
  TORCH_CHECK(cell_counts.is_cuda() && cell_counts.dim() == 1 &&
                  cell_counts.scalar_type() == torch::kInt,
              "cell_counts must have shape (M,) and dtype int32 on a HIP device");
  TORCH_CHECK(cell_starts.is_cuda() && cell_starts.sizes() == cell_counts.sizes() &&
                  cell_starts.scalar_type() == torch::kInt,
              "cell_starts must match counts shape and have dtype int32 on a HIP device");
  TORCH_CHECK(cell_atom_list.is_cuda() && cell_atom_list.dim() == 1 &&
                  cell_atom_list.scalar_type() == torch::kInt,
              "cell_atom_list must have shape (capacity,) and dtype int32 on a HIP device");
  TORCH_CHECK(cell_cursor.is_cuda() && cell_cursor.sizes() == cell_counts.sizes() &&
                  cell_cursor.scalar_type() == torch::kInt,
              "cell_cursor must match counts shape and have dtype int32 on a HIP device");
  TORCH_CHECK(cell_keys.device() == cell_counts.device() &&
                  cell_keys.device() == cell_starts.device() &&
                  cell_keys.device() == cell_atom_list.device() &&
                  cell_keys.device() == cell_cursor.device(),
              "all CSR fill tensors must be on the same device");
  TORCH_CHECK(cell_atom_list.is_contiguous() && cell_cursor.is_contiguous(),
              "CSR fill output and cursor buffers must be contiguous");
  TORCH_CHECK(global_atom_offset >= 0 &&
                  global_atom_offset <= std::numeric_limits<int32_t>::max(),
              "global_atom_offset must fit int32");

  cell_keys = cell_keys.contiguous();
  cell_counts = cell_counts.contiguous();
  cell_starts = cell_starts.contiguous();
  cell_atom_list_fill_hip(cell_keys, cell_counts, cell_starts, cell_atom_list,
                          cell_cursor, global_atom_offset);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("cell_atom_list_fill_into", &cell_atom_list_fill_into,
             "Atomically fill one native HIP CSR atom-list slice");
}
