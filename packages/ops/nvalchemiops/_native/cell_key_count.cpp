#include <torch/extension.h>

void cell_key_count_hip(torch::Tensor cell_keys, torch::Tensor cell_counts);

void cell_key_count_into(torch::Tensor cell_keys, torch::Tensor cell_counts) {
  TORCH_CHECK(cell_keys.is_cuda() && cell_keys.dim() == 1 &&
                  cell_keys.scalar_type() == torch::kInt,
              "cell_keys must have shape (N,) and dtype int32 on a HIP device");
  TORCH_CHECK(cell_counts.is_cuda() && cell_counts.dim() == 1 &&
                  cell_counts.scalar_type() == torch::kInt,
              "cell_counts must have shape (M,) and dtype int32 on a HIP device");
  TORCH_CHECK(cell_keys.device() == cell_counts.device(),
              "cell-key count tensors must be on the same device");
  TORCH_CHECK(cell_counts.is_contiguous(),
              "cell-count output buffer must be contiguous");

  // The output is always overwritten. Read-only keys may be strided, so make
  // their raw-pointer layout explicit without changing caller-owned storage.
  cell_keys = cell_keys.contiguous();
  cell_key_count_hip(cell_keys, cell_counts);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("cell_key_count_into", &cell_key_count_into,
             "Write native HIP cell counts into a preallocated buffer");
}
