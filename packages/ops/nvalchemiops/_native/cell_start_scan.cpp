#include <torch/extension.h>

#include <cstdint>
#include <limits>

size_t cell_start_scan_workspace_size_hip(torch::Tensor cell_counts,
                                          torch::Tensor cell_starts);
void cell_start_scan_hip(torch::Tensor cell_counts, torch::Tensor cell_starts,
                         torch::Tensor workspace, int64_t global_atom_offset);

namespace {

void validate_tensors(torch::Tensor cell_counts, torch::Tensor cell_starts) {
  TORCH_CHECK(cell_counts.is_cuda() && cell_counts.dim() == 1 &&
                  cell_counts.scalar_type() == torch::kInt,
              "cell_counts must have shape (M,) and dtype int32 on a HIP device");
  TORCH_CHECK(cell_starts.is_cuda() && cell_starts.sizes() == cell_counts.sizes() &&
                  cell_starts.scalar_type() == torch::kInt,
              "cell_starts must match counts shape and have dtype int32 on a HIP device");
  TORCH_CHECK(cell_counts.device() == cell_starts.device(),
              "cell-count and cell-start tensors must be on the same device");
  TORCH_CHECK(cell_starts.is_contiguous(),
              "cell-start output buffer must be contiguous");
}

}  // namespace

size_t cell_start_scan_workspace_size(torch::Tensor cell_counts,
                                      torch::Tensor cell_starts) {
  validate_tensors(cell_counts, cell_starts);
  cell_counts = cell_counts.contiguous();
  return cell_start_scan_workspace_size_hip(cell_counts, cell_starts);
}

void cell_start_scan_into(torch::Tensor cell_counts, torch::Tensor cell_starts,
                          torch::Tensor workspace, int64_t global_atom_offset) {
  validate_tensors(cell_counts, cell_starts);
  TORCH_CHECK(workspace.is_cuda() && workspace.dim() == 1 &&
                  workspace.scalar_type() == torch::kByte,
              "cell-start workspace must have shape (W,) and dtype uint8 on a HIP device");
  TORCH_CHECK(workspace.device() == cell_counts.device(),
              "cell-start workspace must be on the same device as counts");
  TORCH_CHECK(workspace.is_contiguous(), "cell-start workspace must be contiguous");
  TORCH_CHECK(global_atom_offset >= 0 &&
                  global_atom_offset <= std::numeric_limits<int32_t>::max(),
              "global_atom_offset must fit int32");

  cell_counts = cell_counts.contiguous();
  const auto required = cell_start_scan_workspace_size_hip(cell_counts, cell_starts);
  TORCH_CHECK(static_cast<size_t>(workspace.numel()) >= required,
              "cell-start workspace is too small");
  cell_start_scan_hip(cell_counts, cell_starts, workspace, global_atom_offset);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("cell_start_scan_workspace_size", &cell_start_scan_workspace_size,
             "Return native HIP cell-start scan workspace bytes");
  module.def("cell_start_scan_into", &cell_start_scan_into,
             "Write native HIP cell starts into a preallocated buffer");
}
