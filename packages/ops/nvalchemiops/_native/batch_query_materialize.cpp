#include <torch/extension.h>

#include <cstdint>
#include <limits>
#include <tuple>

void batch_query_materialize_topology_hip(
    torch::Tensor candidate_matrix, torch::Tensor candidate_shifts,
    torch::Tensor candidate_counts, torch::Tensor public_matrix,
    torch::Tensor public_shifts, torch::Tensor public_counts,
    int64_t fill_value);

void batch_query_materialize_topology_composite_hip(
    torch::Tensor candidate_matrix, torch::Tensor candidate_shifts,
    torch::Tensor candidate_counts, torch::Tensor public_matrix,
    torch::Tensor public_shifts, torch::Tensor public_counts,
    int64_t fill_value, int64_t row_bits, int64_t total_bits,
    int64_t shift_bits, int64_t shift_bias);

std::tuple<int64_t, int64_t>
batch_query_materialize_topology_composite_workspace_size_hip(
    torch::Tensor candidate_matrix, torch::Tensor candidate_counts,
    int64_t total_bits);

void batch_query_materialize_topology_composite_workspace_hip(
    torch::Tensor candidate_matrix, torch::Tensor candidate_shifts,
    torch::Tensor candidate_counts, torch::Tensor public_matrix,
    torch::Tensor public_shifts, torch::Tensor public_counts,
    torch::Tensor keys_a, torch::Tensor keys_b, torch::Tensor order_a,
    torch::Tensor order_b, torch::Tensor row_starts,
    torch::Tensor sort_workspace, torch::Tensor scan_workspace,
    int64_t fill_value, int64_t row_bits, int64_t total_bits,
    int64_t shift_bits, int64_t shift_bias);

namespace {

void validate_tensors(torch::Tensor candidate_matrix,
                      torch::Tensor candidate_shifts,
                      torch::Tensor candidate_counts,
                      torch::Tensor public_matrix,
                      torch::Tensor public_shifts,
                      torch::Tensor public_counts, int64_t fill_value) {
  TORCH_CHECK(candidate_matrix.is_cuda() && candidate_matrix.dim() == 2 &&
                  candidate_matrix.size(1) > 0 &&
                  candidate_matrix.scalar_type() == torch::kInt,
              "candidate_matrix must have shape (N, Kc), positive Kc, and int32 dtype on HIP");
  TORCH_CHECK(candidate_shifts.is_cuda() && candidate_shifts.dim() == 3 &&
                  candidate_shifts.size(0) == candidate_matrix.size(0) &&
                  candidate_shifts.size(1) == candidate_matrix.size(1) &&
                  candidate_shifts.size(2) == 3 &&
                  candidate_shifts.scalar_type() == torch::kInt,
              "candidate_shifts must have shape (N, Kc, 3) and int32 dtype on HIP");
  TORCH_CHECK(candidate_counts.is_cuda() && candidate_counts.dim() == 1 &&
                  candidate_counts.size(0) == candidate_matrix.size(0) &&
                  candidate_counts.scalar_type() == torch::kInt,
              "candidate_counts must have shape (N,) and int32 dtype on HIP");
  TORCH_CHECK(public_matrix.is_cuda() && public_matrix.dim() == 2 &&
                  public_matrix.size(0) == candidate_matrix.size(0) &&
                  public_matrix.size(1) > 0 &&
                  public_matrix.scalar_type() == torch::kInt,
              "public_matrix must have shape (N, Kp), positive Kp, and int32 dtype on HIP");
  TORCH_CHECK(public_shifts.is_cuda() && public_shifts.dim() == 3 &&
                  public_shifts.size(0) == public_matrix.size(0) &&
                  public_shifts.size(1) == public_matrix.size(1) &&
                  public_shifts.size(2) == 3 &&
                  public_shifts.scalar_type() == torch::kInt,
              "public_shifts must have shape (N, Kp, 3) and int32 dtype on HIP");
  TORCH_CHECK(public_counts.is_cuda() && public_counts.dim() == 1 &&
                  public_counts.size(0) == public_matrix.size(0) &&
                  public_counts.scalar_type() == torch::kInt,
              "public_counts must have shape (N,) and int32 dtype on HIP");
  TORCH_CHECK(fill_value >= std::numeric_limits<int32_t>::min() &&
                  fill_value <= std::numeric_limits<int32_t>::max(),
              "fill_value must fit int32");
  const auto tensors = {candidate_matrix, candidate_shifts, candidate_counts,
                        public_matrix, public_shifts, public_counts};
  for (const auto& tensor : tensors) {
    TORCH_CHECK(tensor.device() == candidate_matrix.device(),
                "all topology materialization tensors must share one HIP device");
    TORCH_CHECK(tensor.is_contiguous(),
                "all topology materialization tensors must be contiguous");
  }
}

}  // namespace

void batch_query_materialize_topology_into(
    torch::Tensor candidate_matrix, torch::Tensor candidate_shifts,
    torch::Tensor candidate_counts, torch::Tensor public_matrix,
    torch::Tensor public_shifts, torch::Tensor public_counts,
    int64_t fill_value) {
  validate_tensors(candidate_matrix, candidate_shifts, candidate_counts,
                   public_matrix, public_shifts, public_counts, fill_value);
  batch_query_materialize_topology_hip(
      candidate_matrix, candidate_shifts, candidate_counts, public_matrix,
      public_shifts, public_counts, fill_value);
}

void batch_query_materialize_topology_composite_into(
    torch::Tensor candidate_matrix, torch::Tensor candidate_shifts,
    torch::Tensor candidate_counts, torch::Tensor public_matrix,
    torch::Tensor public_shifts, torch::Tensor public_counts,
    int64_t fill_value, int64_t row_bits, int64_t total_bits,
    int64_t shift_bits, int64_t shift_bias) {
  validate_tensors(candidate_matrix, candidate_shifts, candidate_counts,
                   public_matrix, public_shifts, public_counts, fill_value);
  TORCH_CHECK(row_bits > 0 && row_bits < 63,
              "row_bits must fit the packed int64 key");
  TORCH_CHECK(shift_bits > 0 && shift_bits < 63,
              "shift_bits must fit the packed int64 key");
  TORCH_CHECK(total_bits > 0 && total_bits <= 63,
              "composite key must fit signed int64");
  TORCH_CHECK(total_bits == 2 * row_bits + 3 * shift_bits,
              "composite key bit layout is inconsistent");
  TORCH_CHECK(shift_bias >= 0 && shift_bias < (1LL << shift_bits),
              "shift_bias must fit shift_bits");
  batch_query_materialize_topology_composite_hip(
      candidate_matrix, candidate_shifts, candidate_counts, public_matrix,
      public_shifts, public_counts, fill_value, row_bits, total_bits,
      shift_bits, shift_bias);
}

std::tuple<int64_t, int64_t>
batch_query_materialize_topology_composite_workspace_size(
    torch::Tensor candidate_matrix, torch::Tensor candidate_counts,
    int64_t total_bits) {
  TORCH_CHECK(candidate_matrix.is_cuda() && candidate_matrix.dim() == 2 &&
                  candidate_matrix.scalar_type() == torch::kInt,
              "candidate_matrix must be a CUDA int32 matrix");
  TORCH_CHECK(candidate_counts.is_cuda() && candidate_counts.dim() == 1 &&
                  candidate_counts.size(0) == candidate_matrix.size(0) &&
                  candidate_counts.scalar_type() == torch::kInt,
              "candidate_counts must be a CUDA int32 vector");
  TORCH_CHECK(candidate_matrix.is_contiguous() && candidate_counts.is_contiguous(),
              "workspace size inputs must be contiguous");
  TORCH_CHECK(candidate_matrix.device() == candidate_counts.device(),
              "workspace size inputs must share one device");
  TORCH_CHECK(total_bits > 0 && total_bits <= 63,
              "composite key must fit signed int64");
  return batch_query_materialize_topology_composite_workspace_size_hip(
      candidate_matrix, candidate_counts, total_bits);
}

void batch_query_materialize_topology_composite_workspace_into(
    torch::Tensor candidate_matrix, torch::Tensor candidate_shifts,
    torch::Tensor candidate_counts, torch::Tensor public_matrix,
    torch::Tensor public_shifts, torch::Tensor public_counts,
    torch::Tensor keys_a, torch::Tensor keys_b, torch::Tensor order_a,
    torch::Tensor order_b, torch::Tensor row_starts,
    torch::Tensor sort_workspace, torch::Tensor scan_workspace,
    int64_t fill_value, int64_t row_bits, int64_t total_bits,
    int64_t shift_bits, int64_t shift_bias) {
  validate_tensors(candidate_matrix, candidate_shifts, candidate_counts,
                   public_matrix, public_shifts, public_counts, fill_value);
  const int64_t total = candidate_matrix.numel();
  const auto workspace = {keys_a, keys_b, order_a, order_b, row_starts,
                          sort_workspace, scan_workspace};
  for (const auto& tensor : workspace) {
    TORCH_CHECK(tensor.device() == candidate_matrix.device(),
                "all composite workspace tensors must share one HIP device");
    TORCH_CHECK(tensor.is_contiguous(),
                "all composite workspace tensors must be contiguous");
  }
  TORCH_CHECK(keys_a.scalar_type() == torch::kLong &&
                  keys_b.scalar_type() == torch::kLong,
              "composite key workspace must use int64 tensors");
  TORCH_CHECK(order_a.scalar_type() == torch::kInt &&
                  order_b.scalar_type() == torch::kInt &&
                  row_starts.scalar_type() == torch::kInt,
              "composite order workspace must use int32 tensors");
  TORCH_CHECK(sort_workspace.scalar_type() == torch::kByte &&
                  scan_workspace.scalar_type() == torch::kByte,
              "composite temporary workspace must use uint8 tensors");
  TORCH_CHECK(keys_a.numel() >= total && keys_b.numel() >= total &&
                  order_a.numel() >= total && order_b.numel() >= total &&
                  row_starts.numel() >= candidate_matrix.size(0),
              "composite workspace is too small for candidate topology");
  TORCH_CHECK(row_bits > 0 && row_bits < 63 && shift_bits > 0 &&
                  shift_bits < 63 && total_bits > 0 && total_bits <= 63,
              "invalid composite key bit layout");
  TORCH_CHECK(total_bits == 2 * row_bits + 3 * shift_bits,
              "composite key bit layout is inconsistent");
  TORCH_CHECK(shift_bias >= 0 && shift_bias < (1LL << shift_bits),
              "shift_bias must fit shift_bits");
  batch_query_materialize_topology_composite_workspace_hip(
      candidate_matrix, candidate_shifts, candidate_counts, public_matrix,
      public_shifts, public_counts, keys_a, keys_b, order_a, order_b,
      row_starts, sort_workspace, scan_workspace, fill_value, row_bits,
      total_bits, shift_bits, shift_bias);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("batch_query_materialize_topology_into",
             &batch_query_materialize_topology_into,
             "Materialize native HIP Batch query topology");
  module.def("batch_query_materialize_topology_composite_into",
             &batch_query_materialize_topology_composite_into,
             "Materialize native HIP Batch query topology with one composite key");
  module.def("batch_query_materialize_topology_composite_workspace_size",
             &batch_query_materialize_topology_composite_workspace_size,
             "Query native HIP composite topology workspace sizes");
  module.def("batch_query_materialize_topology_composite_workspace_into",
             &batch_query_materialize_topology_composite_workspace_into,
             "Materialize native HIP composite topology with caller workspace");
}
