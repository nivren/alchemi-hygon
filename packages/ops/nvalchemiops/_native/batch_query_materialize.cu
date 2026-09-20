#include <hip/hip_runtime.h>
#include <complex>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <rocprim/device/device_radix_sort.hpp>
#include <rocprim/device/device_scan.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <tuple>
#include <utility>

namespace {

constexpr int kThreads = 256;

__global__ void prepare_fields_kernel(
    const int32_t* candidate_matrix, const int32_t* candidate_shifts,
    const int32_t* candidate_counts, int32_t* rows, int32_t* columns,
    int32_t* shift_x, int32_t* shift_y, int32_t* shift_z, int64_t atoms,
    int64_t candidate_capacity) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = atoms * candidate_capacity;
  if (index >= total) {
    return;
  }
  const int64_t row = index / candidate_capacity;
  const int64_t slot = index - row * candidate_capacity;
  const bool active = slot < candidate_counts[row];
  rows[index] = active ? static_cast<int32_t>(row) : static_cast<int32_t>(atoms);
  columns[index] = active ? candidate_matrix[index] : 0;
  const int64_t shift_index = index * 3;
  shift_x[index] = active ? candidate_shifts[shift_index + 0] : 0;
  shift_y[index] = active ? candidate_shifts[shift_index + 1] : 0;
  shift_z[index] = active ? candidate_shifts[shift_index + 2] : 0;
}

__global__ void initialize_order_kernel(int32_t* order, int64_t total) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < total) {
    order[index] = static_cast<int32_t>(index);
  }
}

__global__ void prepare_composite_key_kernel(
    const int32_t* candidate_matrix, const int32_t* candidate_shifts,
    const int32_t* candidate_counts, int64_t* keys, int32_t* order,
    int64_t atoms, int64_t candidate_capacity, int row_bits,
    int shift_bits, int64_t shift_bias) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = atoms * candidate_capacity;
  if (index >= total) {
    return;
  }
  const int64_t row = index / candidate_capacity;
  const int64_t slot = index - row * candidate_capacity;
  const bool active = slot < candidate_counts[row];
  const int64_t row_code = active ? row : atoms;
  const int64_t column_code = active ? candidate_matrix[index] : 0;
  const int64_t shift_index = index * 3;
  const int64_t shift_x = active ? candidate_shifts[shift_index + 0] + shift_bias : 0;
  const int64_t shift_y = active ? candidate_shifts[shift_index + 1] + shift_bias : 0;
  const int64_t shift_z = active ? candidate_shifts[shift_index + 2] + shift_bias : 0;

  uint64_t packed = static_cast<uint64_t>(row_code);
  packed = (packed << row_bits) | static_cast<uint64_t>(column_code);
  packed = (packed << shift_bits) | static_cast<uint64_t>(shift_x);
  packed = (packed << shift_bits) | static_cast<uint64_t>(shift_y);
  packed = (packed << shift_bits) | static_cast<uint64_t>(shift_z);
  keys[index] = static_cast<int64_t>(packed);
  order[index] = static_cast<int32_t>(index);
}

__global__ void gather_key_kernel(const int32_t* field, const int32_t* order,
                                  int32_t* keys, int64_t total) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < total) {
    keys[index] = field[order[index]];
  }
}

__global__ void scatter_public_kernel(
    const int32_t* order, const int32_t* columns, const int32_t* shift_x,
    const int32_t* shift_y, const int32_t* shift_z, const int32_t* row_starts,
    const int32_t* candidate_counts, int32_t* public_matrix,
    int32_t* public_shifts, int64_t total,
    int64_t candidate_capacity, int64_t public_capacity) {
  const int64_t pair = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair >= total) {
    return;
  }
  const int32_t original = order[pair];
  const int32_t row = static_cast<int32_t>(original / candidate_capacity);
  const int32_t slot = static_cast<int32_t>(
      original - static_cast<int64_t>(row) * candidate_capacity);
  if (slot >= candidate_counts[row]) {
    return;
  }
  const int32_t rank = static_cast<int32_t>(pair - row_starts[row]);
  const int64_t destination = static_cast<int64_t>(row) * public_capacity + rank;
  public_matrix[destination] = columns[original];
  const int64_t shift_destination = destination * 3;
  public_shifts[shift_destination + 0] = shift_x[original];
  public_shifts[shift_destination + 1] = shift_y[original];
  public_shifts[shift_destination + 2] = shift_z[original];
}

__global__ void scatter_public_interleaved_kernel(
    const int32_t* order, const int32_t* candidate_matrix,
    const int32_t* candidate_shifts, const int32_t* row_starts,
    const int32_t* candidate_counts, int32_t* public_matrix,
    int32_t* public_shifts, int64_t total,
    int64_t candidate_capacity, int64_t public_capacity) {
  const int64_t pair = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair >= total) {
    return;
  }
  const int32_t original = order[pair];
  const int32_t row = static_cast<int32_t>(original / candidate_capacity);
  const int32_t slot = static_cast<int32_t>(
      original - static_cast<int64_t>(row) * candidate_capacity);
  if (slot >= candidate_counts[row]) {
    return;
  }
  const int32_t rank = static_cast<int32_t>(pair - row_starts[row]);
  const int64_t destination = static_cast<int64_t>(row) * public_capacity + rank;
  public_matrix[destination] = candidate_matrix[original];
  const int64_t source_shift = static_cast<int64_t>(original) * 3;
  const int64_t shift_destination = destination * 3;
  public_shifts[shift_destination + 0] = candidate_shifts[source_shift + 0];
  public_shifts[shift_destination + 1] = candidate_shifts[source_shift + 1];
  public_shifts[shift_destination + 2] = candidate_shifts[source_shift + 2];
}

void check_launch() { C10_CUDA_KERNEL_LAUNCH_CHECK(); }

size_t radix_sort_workspace_size(const int32_t* keys_in, int32_t* keys_out,
                                 const int32_t* values_in, int32_t* values_out,
                                 int64_t total, hipStream_t stream) {
  size_t bytes = 0;
  C10_CUDA_CHECK(rocprim::radix_sort_pairs(
      nullptr, bytes, keys_in, keys_out, values_in, values_out, total, 0, 32,
      stream));
  return bytes;
}

void radix_sort_pairs(const int32_t* keys_in, int32_t* keys_out,
                      const int32_t* values_in, int32_t* values_out,
                      int64_t total, void* workspace, size_t workspace_bytes,
                      hipStream_t stream) {
  size_t bytes = workspace_bytes;
  C10_CUDA_CHECK(rocprim::radix_sort_pairs(
      workspace, bytes, keys_in, keys_out, values_in, values_out, total, 0, 32,
      stream));
}

size_t radix_sort_int64_workspace_size(const int64_t* keys_in, int64_t* keys_out,
                                       const int32_t* values_in,
                                       int32_t* values_out, int64_t total,
                                       int end_bit, hipStream_t stream) {
  size_t bytes = 0;
  C10_CUDA_CHECK(rocprim::radix_sort_pairs(
      nullptr, bytes, keys_in, keys_out, values_in, values_out, total, 0,
      end_bit, stream));
  return bytes;
}

void radix_sort_int64_pairs(const int64_t* keys_in, int64_t* keys_out,
                            const int32_t* values_in, int32_t* values_out,
                            int64_t total, void* workspace,
                            size_t workspace_bytes, int end_bit,
                            hipStream_t stream) {
  size_t bytes = workspace_bytes;
  C10_CUDA_CHECK(rocprim::radix_sort_pairs(
      workspace, bytes, keys_in, keys_out, values_in, values_out, total, 0,
      end_bit, stream));
}

}  // namespace

void batch_query_materialize_topology_hip(
    torch::Tensor candidate_matrix, torch::Tensor candidate_shifts,
    torch::Tensor candidate_counts, torch::Tensor public_matrix,
    torch::Tensor public_shifts, torch::Tensor public_counts,
    int64_t fill_value) {
  const int64_t atoms = candidate_matrix.size(0);
  const int64_t candidate_capacity = candidate_matrix.size(1);
  const int64_t public_capacity = public_matrix.size(1);
  TORCH_CHECK(atoms <= std::numeric_limits<int32_t>::max(),
              "number of atoms must fit int32 row/order fields");
  TORCH_CHECK(candidate_capacity <= std::numeric_limits<int32_t>::max(),
              "candidate capacity must fit int32 indexing");
  TORCH_CHECK(public_capacity <= std::numeric_limits<int32_t>::max(),
              "public capacity must fit int32 indexing");
  TORCH_CHECK(atoms == 0 || atoms * candidate_capacity <=
                                  std::numeric_limits<int32_t>::max(),
              "candidate storage is too large for int32 order fields");

  public_matrix.fill_(static_cast<int32_t>(fill_value));
  public_shifts.zero_();
  public_counts.copy_(candidate_counts);
  if (atoms == 0) {
    return;
  }

  const int64_t total = atoms * candidate_capacity;
  const auto int_options = candidate_matrix.options().dtype(torch::kInt);
  auto rows = torch::empty({total}, int_options);
  auto columns = torch::empty({total}, int_options);
  auto shift_x = torch::empty({total}, int_options);
  auto shift_y = torch::empty({total}, int_options);
  auto shift_z = torch::empty({total}, int_options);
  auto order_a = torch::empty({total}, int_options);
  auto order_b = torch::empty({total}, int_options);
  auto keys_a = torch::empty({total}, int_options);
  auto keys_b = torch::empty({total}, int_options);
  auto row_starts = torch::empty({atoms}, int_options);

  const auto stream = at::cuda::getCurrentCUDAStream();
  const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
  hipLaunchKernelGGL(
      prepare_fields_kernel, dim3(blocks), dim3(kThreads), 0, stream,
      candidate_matrix.data_ptr<int32_t>(), candidate_shifts.data_ptr<int32_t>(),
      candidate_counts.data_ptr<int32_t>(), rows.data_ptr<int32_t>(),
      columns.data_ptr<int32_t>(), shift_x.data_ptr<int32_t>(),
      shift_y.data_ptr<int32_t>(), shift_z.data_ptr<int32_t>(), atoms,
      candidate_capacity);
  check_launch();
  hipLaunchKernelGGL(initialize_order_kernel, dim3(blocks), dim3(kThreads), 0,
                     stream, order_a.data_ptr<int32_t>(), total);
  check_launch();

  const size_t sort_bytes = radix_sort_workspace_size(
      keys_a.data_ptr<int32_t>(), keys_b.data_ptr<int32_t>(),
      order_a.data_ptr<int32_t>(), order_b.data_ptr<int32_t>(), total, stream);
  auto sort_workspace = torch::empty(
      {static_cast<int64_t>(sort_bytes)},
      candidate_matrix.options().dtype(torch::kByte));

  const std::array<const int32_t*, 5> fields = {
      shift_z.data_ptr<int32_t>(), shift_y.data_ptr<int32_t>(),
      shift_x.data_ptr<int32_t>(), columns.data_ptr<int32_t>(),
      rows.data_ptr<int32_t>()};
  int32_t* key_input = keys_a.data_ptr<int32_t>();
  int32_t* key_output = keys_b.data_ptr<int32_t>();
  int32_t* order_input = order_a.data_ptr<int32_t>();
  int32_t* order_output = order_b.data_ptr<int32_t>();
  for (const int32_t* field : fields) {
    hipLaunchKernelGGL(gather_key_kernel, dim3(blocks), dim3(kThreads), 0,
                       stream, field, order_input, key_input, total);
    check_launch();
    radix_sort_pairs(
        key_input, key_output, order_input, order_output, total,
        sort_workspace.data_ptr<uint8_t>(), sort_bytes, stream);
    std::swap(key_input, key_output);
    std::swap(order_input, order_output);
  }

  size_t scan_bytes = 0;
  C10_CUDA_CHECK(rocprim::exclusive_scan(
      nullptr, scan_bytes, candidate_counts.data_ptr<int32_t>(),
      row_starts.data_ptr<int32_t>(), 0, atoms, rocprim::plus<int32_t>(),
      stream));
  auto scan_workspace = torch::empty(
      {static_cast<int64_t>(scan_bytes)},
      candidate_matrix.options().dtype(torch::kByte));
  C10_CUDA_CHECK(rocprim::exclusive_scan(
      scan_workspace.data_ptr<uint8_t>(), scan_bytes,
      candidate_counts.data_ptr<int32_t>(), row_starts.data_ptr<int32_t>(), 0,
      atoms, rocprim::plus<int32_t>(), stream));

  const int pair_blocks = static_cast<int>((total + kThreads - 1) / kThreads);
  hipLaunchKernelGGL(
      scatter_public_kernel, dim3(pair_blocks), dim3(kThreads), 0, stream,
      order_input, columns.data_ptr<int32_t>(), shift_x.data_ptr<int32_t>(),
      shift_y.data_ptr<int32_t>(), shift_z.data_ptr<int32_t>(),
      row_starts.data_ptr<int32_t>(), candidate_counts.data_ptr<int32_t>(),
      public_matrix.data_ptr<int32_t>(), public_shifts.data_ptr<int32_t>(),
      total, candidate_capacity, public_capacity);
  check_launch();
}

std::tuple<int64_t, int64_t>
batch_query_materialize_topology_composite_workspace_size_hip(
    torch::Tensor candidate_matrix, torch::Tensor candidate_counts,
    int64_t total_bits) {
  const int64_t atoms = candidate_matrix.size(0);
  const int64_t candidate_capacity = candidate_matrix.size(1);
  if (atoms == 0) {
    return {0, 0};
  }
  const int64_t total = atoms * candidate_capacity;
  const auto stream = at::cuda::getCurrentCUDAStream();
  // Use correctly typed scratch for the rocPRIM query.  Reinterpreting the
  // int32 candidate matrix as int64 keys under-allocates the apparent key
  // storage for large Batch candidates and can make the backend inspect past
  // the tensor even though this is only a temporary-storage query.
  const auto key_options = candidate_matrix.options().dtype(torch::kLong);
  const auto order_options = candidate_counts.options().dtype(torch::kInt);
  auto key_input_storage = torch::empty({total}, key_options);
  auto key_output_storage = torch::empty({total}, key_options);
  auto order_input_storage = torch::empty({total}, order_options);
  auto order_output_storage = torch::empty({total}, order_options);
  const auto key_input = key_input_storage.data_ptr<int64_t>();
  const auto key_output = key_output_storage.data_ptr<int64_t>();
  const auto order_input = order_input_storage.data_ptr<int32_t>();
  const auto order_output = order_output_storage.data_ptr<int32_t>();
  const size_t sort_bytes = radix_sort_int64_workspace_size(
      key_input, key_output, order_input, order_output, total,
      static_cast<int>(total_bits), stream);
  auto scan_input_storage = torch::empty({atoms}, candidate_counts.options());
  auto scan_output_storage = torch::empty({atoms}, candidate_counts.options());
  size_t scan_bytes = 0;
  C10_CUDA_CHECK(rocprim::exclusive_scan(
      nullptr, scan_bytes, scan_input_storage.data_ptr<int32_t>(),
      scan_output_storage.data_ptr<int32_t>(), 0, atoms,
      rocprim::plus<int32_t>(), stream));
  return {static_cast<int64_t>(sort_bytes), static_cast<int64_t>(scan_bytes)};
}

void batch_query_materialize_topology_composite_workspace_hip(
    torch::Tensor candidate_matrix, torch::Tensor candidate_shifts,
    torch::Tensor candidate_counts, torch::Tensor public_matrix,
    torch::Tensor public_shifts, torch::Tensor public_counts,
    torch::Tensor keys_a, torch::Tensor keys_b, torch::Tensor order_a,
    torch::Tensor order_b, torch::Tensor row_starts,
    torch::Tensor sort_workspace, torch::Tensor scan_workspace,
    int64_t fill_value, int64_t row_bits, int64_t total_bits,
    int64_t shift_bits, int64_t shift_bias) {
  const int64_t atoms = candidate_matrix.size(0);
  const int64_t candidate_capacity = candidate_matrix.size(1);
  const int64_t public_capacity = public_matrix.size(1);
  TORCH_CHECK(atoms <= std::numeric_limits<int32_t>::max(),
              "number of atoms must fit int32 row/order fields");
  TORCH_CHECK(candidate_capacity <= std::numeric_limits<int32_t>::max(),
              "candidate capacity must fit int32 indexing");
  TORCH_CHECK(public_capacity <= std::numeric_limits<int32_t>::max(),
              "public capacity must fit int32 indexing");
  TORCH_CHECK(atoms == 0 || atoms * candidate_capacity <=
                                  std::numeric_limits<int32_t>::max(),
              "candidate storage is too large for int32 order fields");
  TORCH_CHECK(row_bits > 0 && row_bits < 63 && shift_bits > 0 &&
                  shift_bits < 63 && total_bits > 0 && total_bits <= 63,
              "invalid composite key bit layout");
  TORCH_CHECK(total_bits == 2 * row_bits + 3 * shift_bits,
              "composite key bit layout is inconsistent");
  TORCH_CHECK(shift_bias >= 0 && shift_bias < (1LL << shift_bits),
              "shift_bias must fit shift_bits");
  const auto workspace_sizes =
      batch_query_materialize_topology_composite_workspace_size_hip(
          candidate_matrix, candidate_counts, total_bits);
  TORCH_CHECK(sort_workspace.numel() >= std::get<0>(workspace_sizes),
              "composite sort workspace is too small");
  TORCH_CHECK(scan_workspace.numel() >= std::get<1>(workspace_sizes),
              "composite scan workspace is too small");

  public_matrix.fill_(static_cast<int32_t>(fill_value));
  public_shifts.zero_();
  public_counts.copy_(candidate_counts);
  if (atoms == 0) {
    return;
  }

  const int64_t total = atoms * candidate_capacity;

  const auto stream = at::cuda::getCurrentCUDAStream();
  const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
  hipLaunchKernelGGL(
      prepare_composite_key_kernel, dim3(blocks), dim3(kThreads), 0, stream,
      candidate_matrix.data_ptr<int32_t>(), candidate_shifts.data_ptr<int32_t>(),
      candidate_counts.data_ptr<int32_t>(), keys_a.data_ptr<int64_t>(),
      order_a.data_ptr<int32_t>(), atoms, candidate_capacity,
      static_cast<int>(row_bits), static_cast<int>(shift_bits), shift_bias);
  check_launch();

  const size_t sort_bytes = radix_sort_int64_workspace_size(
      keys_a.data_ptr<int64_t>(), keys_b.data_ptr<int64_t>(),
      order_a.data_ptr<int32_t>(), order_b.data_ptr<int32_t>(), total,
      static_cast<int>(total_bits), stream);
  radix_sort_int64_pairs(
      keys_a.data_ptr<int64_t>(), keys_b.data_ptr<int64_t>(),
      order_a.data_ptr<int32_t>(), order_b.data_ptr<int32_t>(), total,
      sort_workspace.data_ptr<uint8_t>(), sort_bytes,
      static_cast<int>(total_bits), stream);

  size_t scan_bytes = 0;
  C10_CUDA_CHECK(rocprim::exclusive_scan(
      nullptr, scan_bytes, candidate_counts.data_ptr<int32_t>(),
      row_starts.data_ptr<int32_t>(), 0, atoms, rocprim::plus<int32_t>(),
      stream));
  C10_CUDA_CHECK(rocprim::exclusive_scan(
      scan_workspace.data_ptr<uint8_t>(), scan_bytes,
      candidate_counts.data_ptr<int32_t>(), row_starts.data_ptr<int32_t>(), 0,
      atoms, rocprim::plus<int32_t>(), stream));

  const int pair_blocks = static_cast<int>((total + kThreads - 1) / kThreads);
  hipLaunchKernelGGL(
      scatter_public_interleaved_kernel, dim3(pair_blocks), dim3(kThreads), 0,
      stream, order_b.data_ptr<int32_t>(),
      candidate_matrix.data_ptr<int32_t>(), candidate_shifts.data_ptr<int32_t>(),
      row_starts.data_ptr<int32_t>(), candidate_counts.data_ptr<int32_t>(),
      public_matrix.data_ptr<int32_t>(), public_shifts.data_ptr<int32_t>(),
      total, candidate_capacity, public_capacity);
  check_launch();
}

void batch_query_materialize_topology_composite_hip(
    torch::Tensor candidate_matrix, torch::Tensor candidate_shifts,
    torch::Tensor candidate_counts, torch::Tensor public_matrix,
    torch::Tensor public_shifts, torch::Tensor public_counts,
    int64_t fill_value, int64_t row_bits, int64_t total_bits,
    int64_t shift_bits, int64_t shift_bias) {
  const int64_t atoms = candidate_matrix.size(0);
  const int64_t candidate_capacity = candidate_matrix.size(1);
  const auto workspace_sizes =
      batch_query_materialize_topology_composite_workspace_size_hip(
          candidate_matrix, candidate_counts, total_bits);
  const auto key_options = candidate_matrix.options().dtype(torch::kLong);
  const auto int_options = candidate_matrix.options().dtype(torch::kInt);
  const int64_t total = atoms * candidate_capacity;
  auto keys_a = torch::empty({total}, key_options);
  auto keys_b = torch::empty({total}, key_options);
  auto order_a = torch::empty({total}, int_options);
  auto order_b = torch::empty({total}, int_options);
  auto row_starts = torch::empty({atoms}, int_options);
  auto sort_workspace = torch::empty(
      {std::get<0>(workspace_sizes)}, candidate_matrix.options().dtype(torch::kByte));
  auto scan_workspace = torch::empty(
      {std::get<1>(workspace_sizes)}, candidate_matrix.options().dtype(torch::kByte));
  batch_query_materialize_topology_composite_workspace_hip(
      candidate_matrix, candidate_shifts, candidate_counts, public_matrix,
      public_shifts, public_counts, keys_a, keys_b, order_a, order_b,
      row_starts, sort_workspace, scan_workspace, fill_value, row_bits,
      total_bits, shift_bits, shift_bias);
}
