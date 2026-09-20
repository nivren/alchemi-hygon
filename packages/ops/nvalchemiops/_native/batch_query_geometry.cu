#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

namespace {

constexpr int kThreads = 256;

template <typename scalar_t>
__global__ void batch_query_geometry_kernel(
    const scalar_t* positions, const scalar_t* cells, const int32_t* batch_idx,
    const int32_t* public_matrix, const int32_t* public_shifts,
    const int32_t* public_counts, scalar_t* distances, scalar_t* vectors,
    int64_t atoms, int64_t capacity) {
  const int64_t pair = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = atoms * capacity;
  if (pair >= total) {
    return;
  }
  const int64_t row = pair / capacity;
  const int64_t slot = pair - row * capacity;
  const int64_t vector_offset = pair * 3;
  if (slot >= public_counts[row]) {
    distances[pair] = static_cast<scalar_t>(0);
    vectors[vector_offset + 0] = static_cast<scalar_t>(0);
    vectors[vector_offset + 1] = static_cast<scalar_t>(0);
    vectors[vector_offset + 2] = static_cast<scalar_t>(0);
    return;
  }

  const int32_t column = public_matrix[pair];
  const int32_t system = batch_idx[row];
  const scalar_t* source = positions + row * 3;
  const scalar_t* target = positions + static_cast<int64_t>(column) * 3;
  const scalar_t* geometry = cells + static_cast<int64_t>(system) * 9;
  const int32_t* shift = public_shifts + vector_offset;
  scalar_t distance_sq = static_cast<scalar_t>(0);
  for (int cartesian = 0; cartesian < 3; ++cartesian) {
    scalar_t value = source[cartesian] - target[cartesian];
    value -= static_cast<scalar_t>(shift[0]) * geometry[0 * 3 + cartesian];
    value -= static_cast<scalar_t>(shift[1]) * geometry[1 * 3 + cartesian];
    value -= static_cast<scalar_t>(shift[2]) * geometry[2 * 3 + cartesian];
    vectors[vector_offset + cartesian] = value;
    distance_sq += value * value;
  }
  distances[pair] = sqrt(distance_sq);
}

}  // namespace

void batch_query_geometry_hip(
    torch::Tensor positions, torch::Tensor cells, torch::Tensor batch_idx,
    torch::Tensor public_matrix, torch::Tensor public_shifts,
    torch::Tensor public_counts, torch::Tensor distances, torch::Tensor vectors) {
  const int64_t atoms = positions.size(0);
  const int64_t capacity = public_matrix.size(1);
  if (atoms == 0) {
    distances.zero_();
    vectors.zero_();
    return;
  }
  const int64_t total = atoms * capacity;
  constexpr int threads = kThreads;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(positions.scalar_type(),
                             "batch_query_geometry_hip", [&] {
    batch_query_geometry_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        positions.data_ptr<scalar_t>(), cells.data_ptr<scalar_t>(),
        batch_idx.data_ptr<int32_t>(), public_matrix.data_ptr<int32_t>(),
        public_shifts.data_ptr<int32_t>(), public_counts.data_ptr<int32_t>(),
        distances.data_ptr<scalar_t>(), vectors.data_ptr<scalar_t>(), atoms,
        capacity);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
