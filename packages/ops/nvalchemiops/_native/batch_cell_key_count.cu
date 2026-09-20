#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <hip/hip_runtime.h>

#include <cmath>

template <typename scalar_t>
__global__ void batch_cell_key_count_kernel(
    const scalar_t* positions,
    const scalar_t* inverse_cells,
    const int32_t* cells_per_dimension,
    const bool* pbc,
    const int32_t* batch_idx,
    const int32_t* cell_offsets,
    int32_t* atom_periodic_shifts,
    int32_t* atom_to_cell_mapping,
    int32_t* cell_keys,
    int32_t* cell_counts,
    int64_t num_atoms) {
  const int64_t atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= num_atoms) {
    return;
  }

  const int32_t system = batch_idx[atom];
  const auto inverse_cell = inverse_cells + system * 9;
  const auto dimensions = cells_per_dimension + system * 3;
  const auto system_pbc = pbc + system * 3;
  int32_t coordinate[3];
#pragma unroll
  for (int dimension = 0; dimension < 3; ++dimension) {
    scalar_t fractional = 0;
#pragma unroll
    for (int cartesian = 0; cartesian < 3; ++cartesian) {
      fractional += positions[atom * 3 + cartesian] *
                    inverse_cell[cartesian * 3 + dimension];
    }
    const int32_t shift = system_pbc[dimension]
                              ? static_cast<int32_t>(floor(fractional))
                              : 0;
    const scalar_t wrapped = fractional - static_cast<scalar_t>(shift);
    int32_t index = static_cast<int32_t>(floor(
        wrapped * static_cast<scalar_t>(dimensions[dimension])));
    index = index < 0 ? 0 : index;
    index = index >= dimensions[dimension] ? dimensions[dimension] - 1 : index;
    atom_periodic_shifts[atom * 3 + dimension] = shift;
    atom_to_cell_mapping[atom * 3 + dimension] = index;
    coordinate[dimension] = index;
  }
  const int32_t key = cell_offsets[system] + coordinate[0] + dimensions[0] *
      (coordinate[1] + dimensions[1] * coordinate[2]);
  cell_keys[atom] = key;
  atomicAdd(cell_counts + key, 1);
}

void batch_cell_key_count_hip(
    torch::Tensor positions,
    torch::Tensor inverse_cells,
    torch::Tensor cells_per_dimension,
    torch::Tensor pbc,
    torch::Tensor batch_idx,
    torch::Tensor cell_offsets,
    torch::Tensor atom_periodic_shifts,
    torch::Tensor atom_to_cell_mapping,
    torch::Tensor cell_keys,
    torch::Tensor cell_counts) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (cell_counts.numel() > 0) {
    C10_CUDA_CHECK(hipMemsetAsync(cell_counts.data_ptr<int32_t>(), 0,
                                  cell_counts.nbytes(), stream));
  }
  const auto num_atoms = positions.size(0);
  if (num_atoms == 0) {
    return;
  }
  constexpr int threads = 256;
  const int blocks = static_cast<int>((num_atoms + threads - 1) / threads);
  AT_DISPATCH_FLOATING_TYPES(positions.scalar_type(), "batch_cell_key_count_hip", [&] {
    batch_cell_key_count_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        positions.data_ptr<scalar_t>(), inverse_cells.data_ptr<scalar_t>(),
        cells_per_dimension.data_ptr<int32_t>(), pbc.data_ptr<bool>(),
        batch_idx.data_ptr<int32_t>(), cell_offsets.data_ptr<int32_t>(),
        atom_periodic_shifts.data_ptr<int32_t>(),
        atom_to_cell_mapping.data_ptr<int32_t>(), cell_keys.data_ptr<int32_t>(),
        cell_counts.data_ptr<int32_t>(), num_atoms);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
