#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cmath>

template <typename scalar_t>
__global__ void cell_keys_kernel(
    const scalar_t* positions,
    const scalar_t* inverse_cell,
    const int32_t* dimensions,
    const bool* pbc,
    int32_t* atom_periodic_shifts,
    int32_t* atom_to_cell_mapping,
    int32_t* keys,
    int64_t num_atoms) {
  const int64_t atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= num_atoms) {
    return;
  }

  int32_t coordinate[3];
#pragma unroll
  for (int dimension = 0; dimension < 3; ++dimension) {
    scalar_t fractional = 0;
#pragma unroll
    for (int cartesian = 0; cartesian < 3; ++cartesian) {
      fractional += positions[atom * 3 + cartesian] *
                    inverse_cell[cartesian * 3 + dimension];
    }
    const int32_t shift = pbc[dimension]
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
  keys[atom] = coordinate[0] + dimensions[0] *
      (coordinate[1] + dimensions[1] * coordinate[2]);
}

void cell_keys_cuda(
    torch::Tensor positions,
    torch::Tensor inverse_cell,
    torch::Tensor dimensions,
    torch::Tensor pbc,
    torch::Tensor atom_periodic_shifts,
    torch::Tensor atom_to_cell_mapping,
    torch::Tensor keys) {
  constexpr int threads = 256;
  const auto num_atoms = positions.size(0);
  const int blocks = static_cast<int>((num_atoms + threads - 1) / threads);
  if (blocks == 0) {
    return;
  }
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(positions.scalar_type(), "cell_keys_cuda", [&] {
    cell_keys_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        positions.data_ptr<scalar_t>(),
        inverse_cell.data_ptr<scalar_t>(),
        dimensions.data_ptr<int32_t>(),
        pbc.data_ptr<bool>(),
        atom_periodic_shifts.data_ptr<int32_t>(),
        atom_to_cell_mapping.data_ptr<int32_t>(),
        keys.data_ptr<int32_t>(),
        num_atoms);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
