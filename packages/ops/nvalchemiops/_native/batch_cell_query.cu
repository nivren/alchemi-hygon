#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

__device__ inline int32_t floor_divide(int32_t value, int32_t divisor) {
  int32_t quotient = value / divisor;
  const int32_t remainder = value % divisor;
  if (remainder < 0) {
    --quotient;
  }
  return quotient;
}

template <typename scalar_t>
__global__ void batch_cell_query_kernel(
    const scalar_t* positions, const scalar_t* cells, const bool* pbc,
    scalar_t cutoff, const int32_t* batch_idx,
    const int32_t* cells_per_dimension, const int32_t* neighbor_search_radius,
    const int32_t* cell_offsets, const int32_t* atom_periodic_shifts,
    const int32_t* atom_to_cell_mapping, const int32_t* atoms_per_cell_count,
    const int32_t* cell_atom_start_indices, const int32_t* cell_atom_list,
    int32_t* neighbor_matrix, int32_t* neighbor_matrix_shifts,
    int32_t* num_neighbors, int32_t fill_value, bool half_fill,
    int32_t* error, int64_t num_atoms, int64_t capacity) {
  const int64_t atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= num_atoms) {
    return;
  }

  int32_t* row = neighbor_matrix + atom * capacity;
  int32_t* row_shifts = neighbor_matrix_shifts + atom * capacity * 3;
  for (int64_t slot = 0; slot < capacity; ++slot) {
    row[slot] = fill_value;
    row_shifts[slot * 3 + 0] = 0;
    row_shifts[slot * 3 + 1] = 0;
    row_shifts[slot * 3 + 2] = 0;
  }
  num_neighbors[atom] = 0;

  const int32_t system = batch_idx[atom];
  const int32_t* dimensions = cells_per_dimension + system * 3;
  const int32_t* radius = neighbor_search_radius + system * 3;
  const bool* system_pbc = pbc + system * 3;
  const int32_t source_x = atom_to_cell_mapping[atom * 3 + 0];
  const int32_t source_y = atom_to_cell_mapping[atom * 3 + 1];
  const int32_t source_z = atom_to_cell_mapping[atom * 3 + 2];
  const int32_t cell_offset = cell_offsets[system];
  const scalar_t* geometry = cells + system * 9;
  const scalar_t cutoff_sq = cutoff * cutoff;
  int32_t count = 0;

  for (int32_t dx = -radius[0]; dx <= radius[0]; ++dx) {
    for (int32_t dy = -radius[1]; dy <= radius[1]; ++dy) {
      for (int32_t dz = -radius[2]; dz <= radius[2]; ++dz) {
        const int32_t raw_x = source_x + dx;
        const int32_t raw_y = source_y + dy;
        const int32_t raw_z = source_z + dz;
        int32_t image_x = 0;
        int32_t image_y = 0;
        int32_t image_z = 0;
        int32_t target_x = raw_x;
        int32_t target_y = raw_y;
        int32_t target_z = raw_z;
        const int32_t raw[3] = {raw_x, raw_y, raw_z};
        int32_t target[3] = {target_x, target_y, target_z};
        int32_t image[3] = {image_x, image_y, image_z};
        bool valid = true;
        for (int dimension = 0; dimension < 3; ++dimension) {
          if (system_pbc[dimension]) {
            image[dimension] = floor_divide(raw[dimension], dimensions[dimension]);
            target[dimension] = raw[dimension] -
                               image[dimension] * dimensions[dimension];
          } else if (raw[dimension] < 0 ||
                     raw[dimension] >= dimensions[dimension]) {
            valid = false;
          } else {
            image[dimension] = 0;
            target[dimension] = raw[dimension];
          }
        }
        if (!valid) {
          continue;
        }
        const int32_t target_key =
            cell_offset + target[0] + dimensions[0] *
                                      (target[1] + dimensions[1] * target[2]);
        const int32_t target_count = atoms_per_cell_count[target_key];
        const int32_t target_start = cell_atom_start_indices[target_key];
        for (int32_t rank = 0; rank < target_count; ++rank) {
          const int32_t column = cell_atom_list[target_start + rank];
          int32_t shift[3];
          for (int dimension = 0; dimension < 3; ++dimension) {
            shift[dimension] = atom_periodic_shifts[atom * 3 + dimension] -
                              atom_periodic_shifts[column * 3 + dimension] +
                              image[dimension];
          }
          scalar_t vector[3];
          scalar_t distance_sq = 0;
          for (int cartesian = 0; cartesian < 3; ++cartesian) {
            scalar_t value = positions[atom * 3 + cartesian] -
                             positions[column * 3 + cartesian];
            for (int dimension = 0; dimension < 3; ++dimension) {
              value -= static_cast<scalar_t>(shift[dimension]) *
                       geometry[dimension * 3 + cartesian];
            }
            vector[cartesian] = value;
            distance_sq += value * value;
          }
          if (column == atom && shift[0] == 0 && shift[1] == 0 && shift[2] == 0) {
            continue;
          }
          if (!(distance_sq < cutoff_sq)) {
            continue;
          }
          if (distance_sq == static_cast<scalar_t>(0)) {
            atomicExch(error, 2);
            return;
          }
          if (half_fill) {
            const bool positive_image =
                (shift[0] > 0) ||
                (shift[0] == 0 &&
                 (shift[1] > 0 ||
                  (shift[1] == 0 && shift[2] > 0)));
            if (!(atom < column || (atom == column && positive_image))) {
              continue;
            }
          }
          if (count >= capacity) {
            atomicExch(error, 1);
            return;
          }
          row[count] = column;
          row_shifts[count * 3 + 0] = shift[0];
          row_shifts[count * 3 + 1] = shift[1];
          row_shifts[count * 3 + 2] = shift[2];
          ++count;
        }
      }
    }
  }
  num_neighbors[atom] = count;
}

void batch_cell_query_hip(
    torch::Tensor positions, torch::Tensor cells, torch::Tensor pbc,
    double cutoff, torch::Tensor batch_idx, torch::Tensor cells_per_dimension,
    torch::Tensor neighbor_search_radius, torch::Tensor cell_offsets,
    torch::Tensor atom_periodic_shifts, torch::Tensor atom_to_cell_mapping,
    torch::Tensor atoms_per_cell_count, torch::Tensor cell_atom_start_indices,
    torch::Tensor cell_atom_list, torch::Tensor neighbor_matrix,
    torch::Tensor neighbor_matrix_shifts, torch::Tensor num_neighbors,
    bool half_fill, int64_t fill_value) {
  const auto stream = at::cuda::getCurrentCUDAStream();
  auto error = torch::zeros(
      {1}, torch::TensorOptions().device(positions.device()).dtype(torch::kInt));
  const auto num_atoms = positions.size(0);
  if (num_atoms == 0) {
    neighbor_matrix.fill_(fill_value);
    neighbor_matrix_shifts.zero_();
    num_neighbors.zero_();
    return;
  }
  constexpr int threads = 128;
  const int blocks = static_cast<int>((num_atoms + threads - 1) / threads);
  AT_DISPATCH_FLOATING_TYPES(positions.scalar_type(), "batch_cell_query_hip", [&] {
    batch_cell_query_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        positions.data_ptr<scalar_t>(), cells.data_ptr<scalar_t>(),
        pbc.data_ptr<bool>(), static_cast<scalar_t>(cutoff),
        batch_idx.data_ptr<int32_t>(), cells_per_dimension.data_ptr<int32_t>(),
        neighbor_search_radius.data_ptr<int32_t>(),
        cell_offsets.data_ptr<int32_t>(),
        atom_periodic_shifts.data_ptr<int32_t>(),
        atom_to_cell_mapping.data_ptr<int32_t>(),
        atoms_per_cell_count.data_ptr<int32_t>(),
        cell_atom_start_indices.data_ptr<int32_t>(),
        cell_atom_list.data_ptr<int32_t>(),
        neighbor_matrix.data_ptr<int32_t>(),
        neighbor_matrix_shifts.data_ptr<int32_t>(),
        num_neighbors.data_ptr<int32_t>(), static_cast<int32_t>(fill_value),
        half_fill, error.data_ptr<int32_t>(), num_atoms,
        neighbor_matrix.size(1));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const auto error_code = error.cpu().item<int32_t>();
  TORCH_CHECK(error_code == 0,
              error_code == 1 ? "native HIP Batch query capacity overflow"
                               : "native HIP Batch query found overlapping active pair");
}
