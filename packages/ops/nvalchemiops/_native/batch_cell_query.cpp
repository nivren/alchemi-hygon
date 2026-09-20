#include <torch/extension.h>

#include <cstdint>
#include <limits>

void batch_cell_query_hip(torch::Tensor positions, torch::Tensor cells,
                          torch::Tensor pbc, double cutoff,
                          torch::Tensor batch_idx,
                          torch::Tensor cells_per_dimension,
                          torch::Tensor neighbor_search_radius,
                          torch::Tensor cell_offsets,
                          torch::Tensor atom_periodic_shifts,
                          torch::Tensor atom_to_cell_mapping,
                          torch::Tensor atoms_per_cell_count,
                          torch::Tensor cell_atom_start_indices,
                          torch::Tensor cell_atom_list,
                          torch::Tensor neighbor_matrix,
                          torch::Tensor neighbor_matrix_shifts,
                          torch::Tensor num_neighbors, bool half_fill,
                          int64_t fill_value);

void batch_cell_query_into(
    torch::Tensor positions, torch::Tensor cells, torch::Tensor pbc,
    double cutoff, torch::Tensor batch_idx, torch::Tensor cells_per_dimension,
    torch::Tensor neighbor_search_radius, torch::Tensor cell_offsets,
    torch::Tensor atom_periodic_shifts, torch::Tensor atom_to_cell_mapping,
    torch::Tensor atoms_per_cell_count, torch::Tensor cell_atom_start_indices,
    torch::Tensor cell_atom_list, torch::Tensor neighbor_matrix,
    torch::Tensor neighbor_matrix_shifts, torch::Tensor num_neighbors,
    bool half_fill, int64_t fill_value) {
  TORCH_CHECK(positions.is_cuda() && positions.dim() == 2 &&
                  positions.size(1) == 3 &&
                  (positions.scalar_type() == torch::kFloat ||
                   positions.scalar_type() == torch::kDouble),
              "positions must have shape (N, 3) and float32/float64 dtype on HIP");
  TORCH_CHECK(cells.is_cuda() && cells.dim() == 3 && cells.size(1) == 3 &&
                  cells.size(2) == 3 && cells.scalar_type() == positions.scalar_type(),
              "cells must have shape (B, 3, 3) and match positions dtype on HIP");
  TORCH_CHECK(pbc.is_cuda() && pbc.dim() == 2 && pbc.size(0) == cells.size(0) &&
                  pbc.size(1) == 3 && pbc.scalar_type() == torch::kBool,
              "pbc must have shape (B, 3) and bool dtype on HIP");
  TORCH_CHECK(batch_idx.is_cuda() && batch_idx.dim() == 1 &&
                  batch_idx.size(0) == positions.size(0) &&
                  batch_idx.scalar_type() == torch::kInt,
              "batch_idx must have shape (N,) and int32 dtype on HIP");
  TORCH_CHECK(cells_per_dimension.is_cuda() &&
                  cells_per_dimension.dim() == 2 &&
                  cells_per_dimension.size(0) == cells.size(0) &&
                  cells_per_dimension.size(1) == 3 &&
                  cells_per_dimension.scalar_type() == torch::kInt,
              "cells_per_dimension must have shape (B, 3) and int32 dtype on HIP");
  TORCH_CHECK(neighbor_search_radius.is_cuda() &&
                  neighbor_search_radius.dim() == 2 &&
                  neighbor_search_radius.size(0) == cells.size(0) &&
                  neighbor_search_radius.size(1) == 3 &&
                  neighbor_search_radius.scalar_type() == torch::kInt,
              "neighbor_search_radius must have shape (B, 3) and int32 dtype on HIP");
  TORCH_CHECK(cell_offsets.is_cuda() && cell_offsets.dim() == 1 &&
                  cell_offsets.size(0) == cells.size(0) &&
                  cell_offsets.scalar_type() == torch::kInt,
              "cell_offsets must have shape (B,) and int32 dtype on HIP");
  TORCH_CHECK(atom_periodic_shifts.is_cuda() &&
                  atom_periodic_shifts.dim() == 2 &&
                  atom_periodic_shifts.size(0) == positions.size(0) &&
                  atom_periodic_shifts.size(1) == 3 &&
                  atom_periodic_shifts.scalar_type() == torch::kInt,
              "atom_periodic_shifts must have shape (N, 3) and int32 dtype on HIP");
  TORCH_CHECK(atom_to_cell_mapping.is_cuda() &&
                  atom_to_cell_mapping.dim() == 2 &&
                  atom_to_cell_mapping.size(0) == positions.size(0) &&
                  atom_to_cell_mapping.size(1) == 3 &&
                  atom_to_cell_mapping.scalar_type() == torch::kInt,
              "atom_to_cell_mapping must have shape (N, 3) and int32 dtype on HIP");
  TORCH_CHECK(atoms_per_cell_count.is_cuda() &&
                  atoms_per_cell_count.dim() == 1 &&
                  atoms_per_cell_count.scalar_type() == torch::kInt,
              "atoms_per_cell_count must be a one-dimensional int32 HIP tensor");
  TORCH_CHECK(cell_atom_start_indices.is_cuda() &&
                  cell_atom_start_indices.sizes() == atoms_per_cell_count.sizes() &&
                  cell_atom_start_indices.scalar_type() == torch::kInt,
              "cell_atom_start_indices must match count shape and be int32 on HIP");
  TORCH_CHECK(cell_atom_list.is_cuda() && cell_atom_list.dim() == 1 &&
                  cell_atom_list.scalar_type() == torch::kInt,
              "cell_atom_list must be a one-dimensional int32 HIP tensor");
  TORCH_CHECK(neighbor_matrix.is_cuda() && neighbor_matrix.dim() == 2 &&
                  neighbor_matrix.size(0) == positions.size(0) &&
                  neighbor_matrix.scalar_type() == torch::kInt,
              "neighbor_matrix must have shape (N, K) and int32 dtype on HIP");
  TORCH_CHECK(neighbor_matrix_shifts.is_cuda() &&
                  neighbor_matrix_shifts.dim() == 3 &&
                  neighbor_matrix_shifts.size(0) == neighbor_matrix.size(0) &&
                  neighbor_matrix_shifts.size(1) == neighbor_matrix.size(1) &&
                  neighbor_matrix_shifts.size(2) == 3 &&
                  neighbor_matrix_shifts.scalar_type() == torch::kInt,
              "neighbor_matrix_shifts must have shape (N, K, 3) and int32 dtype");
  TORCH_CHECK(num_neighbors.is_cuda() && num_neighbors.dim() == 1 &&
                  num_neighbors.size(0) == positions.size(0) &&
                  num_neighbors.scalar_type() == torch::kInt,
              "num_neighbors must have shape (N,) and int32 dtype on HIP");
  TORCH_CHECK(fill_value >= std::numeric_limits<int32_t>::min() &&
                  fill_value <= std::numeric_limits<int32_t>::max(),
              "fill_value must fit int32");
  TORCH_CHECK(neighbor_matrix.size(1) > 0,
              "neighbor_matrix must provide positive capacity K");
  const auto tensors = {cells, pbc, batch_idx, cells_per_dimension,
                        neighbor_search_radius, cell_offsets,
                        atom_periodic_shifts, atom_to_cell_mapping,
                        atoms_per_cell_count, cell_atom_start_indices,
                        cell_atom_list, neighbor_matrix,
                        neighbor_matrix_shifts, num_neighbors};
  for (const auto& tensor : tensors) {
    TORCH_CHECK(tensor.device() == positions.device(),
                "all native HIP Batch query tensors must share one device");
    TORCH_CHECK(tensor.is_contiguous(),
                "all native HIP Batch query tensors must be contiguous");
  }
  batch_cell_query_hip(
      positions, cells, pbc, cutoff, batch_idx, cells_per_dimension,
      neighbor_search_radius, cell_offsets, atom_periodic_shifts,
      atom_to_cell_mapping, atoms_per_cell_count, cell_atom_start_indices,
      cell_atom_list, neighbor_matrix, neighbor_matrix_shifts, num_neighbors,
      half_fill, fill_value);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("batch_cell_query_into", &batch_cell_query_into,
             "Enumerate a native HIP Batch cell-list query candidate");
}
