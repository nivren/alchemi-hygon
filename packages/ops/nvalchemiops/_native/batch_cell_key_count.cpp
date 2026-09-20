#include <torch/extension.h>

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
    torch::Tensor cell_counts);

void batch_cell_key_count_into(
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
  const auto systems = inverse_cells.dim() == 3 ? inverse_cells.size(0) : -1;
  TORCH_CHECK(positions.is_cuda() && positions.dim() == 2 && positions.size(1) == 3,
              "positions must have shape (N, 3) on a HIP device");
  TORCH_CHECK(positions.scalar_type() == torch::kFloat ||
                  positions.scalar_type() == torch::kDouble,
              "positions must have dtype float32 or float64");
  TORCH_CHECK(inverse_cells.is_cuda() &&
                  inverse_cells.sizes() == torch::IntArrayRef({systems, 3, 3}),
              "inverse_cells must have shape (B, 3, 3) on a HIP device");
  TORCH_CHECK(inverse_cells.scalar_type() == positions.scalar_type(),
              "inverse_cells dtype must match positions");
  TORCH_CHECK(cells_per_dimension.is_cuda() &&
                  cells_per_dimension.sizes() == torch::IntArrayRef({systems, 3}) &&
                  cells_per_dimension.scalar_type() == torch::kInt,
              "cells_per_dimension must have shape (B, 3) and dtype int32");
  TORCH_CHECK(pbc.is_cuda() && pbc.sizes() == torch::IntArrayRef({systems, 3}) &&
                  pbc.scalar_type() == torch::kBool,
              "pbc must have shape (B, 3) and dtype bool");
  TORCH_CHECK(batch_idx.is_cuda() && batch_idx.sizes() ==
                  torch::IntArrayRef({positions.size(0)}) &&
                  batch_idx.scalar_type() == torch::kInt,
              "batch_idx must have shape (N,) and dtype int32");
  TORCH_CHECK(cell_offsets.is_cuda() && cell_offsets.sizes() ==
                  torch::IntArrayRef({systems}) &&
                  cell_offsets.scalar_type() == torch::kInt,
              "cell_offsets must have shape (B,) and dtype int32");
  TORCH_CHECK(atom_periodic_shifts.is_cuda() && atom_periodic_shifts.sizes() ==
                  torch::IntArrayRef({positions.size(0), 3}) &&
                  atom_periodic_shifts.scalar_type() == torch::kInt,
              "atom_periodic_shifts must have shape (N, 3) and dtype int32");
  TORCH_CHECK(atom_to_cell_mapping.is_cuda() && atom_to_cell_mapping.sizes() ==
                  torch::IntArrayRef({positions.size(0), 3}) &&
                  atom_to_cell_mapping.scalar_type() == torch::kInt,
              "atom_to_cell_mapping must have shape (N, 3) and dtype int32");
  TORCH_CHECK(cell_keys.is_cuda() && cell_keys.sizes() ==
                  torch::IntArrayRef({positions.size(0)}) &&
                  cell_keys.scalar_type() == torch::kInt,
              "cell_keys must have shape (N,) and dtype int32");
  TORCH_CHECK(cell_counts.is_cuda() && cell_counts.dim() == 1 &&
                  cell_counts.scalar_type() == torch::kInt,
              "cell_counts must have shape (M,) and dtype int32");
  TORCH_CHECK(positions.device() == inverse_cells.device() &&
                  positions.device() == cells_per_dimension.device() &&
                  positions.device() == pbc.device() &&
                  positions.device() == batch_idx.device() &&
                  positions.device() == cell_offsets.device() &&
                  positions.device() == atom_periodic_shifts.device() &&
                  positions.device() == atom_to_cell_mapping.device() &&
                  positions.device() == cell_keys.device() &&
                  positions.device() == cell_counts.device(),
              "all batch cell-key/count tensors must be on the same device");
  TORCH_CHECK(atom_periodic_shifts.is_contiguous() &&
                  atom_to_cell_mapping.is_contiguous() && cell_keys.is_contiguous() &&
                  cell_counts.is_contiguous(),
              "batch cell-key/count output buffers must be contiguous");

  positions = positions.contiguous();
  inverse_cells = inverse_cells.contiguous();
  cells_per_dimension = cells_per_dimension.contiguous();
  pbc = pbc.contiguous();
  batch_idx = batch_idx.contiguous();
  cell_offsets = cell_offsets.contiguous();
  batch_cell_key_count_hip(
      positions, inverse_cells, cells_per_dimension, pbc, batch_idx, cell_offsets,
      atom_periodic_shifts, atom_to_cell_mapping, cell_keys, cell_counts);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("batch_cell_key_count_into", &batch_cell_key_count_into,
             "Write fused native HIP Batch cell-key/count outputs");
}
