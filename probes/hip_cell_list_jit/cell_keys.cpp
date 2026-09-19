#include <torch/extension.h>

#include <vector>

void cell_keys_cuda(
    torch::Tensor positions,
    torch::Tensor inverse_cell,
    torch::Tensor dimensions,
    torch::Tensor pbc,
    torch::Tensor atom_periodic_shifts,
    torch::Tensor atom_to_cell_mapping,
    torch::Tensor keys);

std::vector<torch::Tensor> cell_keys(
    torch::Tensor positions,
    torch::Tensor inverse_cell,
    torch::Tensor dimensions,
    torch::Tensor pbc) {
  TORCH_CHECK(positions.is_cuda(), "positions must be on a HIP device");
  TORCH_CHECK(positions.dim() == 2 && positions.size(1) == 3,
              "positions must have shape (N, 3)");
  TORCH_CHECK(inverse_cell.is_cuda() && inverse_cell.sizes() == torch::IntArrayRef({3, 3}),
              "inverse_cell must have shape (3, 3) on the same HIP device");
  TORCH_CHECK(dimensions.is_cuda() && dimensions.numel() == 3,
              "dimensions must have three device values");
  TORCH_CHECK(pbc.is_cuda() && pbc.numel() == 3,
              "pbc must have three device values");
  TORCH_CHECK(positions.scalar_type() == inverse_cell.scalar_type(),
              "positions and inverse_cell must have the same dtype");
  TORCH_CHECK(positions.device() == inverse_cell.device() &&
                  positions.device() == dimensions.device() &&
                  positions.device() == pbc.device(),
              "all inputs must be on the same device");

  positions = positions.contiguous();
  inverse_cell = inverse_cell.contiguous();
  dimensions = dimensions.to(torch::kInt32).contiguous();
  pbc = pbc.to(torch::kBool).contiguous();

  auto int_options = positions.options().dtype(torch::kInt32);
  auto atom_periodic_shifts = torch::empty({positions.size(0), 3}, int_options);
  auto atom_to_cell_mapping = torch::empty({positions.size(0), 3}, int_options);
  auto keys = torch::empty({positions.size(0)}, int_options);
  cell_keys_cuda(
      positions,
      inverse_cell,
      dimensions,
      pbc,
      atom_periodic_shifts,
      atom_to_cell_mapping,
      keys);
  return {atom_periodic_shifts, atom_to_cell_mapping, keys};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("cell_keys", &cell_keys, "HIP cell-list fractional coordinates and keys");
}
