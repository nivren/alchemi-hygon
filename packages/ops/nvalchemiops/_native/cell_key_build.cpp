#include <torch/extension.h>

void cell_key_build_hip(
    torch::Tensor positions,
    torch::Tensor inverse_cell,
    torch::Tensor dimensions,
    torch::Tensor pbc,
    torch::Tensor atom_periodic_shifts,
    torch::Tensor atom_to_cell_mapping,
    torch::Tensor cell_keys);

void cell_key_build_into(
    torch::Tensor positions,
    torch::Tensor inverse_cell,
    torch::Tensor dimensions,
    torch::Tensor pbc,
    torch::Tensor atom_periodic_shifts,
    torch::Tensor atom_to_cell_mapping,
    torch::Tensor cell_keys) {
  TORCH_CHECK(positions.is_cuda(), "positions must be on a HIP device");
  TORCH_CHECK(positions.dim() == 2 && positions.size(1) == 3,
              "positions must have shape (N, 3)");
  TORCH_CHECK(positions.scalar_type() == torch::kFloat ||
                  positions.scalar_type() == torch::kDouble,
              "positions must have dtype float32 or float64");
  TORCH_CHECK(inverse_cell.is_cuda() && inverse_cell.sizes() ==
                  torch::IntArrayRef({3, 3}),
              "inverse_cell must have shape (3, 3) on a HIP device");
  TORCH_CHECK(inverse_cell.scalar_type() == positions.scalar_type(),
              "inverse_cell dtype must match positions");
  TORCH_CHECK(dimensions.is_cuda() && dimensions.sizes() ==
                  torch::IntArrayRef({3}) &&
                  dimensions.scalar_type() == torch::kInt,
              "dimensions must have shape (3) and dtype int32 on a HIP device");
  TORCH_CHECK(pbc.is_cuda() && pbc.sizes() == torch::IntArrayRef({3}) &&
                  pbc.scalar_type() == torch::kBool,
              "pbc must have shape (3) and dtype bool on a HIP device");
  TORCH_CHECK(atom_periodic_shifts.is_cuda() &&
                  atom_periodic_shifts.sizes() ==
                      torch::IntArrayRef({positions.size(0), 3}) &&
                  atom_periodic_shifts.scalar_type() == torch::kInt,
              "atom_periodic_shifts must have shape (N, 3) and dtype int32");
  TORCH_CHECK(atom_to_cell_mapping.is_cuda() &&
                  atom_to_cell_mapping.sizes() ==
                      torch::IntArrayRef({positions.size(0), 3}) &&
                  atom_to_cell_mapping.scalar_type() == torch::kInt,
              "atom_to_cell_mapping must have shape (N, 3) and dtype int32");
  TORCH_CHECK(cell_keys.is_cuda() &&
                  cell_keys.sizes() == torch::IntArrayRef({positions.size(0)}) &&
                  cell_keys.scalar_type() == torch::kInt,
              "cell_keys must have shape (N) and dtype int32");
  TORCH_CHECK(positions.device() == inverse_cell.device() &&
                  positions.device() == dimensions.device() &&
                  positions.device() == pbc.device() &&
                  positions.device() == atom_periodic_shifts.device() &&
                  positions.device() == atom_to_cell_mapping.device() &&
                  positions.device() == cell_keys.device(),
              "all cell-key tensors must be on the same device");
  TORCH_CHECK(atom_periodic_shifts.is_contiguous() &&
                  atom_to_cell_mapping.is_contiguous() && cell_keys.is_contiguous(),
              "cell-key output buffers must be contiguous");

  // Read-only inputs may be strided.  Normalize them before handing raw
  // pointers to the one-thread-per-atom kernel; output mutation remains in
  // the caller-owned contiguous buffers.
  positions = positions.contiguous();
  inverse_cell = inverse_cell.contiguous();
  dimensions = dimensions.contiguous();
  pbc = pbc.contiguous();

  cell_key_build_hip(
      positions,
      inverse_cell,
      dimensions,
      pbc,
      atom_periodic_shifts,
      atom_to_cell_mapping,
      cell_keys);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "cell_key_build_into",
      &cell_key_build_into,
      "Write native HIP cell-list key outputs into preallocated buffers");
}
