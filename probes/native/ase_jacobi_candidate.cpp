// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
#include <tuple>

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
ase_jacobi_candidate(at::Tensor input, int64_t max_sweeps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("eigh", &ase_jacobi_candidate,
             "Isolated dynamic FP64 full-eigensystem Jacobi prototype");
}
