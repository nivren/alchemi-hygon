// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
#include <tuple>
#include <vector>
using Result = std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor>;
Result ase_jacobi_candidate(at::Tensor input, int64_t max_sweeps);
std::vector<Result> ase_jacobi_heterogeneous(std::vector<at::Tensor> inputs, int64_t max_sweeps);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,module) {
  module.def("eigh",&ase_jacobi_candidate,pybind11::call_guard<pybind11::gil_scoped_release>());
  module.def("heterogeneous",&ase_jacobi_heterogeneous,pybind11::call_guard<pybind11::gil_scoped_release>());
}
