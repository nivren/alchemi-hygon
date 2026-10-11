// SPDX-FileCopyrightText: Copyright (c) 2026 alchemi-hygon contributors
// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
#include <tuple>

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
bfgs_jacobi_eigh(at::Tensor input, int64_t max_sweeps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("eigh", &bfgs_jacobi_eigh,
             pybind11::call_guard<pybind11::gil_scoped_release>(),
             "Device-local dynamic FP64 symmetric Jacobi eigensystem");
}
