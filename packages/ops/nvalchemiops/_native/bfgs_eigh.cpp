// SPDX-FileCopyrightText: Copyright (c) 2026 alchemi-hygon contributors
// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
#include <string>
#include <tuple>

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
bfgs_eigh_hip(at::Tensor input, const std::string& uplo, int64_t max_sweeps);
std::string bfgs_rocsolver_version();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("eigh", &bfgs_eigh_hip,
             "Forward-only FP64 BFGS eigensystem on the current HIP stream");
  module.def("rocsolver_version", &bfgs_rocsolver_version);
}
