// SPDX-FileCopyrightText: Copyright (c) 2026 alchemi-hygon contributors
// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>

#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>

namespace {
void check(rocblas_status status, const char* call) {
  TORCH_CHECK(status == rocblas_status_success, call,
              " failed with rocBLAS/rocSOLVER API status ", static_cast<int>(status));
}

// Handles are never shared between host threads or devices. rocBLAS owns its
// workspace. Normal calls check every status; shutdown cleanup must not throw.
struct Handle {
  rocblas_handle value = nullptr;
  int device;
  explicit Handle(int device_index) : device(device_index) {
    check(rocblas_create_handle(&value), "rocblas_create_handle");
  }
  ~Handle() noexcept {
    int previous = -1;
    if (hipGetDevice(&previous) == hipSuccess &&
        hipSetDevice(device) == hipSuccess) {
      rocblas_destroy_handle(value);
      hipSetDevice(previous);
    }
  }
};

rocblas_handle current_handle(int device) {
  thread_local std::unordered_map<int, std::unique_ptr<Handle>> handles;
  auto& owner = handles[device];
  if (!owner) owner = std::make_unique<Handle>(device);
  check(rocblas_set_stream(owner->value, at::cuda::getCurrentCUDAStream(device)),
        "rocblas_set_stream");
  return owner->value;
}
}  // namespace

std::string bfgs_rocsolver_version() {
  char version[128] = {};
  check(rocsolver_get_version_string(version, sizeof(version)),
        "rocsolver_get_version_string");
  return version;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
bfgs_eigh_hip(at::Tensor input, const std::string& uplo, int64_t max_sweeps) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kDouble &&
                  input.is_contiguous() && input.layout() == at::kStrided &&
                  !input.requires_grad(),
              "BFGS eigh requires contiguous, non-grad FP64 HIP input");
  TORCH_CHECK((input.dim() == 2 || input.dim() == 3) &&
                  input.size(-2) == input.size(-1) && input.numel() > 0,
              "BFGS eigh requires nonempty [n,n] or [B,n,n]");
  TORCH_CHECK(uplo == "L" || uplo == "U", "UPLO must be L or U");
  const int64_t n64 = input.size(-1);
  const int64_t count64 = input.dim() == 2 ? 1 : input.size(0);
  TORCH_CHECK(n64 <= std::numeric_limits<rocblas_int>::max() &&
                  count64 <= std::numeric_limits<rocblas_int>::max() &&
                  max_sweeps > 0 && max_sweeps <= std::numeric_limits<rocblas_int>::max(),
              "BFGS eigh dimensions and sweep limit must fit positive int32");
  const c10::cuda::CUDAGuard guard(input.device());
  const auto handle = current_handle(input.get_device());
  const auto n = static_cast<rocblas_int>(n64);
  const auto count = static_cast<rocblas_int>(count64);
  auto work = input.clone(at::MemoryFormat::Contiguous);
  auto shape = input.sizes().vec();
  shape.pop_back();
  auto w = at::empty(shape, input.options());
  auto info = at::zeros({count64}, input.options().dtype(at::kInt));
  auto residual = at::zeros({count64}, input.options());
  auto sweeps = at::zeros({count64}, input.options().dtype(at::kInt));
  const auto triangle = uplo == "L" ? rocblas_fill_upper : rocblas_fill_lower;
  // M4: the admitted DTK DC solver faults or returns an incorrect spectrum on
  // a captured 285-DOF CSP Hessian with a large repeated-alpha eigenspace.
  // Use checked native Jacobi for every BFGS width, without spectral changes
  // or a failure-triggered fallback. See reports/g2-ase-bfgs-m4.md.
  check(rocsolver_dsyevj_strided_batched(
            handle, rocblas_esort_ascending, rocblas_evect_original, triangle,
            n, work.data_ptr<double>(), n, n64 * n64, 1e-15,
            residual.data_ptr<double>(), static_cast<rocblas_int>(max_sweeps),
            sweeps.data_ptr<rocblas_int>(), w.data_ptr<double>(), n,
            info.data_ptr<rocblas_int>(), count),
        "rocsolver_dsyevj_strided_batched");
  // Column-major vectors become rows in the row-major view; return columns.
  auto vectors = work.transpose(-1, -2).contiguous();
  return {w, vectors, info, residual, sweeps};
}
