// SPDX-License-Identifier: Apache-2.0
// Original probe implementation. Not a registered or supported product backend.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>

namespace {
constexpr int threads = 256;

__global__ void prepare_scale(const double* input, double* shift, double* norm0, int n, int64_t batch) {
  const int lane = threadIdx.x;
  const int64_t square = int64_t(n)*n;
  __shared__ double trace[threads], total[threads];
  for (int64_t b = blockIdx.x; b < batch; b += gridDim.x) {
    const int64_t base = b*square;
    double local_trace = 0.0, local_total = 0.0;
    for (int64_t k = lane; k < int64_t(n)*n; k += blockDim.x) {
      const int row = k/n, col = k % n;
      const double value = row >= col ? input[base+k] : input[base+int64_t(col)*n+row];
      local_total += value*value;
      if (row == col) local_trace += value/n;
    }
    trace[lane] = local_trace; total[lane] = local_total;
    __syncthreads();
    for (int distance = threads/2; distance > 0; distance /= 2) {
      if (lane < distance) {
        trace[lane] += trace[lane+distance]; total[lane] += total[lane+distance];
      }
      __syncthreads();
    }
    if (lane == 0) { shift[b] = trace[0]; norm0[b] = sqrt(total[0]); }
    __syncthreads();
  }
}

__global__ void initialize(const double* input, double* a, double* a_other,
                           double* v, double* v_other, const double* shift, int n, int64_t elements) {
  const int64_t square = int64_t(n)*n;
  for (int64_t index = int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
       index < elements; index += int64_t(gridDim.x)*blockDim.x) {
    const int64_t b = index/square, k = index%square, offset = b*square;
    const int row = k % n, col = k / n;
    const double value = row >= col ? input[offset + int64_t(row)*n + col]
                                   : input[offset + int64_t(col)*n + row];
    a[offset + k] = a_other[offset + k] = row == col ? value-shift[b] : value;
    v[offset + k] = v_other[offset + k] = row == col ? 1.0 : 0.0;
  }
}

__global__ void rotations(const double* a, double* c, double* s, double* change,
                          int* partner, const int* info, int n, int round, int64_t pair_count) {
  const int m = n + (n % 2);
  for (int64_t index = int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
       index < pair_count; index += int64_t(gridDim.x)*blockDim.x) {
    const int64_t b = index/(m/2);
    if (info[b] != 1) continue;
    const int k = index%(m/2);
    int p = k == 0 ? 0 : 1 + (int64_t(k) - 1 + round) % (m - 1);
    int q = 1 + (int64_t(m) - 2 - k + round) % (m - 1);
    if (p > q) { const int tmp = p; p = q; q = tmp; }
    const int64_t meta = b*n, base = b*int64_t(n)*n;
    if (q >= n) {
      c[meta+p] = 1.0; s[meta+p] = change[meta+p] = 0.0;
      partner[meta+p] = p;
      continue;
    }
    const double off = a[base + p + int64_t(q)*n];
    double t = 0.0;
    if (off != 0.0) {
      const double delta = 0.5*a[base + q + int64_t(q)*n] - 0.5*a[base + p + int64_t(p)*n];
      t = off / (delta + copysign(hypot(delta, off), delta));
    }
    const double cosine = 1.0 / sqrt(1.0 + t*t);
    const double sine = t*cosine;
    c[meta+p] = c[meta+q] = cosine;
    s[meta+p] = -sine; s[meta+q] = sine;
    change[meta+p] = -t*off; change[meta+q] = t*off;
    partner[meta+p] = q; partner[meta+q] = p;
  }
}

__global__ void apply_rotations(const double* a, const double* v, double* out_a,
                                double* out_v, const double* c, const double* s,
                                const double* change, const int* partner,
                                const int* info, int n, int64_t elements) {
  const int64_t square = int64_t(n)*n;
  for (int64_t index = int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
       index < elements; index += int64_t(gridDim.x)*blockDim.x) {
    const int64_t b = index/square, k = index%square;
    if (info[b] != 1) continue;
    const int64_t base = b*square, meta = b*n;
    const int row = k % n, col = k / n;
    // Both triangles use the same arithmetic ordering, preserving exact symmetry.
    const int i = row < col ? row : col, j = row < col ? col : row;
    const int pi = partner[meta+i], pj = partner[meta+j];
    double value;
    if (i == j) {
      value = a[base + k] + change[meta+i];
    } else if (pi == j) {
      value = 0.0;
    } else {
      const double ci = c[meta+i], cj = c[meta+j];
      const double si = s[meta+i], sj = s[meta+j];
      const int pr = row <= col ? pi : pj, pc = row <= col ? pj : pi;
      // A is exactly symmetric. Read the equivalent triangle whose row index
      // varies along adjacent lanes, retaining the canonical arithmetic order.
      const double direct = a[base+row+int64_t(col)*n];
      const double col_mix = a[base+row+int64_t(pc)*n];
      const double row_mix = a[base+pr+int64_t(col)*n];
      const double right = row <= col ? col_mix : row_mix;
      const double left = row <= col ? row_mix : col_mix;
      const double cross = a[base+pr+int64_t(pc)*n];
      value = ci*(cj*direct + sj*right) + si*(cj*left + sj*cross);
    }
    out_a[base+k] = value;
    out_v[base+k] = c[meta+col]*v[base+k]
                    + s[meta+col]*v[base+row+int64_t(partner[meta+col])*n];
  }
}

__global__ void convergence(const double* a, const double* v, double* a_other,
                            double* v_other, double* norm0, double* residual,
                            int* info, int* sweeps, int n, int sweep, int64_t batch) {
  const int lane = threadIdx.x;
  __shared__ double off[threads], total[threads];
  for (int64_t b = blockIdx.x; b < batch; b += gridDim.x) {
    if (info[b] != 1) continue;
    const int64_t base = b*int64_t(n)*n;
    double local_off = 0.0, local_total = 0.0;
    for (int64_t k = lane; k < int64_t(n)*n; k += blockDim.x) {
      const double value = a[base+k];
      const double square = value*value;
      local_total += square;
      if (k % n != k / n) local_off += square;
    }
    off[lane] = local_off; total[lane] = local_total;
    __syncthreads();
    for (int distance = threads/2; distance > 0; distance /= 2) {
      if (lane < distance) {
        off[lane] += off[lane+distance];
        total[lane] += total[lane+distance];
      }
      __syncthreads();
    }
    if (lane == 0) {
      residual[b] = norm0[b] == 0.0 ? sqrt(off[0]) : sqrt(off[0])/norm0[b];
      sweeps[b] = sweep;
      if (!isfinite(norm0[b]) || !isfinite(total[0]) || !isfinite(residual[b])) info[b] = 2;
      else if (residual[b] <= 1e-15) info[b] = 0;
    }
    __syncthreads();
    // Frozen members can skip future kernels irrespective of ping-pong parity.
    if (info[b] == 0) {
      for (int64_t k = lane; k < int64_t(n)*n; k += blockDim.x) {
        a_other[base+k] = a[base+k]; v_other[base+k] = v[base+k];
      }
    }
    __syncthreads();
  }
}

__global__ void sort_vectors(const double* v, const int64_t* permutation,
                             double* out, int n, int64_t elements) {
  const int64_t square = int64_t(n)*n;
  for (int64_t index = int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
       index < elements; index += int64_t(gridDim.x)*blockDim.x) {
    const int64_t b = index/square, k = index%square;
    const int row = k/n, col = k % n;
    out[index] = v[b*square+row+permutation[b*n+col]*n];
  }
}
}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
ase_jacobi_candidate(at::Tensor input, int64_t max_sweeps) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == at::kDouble &&
              input.layout() == at::kStrided && input.is_contiguous() && !input.requires_grad(),
              "prototype requires contiguous, non-grad FP64 HIP input");
  TORCH_CHECK((input.dim() == 2 || input.dim() == 3) && input.size(-2) == input.size(-1) && input.numel() > 0,
              "prototype requires [n,n] or [B,n,n]");
  const int64_t n64 = input.size(-1), batch = input.dim() == 2 ? 1 : input.size(0);
  TORCH_CHECK(n64 > 0 && n64 < std::numeric_limits<int>::max() && batch > 0 &&
              n64 <= std::numeric_limits<int64_t>::max()/n64 &&
              batch <= std::numeric_limits<int64_t>::max()/(n64*n64) &&
              max_sweeps >= 1 && max_sweeps <= 100, "unrepresentable size or invalid sweep limit");
  const int n = static_cast<int>(n64);
  const int64_t elements = batch*n64*n64, pair_count = batch*((n64+n64%2)/2);
  const c10::cuda::CUDAGuard guard(input.device());
  const auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
  auto a = at::empty_like(input), a_other = at::empty_like(input);
  auto v = at::empty_like(input), v_other = at::empty_like(input);
  auto c = at::empty({batch,n}, input.options()), s = at::empty_like(c), change = at::empty_like(c);
  auto partner = at::empty({batch,n}, input.options().dtype(at::kInt));
  auto norm0 = at::empty({batch}, input.options()), residual = at::empty_like(norm0);
  auto shift = at::empty_like(norm0);
  auto info = at::ones({batch}, input.options().dtype(at::kInt)), sweeps = at::zeros_like(info);
  const int64_t max_grid = at::cuda::getCurrentDeviceProperties()->maxGridSize[0];
  const auto grid_for = [max_grid](int64_t count) {
    return static_cast<unsigned int>(std::min((count-1)/threads+1, max_grid));
  };
  const unsigned int matrix_grid = static_cast<unsigned int>(std::min(batch,max_grid));
  const auto grid = grid_for(elements), pair_grid = grid_for(pair_count);
  prepare_scale<<<matrix_grid,threads,0,stream>>>(input.data_ptr<double>(),shift.data_ptr<double>(),norm0.data_ptr<double>(),n,batch);
  initialize<<<grid,threads,0,stream>>>(input.data_ptr<double>(), a.data_ptr<double>(),
      a_other.data_ptr<double>(), v.data_ptr<double>(), v_other.data_ptr<double>(), shift.data_ptr<double>(), n, elements);
  convergence<<<matrix_grid,threads,0,stream>>>(a.data_ptr<double>(), v.data_ptr<double>(),
      a_other.data_ptr<double>(), v_other.data_ptr<double>(), norm0.data_ptr<double>(),
      residual.data_ptr<double>(), info.data_ptr<int>(), sweeps.data_ptr<int>(), n, 0, batch);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  bool unfinished = info.eq(1).any().item<bool>();  // Explicit scalar-only control transfer.
  for (int sweep = 1; sweep <= max_sweeps && unfinished; ++sweep) {
    for (int round = 0; round < n+n%2-1; ++round) {
      rotations<<<pair_grid,threads,0,stream>>>(a.data_ptr<double>(), c.data_ptr<double>(),
          s.data_ptr<double>(), change.data_ptr<double>(), partner.data_ptr<int>(), info.data_ptr<int>(), n, round, pair_count);
      apply_rotations<<<grid,threads,0,stream>>>(a.data_ptr<double>(), v.data_ptr<double>(),
          a_other.data_ptr<double>(), v_other.data_ptr<double>(), c.data_ptr<double>(),
          s.data_ptr<double>(), change.data_ptr<double>(), partner.data_ptr<int>(), info.data_ptr<int>(), n, elements);
      std::swap(a, a_other); std::swap(v, v_other);
    }
    convergence<<<matrix_grid,threads,0,stream>>>(a.data_ptr<double>(), v.data_ptr<double>(),
        a_other.data_ptr<double>(), v_other.data_ptr<double>(), norm0.data_ptr<double>(),
        residual.data_ptr<double>(), info.data_ptr<int>(), sweeps.data_ptr<int>(), n, sweep, batch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    unfinished = info.eq(1).any().item<bool>();
  }
  // Stable device sort replaces the fixed 512-slot bitonic sorter. Equal
  // values retain the original column order; no epsilon or spectral changes.
  auto sorted = at::sort(a.view({batch,n64,n64}).diagonal(0,-2,-1),
                         std::optional<bool>(true), -1, false);
  auto values = std::get<0>(sorted).contiguous();
  auto permutation = std::get<1>(sorted).contiguous();
  values.add_(shift.unsqueeze(-1));
  auto shape = input.sizes().vec(); shape.pop_back();
  auto w = values.reshape(shape), output = at::empty_like(input);
  sort_vectors<<<grid,threads,0,stream>>>(v.data_ptr<double>(), permutation.data_ptr<int64_t>(), output.data_ptr<double>(), n, elements);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {w, output, info, residual, sweeps};
}
