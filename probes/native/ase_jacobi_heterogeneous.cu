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
#include <vector>

namespace {
constexpr int threads = 256;

__global__ void prepare_scale(const double* input, double* shift, double* norm0, const int64_t* meta, int64_t batch) {
  const int lane = threadIdx.x;
  __shared__ double trace[threads], total[threads];
  for (int64_t b = blockIdx.x; b < batch; b += gridDim.x) {
    const int n = meta[b*3];
    const int64_t base = meta[b*3+1];
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
                           double* v, double* v_other, const double* shift, const int64_t* meta, const int64_t* jobs, int64_t job_count) {
  for (int64_t job = blockIdx.x; job < job_count; job += gridDim.x) {
    const int64_t b = jobs[job*2], k = jobs[job*2+1]+threadIdx.x;
    const int n = meta[b*3];
    const int64_t offset = meta[b*3+1];
    if (k >= int64_t(n)*n) continue;
    const int row = k % n, col = k / n;
    const double value = row >= col ? input[offset + int64_t(row)*n + col]
                                   : input[offset + int64_t(col)*n + row];
    a[offset + k] = a_other[offset + k] = row == col ? value-shift[b] : value;
    v[offset + k] = v_other[offset + k] = row == col ? 1.0 : 0.0;
  }
}

__global__ void rotations(const double* a, double* c, double* s, double* change,
                          int* partner, const int* info, const int64_t* meta, const int64_t* jobs, int64_t job_count, int round) {
  for (int64_t job = blockIdx.x; job < job_count; job += gridDim.x) {
    const int64_t b = jobs[job*2];
    const int n = meta[b*3], m = n + n%2;
    const int64_t k = jobs[job*2+1]+threadIdx.x;
    if (info[b] != 1 || round >= m-1 || k >= m/2) continue;
    int p = k == 0 ? 0 : 1 + (int64_t(k) - 1 + round) % (m - 1);
    int q = 1 + (int64_t(m) - 2 - k + round) % (m - 1);
    if (p > q) { const int tmp = p; p = q; q = tmp; }
    const int64_t vec = meta[b*3+2], base = meta[b*3+1];
    if (q >= n) {
      c[vec+p] = 1.0; s[vec+p] = change[vec+p] = 0.0;
      partner[vec+p] = p;
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
    c[vec+p] = c[vec+q] = cosine;
    s[vec+p] = -sine; s[vec+q] = sine;
    change[vec+p] = -t*off; change[vec+q] = t*off;
    partner[vec+p] = q; partner[vec+q] = p;
  }
}

__global__ void apply_rotations(const double* a, const double* v, double* out_a,
                                double* out_v, const double* c, const double* s,
                                const double* change, const int* partner,
                                const int* info, const int64_t* meta, const int64_t* jobs, int64_t job_count, int round) {
  for (int64_t job = blockIdx.x; job < job_count; job += gridDim.x) {
    const int64_t b = jobs[job*2], k = jobs[job*2+1]+threadIdx.x;
    const int n = meta[b*3];
    if (info[b] != 1 || round >= n+n%2-1 || k >= int64_t(n)*n) continue;
    const int64_t base = meta[b*3+1], vec = meta[b*3+2];
    const int row = k % n, col = k / n;
    // Both triangles use the same arithmetic ordering, preserving exact symmetry.
    const int i = row < col ? row : col, j = row < col ? col : row;
    const int pi = partner[vec+i], pj = partner[vec+j];
    double value;
    if (i == j) {
      value = a[base + k] + change[vec+i];
    } else if (pi == j) {
      value = 0.0;
    } else {
      const double ci = c[vec+i], cj = c[vec+j];
      const double si = s[vec+i], sj = s[vec+j];
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
    out_v[base+k] = c[vec+col]*v[base+k]
                    + s[vec+col]*v[base+row+int64_t(partner[vec+col])*n];
  }
}

__global__ void convergence(const double* a, const double* v, double* a_other,
                            double* v_other, double* norm0, double* residual,
                            int* info, int* sweeps, const int64_t* meta, int sweep, int64_t batch) {
  const int lane = threadIdx.x;
  __shared__ double off[threads], total[threads];
  for (int64_t b = blockIdx.x; b < batch; b += gridDim.x) {
    if (info[b] != 1) continue;
    const int n = meta[b*3];
    const int64_t base = meta[b*3+1];
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

using Result = std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor>;
std::vector<Result> ase_jacobi_heterogeneous(std::vector<at::Tensor> inputs, int64_t max_sweeps) {
  TORCH_CHECK(!inputs.empty(), "heterogeneous prototype requires nonempty list");
  const auto options = inputs[0].options();
  TORCH_CHECK(inputs[0].is_cuda(), "heterogeneous prototype requires HIP");
  const c10::cuda::CUDAGuard guard(inputs[0].device());
  const auto stream = at::cuda::getCurrentCUDAStream(inputs[0].get_device());
  TORCH_CHECK(max_sweeps >= 1 && max_sweeps <= 100, "invalid sweep limit");
  std::vector<int64_t> metadata, element_jobs, pair_jobs, starts, vector_starts, matrix_starts;
  std::vector<at::Tensor> flat_inputs;
  int64_t elements=0, vectors=0, batch=0;
  int max_rounds=0;
  for (const auto& input : inputs) {
    TORCH_CHECK(input.device() == inputs[0].device() && input.scalar_type() == at::kDouble &&
                input.layout() == at::kStrided && input.is_contiguous() && !input.requires_grad() &&
                (input.dim()==2 || input.dim()==3) && input.size(-2)==input.size(-1) && input.numel()>0,
                "requires same-device contiguous non-grad FP64 [n,n]/[B,n,n]");
    const int64_t n=input.size(-1), count=input.dim()==2?1:input.size(0);
    TORCH_CHECK(n>0 && n<std::numeric_limits<int>::max() && count>0 &&
                n<=std::numeric_limits<int64_t>::max()/n &&
                count<=std::numeric_limits<int64_t>::max()/(n*n) &&
                elements<=std::numeric_limits<int64_t>::max()-count*n*n &&
                vectors<=std::numeric_limits<int64_t>::max()-count*n,
                "unrepresentable dimensions");
    starts.push_back(elements); vector_starts.push_back(vectors); matrix_starts.push_back(batch);
    max_rounds=std::max(max_rounds,static_cast<int>(n+n%2-1));
    for (int64_t j=0;j<count;++j) {
      metadata.insert(metadata.end(),{n,elements,vectors});
      for (int64_t k=0;k<n*n;k+=threads) element_jobs.insert(element_jobs.end(),{batch,k});
      for (int64_t k=0;k<(n+n%2)/2;k+=threads) pair_jobs.insert(pair_jobs.end(),{batch,k});
      elements+=n*n; vectors+=n; ++batch;
    }
    flat_inputs.push_back(input.flatten());
  }
  const auto control=[&](std::vector<int64_t>& values) {
    return at::from_blob(values.data(),{static_cast<int64_t>(values.size())},at::TensorOptions().dtype(at::kLong))
      .to(inputs[0].device(),at::kLong,false,true);
  };
  auto meta=control(metadata), jobs=control(element_jobs), pairs=control(pair_jobs);
  auto input=at::cat(flat_inputs);
  auto a=at::empty_like(input), other=at::empty_like(input), v=at::empty_like(input), other_v=at::empty_like(input);
  auto c=at::empty({vectors},options), s=at::empty_like(c), change=at::empty_like(c);
  auto partner=at::empty({vectors},options.dtype(at::kInt));
  auto norm=at::empty({batch},options), residual=at::empty_like(norm), shift=at::empty_like(norm);
  auto info=at::ones({batch},options.dtype(at::kInt)), sweeps=at::zeros_like(info);
  const int64_t max_grid=at::cuda::getCurrentDeviceProperties()->maxGridSize[0];
  const unsigned matrix_grid=std::min(batch,max_grid), grid=std::min(static_cast<int64_t>(element_jobs.size()/2),max_grid);
  const unsigned pair_grid=std::min(static_cast<int64_t>(pair_jobs.size()/2),max_grid);
  prepare_scale<<<matrix_grid,threads,0,stream>>>(input.data_ptr<double>(),shift.data_ptr<double>(),norm.data_ptr<double>(),meta.data_ptr<int64_t>(),batch);
  initialize<<<grid,threads,0,stream>>>(input.data_ptr<double>(),a.data_ptr<double>(),other.data_ptr<double>(),v.data_ptr<double>(),other_v.data_ptr<double>(),shift.data_ptr<double>(),meta.data_ptr<int64_t>(),jobs.data_ptr<int64_t>(),element_jobs.size()/2);
  const auto check=[&](int sweep) {
    convergence<<<matrix_grid,threads,0,stream>>>(a.data_ptr<double>(),v.data_ptr<double>(),other.data_ptr<double>(),other_v.data_ptr<double>(),norm.data_ptr<double>(),residual.data_ptr<double>(),info.data_ptr<int>(),sweeps.data_ptr<int>(),meta.data_ptr<int64_t>(),sweep,batch);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return info.eq(1).any().item<bool>();
  };
  bool unfinished=check(0);
  for(int sweep=1;sweep<=max_sweeps && unfinished;++sweep) {
    for(int round=0;round<max_rounds;++round) {
      rotations<<<pair_grid,threads,0,stream>>>(a.data_ptr<double>(),c.data_ptr<double>(),s.data_ptr<double>(),change.data_ptr<double>(),partner.data_ptr<int>(),info.data_ptr<int>(),meta.data_ptr<int64_t>(),pairs.data_ptr<int64_t>(),pair_jobs.size()/2,round);
      apply_rotations<<<grid,threads,0,stream>>>(a.data_ptr<double>(),v.data_ptr<double>(),other.data_ptr<double>(),other_v.data_ptr<double>(),c.data_ptr<double>(),s.data_ptr<double>(),change.data_ptr<double>(),partner.data_ptr<int>(),info.data_ptr<int>(),meta.data_ptr<int64_t>(),jobs.data_ptr<int64_t>(),element_jobs.size()/2,round);
      // Each local round count is odd. Inactive tails are even, so the global
      // current buffer at sweep end contains each matrix's local final buffer.
      std::swap(a,other); std::swap(v,other_v);
    }
    unfinished=check(sweep);
  }
  std::vector<Result> results;
  for(size_t group=0;group<inputs.size();++group) {
    const auto& source=inputs[group];
    const int64_t n=source.size(-1), count=source.dim()==2?1:source.size(0);
    auto group_a=a.narrow(0,starts[group],count*n*n).view({count,n,n});
    auto sorted=at::sort(group_a.diagonal(0,-2,-1),std::optional<bool>(true),-1,false);
    auto values=std::get<0>(sorted).contiguous(), permutation=std::get<1>(sorted).contiguous();
    values.add_(shift.narrow(0,matrix_starts[group],count).unsqueeze(-1));
    auto shape=source.sizes().vec(); shape.pop_back();
    auto output=at::empty_like(source);
    const unsigned sort_grid=std::min((count*n*n-1)/threads+1,max_grid);
    sort_vectors<<<sort_grid,threads,0,stream>>>(v.data_ptr<double>()+starts[group],permutation.data_ptr<int64_t>(),output.data_ptr<double>(),n,count*n*n);
    results.emplace_back(values.reshape(shape),output,info.narrow(0,matrix_starts[group],count),residual.narrow(0,matrix_starts[group],count),sweeps.narrow(0,matrix_starts[group],count));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return results;
}
