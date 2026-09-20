#include <torch/extension.h>

#include <cstdint>

void batch_query_geometry_hip(
    torch::Tensor positions, torch::Tensor cells, torch::Tensor batch_idx,
    torch::Tensor public_matrix, torch::Tensor public_shifts,
    torch::Tensor public_counts, torch::Tensor distances, torch::Tensor vectors);

void batch_query_geometry_into(
    torch::Tensor positions, torch::Tensor cells, torch::Tensor batch_idx,
    torch::Tensor public_matrix, torch::Tensor public_shifts,
    torch::Tensor public_counts, torch::Tensor distances, torch::Tensor vectors) {
  TORCH_CHECK(positions.is_cuda() && positions.dim() == 2 &&
                  positions.size(1) == 3 &&
                  (positions.scalar_type() == torch::kFloat ||
                   positions.scalar_type() == torch::kDouble),
              "positions must have shape (N, 3) and float32/float64 dtype on HIP");
  TORCH_CHECK(cells.is_cuda() && cells.dim() == 3 && cells.size(1) == 3 &&
                  cells.size(2) == 3 && cells.scalar_type() == positions.scalar_type(),
              "cells must have shape (B, 3, 3) and match positions dtype on HIP");
  TORCH_CHECK(batch_idx.is_cuda() && batch_idx.dim() == 1 &&
                  batch_idx.size(0) == positions.size(0) &&
                  batch_idx.scalar_type() == torch::kInt,
              "batch_idx must have shape (N,) and int32 dtype on HIP");
  TORCH_CHECK(public_matrix.is_cuda() && public_matrix.dim() == 2 &&
                  public_matrix.size(0) == positions.size(0) &&
                  public_matrix.scalar_type() == torch::kInt,
              "public_matrix must have shape (N, K) and int32 dtype on HIP");
  TORCH_CHECK(public_shifts.is_cuda() && public_shifts.dim() == 3 &&
                  public_shifts.size(0) == public_matrix.size(0) &&
                  public_shifts.size(1) == public_matrix.size(1) &&
                  public_shifts.size(2) == 3 &&
                  public_shifts.scalar_type() == torch::kInt,
              "public_shifts must have shape (N, K, 3) and int32 dtype on HIP");
  TORCH_CHECK(public_counts.is_cuda() && public_counts.dim() == 1 &&
                  public_counts.size(0) == positions.size(0) &&
                  public_counts.scalar_type() == torch::kInt,
              "public_counts must have shape (N,) and int32 dtype on HIP");
  TORCH_CHECK(distances.is_cuda() && distances.sizes() == public_matrix.sizes() &&
                  distances.scalar_type() == positions.scalar_type(),
              "distances must have shape (N, K) and match positions dtype on HIP");
  TORCH_CHECK(vectors.is_cuda() && vectors.dim() == 3 &&
                  vectors.size(0) == public_matrix.size(0) &&
                  vectors.size(1) == public_matrix.size(1) && vectors.size(2) == 3 &&
                  vectors.scalar_type() == positions.scalar_type(),
              "vectors must have shape (N, K, 3) and match positions dtype on HIP");
  const auto tensors = {cells, batch_idx, public_matrix, public_shifts,
                        public_counts, distances, vectors};
  for (const auto& tensor : tensors) {
    TORCH_CHECK(tensor.device() == positions.device(),
                "all native HIP geometry tensors must share one device");
    TORCH_CHECK(tensor.is_contiguous(),
                "all native HIP geometry tensors must be contiguous");
  }
  TORCH_CHECK(cells.size(0) > 0, "native HIP geometry requires at least one system");
  batch_query_geometry_hip(positions, cells, batch_idx, public_matrix,
                           public_shifts, public_counts, distances, vectors);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("batch_query_geometry_into", &batch_query_geometry_into,
             "Materialize Batch query geometry with a native HIP kernel");
}
