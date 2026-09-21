# Memory Layout and Stream Contracts

## 1. Row-Major (PyTorch) vs Column-Major (rocSOLVER)

PyTorch tensors use C-contiguous row-major order:
$$\text{offset}(i, j) = i \cdot \text{stride}_0 + j$$

LAPACK, BLAS, and rocSOLVER routines expect Fortran column-major order:
$$\text{offset}(i, j) = j \cdot \text{stride}_1 + i$$

For a symmetric matrix, reading the lower triangle in row-major memory is
mathematically equivalent to reading the upper triangle in column-major memory.
Therefore the uplo flag must be inverted at the interface boundary.

rocSOLVER writes eigenvectors into columns of the column-major input memory.
When viewed as row-major PyTorch memory, this can appear transposed. Restore the
PyTorch layout when required:

    eigenvectors = A_work.transpose(-1, -2).contiguous()

## 2. Stream and Device Guarding

Operations in a PyTorch extension must not block unrelated streams or cross
device boundaries.

### Thread-Local Handle Cache

Use a thread-local map keyed by device ID. On first use, set the device,
create the rocblas handle, and check the returned status. On every call, bind
the handle to the current PyTorch stream.

### Stream Synchronization Contract

1. Extract the current stream for the input device.
2. Bind the handle before enqueueing solver kernels.
3. Do not call hipStreamSynchronize or torch.cuda.synchronize inside the
   kernel wrapper; preserve PyTorch asynchronous execution semantics.
