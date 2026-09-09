#pragma once

#include <cassert>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>

#include "../classes/cudaBuffer.cuh"

namespace CUDA {

const int vecAddThreadsPerBlock = 1024;

template <typename T, typename Op>
__global__ void vectorCombineKernel(const T *a, const T *b, T *c, int n,
                                    Op op) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n)
    c[i] = op(a[i], b[i]);
}

template <typename T, typename Op>
CudaBuffer<T> vectorCombine(const CudaBuffer<T> &da, const CudaBuffer<T> &db,
                            Op op) {
  assert(da.n == db.n);
  const int n = static_cast<int>(da.n);
  CudaBuffer<T> dc(da.n);

  vectorCombineKernel<<<(n + vecAddThreadsPerBlock - 1) / vecAddThreadsPerBlock,
                        vecAddThreadsPerBlock>>>(da.ptr, db.ptr, dc.ptr, n, op);

  cudaError_t cudaError = cudaGetLastError();
  if (cudaError != cudaSuccess) {
    std::cout << "kernel error  : " << cudaGetErrorString(cudaError);
    std::exit(-1);
  }

  cudaDeviceSynchronize();
  return dc;
}

template <typename T, typename Op>
void vectorCombineInto(const CudaBuffer<T> &da, const CudaBuffer<T> &db,
                       CudaBuffer<T> &dc, Op op) {
  assert(da.n == db.n && da.n == dc.n);
  const int n = static_cast<int>(da.n);

  vectorCombineKernel<<<(n + vecAddThreadsPerBlock - 1) / vecAddThreadsPerBlock,
                        vecAddThreadsPerBlock>>>(da.ptr, db.ptr, dc.ptr, n, op);

  cudaError_t cudaError = cudaGetLastError();
  if (cudaError != cudaSuccess) {
    std::cout << "kernel error  : " << cudaGetErrorString(cudaError);
    std::exit(-1);
  }

  cudaDeviceSynchronize();
}

template <typename T>
__global__ void addRowBiasKernel(T *mat, const T *bias, int rows, int cols) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < rows * cols)
    mat[i] += bias[i % cols];
}

template <typename T>
void addRowBias(CudaBuffer<T> &mat, const CudaBuffer<T> &bias, int rows,
                int cols) {
  assert(mat.n == static_cast<size_t>(rows) * cols);
  assert(bias.n == static_cast<size_t>(cols));
  const int total = rows * cols;
  addRowBiasKernel<<<(total + vecAddThreadsPerBlock - 1) / vecAddThreadsPerBlock,
                     vecAddThreadsPerBlock>>>(mat.ptr, bias.ptr, rows, cols);

  cudaError_t cudaError = cudaGetLastError();
  if (cudaError != cudaSuccess) {
    std::cout << "kernel error  : " << cudaGetErrorString(cudaError);
    std::exit(-1);
  }

  cudaDeviceSynchronize();
}

}; // namespace CUDA
