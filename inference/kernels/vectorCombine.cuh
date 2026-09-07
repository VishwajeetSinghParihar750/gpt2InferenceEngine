#pragma once

#include <cstdlib>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>

namespace CUDA {

const int vecAddThreadsPerBlock = 1024;

template <typename T, typename Op>
__global__ void vectorCombineKernel(const T *a, const T *b, T *c, int n,
                                    Op op) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n)
    c[i] = op(a[i], b[i]);
}

// op needs to be __host__
// Caller owns the returned pointer (cudaFree).
template <typename T, typename Op>
T *vectorCombine(const T *da, const T *db, int n, Op op) {

  T *dc;
  cudaMalloc(&dc, n * sizeof(T));

  vectorCombineKernel<<<(n + vecAddThreadsPerBlock - 1) / vecAddThreadsPerBlock,
                        vecAddThreadsPerBlock>>>(da, db, dc, n, op);

  cudaError_t cudaError = cudaGetLastError();
  if (cudaError != cudaSuccess) {
    std::cout << "kernel error  : " << cudaGetErrorString(cudaError);
    std::exit(-1);
  }

  cudaDeviceSynchronize();

  return dc;
}

// Writes into dc (must be device memory of length n). Returns dc.
template <typename T, typename Op>
T *vectorCombineInto(const T *da, const T *db, T *dc, int n, Op op) {

  vectorCombineKernel<<<(n + vecAddThreadsPerBlock - 1) / vecAddThreadsPerBlock,
                        vecAddThreadsPerBlock>>>(da, db, dc, n, op);

  cudaError_t cudaError = cudaGetLastError();
  if (cudaError != cudaSuccess) {
    std::cout << "kernel error  : " << cudaGetErrorString(cudaError);
    std::exit(-1);
  }

  cudaDeviceSynchronize();

  return dc;
}

// mat: rows x cols row-major. Adds bias[col] to every row in place.
template <typename T>
__global__ void addRowBiasKernel(T *mat, const T *bias, int rows, int cols) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < rows * cols)
    mat[i] += bias[i % cols];
}

template <typename T>
T *addRowBias(T *mat, const T *bias, int rows, int cols) {
  const int total = rows * cols;
  addRowBiasKernel<<<(total + vecAddThreadsPerBlock - 1) / vecAddThreadsPerBlock,
                     vecAddThreadsPerBlock>>>(mat, bias, rows, cols);

  cudaError_t cudaError = cudaGetLastError();
  if (cudaError != cudaSuccess) {
    std::cout << "kernel error  : " << cudaGetErrorString(cudaError);
    std::exit(-1);
  }

  cudaDeviceSynchronize();

  return mat;
}

}; // namespace CUDA
