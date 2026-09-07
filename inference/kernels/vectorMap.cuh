#pragma once

#include <cstdlib>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>

namespace CUDA {

const int vecMapThreadsPerBlock = 1024;

template <typename T, typename Op>
__global__ void vectorMapKernel(const T *a, T *b, int n, Op op) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n)
    b[i] = op(a[i]);
}

// op needs to be __host__ __device__
// Caller owns the returned pointer (cudaFree).
template <typename T, typename Op>
T *vectorMap(const T *da, int n, Op op) {

  T *db;
  cudaMalloc(&db, n * sizeof(T));

  vectorMapKernel<<<(n + vecMapThreadsPerBlock - 1) / vecMapThreadsPerBlock,
                    vecMapThreadsPerBlock>>>(da, db, n, op);

  cudaError_t cudaError = cudaGetLastError();
  if (cudaError != cudaSuccess) {
    std::cout << "kernel error  : " << cudaGetErrorString(cudaError);
    std::exit(-1);
  }

  cudaDeviceSynchronize();

  return db;
}

template <typename T, typename Op>
__global__ void vectorMapInPlaceKernel(T *a, int n, Op op) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n)
    a[i] = op(a[i]);
}

// op needs to be __host__ __device__
// Mutates da in place and returns it.
template <typename T, typename Op>
T *vectorMapInPlace(T *da, int n, Op op) {

  vectorMapInPlaceKernel<<<(n + vecMapThreadsPerBlock - 1) /
                               vecMapThreadsPerBlock,
                           vecMapThreadsPerBlock>>>(da, n, op);

  cudaError_t cudaError = cudaGetLastError();
  if (cudaError != cudaSuccess) {
    std::cout << "kernel error  : " << cudaGetErrorString(cudaError);
    std::exit(-1);
  }

  cudaDeviceSynchronize();

  return da;
}

}; // namespace CUDA
