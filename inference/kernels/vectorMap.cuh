#pragma once

#include <cstdlib>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>

#include "../classes/cudaBuffer.cuh"

namespace CUDA {

const int vecMapThreadsPerBlock = 1024;

template <typename T, typename Op>
__global__ void vectorMapKernel(const T *a, T *b, int n, Op op) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n)
    b[i] = op(a[i]);
}

template <typename T, typename Op>
CudaBuffer<T> vectorMap(const CudaBuffer<T> &da, Op op) {
  const int n = static_cast<int>(da.n);
  CudaBuffer<T> db(da.n);

  vectorMapKernel<<<(n + vecMapThreadsPerBlock - 1) / vecMapThreadsPerBlock,
                    vecMapThreadsPerBlock>>>(da.ptr, db.ptr, n, op);

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

template <typename T, typename Op>
void vectorMapInPlace(CudaBuffer<T> &da, Op op) {
  const int n = static_cast<int>(da.n);

  vectorMapInPlaceKernel<<<(n + vecMapThreadsPerBlock - 1) /
                               vecMapThreadsPerBlock,
                           vecMapThreadsPerBlock>>>(da.ptr, n, op);

  cudaError_t cudaError = cudaGetLastError();
  if (cudaError != cudaSuccess) {
    std::cout << "kernel error  : " << cudaGetErrorString(cudaError);
    std::exit(-1);
  }

  cudaDeviceSynchronize();
}

}; // namespace CUDA
