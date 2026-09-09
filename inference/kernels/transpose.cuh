#pragma once

#include <cassert>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>

#include "../classes/cudaBuffer.cuh"

namespace CUDA {

const int transposeThreadsPerBlock = 1024;

template <typename T>
__global__ void transposeKernel(const T *a, T *result, int n, int m) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int total = n * m;
  if (i < total) {
    int row = i / m;
    int col = i % m;
    result[col * n + row] = a[i];
  }
}

// da is n x m row-major; result is m x n.
template <typename T>
CudaBuffer<T> Transpose(const CudaBuffer<T> &da, int n, int m) {
  assert(da.n == static_cast<size_t>(n) * m);
  CudaBuffer<T> result(static_cast<size_t>(m) * n);

  const int total = n * m;
  transposeKernel<<<(total + transposeThreadsPerBlock - 1) /
                        transposeThreadsPerBlock,
                    transposeThreadsPerBlock>>>(da.ptr, result.ptr, n, m);

  cudaError_t cudaError = cudaGetLastError();
  if (cudaError != cudaSuccess) {
    std::cout << "kernel error  : " << cudaGetErrorString(cudaError);
    std::exit(-1);
  }

  cudaDeviceSynchronize();
  return result;
}

}; // namespace CUDA
