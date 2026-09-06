#pragma once

#include <cstdlib>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>

namespace CUDA {

const int transposeThreadsPerBlock = 1024;

// a: n x m flattened row-major -> result: m x n flattened row-major
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

// Caller owns the returned pointer (cudaFree).
template <typename T>
T *Transpose(const T *da, int n, int m) {

  T *result;
  cudaMalloc(&result, static_cast<size_t>(m) * n * sizeof(T));

  const int total = n * m;
  transposeKernel<<<(total + transposeThreadsPerBlock - 1) /
                        transposeThreadsPerBlock,
                    transposeThreadsPerBlock>>>(da, result, n, m);

  cudaError_t cudaError = cudaGetLastError();
  if (cudaError != cudaSuccess) {
    std::cout << "kernel error  : " << cudaGetErrorString(cudaError);
    std::exit(-1);
  }

  cudaDeviceSynchronize();

  return result;
}

}; // namespace CUDA
