#pragma once

#include <cassert>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>

#include "../classes/cudaBuffer.cuh"

namespace CUDA {

const int packHeadThreadsPerBlock = 1024;

template <typename T>
__global__ void packHeadKernel(const T *headOut, T *result, int numTokens,
                               int headDim, int embedDim, int headIdx) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int total = numTokens * headDim;
  if (i < total) {
    int j = i / headDim;
    int k = i % headDim;
    result[j * embedDim + headIdx * headDim + k] = headOut[i];
  }
}

template <typename T>
void packHead(const CudaBuffer<T> &headOut, CudaBuffer<T> &result,
              int numTokens, int headDim, int embedDim, int headIdx) {
  assert(headOut.n == static_cast<size_t>(numTokens) * headDim);
  assert(result.n == static_cast<size_t>(numTokens) * embedDim);

  const int total = numTokens * headDim;
  packHeadKernel<<<(total + packHeadThreadsPerBlock - 1) /
                       packHeadThreadsPerBlock,
                   packHeadThreadsPerBlock>>>(headOut.ptr, result.ptr, numTokens,
                                              headDim, embedDim, headIdx);

  cudaError_t cudaError = cudaGetLastError();
  if (cudaError != cudaSuccess) {
    std::cout << "kernel error  : " << cudaGetErrorString(cudaError);
    std::exit(-1);
  }

  cudaDeviceSynchronize();
}

}; // namespace CUDA
