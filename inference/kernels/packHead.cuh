#pragma once

#include <cstdlib>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>

namespace CUDA {

const int packHeadThreadsPerBlock = 1024;

// headOut: numTokens x headDim
// result:  numTokens x embedDim, writes into columns [headIdx*headDim, (headIdx+1)*headDim)
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
T *packHead(const T *headOut, T *result, int numTokens, int headDim,
            int embedDim, int headIdx) {

  const int total = numTokens * headDim;
  packHeadKernel<<<(total + packHeadThreadsPerBlock - 1) /
                       packHeadThreadsPerBlock,
                   packHeadThreadsPerBlock>>>(headOut, result, numTokens,
                                              headDim, embedDim, headIdx);

  cudaError_t cudaError = cudaGetLastError();
  if (cudaError != cudaSuccess) {
    std::cout << "kernel error  : " << cudaGetErrorString(cudaError);
    std::exit(-1);
  }

  cudaDeviceSynchronize();

  return result;
}

}; // namespace CUDA
