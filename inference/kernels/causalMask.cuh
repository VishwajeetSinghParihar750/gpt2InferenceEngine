#pragma once

#include <cassert>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <math_constants.h>

#include "../classes/cudaBuffer.cuh"

namespace CUDA {

const int causalMaskThreadsPerBlock = 1024;

template <typename T>
__global__ void causalMaskKernel(T *scores, int numTokens) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int total = numTokens * numTokens;
  if (idx < total) {
    int i = idx / numTokens;
    int j = idx % numTokens;
    if (j > i)
      scores[idx] = -CUDART_INF;
  }
}

template <typename T>
void causalMask(CudaBuffer<T> &scores, int numTokens) {
  assert(scores.n == static_cast<size_t>(numTokens) * numTokens);
  const int total = numTokens * numTokens;
  causalMaskKernel<<<(total + causalMaskThreadsPerBlock - 1) /
                         causalMaskThreadsPerBlock,
                     causalMaskThreadsPerBlock>>>(scores.ptr, numTokens);

  cudaError_t cudaError = cudaGetLastError();
  if (cudaError != cudaSuccess) {
    std::cout << "kernel error  : " << cudaGetErrorString(cudaError);
    std::exit(-1);
  }

  cudaDeviceSynchronize();
}

}; // namespace CUDA
