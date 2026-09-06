#pragma once

#include <cstdlib>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <math_constants.h>

namespace CUDA {

const int causalMaskThreadsPerBlock = 1024;

// scores: NUM_TOKENS x NUM_TOKENS row-major. Sets scores[i, j] = -inf for j > i.
template <typename T>
__global__ void causalMaskKernel(T *scores, int numTokens) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int total = numTokens * numTokens;
  if (idx < total) {
    int i = idx / numTokens;
    int j = idx % numTokens;
    if (j > i)
      scores[idx] = -CUDART_INF; // device -inf (double)
  }
}

// Mutates scores in place.
template <typename T>
T *causalMask(T *scores, int numTokens) {

  const int total = numTokens * numTokens;
  causalMaskKernel<<<(total + causalMaskThreadsPerBlock - 1) /
                         causalMaskThreadsPerBlock,
                     causalMaskThreadsPerBlock>>>(scores, numTokens);

  cudaError_t cudaError = cudaGetLastError();
  if (cudaError != cudaSuccess) {
    std::cout << "kernel error  : " << cudaGetErrorString(cudaError);
    std::exit(-1);
  }

  cudaDeviceSynchronize();

  return scores;
}

}; // namespace CUDA
