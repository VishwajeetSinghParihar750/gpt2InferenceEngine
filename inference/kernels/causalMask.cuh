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
__global__ void causalMaskKernel(T *scores, int rows, int cols,
                                 int queryPosOffset) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int total = rows * cols;
  if (idx < total) {
    int i = idx / cols;
    int j = idx % cols;
    if (j > queryPosOffset + i)
      scores[idx] = -CUDART_INF;
  }
}

template <typename T>
void causalMask(CudaBuffer<T> &scores, int rows, int cols,
                int queryPosOffset = 0) {
  assert(scores.n == static_cast<size_t>(rows) * cols);
  const int total = rows * cols;
  causalMaskKernel<<<(total + causalMaskThreadsPerBlock - 1) /
                         causalMaskThreadsPerBlock,
                     causalMaskThreadsPerBlock>>>(scores.ptr, rows, cols,
                                                  queryPosOffset);

  cudaError_t cudaError = cudaGetLastError();
  if (cudaError != cudaSuccess) {
    std::cout << "kernel error  : " << cudaGetErrorString(cudaError);
    std::exit(-1);
  }

  cudaDeviceSynchronize();
}

template <typename T>
void causalMask(CudaBuffer<T> &scores, int numTokens) {
  causalMask(scores, numTokens, numTokens, 0);
}

}; // namespace CUDA
