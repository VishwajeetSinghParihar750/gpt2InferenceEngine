#pragma once

#include <cassert>
#include <cstdlib>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>

#include "../classes/cudaBuffer.cuh"

namespace CUDA {

const int tileRows = 32;
const int tileCols = tileRows;

template <typename T>
__global__ void matMulKernel(const T *a, const T *b, T *c, int an, int am,
                             int bn, int bm) {

  int tileLoops = (am + blockDim.x - 1) / blockDim.x;

  __shared__ T sharedA[tileRows][tileCols];
  __shared__ T sharedB[tileRows][tileCols];

  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  T sum = 0;

  for (int i = 0; i < tileLoops; i++) {
    int aCol = blockDim.x * i + threadIdx.x;
    int bRow = blockDim.y * i + threadIdx.y;

    sharedA[threadIdx.y][threadIdx.x] =
        (row < an && aCol < am) ? a[row * am + aCol] : 0;
    sharedB[threadIdx.y][threadIdx.x] =
        (bRow < bn && col < bm) ? b[bRow * bm + col] : 0;

    __syncthreads();

    for (int l = 0; l < blockDim.x; l++) {
      sum += sharedA[threadIdx.y][l] * sharedB[l][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < an && col < bm)
    c[row * bm + col] = sum;
}

template <typename T>
CudaBuffer<T> MatMul(const CudaBuffer<T> &da, const CudaBuffer<T> &db,
                     const int an, const int am, const int bn, const int bm) {

  assert(am == bn);
  assert(da.n == static_cast<size_t>(an) * am);
  assert(db.n == static_cast<size_t>(bn) * bm);

  CudaBuffer<T> dc(static_cast<size_t>(an) * bm);
  dc.zero();

  dim3 threadsPerBlock(tileCols, tileRows);
  dim3 blocksPerGrid((bm + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (an + threadsPerBlock.y - 1) / threadsPerBlock.y);

  matMulKernel<<<blocksPerGrid, threadsPerBlock>>>(da.ptr, db.ptr, dc.ptr, an,
                                                   am, bn, bm);

  cudaError_t cudaError = cudaGetLastError();
  if (cudaError != cudaSuccess) {
    std::cout << "kernel error  : " << cudaGetErrorString(cudaError);
    std::exit(-1);
  }

  cudaDeviceSynchronize();
  return dc;
}

}; // namespace CUDA
