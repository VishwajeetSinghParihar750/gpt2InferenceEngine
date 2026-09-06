#pragma once

#include <cassert>
#include <cstdlib>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>

namespace CUDA {

  const int tileRows = 32;
  const int tileCols = tileRows;

  template<typename T>
  __global__ void matMulKernel(const T *a,const T *b, T *c/*  */, int an, int am,
                              int bn, int bm) {

    int tileLoops = (am + blockDim.x - 1) / blockDim.x;

    __shared__ T sharedA[tileRows][tileCols];
    __shared__ T sharedB[tileRows][tileCols];

    int x = blockDim.x * blockIdx.x, y = blockDim.y * blockIdx.y;
    T sum = 0;

    for (int i = 0; i < tileLoops; i++) {

      int aiy = (y + threadIdx.y);
      int aix = blockDim.x * i + threadIdx.x;

      int ai = aiy * am + aix;

      int biy = (blockDim.y * i + threadIdx.y);
      int bix = x + threadIdx.x;
      int bi = biy * bm + bix;

      sharedA[threadIdx.y][threadIdx.x] = (aiy < an && aix < am) ? a[ai] : 0;
      sharedB[threadIdx.y][threadIdx.x] = (biy < bn && bix < bm) ? b[bi] : 0;

      __syncthreads();

      for (int l = 0; l < blockDim.x; l++) {
        sum += sharedA[threadIdx.y][l] * sharedB[l][threadIdx.x];
      }

      __syncthreads();
    }

    if (x < an && y < bm)
      c[x * bm + y] = sum;
  }

  // Caller owns the returned pointer (delete[]).
  template<typename T>
  T* MatMul(const T* da, size_t aSize, const T* db, size_t bSize,
            const int an, const int am, const int bn, const int bm) {


    T *dc;
    cudaMalloc(&dc, static_cast<size_t>(an) * bm * sizeof(T));
    cudaMemset(dc, 0, static_cast<size_t>(an) * bm * sizeof(T));

    dim3 threadsPerBlock(tileRows, tileCols);
    dim3 blocksPerGrid((bm + threadsPerBlock.x - 1) / threadsPerBlock.x,
                      (an + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matMulKernel<<<blocksPerGrid, threadsPerBlock>>>(da, db, dc, an, am, bn, bm);

    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
      std::cout << "kernel error  : " << cudaGetErrorString(cudaError);
      std::exit(-1);
    }

    cudaDeviceSynchronize();

    return dc;
  }
};
