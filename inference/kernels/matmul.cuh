#pragma once

#include <cassert>
#include <cstdlib>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <span>
#include <vector>

namespace CUDA {

  const int tileRows = 32;
  const int tileCols = tileRows;

  template<typename T>
  __global__ void matMulKernel(T *a, T *b, T *c/*  */, int an, int am,
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

  template<typename T>
  std::vector<T> MatMul( std::span<const T> a, const std::span<const T> &b,
              const int an, const int am, const int bn, const int bm) {

    std::vector<T> c(an * bm);

    T *da, *db, *dc;
    cudaMalloc(&da, a.size() * sizeof(T));
    cudaMalloc(&db, b.size() * sizeof(T));
    cudaMalloc(&dc, c.size() * sizeof(T));

    cudaMemcpy(da, a.data(), a.size() * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b.data(), b.size() * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemset(dc, 0, c.size() * sizeof(T));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // call
    dim3 threadsPerBlock(tileRows, tileCols);
    dim3 blocksPerGrid((bm + threadsPerBlock.x - 1) / threadsPerBlock.x,
                      (an + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEventRecord(start);
    matMulKernel<<<blocksPerGrid, threadsPerBlock>>>(da, db, dc, an, am, bn, bm);
    cudaEventRecord(stop);

    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess) {
      std::cout << "kernel error  : " << cudaGetErrorString(cudaError);
      std::exit(-1);
    }

    cudaDeviceSynchronize();

    float time = 0;
    cudaEventElapsedTime(&time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(c.data(), dc, an * bm * sizeof(T), cudaMemcpyDeviceToHost);

    std::cout << "elapsed time in ms: " << time << std::endl;

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

    return c;
  }
}