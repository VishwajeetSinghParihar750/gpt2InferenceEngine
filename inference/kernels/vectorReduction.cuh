#pragma once

#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <numeric>
#include <vector>

#include "../classes/cudaBuffer.cuh"

namespace CUDA {

const int perBlock = 1024;

template <typename Op>
__global__ void vectorReductionBlock(const double *a, double *b, int n, Op op,
                                     double defaultValue) {

  __shared__ double data[perBlock];

  int gi = blockDim.x * blockIdx.x + threadIdx.x;

  data[threadIdx.x] = gi < n ? a[gi] : defaultValue;

  __syncthreads();

  int len = perBlock;

  for (int stride = len / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      data[threadIdx.x] = op(data[threadIdx.x], data[threadIdx.x + stride]);
    }

    __syncthreads();
  }

  if (threadIdx.x == 0)
    b[blockIdx.x] = data[threadIdx.x];
}

template <typename KernelOp, typename AccumulateOp>
double vectorReduction(const CudaBuffer<double> &a, KernelOp kernelOp,
                       AccumulateOp accumulateOp, double kernelDefaultValue,
                       double accumulateDefaultValue) {

  const int n = static_cast<int>(a.n);
  const int blocks = (n + perBlock - 1) / perBlock;

  CudaBuffer<double> deviceResult(static_cast<size_t>(blocks));

  vectorReductionBlock<<<blocks, perBlock>>>(a.ptr, deviceResult.ptr, n,
                                             kernelOp, kernelDefaultValue);

  std::vector<double> result(blocks);
  deviceResult.copyToHost(result.data());

  return std::accumulate(result.begin(), result.end(), accumulateDefaultValue,
                         accumulateOp);
}

template <typename KernelOp, typename AccumulateOp>
double vectorReduction(const CudaBuffer<double> &a, KernelOp kernelOp,
                       AccumulateOp accumulateOp, double kernelDefaultValue) {
  return vectorReduction(a, kernelOp, accumulateOp, kernelDefaultValue,
                         kernelDefaultValue);
}

template <typename KernelOp>
double vectorReduction(const CudaBuffer<double> &a, KernelOp kernelOp,
                       double defaultValue) {
  return vectorReduction(a, kernelOp, kernelOp, defaultValue, defaultValue);
}

}; // namespace CUDA
