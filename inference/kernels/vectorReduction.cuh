#pragma once

#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <numeric>
#include <vector>

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

// kernelOp needs to be __host__ __device__; accumulateOp is host-only.
template <typename KernelOp, typename AccumulateOp>
double vectorReduction(const double *a, int n, KernelOp kernelOp,
                       AccumulateOp accumulateOp, double kernelDefaultValue,
                       double accumulateDefaultValue) {

  const int blocks = (n + perBlock - 1) / perBlock;

  double *deviceResult;
  cudaMalloc(&deviceResult, blocks * sizeof(double));

  vectorReductionBlock<<<blocks, perBlock>>>(a, deviceResult, n, kernelOp,
                                             kernelDefaultValue);

  std::vector<double> result(blocks);
  cudaMemcpy(result.data(), deviceResult, blocks * sizeof(double),
             cudaMemcpyDeviceToHost);

  cudaFree(deviceResult);

  return std::accumulate(result.begin(), result.end(), accumulateDefaultValue,
                         accumulateOp);
}

// accumulateDefaultValue defaults to kernelDefaultValue.
template <typename KernelOp, typename AccumulateOp>
double vectorReduction(const double *a, int n, KernelOp kernelOp,
                       AccumulateOp accumulateOp, double kernelDefaultValue) {
  return vectorReduction(a, n, kernelOp, accumulateOp, kernelDefaultValue,
                         kernelDefaultValue);
}

// accumulateOp and its defaultValue default to the kernel's.
template <typename KernelOp>
double vectorReduction(const double *a, int n, KernelOp kernelOp,
                       double defaultValue) {
  return vectorReduction(a, n, kernelOp, kernelOp, defaultValue, defaultValue);
}

}; // namespace CUDA
