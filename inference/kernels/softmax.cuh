#pragma once

#include <cassert>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <limits>

#include "../classes/cudaBuffer.cuh"
#include "vectorMap.cuh"
#include "vectorReduction.cuh"

namespace CUDA {

CudaBuffer<double> SoftMax(const CudaBuffer<double> &input) {
  const int len = static_cast<int>(input.n);

  auto mxLambda = [] __host__ __device__(const double &a,
                                         const double &b) -> double {
    return a > b ? a : b;
  };
  auto sumLambda = [] __host__ __device__(const double &a,
                                          const double &b) -> double {
    return a + b;
  };

  double max = CUDA::vectorReduction(input, mxLambda,
                                     std::numeric_limits<double>::lowest());

  auto softmax = CUDA::vectorMap(
      input, [max] __host__ __device__(const double &a) -> double {
        return exp(a - max);
      });

  double sum = CUDA::vectorReduction(softmax, sumLambda, 0.0);

  CUDA::vectorMapInPlace(softmax, [sum] __device__ __host__(double &v) {
    v /= sum;
    return v;
  });

  return softmax;
}

void SoftMaxInPlace(CudaBuffer<double> &input) {
  auto mxLambda = [] __host__ __device__(const double &a,
                                         const double &b) -> double {
    return a > b ? a : b;
  };
  auto sumLambda = [] __host__ __device__(const double &a,
                                          const double &b) -> double {
    return a + b;
  };

  double max = CUDA::vectorReduction(input, mxLambda,
                                     std::numeric_limits<double>::lowest());

  CUDA::vectorMapInPlace(input, [max] __device__ __host__(double &v) {
    v = exp(v - max);
    return v;
  });

  double sum = CUDA::vectorReduction(input, sumLambda, 0.0);

  CUDA::vectorMapInPlace(input, [sum] __device__ __host__(double &v) {
    v /= sum;
    return v;
  });
}

void SoftMaxRows(CudaBuffer<double> &scores, int rows, int cols) {
  assert(scores.n == static_cast<size_t>(rows) * cols);
  for (int i = 0; i < rows; i++) {
    auto row = scores.slice(static_cast<size_t>(i) * cols, cols);
    SoftMaxInPlace(row);
  }
}

}; // namespace CUDA
