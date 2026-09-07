#pragma once

#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <limits>

#include "vectorMap.cuh"
#include "vectorReduction.cuh"

namespace CUDA {

// Caller owns the returned pointer (cudaFree).
double *SoftMax(const double *input, int len) // input: 1d
{
  auto mxLambda = [] __host__ __device__(const double &a,
                                         const double &b) -> double {
    return a > b ? a : b;
  };
  auto sumLambda = [] __host__ __device__(const double &a,
                                          const double &b) -> double {
    return a + b;
  };

  double max = CUDA::vectorReduction(input, len, mxLambda,
                                     std::numeric_limits<double>::lowest());

  auto softmaxLambda = [max] __host__ __device__(const double &a) -> double {
    return exp(a - max);
  };

  double *softmax = CUDA::vectorMap(input, len, softmaxLambda);

  double sum = CUDA::vectorReduction(softmax, len, sumLambda, 0.0);

  CUDA::vectorMapInPlace(softmax, len,
                         [sum] __device__ __host__(double &v) {
                           v /= sum;
                           return v;
                         });

  return softmax;
}

// Mutates input in place and returns it.
double *SoftMaxInPlace(double *input, int len) {
  auto mxLambda = [] __host__ __device__(const double &a,
                                         const double &b) -> double {
    return a > b ? a : b;
  };
  auto sumLambda = [] __host__ __device__(const double &a,
                                          const double &b) -> double {
    return a + b;
  };

  double max = CUDA::vectorReduction(input, len, mxLambda,
                                     std::numeric_limits<double>::lowest());

  CUDA::vectorMapInPlace(input, len, [max] __device__ __host__(double &v) {
    v = exp(v - max);
    return v;
  });

  double sum = CUDA::vectorReduction(input, len, sumLambda, 0.0);

  CUDA::vectorMapInPlace(input, len, [sum] __device__ __host__(double &v) {
    v /= sum;
    return v;
  });

  return input;
}

// scores: rows x cols row-major. Softmax applied per row, in place.
double *SoftMaxRows(double *scores, int rows, int cols) {
  for (int i = 0; i < rows; i++)
    SoftMaxInPlace(scores + i * cols, cols);
  return scores;
}

}; // namespace CUDA
