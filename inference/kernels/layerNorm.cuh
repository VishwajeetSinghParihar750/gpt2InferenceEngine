#pragma once

#include <cmath>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#include "vectorCombine.cuh"
#include "vectorMap.cuh"
#include "vectorReduction.cuh"

namespace CUDA {

const int layerNormThreadsPerBlock = 1024;

// Caller owns the returned pointer (cudaFree).
template <typename T>
T *LayerNorm(const T *originalEmbedding, const T *gamma, const T *beta, int len,
             T epsilon) {

  auto add = [] __device__ __host__(const double &a, const double &b) -> double {
    return a + b;
  };

  double sum = CUDA::vectorReduction(originalEmbedding, len, add, 0.0);
  double mean = sum / len;

  double *squaredDiffs = CUDA::vectorMap(
      originalEmbedding, len, [mean] __device__ __host__(const double &x) -> double {
        return (x - mean) * (x - mean);
      });

  double sumSquaredDiffs = CUDA::vectorReduction(squaredDiffs, len, add, 0.0);
  double variance = sumSquaredDiffs / len;

  cudaFree(squaredDiffs);

  double modifiedStandardDeviation = sqrt(variance + epsilon);

  double *normalized = CUDA::vectorMap(
      originalEmbedding, len,
      [mean, modifiedStandardDeviation] __device__ __host__(
          const double &x) -> double {
        return (x - mean) / modifiedStandardDeviation;
      });

  double *scaled = CUDA::vectorCombine(
      normalized, gamma, len,
      [] __device__ __host__(const double &a, const double &b) -> double {
        return a * b;
      });
  cudaFree(normalized);

  T *output = CUDA::vectorCombine(scaled, beta, len, add);
  cudaFree(scaled);

  return output;
}

}; // namespace CUDA
