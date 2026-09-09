#pragma once

#include <cassert>
#include <cmath>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#include "../classes/cudaBuffer.cuh"
#include "vectorCombine.cuh"
#include "vectorMap.cuh"
#include "vectorReduction.cuh"

namespace CUDA {

template <typename T>
CudaBuffer<T> LayerNorm(const CudaBuffer<T> &originalEmbedding,
                        const CudaBuffer<T> &gamma, const CudaBuffer<T> &beta,
                        T epsilon) {
  const int len = static_cast<int>(originalEmbedding.n);
  assert(gamma.n == originalEmbedding.n && beta.n == originalEmbedding.n);

  auto add = [] __device__ __host__(const double &a, const double &b) -> double {
    return a + b;
  };

  double sum = CUDA::vectorReduction(originalEmbedding, add, 0.0);
  double mean = sum / len;

  auto squaredDiffs = CUDA::vectorMap(
      originalEmbedding,
      [mean] __device__ __host__(const double &x) -> double {
        return (x - mean) * (x - mean);
      });

  double sumSquaredDiffs = CUDA::vectorReduction(squaredDiffs, add, 0.0);
  double variance = sumSquaredDiffs / len;

  double modifiedStandardDeviation = sqrt(variance + epsilon);

  auto normalized = CUDA::vectorMap(
      originalEmbedding,
      [mean, modifiedStandardDeviation] __device__ __host__(
          const double &x) -> double {
        return (x - mean) / modifiedStandardDeviation;
      });

  auto scaled = CUDA::vectorCombine(
      normalized, gamma,
      [] __device__ __host__(const double &a, const double &b) -> double {
        return a * b;
      });

  return CUDA::vectorCombine(scaled, beta, add);
}

}; // namespace CUDA
