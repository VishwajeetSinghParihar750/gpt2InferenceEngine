#pragma once

#include "../classes/cudaBuffer.cuh"
#include "vectorCombine.cuh"
#include "vectorReduction.cuh"

namespace CUDA {

double DotProduct(const CudaBuffer<double> &a, const CudaBuffer<double> &b) {
  auto mul = [] __host__ __device__(const double &x, const double &y) -> double {
    return x * y;
  };
  auto add = [] __host__ __device__(const double &x, const double &y) -> double {
    return x + y;
  };

  auto products = CUDA::vectorCombine(a, b, mul);
  return CUDA::vectorReduction(products, add, 0.0);
}

}; // namespace CUDA
