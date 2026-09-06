#pragma once

#include "vectorCombine.cuh"
#include "vectorReduction.cuh"

namespace CUDA {

double DotProduct(const double *a, const double *b, int len) // a, b: 1d
{
  auto mul = [] __host__ __device__(const double &x, const double &y) -> double {
    return x * y;
  };
  auto add = [] __host__ __device__(const double &x, const double &y) -> double {
    return x + y;
  };

  double *products = CUDA::vectorCombine(a, b, len, mul);
  double output = CUDA::vectorReduction(products, len, add, 0.0);
  cudaFree(products);

  return output;
}

}; // namespace CUDA
