#pragma once

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#include "../classes/cudaBuffer.cuh"
#include "matmul.cuh"
#include "vectorCombine.cuh"
#include "vectorMap.cuh"

namespace CUDA {

// weights: countNeurons x inputSize, inputs: inputSize, biases: countNeurons
template <typename T>
CudaBuffer<T> ForwardPass(const CudaBuffer<T> &weights,
                          const CudaBuffer<T> &biases,
                          const CudaBuffer<T> &inputs, bool gelu = false) {
  const int countNeurons = static_cast<int>(biases.n);
  const int inputSize = static_cast<int>(inputs.n);
  assert(weights.n == static_cast<size_t>(countNeurons) * inputSize);

  auto output =
      MatMul<T>(weights, inputs, countNeurons, inputSize, inputSize, 1);

  addRowBias(output, biases, 1, countNeurons);

  if (gelu) {
    auto geluNew = [] __host__ __device__(T x) -> T {
      return T(.5) * x *
             (T(1) + tanh(sqrt(T(2.0) / M_PI) *
                          (x + T(0.044715) * x * x * x)));
    };
    vectorMapInPlace(output, [geluNew] __device__ __host__(T &v) {
      v = geluNew(v);
      return v;
    });
  }

  return output;
}

}; // namespace CUDA
