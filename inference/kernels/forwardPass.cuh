#pragma once

#include <cmath>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#include "matmul.cuh"
#include "vectorCombine.cuh"
#include "vectorMap.cuh"

namespace CUDA {

// weights: countNeurons x inputSize, inputs: inputSize, biases: countNeurons
// Caller owns the returned pointer (cudaFree).
template <typename T>
T *ForwardPass(const T *weights, const T *biases, int countNeurons,
               const T *inputs, int inputSize, bool gelu = false) {

  // out = weights @ inputs  (countNeurons x 1)
  T *output = MatMul<T>(weights, static_cast<size_t>(countNeurons) * inputSize,
                        inputs, static_cast<size_t>(inputSize), countNeurons,
                        inputSize, inputSize, 1);

  addRowBias(output, biases, 1, countNeurons);

  if (gelu) {
    auto geluNew = [] __host__ __device__(T x) -> T {
      return T(.5) * x *
             (T(1) + tanh(sqrt(T(2.0) / M_PI) *
                          (x + T(0.044715) * x * x * x)));
    };
    vectorMapInPlace(output, countNeurons, [geluNew] __device__ __host__(T &v) {
      v = geluNew(v);
      return v;
    });
  }

  return output;
}

}; // namespace CUDA
