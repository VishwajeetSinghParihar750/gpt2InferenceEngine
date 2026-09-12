#pragma once

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <limits>
#include <math_constants.h>
#include <numeric>
#include <vector>

#include "buffer.cuh"

namespace CUDA {

constexpr int kThreads = 1024;
constexpr int tileRows = 32;
constexpr int tileCols = tileRows;

// --- matmul ---

template <typename T>
__global__ void matMulKernel(const T *a, const T *b, T *c, int an, int am,
                             int bn, int bm, int lda, int aCol0) {
  int tileLoops = (am + blockDim.x - 1) / blockDim.x;

  __shared__ T sharedA[tileRows][tileCols];
  __shared__ T sharedB[tileRows][tileCols];

  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  T sum = 0;

  for (int i = 0; i < tileLoops; i++) {
    int aCol = blockDim.x * i + threadIdx.x;
    int bRow = blockDim.y * i + threadIdx.y;

    sharedA[threadIdx.y][threadIdx.x] =
        (row < an && aCol < am) ? a[row * lda + aCol0 + aCol] : 0;
    sharedB[threadIdx.y][threadIdx.x] =
        (bRow < bn && col < bm) ? b[bRow * bm + col] : 0;

    __syncthreads();

    for (int l = 0; l < blockDim.x; l++) {
      sum += sharedA[threadIdx.y][l] * sharedB[l][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < an && col < bm)
    c[row * bm + col] = sum;
}

// A is an x am (optionally a view into wider rows: stride lda, start col aCol0).
template <typename T>
CudaBuffer<T> MatMul(const CudaBuffer<T> &da, const CudaBuffer<T> &db,
                     const int an, const int am, const int bn, const int bm,
                     int lda = -1, int aCol0 = 0) {
  assert(am == bn);
  if (lda < 0)
    lda = am;
  assert(aCol0 >= 0 && aCol0 + am <= lda);
  assert(da.n >= static_cast<size_t>(an) * lda);
  assert(db.n == static_cast<size_t>(bn) * bm);

  CudaBuffer<T> dc(static_cast<size_t>(an) * bm);
  dc.zero();

  dim3 threadsPerBlock(tileCols, tileRows);
  dim3 blocksPerGrid((bm + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (an + threadsPerBlock.y - 1) / threadsPerBlock.y);

  matMulKernel<<<blocksPerGrid, threadsPerBlock>>>(da.ptr, db.ptr, dc.ptr, an,
                                                   am, bn, bm, lda, aCol0);

  cudaError_t cudaError = cudaGetLastError();
  if (cudaError != cudaSuccess) {
    std::cout << "kernel error  : " << cudaGetErrorString(cudaError);
    std::exit(-1);
  }

  cudaDeviceSynchronize();
  return dc;
}

// Copy one head (strided in src) into a contiguous dst buffer.
template <typename T>
__global__ void copyStridedHeadKernel(const T *src, T *dst, int numTokens,
                                      int headDim, int lda, int col0) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int total = numTokens * headDim;
  if (i < total) {
    int tok = i / headDim;
    int d = i % headDim;
    dst[i] = src[tok * lda + col0 + d];
  }
}

template <typename T>
void copyStridedHead(const CudaBuffer<T> &src, CudaBuffer<T> &dst,
                     int numTokens, int headDim, int lda, int col0) {
  assert(dst.n >= static_cast<size_t>(numTokens) * headDim);
  assert(src.n >= static_cast<size_t>(numTokens) * lda);
  assert(col0 + headDim <= lda);

  const int total = numTokens * headDim;
  copyStridedHeadKernel<<<(total + kThreads - 1) / kThreads, kThreads>>>(
      src.ptr, dst.ptr, numTokens, headDim, lda, col0);

  cudaError_t cudaError = cudaGetLastError();
  if (cudaError != cudaSuccess) {
    std::cout << "kernel error  : " << cudaGetErrorString(cudaError);
    std::exit(-1);
  }

  cudaDeviceSynchronize();
}

// --- transpose ---

template <typename T>
__global__ void transposeKernel(const T *a, T *result, int n, int m) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int total = n * m;
  if (i < total) {
    int row = i / m;
    int col = i % m;
    result[col * n + row] = a[i];
  }
}

// da is n x m row-major; result is m x n.
template <typename T>
CudaBuffer<T> Transpose(const CudaBuffer<T> &da, int n, int m) {
  assert(da.n == static_cast<size_t>(n) * m);
  CudaBuffer<T> result(static_cast<size_t>(m) * n);

  const int total = n * m;
  transposeKernel<<<(total + kThreads - 1) / kThreads, kThreads>>>(
      da.ptr, result.ptr, n, m);

  cudaError_t cudaError = cudaGetLastError();
  if (cudaError != cudaSuccess) {
    std::cout << "kernel error  : " << cudaGetErrorString(cudaError);
    std::exit(-1);
  }

  cudaDeviceSynchronize();
  return result;
}

// --- vector combine / bias ---

template <typename T, typename Op>
__global__ void vectorCombineKernel(const T *a, const T *b, T *c, int n,
                                    Op op) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n)
    c[i] = op(a[i], b[i]);
}

template <typename T, typename Op>
CudaBuffer<T> vectorCombine(const CudaBuffer<T> &da, const CudaBuffer<T> &db,
                            Op op) {
  assert(da.n == db.n);
  const int n = static_cast<int>(da.n);
  CudaBuffer<T> dc(da.n);

  vectorCombineKernel<<<(n + kThreads - 1) / kThreads, kThreads>>>(
      da.ptr, db.ptr, dc.ptr, n, op);

  cudaError_t cudaError = cudaGetLastError();
  if (cudaError != cudaSuccess) {
    std::cout << "kernel error  : " << cudaGetErrorString(cudaError);
    std::exit(-1);
  }

  cudaDeviceSynchronize();
  return dc;
}

template <typename T, typename Op>
void vectorCombineInto(const CudaBuffer<T> &da, const CudaBuffer<T> &db,
                       CudaBuffer<T> &dc, Op op) {
  assert(da.n == db.n && da.n == dc.n);
  const int n = static_cast<int>(da.n);

  vectorCombineKernel<<<(n + kThreads - 1) / kThreads, kThreads>>>(
      da.ptr, db.ptr, dc.ptr, n, op);

  cudaError_t cudaError = cudaGetLastError();
  if (cudaError != cudaSuccess) {
    std::cout << "kernel error  : " << cudaGetErrorString(cudaError);
    std::exit(-1);
  }

  cudaDeviceSynchronize();
}

template <typename T>
__global__ void addRowBiasKernel(T *mat, const T *bias, int rows, int cols) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < rows * cols)
    mat[i] += bias[i % cols];
}

template <typename T>
void addRowBias(CudaBuffer<T> &mat, const CudaBuffer<T> &bias, int rows,
                int cols) {
  assert(mat.n == static_cast<size_t>(rows) * cols);
  assert(bias.n == static_cast<size_t>(cols));
  const int total = rows * cols;
  addRowBiasKernel<<<(total + kThreads - 1) / kThreads, kThreads>>>(
      mat.ptr, bias.ptr, rows, cols);

  cudaError_t cudaError = cudaGetLastError();
  if (cudaError != cudaSuccess) {
    std::cout << "kernel error  : " << cudaGetErrorString(cudaError);
    std::exit(-1);
  }

  cudaDeviceSynchronize();
}

// --- vector map ---

template <typename T, typename Op>
__global__ void vectorMapKernel(const T *a, T *b, int n, Op op) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n)
    b[i] = op(a[i]);
}

template <typename T, typename Op>
CudaBuffer<T> vectorMap(const CudaBuffer<T> &da, Op op) {
  const int n = static_cast<int>(da.n);
  CudaBuffer<T> db(da.n);

  vectorMapKernel<<<(n + kThreads - 1) / kThreads, kThreads>>>(da.ptr, db.ptr,
                                                               n, op);

  cudaError_t cudaError = cudaGetLastError();
  if (cudaError != cudaSuccess) {
    std::cout << "kernel error  : " << cudaGetErrorString(cudaError);
    std::exit(-1);
  }

  cudaDeviceSynchronize();
  return db;
}

template <typename T, typename Op>
__global__ void vectorMapInPlaceKernel(T *a, int n, Op op) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < n)
    a[i] = op(a[i]);
}

template <typename T, typename Op>
void vectorMapInPlace(CudaBuffer<T> &da, Op op) {
  const int n = static_cast<int>(da.n);

  vectorMapInPlaceKernel<<<(n + kThreads - 1) / kThreads, kThreads>>>(da.ptr, n,
                                                                      op);

  cudaError_t cudaError = cudaGetLastError();
  if (cudaError != cudaSuccess) {
    std::cout << "kernel error  : " << cudaGetErrorString(cudaError);
    std::exit(-1);
  }

  cudaDeviceSynchronize();
}

// --- vector reduction ---

template <typename Op>
__global__ void vectorReductionBlock(const float *a, float *b, int n, Op op,
                                     float defaultValue) {
  __shared__ float data[kThreads];

  int gi = blockDim.x * blockIdx.x + threadIdx.x;

  data[threadIdx.x] = gi < n ? a[gi] : defaultValue;

  __syncthreads();

  int len = kThreads;

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
float vectorReduction(const CudaBuffer<float> &a, KernelOp kernelOp,
                      AccumulateOp accumulateOp, float kernelDefaultValue,
                      float accumulateDefaultValue) {
  const int n = static_cast<int>(a.n);
  const int blocks = (n + kThreads - 1) / kThreads;

  CudaBuffer<float> deviceResult(static_cast<size_t>(blocks));

  vectorReductionBlock<<<blocks, kThreads>>>(a.ptr, deviceResult.ptr, n,
                                             kernelOp, kernelDefaultValue);

  std::vector<float> result(blocks);
  deviceResult.copyToHost(result.data());

  return std::accumulate(result.begin(), result.end(), accumulateDefaultValue,
                         accumulateOp);
}

template <typename KernelOp, typename AccumulateOp>
float vectorReduction(const CudaBuffer<float> &a, KernelOp kernelOp,
                      AccumulateOp accumulateOp, float kernelDefaultValue) {
  return vectorReduction(a, kernelOp, accumulateOp, kernelDefaultValue,
                         kernelDefaultValue);
}

template <typename KernelOp>
float vectorReduction(const CudaBuffer<float> &a, KernelOp kernelOp,
                      float defaultValue) {
  return vectorReduction(a, kernelOp, kernelOp, defaultValue, defaultValue);
}

// --- softmax ---

void SoftMaxInPlace(CudaBuffer<float> &input) {
  auto mxLambda = [] __host__ __device__(const float &a,
                                         const float &b) -> float {
    return a > b ? a : b;
  };
  auto sumLambda = [] __host__ __device__(const float &a,
                                          const float &b) -> float {
    return a + b;
  };

  float max = vectorReduction(input, mxLambda,
                              std::numeric_limits<float>::lowest());

  vectorMapInPlace(input, [max] __device__ __host__(float &v) {
    v = expf(v - max);
    return v;
  });

  float sum = vectorReduction(input, sumLambda, 0.0f);

  vectorMapInPlace(input, [sum] __device__ __host__(float &v) {
    v /= sum;
    return v;
  });
}

void SoftMaxRows(CudaBuffer<float> &scores, int rows, int cols) {
  assert(scores.n == static_cast<size_t>(rows) * cols);
  for (int i = 0; i < rows; i++) {
    auto row = scores.slice(static_cast<size_t>(i) * cols, cols);
    SoftMaxInPlace(row);
  }
}

// --- layer norm ---

template <typename T>
CudaBuffer<T> LayerNorm(const CudaBuffer<T> &originalEmbedding,
                        const CudaBuffer<T> &gamma, const CudaBuffer<T> &beta,
                        T epsilon) {
  const int len = static_cast<int>(originalEmbedding.n);
  assert(gamma.n == originalEmbedding.n && beta.n == originalEmbedding.n);

  auto add = [] __device__ __host__(const T &a, const T &b) -> T {
    return a + b;
  };

  T sum = vectorReduction(originalEmbedding, add, T(0));
  T mean = sum / static_cast<T>(len);

  auto squaredDiffs = vectorMap(
      originalEmbedding,
      [mean] __device__ __host__(const T &x) -> T {
        return (x - mean) * (x - mean);
      });

  T sumSquaredDiffs = vectorReduction(squaredDiffs, add, T(0));
  T variance = sumSquaredDiffs / static_cast<T>(len);

  T modifiedStandardDeviation = sqrt(variance + epsilon);

  auto normalized = vectorMap(
      originalEmbedding,
      [mean, modifiedStandardDeviation] __device__ __host__(const T &x) -> T {
        return (x - mean) / modifiedStandardDeviation;
      });

  auto scaled = vectorCombine(
      normalized, gamma,
      [] __device__ __host__(const T &a, const T &b) -> T { return a * b; });

  return vectorCombine(scaled, beta, add);
}

// --- causal mask ---

template <typename T>
__global__ void causalMaskKernel(T *scores, int rows, int cols,
                                 int queryPosOffset) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int total = rows * cols;
  if (idx < total) {
    int i = idx / cols;
    int j = idx % cols;
    if (j > queryPosOffset + i)
      scores[idx] = -CUDART_INF_F;
  }
}

template <typename T>
void causalMask(CudaBuffer<T> &scores, int rows, int cols,
                int queryPosOffset = 0) {
  assert(scores.n == static_cast<size_t>(rows) * cols);
  const int total = rows * cols;
  causalMaskKernel<<<(total + kThreads - 1) / kThreads, kThreads>>>(
      scores.ptr, rows, cols, queryPosOffset);

  cudaError_t cudaError = cudaGetLastError();
  if (cudaError != cudaSuccess) {
    std::cout << "kernel error  : " << cudaGetErrorString(cudaError);
    std::exit(-1);
  }

  cudaDeviceSynchronize();
}

template <typename T>
void causalMask(CudaBuffer<T> &scores, int numTokens) {
  causalMask(scores, numTokens, numTokens, 0);
}

// --- pack attention head ---

template <typename T>
__global__ void packHeadKernel(const T *headOut, T *result, int numTokens,
                               int headDim, int embedDim, int headIdx) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int total = numTokens * headDim;
  if (i < total) {
    int j = i / headDim;
    int k = i % headDim;
    result[j * embedDim + headIdx * headDim + k] = headOut[i];
  }
}

template <typename T>
void packHead(const CudaBuffer<T> &headOut, CudaBuffer<T> &result,
              int numTokens, int headDim, int embedDim, int headIdx) {
  assert(headOut.n == static_cast<size_t>(numTokens) * headDim);
  assert(result.n == static_cast<size_t>(numTokens) * embedDim);

  const int total = numTokens * headDim;
  packHeadKernel<<<(total + kThreads - 1) / kThreads, kThreads>>>(
      headOut.ptr, result.ptr, numTokens, headDim, embedDim, headIdx);

  cudaError_t cudaError = cudaGetLastError();
  if (cudaError != cudaSuccess) {
    std::cout << "kernel error  : " << cudaGetErrorString(cudaError);
    std::exit(-1);
  }

  cudaDeviceSynchronize();
}

// --- MLP linear (+ optional GELU) ---

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
