#pragma once

#include <cmath>
#include <cuda_runtime.h>

#include "../constants.hh"
#include "../kernels/causalMask.cuh"
#include "../kernels/forwardPass.cuh"
#include "../kernels/layerNorm.cuh"
#include "../kernels/matmul.cuh"
#include "../kernels/packHead.cuh"
#include "../kernels/softmax.cuh"
#include "../kernels/transpose.cuh"
#include "../kernels/vectorCombine.cuh"
#include "../kernels/vectorMap.cuh"
#include "cudaBuffer.cuh"
#include "weights.cuh"

struct Transformer {

  static CudaBuffer<double> Attention(
      const CudaBuffer<double> &embeddings, const CudaBuffer<double> &qWeights,
      const CudaBuffer<double> &kWeights, const CudaBuffer<double> &vWeights,
      const CudaBuffer<double> &qBiases, const CudaBuffer<double> &kBiases,
      const CudaBuffer<double> &vBiases, int NUM_TOKENS, int embedDim,
      int headDim) {
    auto project = [&](const CudaBuffer<double> &weights,
                       const CudaBuffer<double> &biases) -> CudaBuffer<double> {
      auto wT = CUDA::Transpose(weights, headDim, embedDim);
      auto proj = CUDA::MatMul<double>(embeddings, wT, NUM_TOKENS, embedDim,
                                       embedDim, headDim);
      CUDA::addRowBias(proj, biases, NUM_TOKENS, headDim);
      return proj;
    };

    auto qProjections = project(qWeights, qBiases);
    auto kProjections = project(kWeights, kBiases);
    auto vProjections = project(vWeights, vBiases);

    auto kTranspose = CUDA::Transpose(kProjections, NUM_TOKENS, headDim);

    auto qkTranspose = CUDA::MatMul<double>(
        qProjections, kTranspose, NUM_TOKENS, headDim, headDim, NUM_TOKENS);

    qProjections = CudaBuffer<double>();
    kProjections = CudaBuffer<double>();
    kTranspose = CudaBuffer<double>();

    double dimensionsRoot = sqrt(headDim);

    CUDA::vectorMapInPlace(qkTranspose,
                           [dimensionsRoot] __device__ __host__(double &v) {
                             v /= dimensionsRoot;
                             return v;
                           });

    CUDA::causalMask(qkTranspose, NUM_TOKENS);
    CUDA::SoftMaxRows(qkTranspose, NUM_TOKENS, NUM_TOKENS);

    return CUDA::MatMul<double>(qkTranspose, vProjections, NUM_TOKENS,
                                NUM_TOKENS, NUM_TOKENS, headDim);
  }

  static CudaBuffer<double> MultiHeadAttention(
      const CudaBuffer<double> &embeddings, const CudaBuffer<double> &qWeights,
      const CudaBuffer<double> &kWeights, const CudaBuffer<double> &vWeights,
      const CudaBuffer<double> &oWeights, const CudaBuffer<double> &oBiases,
      const CudaBuffer<double> &qBiases, const CudaBuffer<double> &kBiases,
      const CudaBuffer<double> &vBiases, int NUM_TOKENS, int embedDim,
      int heads, int headDim) {
    CudaBuffer<double> packed(static_cast<size_t>(NUM_TOKENS) * embedDim);
    packed.zero();

    int weightBlockSize = embedDim * headDim;

    for (int h = 0; h < heads; h++) {
      auto qHead = qWeights.slice(static_cast<size_t>(h) * weightBlockSize,
                                  weightBlockSize);
      auto kHead = kWeights.slice(static_cast<size_t>(h) * weightBlockSize,
                                  weightBlockSize);
      auto vHead = vWeights.slice(static_cast<size_t>(h) * weightBlockSize,
                                  weightBlockSize);

      auto qBiasHead = qBiases.slice(static_cast<size_t>(h) * headDim, headDim);
      auto kBiasHead = kBiases.slice(static_cast<size_t>(h) * headDim, headDim);
      auto vBiasHead = vBiases.slice(static_cast<size_t>(h) * headDim, headDim);

      auto curResult =
          Attention(embeddings, qHead, kHead, vHead, qBiasHead, kBiasHead,
                    vBiasHead, NUM_TOKENS, embedDim, headDim);

      CUDA::packHead(curResult, packed, NUM_TOKENS, headDim, embedDim, h);
    }

    auto projectionResult = CUDA::MatMul<double>(packed, oWeights, NUM_TOKENS,
                                                 embedDim, embedDim, embedDim);

    CUDA::addRowBias(projectionResult, oBiases, NUM_TOKENS, embedDim);
    return projectionResult;
  }

  static CudaBuffer<double>
  MLP(const CudaBuffer<double> &embeddings, const CudaBuffer<double> &l1Weights,
      const CudaBuffer<double> &l1Biases, const CudaBuffer<double> &l2Weights,
      const CudaBuffer<double> &l2Biases, int NUM_TOKENS, int dimensions) {
    CudaBuffer<double> result(static_cast<size_t>(NUM_TOKENS) * dimensions);

    for (int i = 0; i < NUM_TOKENS; i++) {
      auto tokenEmbedding =
          embeddings.slice(static_cast<size_t>(i) * dimensions, dimensions);
      auto hiddenOut =
          CUDA::ForwardPass(l1Weights, l1Biases, tokenEmbedding, true);
      auto out = CUDA::ForwardPass(l2Weights, l2Biases, hiddenOut);

      result.copyFrom(out, static_cast<size_t>(i) * dimensions);
    }

    return result;
  }

  static CudaBuffer<double> loop(const BlockWeights &input,
                                 const CudaBuffer<double> &embeddings,
                                 int NUM_TOKENS) {
    constexpr int D = N_EMBD;
    constexpr int HEAD_DIM = N_EMBD / N_HEAD;

    auto add = [] __host__ __device__(const double &x, const double &y)
        -> double { return x + y; };

    CudaBuffer<double> layerNormedEmbeddings(static_cast<size_t>(NUM_TOKENS) *
                                             D);

    for (int i = 0; i < NUM_TOKENS; i++) {
      auto token = embeddings.slice(static_cast<size_t>(i) * D, D);
      auto normed =
          CUDA::LayerNorm(token, input.ln1.gamma, input.ln1.beta, EPSILON);
      layerNormedEmbeddings.copyFrom(normed, static_cast<size_t>(i) * D);
    }

    auto attentionResult = MultiHeadAttention(
        layerNormedEmbeddings, input.attn.q, input.attn.k, input.attn.v,
        input.attn.o, input.attn.ob, input.attn.qb, input.attn.kb,
        input.attn.vb, NUM_TOKENS, D, N_HEAD, HEAD_DIM);

    layerNormedEmbeddings = CudaBuffer<double>();

    CudaBuffer<double> normedForMLP(static_cast<size_t>(NUM_TOKENS) * D);

    for (int i = 0; i < NUM_TOKENS; i++) {
      auto embRow = embeddings.slice(static_cast<size_t>(i) * D, D);
      auto attnRow = attentionResult.slice(static_cast<size_t>(i) * D, D);
      auto withResidual = CUDA::vectorCombine(embRow, attnRow, add);

      attentionResult.copyFrom(withResidual, static_cast<size_t>(i) * D);

      auto residualView = attentionResult.slice(static_cast<size_t>(i) * D, D);
      auto normed = CUDA::LayerNorm(residualView, input.ln2.gamma,
                                    input.ln2.beta, EPSILON);

      normedForMLP.copyFrom(normed, static_cast<size_t>(i) * D);
    }

    auto MLPResult = MLP(normedForMLP, input.mlp.fc, input.mlp.fcb,
                         input.mlp.proj, input.mlp.projb, NUM_TOKENS, D);

    normedForMLP = CudaBuffer<double>();

    for (int i = 0; i < NUM_TOKENS; i++) {
      auto attnRow = attentionResult.slice(static_cast<size_t>(i) * D, D);
      auto mlpRow = MLPResult.slice(static_cast<size_t>(i) * D, D);
      CUDA::vectorCombineInto(attnRow, mlpRow, mlpRow, add);
    }

    return MLPResult;
  }
};
