#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <regex>
#include "constants.hh"
#include "include/json.hpp"
#include "kernels/matmul.cuh"
#include "kernels/vectorCombine.cuh"
#include "kernels/softmax.cuh"
#include "kernels/layerNorm.cuh"
#include "kernels/dotProduct.cuh"
#include "kernels/transpose.cuh"
#include "kernels/causalMask.cuh"
#include "kernels/vectorMap.cuh"
#include "kernels/packHead.cuh"
#include "kernels/forwardPass.cuh"

// Caller owns returned pointer (cudaFree).
double *Attention(
    const double *embeddings, int NUM_TOKENS,
    int embedDim, // embeddings: 2d (NUM_TOKENS x embedDim) flattened as 1d
    const double *qWeights, // 2d (headDim x embedDim) flattened as 1d, transposed
    const double *kWeights, // 2d (headDim x embedDim) flattened as 1d, transposed
    const double *vWeights, // 2d (headDim x embedDim) flattened as 1d, transposed
    const double *qBiases,  // 1d, this head's slice (headDim)
    const double *kBiases,  // 1d, this head's slice (headDim)
    const double *vBiases,  // 1d, this head's slice (headDim)
    int headDim) {
  // q/k/vWeights are headDim x embedDim; project as emb @ W^T + bias
  auto project = [&](const double *weights, const double *biases) -> double * {
    double *wT = CUDA::Transpose(weights, headDim, embedDim);
    double *proj = CUDA::MatMul<double>(
        embeddings, static_cast<size_t>(NUM_TOKENS) * embedDim, wT,
        static_cast<size_t>(embedDim) * headDim, NUM_TOKENS, embedDim, embedDim,
        headDim);
    cudaFree(wT);
    CUDA::addRowBias(proj, biases, NUM_TOKENS, headDim);
    return proj;
  };

  double *qProjections = project(qWeights, qBiases);
  double *kProjections = project(kWeights, kBiases);
  double *vProjections = project(vWeights, vBiases);

  double *kTranspose =
      CUDA::Transpose(kProjections, NUM_TOKENS, headDim); // (headDim x NUM_TOKENS)

  double *qkTranspose = CUDA::MatMul<double>(
      qProjections, static_cast<size_t>(NUM_TOKENS) * headDim, kTranspose,
      static_cast<size_t>(headDim) * NUM_TOKENS, NUM_TOKENS, headDim, headDim,
      NUM_TOKENS);

  cudaFree( qProjections);
  cudaFree( kProjections);
  cudaFree( kTranspose);

  double dimensionsRoot = sqrt(headDim);

  CUDA::vectorMapInPlace(qkTranspose, NUM_TOKENS * NUM_TOKENS,
     [dimensionsRoot] __device__ __host__(double & v){
          v /= dimensionsRoot;
          return v;
  });

  // causal mask: token i cannot attend to future tokens j > i
  CUDA::causalMask(qkTranspose, NUM_TOKENS);

  CUDA::SoftMaxRows(qkTranspose, NUM_TOKENS, NUM_TOKENS);

  double *result = CUDA::MatMul<double>(
      qkTranspose, static_cast<size_t>(NUM_TOKENS) * NUM_TOKENS, vProjections,
      static_cast<size_t>(NUM_TOKENS) * headDim, NUM_TOKENS, NUM_TOKENS,
      NUM_TOKENS, headDim);

  cudaFree(qkTranspose) ;
  cudaFree(vProjections);

  return result;
}

// Caller owns returned pointer (cudaFree).
double *MultiHeadAttention(
    const double *embeddings, int NUM_TOKENS,
    int embedDim, // embeddings: 2d (NUM_TOKENS x embedDim) flattened as 1d
    const double *qWeights, // 3d (heads x headDim x embedDim), transposed
    const double *kWeights, // 3d (heads x headDim x embedDim), transposed
    const double *vWeights, // 3d (heads x headDim x embedDim), transposed
    const double *oWeights, // 2d (embedDim x embedDim) flattened as 1d
    int heads, int headDim,
    const double *oBiases, // embedDim 1d
    const double *qBiases, // 2d (heads x headDim) flattened as 1d
    const double *kBiases, // 2d (heads x headDim) flattened as 1d
    const double *vBiases) // 2d (heads x headDim) flattened as 1d
{
  double *result;
  cudaMalloc(&result, static_cast<size_t>(NUM_TOKENS) * embedDim * sizeof(double));
  cudaMemset(result, 0, static_cast<size_t>(NUM_TOKENS) * embedDim * sizeof(double));

  int weightBlockSize = embedDim * headDim;

  for (int h = 0; h < heads; h++) {
    const double *qHead = qWeights + h * weightBlockSize;
    const double *kHead = kWeights + h * weightBlockSize;
    const double *vHead = vWeights + h * weightBlockSize;

    const double *qBiasHead = qBiases + h * headDim;
    const double *kBiasHead = kBiases + h * headDim;
    const double *vBiasHead = vBiases + h * headDim;

    double *curResult =
        Attention(embeddings, NUM_TOKENS, embedDim, qHead, kHead, vHead,
                  qBiasHead, kBiasHead, vBiasHead, headDim);

    CUDA::packHead(curResult, result, NUM_TOKENS, headDim, embedDim, h);

    cudaFree(curResult);
  }

  double *projectionResult = CUDA::MatMul<double>(
      result, static_cast<size_t>(NUM_TOKENS) * embedDim, oWeights,
      static_cast<size_t>(embedDim) * embedDim, NUM_TOKENS, embedDim, embedDim,
      embedDim);

  cudaFree(result);

  CUDA::addRowBias(projectionResult, oBiases, NUM_TOKENS, embedDim);

  return projectionResult;
}

// Caller owns returned pointer (cudaFree).
double *MLP(const double *embeddings, int NUM_TOKENS,
            int dimensions, // embeddings: 2d (NUM_TOKENS x dimensions)
            const double *l1Weights, // 2d (hidden x dimensions), transposed
            const double *l1Biases, int l1BiasesSize, // 1d
            const double *l2Weights, // 2d (dimensions x hidden), transposed
            const double *l2Biases)  // 1d
{
  double *result;
  cudaMalloc(&result, static_cast<size_t>(NUM_TOKENS) * dimensions * sizeof(double));

  for (int i = 0; i < NUM_TOKENS; i++) {
    const double *tokenEmbedding = embeddings + i * dimensions; // 1d
    double *hiddenOut = CUDA::ForwardPass(
        l1Weights, l1Biases, l1BiasesSize, tokenEmbedding, dimensions, true);
    double *out = CUDA::ForwardPass(l2Weights, l2Biases, dimensions, hiddenOut,
                                    l1BiasesSize);
    cudaFree(hiddenOut);

    cudaMemcpy(result + i * dimensions, out,
               static_cast<size_t>(dimensions) * sizeof(double),
               cudaMemcpyDeviceToDevice);
    cudaFree(out);
  }

  return result;
}

struct TransformerInput {
  static constexpr int EMBEDDING_DIMENSION = 768;
  static constexpr int HEADS = 12;
  static constexpr int LAYERS = 12;

  static constexpr double EPSILON_ATTENTION = 1e-5;
  static constexpr double EPSILON_MLP = 1e-5;

  static constexpr int HEAD_DIMENSION = 64;

  // double* for GPU friendliness, plus explicit sizes for safety
  double *qWeights; // 3D (heads x headDim x embedDim), flattened as 1D, transposed
  double *kWeights; // 3D (heads x headDim x embedDim), flattened as 1D, transposed
  double *vWeights; // 3D (heads x headDim x embedDim), flattened as 1D, transposed

  double *qBiases; // 2D (heads x headDim), flattened as 1D
  double *kBiases; // 2D (heads x headDim), flattened as 1D
  double *vBiases; // 2D (heads x headDim), flattened as 1D

  double *oWeights; // 2D (embedDim x embedDim), flattened as 1D
  double *oBiases;

  double *l1Weights; // 2D (hidden x embedDim), flattened as 1D, transposed
  double *l1Biases;  // 1D

  double *l2Weights; // 2D (embedDim x hidden), flattened as 1D, transposed
  double *l2Biases;  // 1D

  double *gammaAttention; // 1D
  double *gammaMLP;       // 1D

  double *betaAttention; // 1D
  double *betaMLP;       // 1D

  // Sizes for all pointers (adapt as needed for real shape calculation)
  size_t qWeightsSize, kWeightsSize, vWeightsSize;
  size_t qBiasesSize, kBiasesSize, vBiasesSize;
  size_t oWeightsSize, oBiasesSize;
  size_t l1WeightsSize, l1BiasesSize;
  size_t l2WeightsSize, l2BiasesSize;
  size_t gammaAttentionSize, gammaMLPSize;
  size_t betaAttentionSize, betaMLPSize;

  TransformerInput() = default;
  // Note: ownership is not handled here; you may want to manage with smart pointers if needed!
  TransformerInput(double *qWeights, size_t qWeightsSize, double *kWeights,
                   size_t kWeightsSize, double *vWeights, size_t vWeightsSize,
                   double *qBiases, size_t qBiasesSize, double *kBiases,
                   size_t kBiasesSize, double *vBiases, size_t vBiasesSize,
                   double *oWeights, size_t oWeightsSize, double *oBiases,
                   size_t oBiasesSize, double *l1Weights, size_t l1WeightsSize,
                   double *l1Biases, size_t l1BiasesSize, double *l2Weights,
                   size_t l2WeightsSize, double *l2Biases, size_t l2BiasesSize,
                   double *gammaAttention, size_t gammaAttentionSize,
                   double *gammaMLP, size_t gammaMLPSize, double *betaAttention,
                   size_t betaAttentionSize, double *betaMLP,
                   size_t betaMLPSize)
      : qWeights(qWeights), qWeightsSize(qWeightsSize), kWeights(kWeights),
        kWeightsSize(kWeightsSize), vWeights(vWeights),
        vWeightsSize(vWeightsSize), qBiases(qBiases), qBiasesSize(qBiasesSize),
        kBiases(kBiases), kBiasesSize(kBiasesSize), vBiases(vBiases),
        vBiasesSize(vBiasesSize), oWeights(oWeights), oWeightsSize(oWeightsSize),
        oBiases(oBiases), oBiasesSize(oBiasesSize), l1Weights(l1Weights),
        l1WeightsSize(l1WeightsSize), l1Biases(l1Biases),
        l1BiasesSize(l1BiasesSize), l2Weights(l2Weights),
        l2WeightsSize(l2WeightsSize), l2Biases(l2Biases),
        l2BiasesSize(l2BiasesSize), gammaAttention(gammaAttention),
        gammaAttentionSize(gammaAttentionSize), gammaMLP(gammaMLP),
        gammaMLPSize(gammaMLPSize), betaAttention(betaAttention),
        betaAttentionSize(betaAttentionSize), betaMLP(betaMLP),
        betaMLPSize(betaMLPSize) {}
};

// Caller owns returned pointer (cudaFree).
double *Transformer(const TransformerInput &input, const int NUM_TOKENS,
                    const double *embeddings // 2D (NUM_TOKENS x embedDim), device
) {
  constexpr int D = TransformerInput::EMBEDDING_DIMENSION;
  const size_t rowBytes = static_cast<size_t>(D) * sizeof(double);

  auto add = [] __host__ __device__(const double &x, const double &y) -> double {
    return x + y;
  };

  double *layerNormedEmbeddings;
  cudaMalloc(&layerNormedEmbeddings,
             static_cast<size_t>(NUM_TOKENS) * D * sizeof(double));

  for (int i = 0; i < NUM_TOKENS; i++) {
    double *normed = CUDA::LayerNorm(
        embeddings + i * D, input.gammaAttention, input.betaAttention, D,
        TransformerInput::EPSILON_ATTENTION);
    cudaMemcpy(layerNormedEmbeddings + i * D, normed, rowBytes,
               cudaMemcpyDeviceToDevice);
    cudaFree(normed);
  }

  double *attentionResult = MultiHeadAttention(
      layerNormedEmbeddings, NUM_TOKENS, D, input.qWeights, input.kWeights,
      input.vWeights, input.oWeights, TransformerInput::HEADS,
      TransformerInput::HEAD_DIMENSION, input.oBiases, input.qBiases,
      input.kBiases, input.vBiases);

  cudaFree(layerNormedEmbeddings);

  double *normedForMLP;
  cudaMalloc(&normedForMLP, static_cast<size_t>(NUM_TOKENS) * D * sizeof(double));

  for (int i = 0; i < NUM_TOKENS; i++) {
    double *withResidual = CUDA::vectorCombine(
        embeddings + i * D, attentionResult + i * D, D, add);

    cudaMemcpy(attentionResult + i * D, withResidual, rowBytes,
               cudaMemcpyDeviceToDevice);

    double *normed =
        CUDA::LayerNorm(withResidual, input.gammaMLP, input.betaMLP, D,
                        TransformerInput::EPSILON_MLP);
    cudaFree(withResidual);

    cudaMemcpy(normedForMLP + i * D, normed, rowBytes, cudaMemcpyDeviceToDevice);
    cudaFree(normed);
  }

  double *MLPResult =
      MLP(normedForMLP, NUM_TOKENS, D, input.l1Weights, input.l1Biases,
          static_cast<int>(input.l1BiasesSize), input.l2Weights, input.l2Biases);

  cudaFree(normedForMLP);

  for (int i = 0; i < NUM_TOKENS; i++) {
    CUDA::vectorCombineInto(attentionResult + i * D, MLPResult + i * D,
                            MLPResult + i * D, D, add);
  }

  cudaFree(attentionResult);

  return MLPResult;
}

struct GptWeights {
  // Host array; members point to device buffers.
  TransformerInput *transformerWeights;
  double *finalLayerNormWeights;
  double *finalLayerNormBiases;
  double *wpeWeights;
  double *wteWeights;
  size_t numTransformerLayers;
  size_t finalLayerNormWeightsSize;
  size_t finalLayerNormBiasesSize;
  size_t wpeWeightsSize;
  size_t wteWeightsSize;
};

void InputVectorFromFilePtr(double *ptr, std::ifstream &stream, size_t count) {
  double inp;
  size_t idx = 0;
  while (idx < count && stream >> inp) {
    ptr[idx++] = inp;
  }
}


GptWeights LoadWeights() {
  constexpr size_t LAYERS = TransformerInput::LAYERS;
  constexpr size_t D = TransformerInput::EMBEDDING_DIMENSION;
  constexpr size_t HIDDEN = MLP_HIDDEN;
  constexpr size_t FINAL_LN_SIZE = D;
  constexpr size_t WPE_SIZE = N_CTX * D;
  constexpr size_t WTE_SIZE = VOCAB_SIZE * D;
  constexpr size_t ATTN_WEIGHT_SIZE = D * D;
  constexpr size_t ATTN_BIAS_SIZE = D;
  constexpr size_t L1_WEIGHT_SIZE = HIDDEN * D;
  constexpr size_t L2_WEIGHT_SIZE = D * HIDDEN;

  // Host array of layer configs; each weight/bias pointer inside is device memory.
  TransformerInput *hostTransformerWeights = new TransformerInput[LAYERS];

  auto copyToDevice = [](const double *hostPtr, size_t count) -> double * {
    double *devPtr = nullptr;
    cudaMalloc(&devPtr, count * sizeof(double));
    cudaMemcpy(devPtr, hostPtr, count * sizeof(double), cudaMemcpyHostToDevice);
    return devPtr;
  };

  // For each layer, load from disk to CPU and then copy to device
  for (size_t i = 0; i < LAYERS; i++) {
    double *gammaAttention = new double[D];
    double *betaAttention = new double[D];
    double *gammaMLP = new double[D];
    double *betaMLP = new double[D];
    double *l1Weights = new double[L1_WEIGHT_SIZE];
    double *l1Biases = new double[HIDDEN];
    double *l2Weights = new double[L2_WEIGHT_SIZE];
    double *l2Biases = new double[D];
    double *qWeights = new double[ATTN_WEIGHT_SIZE];
    double *kWeights = new double[ATTN_WEIGHT_SIZE];
    double *vWeights = new double[ATTN_WEIGHT_SIZE];
    double *qBiases = new double[ATTN_BIAS_SIZE];
    double *kBiases = new double[ATTN_BIAS_SIZE];
    double *vBiases = new double[ATTN_BIAS_SIZE];
    double *oWeights = new double[ATTN_WEIGHT_SIZE];
    double *oBiases = new double[ATTN_BIAS_SIZE];

    // LayerNorm 1
    {
      std::ifstream ln1Weights("../weights/transformer.h." + std::to_string(i) +
                               ".ln_1.weight.txt");
      InputVectorFromFilePtr(gammaAttention, ln1Weights, D);
      std::ifstream ln1Biases("../weights/transformer.h." + std::to_string(i) +
                              ".ln_1.bias.txt");
      InputVectorFromFilePtr(betaAttention, ln1Biases, D);
    }

    // LayerNorm 2
    {
      std::ifstream ln2Weights("../weights/transformer.h." + std::to_string(i) +
                               ".ln_2.weight.txt");
      InputVectorFromFilePtr(gammaMLP, ln2Weights, D);
      std::ifstream ln2Biases("../weights/transformer.h." + std::to_string(i) +
                              ".ln_2.bias.txt");
      InputVectorFromFilePtr(betaMLP, ln2Biases, D);
    }

    // MLP Layer 1
    {
      std::ifstream mlpL1Weights("../weights/transformer.h." +
                                 std::to_string(i) + ".mlp.c_fc.weight.txt");
      InputVectorFromFilePtr(l1Weights, mlpL1Weights, L1_WEIGHT_SIZE);
      std::ifstream mlpL1Biases("../weights/transformer.h." + std::to_string(i) +
                                ".mlp.c_fc.bias.txt");
      InputVectorFromFilePtr(l1Biases, mlpL1Biases, HIDDEN);
    }

    // MLP Layer 2
    {
      std::ifstream mlpL2Weights("../weights/transformer.h." +
                                 std::to_string(i) + ".mlp.c_proj.weight.txt");
      InputVectorFromFilePtr(l2Weights, mlpL2Weights, L2_WEIGHT_SIZE);
      std::ifstream mlpL2Biases("../weights/transformer.h." + std::to_string(i) +
                                ".mlp.c_proj.bias.txt");
      InputVectorFromFilePtr(l2Biases, mlpL2Biases, D);
    }

    // Transpose for device
    {
      double *l1T = CUDA::Transpose(l1Weights, D, HIDDEN);
      delete[] l1Weights;
      l1Weights = l1T;

      double *l2T = CUDA::Transpose(l2Weights, HIDDEN, D);
      delete[] l2Weights;
      l2Weights = l2T;
    }

    // Attention QKV weights and biases
    {
      std::ifstream qkvWeightsFile("../weights/transformer.h." +
                                   std::to_string(i) +
                                   ".attn.c_attn.weight.txt");
      size_t qIdx = 0, kIdx = 0, vIdx = 0;
      for (size_t j = 0; j < D; j++) {
        InputVectorFromFilePtr(qWeights + qIdx, qkvWeightsFile, D);
        qIdx += D;
        InputVectorFromFilePtr(kWeights + kIdx, qkvWeightsFile, D);
        kIdx += D;
        InputVectorFromFilePtr(vWeights + vIdx, qkvWeightsFile, D);
        vIdx += D;
      }

      double *qT = CUDA::Transpose(qWeights, D, D);
      delete[] qWeights;
      qWeights = qT;
      double *kT = CUDA::Transpose(kWeights, D, D);
      delete[] kWeights;
      kWeights = kT;
      double *vT = CUDA::Transpose(vWeights, D, D);
      delete[] vWeights;
      vWeights = vT;

      std::ifstream qkvBiases("../weights/transformer.h." + std::to_string(i) +
                              ".attn.c_attn.bias.txt");
      InputVectorFromFilePtr(qBiases, qkvBiases, D);
      InputVectorFromFilePtr(kBiases, qkvBiases, D);
      InputVectorFromFilePtr(vBiases, qkvBiases, D);
    }

    // Attention output projection
    {
      std::ifstream attnOutputProjWeights(
          "../weights/transformer.h." + std::to_string(i) +
          ".attn.c_proj.weight.txt");
      InputVectorFromFilePtr(oWeights, attnOutputProjWeights, ATTN_WEIGHT_SIZE);

      std::ifstream attnOutputProjBiases("../weights/transformer.h." +
                                         std::to_string(i) +
                                         ".attn.c_proj.bias.txt");
      InputVectorFromFilePtr(oBiases, attnOutputProjBiases, ATTN_BIAS_SIZE);
    }

    // Move weights to device
    double *gammaAttention_device = copyToDevice(gammaAttention, D);
    double *betaAttention_device = copyToDevice(betaAttention, D);
    double *gammaMLP_device = copyToDevice(gammaMLP, D);
    double *betaMLP_device = copyToDevice(betaMLP, D);
    double *l1Weights_device = copyToDevice(l1Weights, L1_WEIGHT_SIZE);
    double *l1Biases_device = copyToDevice(l1Biases, HIDDEN);
    double *l2Weights_device = copyToDevice(l2Weights, L2_WEIGHT_SIZE);
    double *l2Biases_device = copyToDevice(l2Biases, D);
    double *qWeights_device = copyToDevice(qWeights, ATTN_WEIGHT_SIZE);
    double *kWeights_device = copyToDevice(kWeights, ATTN_WEIGHT_SIZE);
    double *vWeights_device = copyToDevice(vWeights, ATTN_WEIGHT_SIZE);
    double *qBiases_device = copyToDevice(qBiases, ATTN_BIAS_SIZE);
    double *kBiases_device = copyToDevice(kBiases, ATTN_BIAS_SIZE);
    double *vBiases_device = copyToDevice(vBiases, ATTN_BIAS_SIZE);
    double *oWeights_device = copyToDevice(oWeights, ATTN_WEIGHT_SIZE);
    double *oBiases_device = copyToDevice(oBiases, ATTN_BIAS_SIZE);

    // Free host memory
    delete[] gammaAttention;
    delete[] betaAttention;
    delete[] gammaMLP;
    delete[] betaMLP;
    delete[] l1Weights;
    delete[] l1Biases;
    delete[] l2Weights;
    delete[] l2Biases;
    delete[] qWeights;
    delete[] kWeights;
    delete[] vWeights;
    delete[] qBiases;
    delete[] kBiases;
    delete[] vBiases;
    delete[] oWeights;
    delete[] oBiases;

    hostTransformerWeights[i] = TransformerInput{
        qWeights_device, ATTN_WEIGHT_SIZE, kWeights_device, ATTN_WEIGHT_SIZE,
        vWeights_device, ATTN_WEIGHT_SIZE, qBiases_device, ATTN_BIAS_SIZE,
        kBiases_device, ATTN_BIAS_SIZE, vBiases_device, ATTN_BIAS_SIZE,
        oWeights_device, ATTN_WEIGHT_SIZE, oBiases_device, ATTN_BIAS_SIZE,
        l1Weights_device, L1_WEIGHT_SIZE, l1Biases_device, HIDDEN,
        l2Weights_device, L2_WEIGHT_SIZE, l2Biases_device, D,
        gammaAttention_device, D, gammaMLP_device, D,
        betaAttention_device, D, betaMLP_device, D};
  }

  // Keep TransformerInput structs on host; weight buffers inside are on device.
  // (Host must be able to index weights.transformerWeights[i].)

  // Load final layer norm and embeddings to CPU
  double *finalLayerNormWeights = new double[FINAL_LN_SIZE];
  double *finalLayerNormBiases = new double[FINAL_LN_SIZE];
  double *wpeWeights = new double[WPE_SIZE];
  double *wteWeights = new double[WTE_SIZE];

  {
    std::ifstream lnWeights("../weights/transformer.ln_f.weight.txt");
    InputVectorFromFilePtr(finalLayerNormWeights, lnWeights, FINAL_LN_SIZE);
  }
  {
    std::ifstream lnBiases("../weights/transformer.ln_f.bias.txt");
    InputVectorFromFilePtr(finalLayerNormBiases, lnBiases, FINAL_LN_SIZE);
  }
  {
    std::ifstream wpeWeightsFile("../weights/transformer.wpe.weight.txt");
    InputVectorFromFilePtr(wpeWeights, wpeWeightsFile, WPE_SIZE);
    std::ifstream wteWeightsFile("../weights/transformer.wte.weight.txt");
    InputVectorFromFilePtr(wteWeights, wteWeightsFile, WTE_SIZE);
  }

  // Copy to GPU
  double *finalLayerNormWeights_device =
      copyToDevice(finalLayerNormWeights, FINAL_LN_SIZE);
  double *finalLayerNormBiases_device =
      copyToDevice(finalLayerNormBiases, FINAL_LN_SIZE);
  double *wpeWeights_device = copyToDevice(wpeWeights, WPE_SIZE);
  double *wteWeights_device = copyToDevice(wteWeights, WTE_SIZE);

  delete[] finalLayerNormWeights;
  delete[] finalLayerNormBiases;
  delete[] wpeWeights;
  delete[] wteWeights;

  // Host structs whose members point into VRAM.
  return {hostTransformerWeights,
          finalLayerNormWeights_device,
          finalLayerNormBiases_device,
          wpeWeights_device,
          wteWeights_device,
          LAYERS,
          FINAL_LN_SIZE,
          FINAL_LN_SIZE,
          WPE_SIZE,
          WTE_SIZE};
}

// return next tokens id
int GPT(const GptWeights &weights, const int numEmbeddings,
        const double *embeddings) {
  double *result =
      Transformer(weights.transformerWeights[0], numEmbeddings, embeddings);
  for (int i = 1; i < N_LAYER; i++) {
    double *next =
        Transformer(weights.transformerWeights[i], numEmbeddings, result);
    cudaFree(result);
    result = next;
  }

  const double *lastTokenEmbedding = result + (numEmbeddings - 1) * N_EMBD;

  double *layerNormedResult =
      CUDA::LayerNorm(lastTokenEmbedding, weights.finalLayerNormWeights,
                      weights.finalLayerNormBiases, N_EMBD, EPSILON);

  cudaFree(result);

  // logits = wte @ lastToken  (VOCAB_SIZE x 1)
  double *logits = CUDA::MatMul<double>(
      weights.wteWeights, static_cast<size_t>(VOCAB_SIZE) * N_EMBD,
      layerNormedResult, static_cast<size_t>(N_EMBD), VOCAB_SIZE, N_EMBD,
      N_EMBD, 1);
  cudaFree(layerNormedResult);

  CUDA::SoftMaxInPlace(logits, VOCAB_SIZE);

  std::vector<double> hostProbs(VOCAB_SIZE);
  cudaMemcpy(hostProbs.data(), logits, VOCAB_SIZE * sizeof(double),
             cudaMemcpyDeviceToHost);
  cudaFree(logits);

  int maxProbTokenId = static_cast<int>(
      std::max_element(hostProbs.begin(), hostProbs.end()) - hostProbs.begin());

  return maxProbTokenId;
}

std::vector<std::string> gpt2Tokens;
std::unordered_map<std::string, int> gpt2TokenToTokenId;
std::map<std::pair<std::string, std::string>, int> gpt2Merges;

void ParseMerges(const nlohmann::json &mergesJson) {
  gpt2Merges.clear();
  int priority = 0;
  for (const auto &merge : mergesJson) {
    gpt2Merges[{merge[0].get<std::string>(), merge[1].get<std::string>()}] =
        priority++;
  }
}

std::string GetTokenFromTokenId(int tokenId) {
  assert(!(tokenId < 0 || tokenId >= static_cast<int>(gpt2Tokens.size())));
  auto token = gpt2Tokens[tokenId];
  return token;
}

std::string GetPrintableToken(std::string token) {
  std::string decoded;
  decoded.reserve(token.size());
  const std::string gpt2Space = "\u0120";   // "Ġ"
  const std::string gpt2Newline = "\u010A"; // "Ċ"
  for (size_t i = 0; i < token.size();) {
    if (token.compare(i, gpt2Space.size(), gpt2Space) == 0) {
      decoded += ' ';
      i += gpt2Space.size();
    } else if (token.compare(i, gpt2Newline.size(), gpt2Newline) == 0) {
      decoded += '\n';
      i += gpt2Newline.size();
    } else {
      decoded += token[i++];
    }
  }
  return decoded;
}

int GetTokenIdFromToken(std::string token) {
  auto it = gpt2TokenToTokenId.find(token);
  assert(it != gpt2TokenToTokenId.end());
  return it->second;
}

// Caller owns returned pointer (cudaFree). Size is always N_EMBD.
double *GetEmbeddingFromTokenId(int tokenId, const double *wteWeights,
                                const double *wpeWeights, const int position) {
  auto add = [] __host__ __device__(const double &x, const double &y) -> double {
    return x + y;
  };
  return CUDA::vectorCombine(wteWeights + tokenId * N_EMBD,
                             wpeWeights + position * N_EMBD, N_EMBD, add);
}



std::vector<int> Tokenize(std::string input) {

  // regex based chunks, then chunks split into chars
  std::regex rg(" ?[A-Za-z]+| ?[0-9]+| ?[^ A-Za-z0-9]+|\\s+");
  std::sregex_iterator it(input.begin(), input.end(), rg);
  std::sregex_iterator end;

  std::vector<std::vector<std::string>> preChunksSplit;
  

  for (; it != end; it++) {
    std::smatch match = *it;
    std::string chunkStr = match.str();

    preChunksSplit.emplace_back();

    size_t startIdx = 0;
    if (!chunkStr.empty() && chunkStr[0] == ' ') {
      preChunksSplit.back().push_back("\u0120"); // Ġ, as one unit
      startIdx = 1;
    }
    for (size_t i = startIdx; i < chunkStr.size(); i++) {
      preChunksSplit.back().push_back(std::string(1, chunkStr[i]));
    }
  }

  // run bpe per chunk
  for (auto &chunk : preChunksSplit) {

    while (true) {

      int len = chunk.size();
      int mergePosition = -1;
      int mergePriority = 1e9;
      for (int i = 0; i < len - 1; i++) {

        int priority = 1e9;
        if (gpt2Merges.contains(std::make_pair(chunk[i], chunk[i + 1]))) {
          priority = gpt2Merges[std::make_pair(chunk[i], chunk[i + 1])];
        }

        if (priority < mergePriority) {
          mergePriority = priority;
          mergePosition = i;
        }
      }

      if (mergePosition == -1)
        break;
      else {
        chunk[mergePosition] += chunk[mergePosition + 1];
        chunk.erase(chunk.begin() + mergePosition + 1);
      }
    }
  }

  std::vector<int> result;

  for (const auto &i : preChunksSplit) {
    for (const auto &token : i) {
      result.push_back(GetTokenIdFromToken(token));
    }
  }

  return result;
}


// Caller owns returned pointer (cudaFree). *countTokens set to token count.
double *GenerateEmbeddings(std::string input, const double *wteWeights,
                           const double *wpeWeights, size_t &countTokens) {

  auto tokenIds = Tokenize(input);
  countTokens = tokenIds.size();

  double *result;
  cudaMalloc(&result, countTokens * N_EMBD * sizeof(double));

  auto add = [] __host__ __device__(const double &x, const double &y) -> double {
    return x + y;
  };

  for (size_t i = 0; i < countTokens; i++) {
    double *embedding = CUDA::vectorCombine(
        wteWeights + tokenIds[i] * N_EMBD,
        wpeWeights + static_cast<int>(i) * N_EMBD, N_EMBD, add);

    cudaMemcpy(result + i * N_EMBD, embedding, N_EMBD * sizeof(double),
               cudaMemcpyDeviceToDevice);
    cudaFree(embedding);
  }

  return result;
}



void LoadVocab() {
  using json = nlohmann::json;

  std::ifstream f("../weights/tokenizer/tokenizer.json");
  assert(f.is_open() && "failed to open tokenizer.json");

  const json data = json::parse(f);

  gpt2Tokens.assign(VOCAB_SIZE, "");
  gpt2TokenToTokenId.clear();

  for (const auto &[token, tokenIdJson] : data["model"]["vocab"].items()) {
    const int tokenId = tokenIdJson.get<int>();
    assert(tokenId >= 0 && tokenId < VOCAB_SIZE);
    gpt2Tokens[tokenId] = token;
    gpt2TokenToTokenId[token] = tokenId;
  }

  if (data.contains("added_tokens")) {
    for (const auto &added : data["added_tokens"]) {
      const int tokenId = added["id"].get<int>();
      const std::string content = added["content"].get<std::string>();
      if (tokenId >= static_cast<int>(gpt2Tokens.size()))
        gpt2Tokens.resize(tokenId + 1);
      gpt2Tokens[tokenId] = content;
      gpt2TokenToTokenId[content] = tokenId;
    }
  }

  ParseMerges(data["model"]["merges"]);
}

int main() {

  std::cout << "loading weights ... " << std::endl;

  auto weights = LoadWeights();

  std::cout << " weights loaded ... " << std::endl;

  std::cout << "loading vocab ... " << std::endl;

  LoadVocab();

  std::cout << "vocab loaded... " << std::endl;

  while (true) {

    std::string input;
    std::cout << "\nEnter text: ";
    std::getline(std::cin, input);
    std::cout << "You entered: " << input << std::endl;

    size_t numTokens = 0;
    double *embeddings =
        GenerateEmbeddings(input, weights.wteWeights, weights.wpeWeights,
                           numTokens);

    int tokenToGenerate = 20;
    while (tokenToGenerate--) {
      auto nextToken = GPT(weights, static_cast<int>(numTokens), embeddings);
      if (tokenToGenerate) {
        double *nextTokenEmbedding = GetEmbeddingFromTokenId(
            nextToken, weights.wteWeights, weights.wpeWeights,
            static_cast<int>(numTokens));

        double *grown;
        cudaMalloc(&grown, (numTokens + 1) * N_EMBD * sizeof(double));
        cudaMemcpy(grown, embeddings, numTokens * N_EMBD * sizeof(double),
                   cudaMemcpyDeviceToDevice);
        cudaMemcpy(grown + numTokens * N_EMBD, nextTokenEmbedding,
                   N_EMBD * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaFree(nextTokenEmbedding);
        cudaFree(embeddings);
        embeddings = grown;
        numTokens++;
      }

      std::cout << GetPrintableToken(GetTokenFromTokenId(nextToken));
      std::cout.flush();
    }

    cudaFree(embeddings);
  }
}
