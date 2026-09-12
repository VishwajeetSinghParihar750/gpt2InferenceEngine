#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "buffer.cuh"
#include "constants.hh"
#include "ops.cuh"
#include "tokenizer.cuh"

// Free function so CUDA extended lambdas can live in private methods.
inline __host__ __device__ double addDoubles(const double &x, const double &y) {
  return x + y;
}

struct AttnWeights {
  CudaBuffer<double> q, k, v, o, qb, kb, vb, ob;
};
struct MlpWeights {
  CudaBuffer<double> fc, proj, fcb, projb;
};
struct LnWeights {
  CudaBuffer<double> gamma, beta;
};
struct BlockWeights {
  LnWeights ln1, ln2;
  AttnWeights attn;
  MlpWeights mlp;
};

inline void InputVectorFromFilePtr(double *ptr, std::ifstream &stream,
                                   size_t count) {
  double inp;
  size_t idx = 0;
  while (idx < count && stream >> inp) {
    ptr[idx++] = inp;
  }
}

struct Transformer {
  std::vector<BlockWeights> weights;
  // kvCache: per layer, head-major [head][pos][headDim]
  CudaBuffer<double> kCache[N_LAYER], vCache[N_LAYER];
  int kvCacheLen = 0;

  CudaBuffer<double> Attention(const CudaBuffer<double> &embeddings,
                               int blockIdx, int headIdx, int totalTokens) {
    constexpr int embedDim = N_EMBD;
    constexpr int headDim = N_EMBD / N_HEAD;
    const auto &attn = weights[blockIdx].attn;
    const int weightBlockSize = embedDim * headDim;

    const int cached = kvCacheLen;
    const int newCount = totalTokens - cached;
    assert(newCount > 0);
    assert(static_cast<size_t>(newCount) * embedDim == embeddings.n);

    auto qWeights = attn.q.slice(static_cast<size_t>(headIdx) * weightBlockSize,
                                 weightBlockSize);
    auto kWeights = attn.k.slice(static_cast<size_t>(headIdx) * weightBlockSize,
                                 weightBlockSize);
    auto vWeights = attn.v.slice(static_cast<size_t>(headIdx) * weightBlockSize,
                                 weightBlockSize);
    auto qBiases =
        attn.qb.slice(static_cast<size_t>(headIdx) * headDim, headDim);
    auto kBiases =
        attn.kb.slice(static_cast<size_t>(headIdx) * headDim, headDim);
    auto vBiases =
        attn.vb.slice(static_cast<size_t>(headIdx) * headDim, headDim);

    auto project = [&](const CudaBuffer<double> &w,
                       const CudaBuffer<double> &biases) -> CudaBuffer<double> {
      auto wT = CUDA::Transpose(w, headDim, embedDim);
      auto proj = CUDA::MatMul<double>(embeddings, wT, newCount, embedDim,
                                       embedDim, headDim);
      CUDA::addRowBias(proj, biases, newCount, headDim);
      return proj;
    };

    auto qProjections = project(qWeights, qBiases);
    auto kProjections = project(kWeights, kBiases);
    auto vProjections = project(vWeights, vBiases);

    const size_t headBase =
        static_cast<size_t>(headIdx) * N_CTX * headDim;
    kCache[blockIdx].copyFrom(kProjections,
                              headBase + static_cast<size_t>(cached) * headDim);
    vCache[blockIdx].copyFrom(vProjections,
                              headBase + static_cast<size_t>(cached) * headDim);

    auto kAll = kCache[blockIdx].slice(
        headBase, static_cast<size_t>(totalTokens) * headDim);
    auto vAll = vCache[blockIdx].slice(
        headBase, static_cast<size_t>(totalTokens) * headDim);

    auto kTranspose = CUDA::Transpose(kAll, totalTokens, headDim);

    auto qkTranspose = CUDA::MatMul<double>(
        qProjections, kTranspose, newCount, headDim, headDim, totalTokens);

    double dimensionsRoot = sqrt(headDim);

    CUDA::vectorMapInPlace(qkTranspose,
                           [dimensionsRoot] __device__ __host__(double &v) {
                             v /= dimensionsRoot;
                             return v;
                           });

    CUDA::causalMask(qkTranspose, newCount, totalTokens, cached);
    CUDA::SoftMaxRows(qkTranspose, newCount, totalTokens);

    return CUDA::MatMul<double>(qkTranspose, vAll, newCount, totalTokens,
                                totalTokens, headDim);
  }

  CudaBuffer<double> MultiHeadAttention(const CudaBuffer<double> &embeddings,
                                        int blockIdx, int totalTokens) {
    constexpr int embedDim = N_EMBD;
    constexpr int headDim = N_EMBD / N_HEAD;
    const auto &attn = weights[blockIdx].attn;
    const int newCount = totalTokens - kvCacheLen;

    CudaBuffer<double> packed(static_cast<size_t>(newCount) * embedDim);
    packed.zero();

    for (int h = 0; h < N_HEAD; h++) {
      auto curResult = Attention(embeddings, blockIdx, h, totalTokens);
      CUDA::packHead(curResult, packed, newCount, headDim, embedDim, h);
    }

    auto projectionResult = CUDA::MatMul<double>(packed, attn.o, newCount,
                                                 embedDim, embedDim, embedDim);

    CUDA::addRowBias(projectionResult, attn.ob, newCount, embedDim);
    return projectionResult;
  }

  CudaBuffer<double> MLP(const CudaBuffer<double> &embeddings, int blockIdx,
                         int NUM_TOKENS) {
    constexpr int dimensions = N_EMBD;
    const auto &mlp = weights[blockIdx].mlp;
    CudaBuffer<double> result(static_cast<size_t>(NUM_TOKENS) * dimensions);

    for (int i = 0; i < NUM_TOKENS; i++) {
      auto tokenEmbedding =
          embeddings.slice(static_cast<size_t>(i) * dimensions, dimensions);
      auto hiddenOut =
          CUDA::ForwardPass(mlp.fc, mlp.fcb, tokenEmbedding, true);
      auto out = CUDA::ForwardPass(mlp.proj, mlp.projb, hiddenOut);

      result.copyFrom(out, static_cast<size_t>(i) * dimensions);
    }

    return result;
  }

  // embeddings: only the new tokens for this forward (length newCount)
  CudaBuffer<double> loopBlock(int blockIdx,
                               const CudaBuffer<double> &embeddings,
                               int totalTokens) {
    constexpr int D = N_EMBD;
    const auto &input = weights[blockIdx];
    const int newCount = totalTokens - kvCacheLen;

    auto add = [] __host__ __device__(const double &x, const double &y)
        -> double { return x + y; };

    CudaBuffer<double> layerNormedEmbeddings(static_cast<size_t>(newCount) * D);

    for (int i = 0; i < newCount; i++) {
      auto token = embeddings.slice(static_cast<size_t>(i) * D, D);
      auto normed =
          CUDA::LayerNorm(token, input.ln1.gamma, input.ln1.beta, EPSILON);
      layerNormedEmbeddings.copyFrom(normed, static_cast<size_t>(i) * D);
    }

    auto attentionResult =
        MultiHeadAttention(layerNormedEmbeddings, blockIdx, totalTokens);

    layerNormedEmbeddings = CudaBuffer<double>();

    CudaBuffer<double> normedForMLP(static_cast<size_t>(newCount) * D);

    for (int i = 0; i < newCount; i++) {
      auto embRow = embeddings.slice(static_cast<size_t>(i) * D, D);
      auto attnRow = attentionResult.slice(static_cast<size_t>(i) * D, D);
      auto withResidual = CUDA::vectorCombine(embRow, attnRow, add);

      attentionResult.copyFrom(withResidual, static_cast<size_t>(i) * D);

      auto residualView = attentionResult.slice(static_cast<size_t>(i) * D, D);
      auto normed = CUDA::LayerNorm(residualView, input.ln2.gamma,
                                    input.ln2.beta, EPSILON);

      normedForMLP.copyFrom(normed, static_cast<size_t>(i) * D);
    }

    auto MLPResult = MLP(normedForMLP, blockIdx, newCount);

    normedForMLP = CudaBuffer<double>();

    for (int i = 0; i < newCount; i++) {
      auto attnRow = attentionResult.slice(static_cast<size_t>(i) * D, D);
      auto mlpRow = MLPResult.slice(static_cast<size_t>(i) * D, D);
      CUDA::vectorCombineInto(attnRow, mlpRow, mlpRow, add);
    }

    return MLPResult;
  }

public:
  Transformer() : weights(N_LAYER) {
    for (auto &i : kCache)
      i.resize(N_CTX * N_EMBD);
    for (auto &i : vCache)
      i.resize(N_CTX * N_EMBD);

    std::cout << "Loading weights ... " << std::endl;

    constexpr size_t D = N_EMBD;
    constexpr size_t HIDDEN = MLP_HIDDEN;
    constexpr size_t ATTN_WEIGHT_SIZE = D * D;
    constexpr size_t ATTN_BIAS_SIZE = D;
    constexpr size_t L1_WEIGHT_SIZE = HIDDEN * D;
    constexpr size_t L2_WEIGHT_SIZE = D * HIDDEN;

    auto copyToDevice = [](const double *hostPtr,
                           size_t count) -> CudaBuffer<double> {
      return CudaBuffer<double>(hostPtr, count);
    };

    // Host buffer -> device -> transpose on GPU. Frees hostPtr.
    auto transposeHostToDevice = [&](double *hostPtr, size_t count, int n,
                                     int m) -> CudaBuffer<double> {
      CudaBuffer<double> dev(hostPtr, count);
      delete[] hostPtr;
      return CUDA::Transpose(dev, n, m);
    };

    for (size_t i = 0; i < static_cast<size_t>(N_LAYER); i++) {
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

      {
        std::ifstream ln1Weights("../weights/transformer.h." +
                                 std::to_string(i) + ".ln_1.weight.txt");
        InputVectorFromFilePtr(gammaAttention, ln1Weights, D);
        std::ifstream ln1Biases("../weights/transformer.h." +
                                std::to_string(i) + ".ln_1.bias.txt");
        InputVectorFromFilePtr(betaAttention, ln1Biases, D);
      }

      {
        std::ifstream ln2Weights("../weights/transformer.h." +
                                 std::to_string(i) + ".ln_2.weight.txt");
        InputVectorFromFilePtr(gammaMLP, ln2Weights, D);
        std::ifstream ln2Biases("../weights/transformer.h." +
                                std::to_string(i) + ".ln_2.bias.txt");
        InputVectorFromFilePtr(betaMLP, ln2Biases, D);
      }

      {
        std::ifstream mlpL1Weights("../weights/transformer.h." +
                                   std::to_string(i) + ".mlp.c_fc.weight.txt");
        InputVectorFromFilePtr(l1Weights, mlpL1Weights, L1_WEIGHT_SIZE);
        std::ifstream mlpL1Biases("../weights/transformer.h." +
                                  std::to_string(i) + ".mlp.c_fc.bias.txt");
        InputVectorFromFilePtr(l1Biases, mlpL1Biases, HIDDEN);
      }

      {
        std::ifstream mlpL2Weights("../weights/transformer.h." +
                                   std::to_string(i) +
                                   ".mlp.c_proj.weight.txt");
        InputVectorFromFilePtr(l2Weights, mlpL2Weights, L2_WEIGHT_SIZE);
        std::ifstream mlpL2Biases("../weights/transformer.h." +
                                  std::to_string(i) + ".mlp.c_proj.bias.txt");
        InputVectorFromFilePtr(l2Biases, mlpL2Biases, D);
      }

      auto l1Weights_device =
          transposeHostToDevice(l1Weights, L1_WEIGHT_SIZE, D, HIDDEN);
      l1Weights = nullptr;
      auto l2Weights_device =
          transposeHostToDevice(l2Weights, L2_WEIGHT_SIZE, HIDDEN, D);
      l2Weights = nullptr;

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

        std::ifstream qkvBiases("../weights/transformer.h." +
                                std::to_string(i) + ".attn.c_attn.bias.txt");
        InputVectorFromFilePtr(qBiases, qkvBiases, D);
        InputVectorFromFilePtr(kBiases, qkvBiases, D);
        InputVectorFromFilePtr(vBiases, qkvBiases, D);
      }

      auto qWeights_device =
          transposeHostToDevice(qWeights, ATTN_WEIGHT_SIZE, D, D);
      qWeights = nullptr;
      auto kWeights_device =
          transposeHostToDevice(kWeights, ATTN_WEIGHT_SIZE, D, D);
      kWeights = nullptr;
      auto vWeights_device =
          transposeHostToDevice(vWeights, ATTN_WEIGHT_SIZE, D, D);
      vWeights = nullptr;

      {
        std::ifstream attnOutputProjWeights("../weights/transformer.h." +
                                            std::to_string(i) +
                                            ".attn.c_proj.weight.txt");
        InputVectorFromFilePtr(oWeights, attnOutputProjWeights,
                               ATTN_WEIGHT_SIZE);

        std::ifstream attnOutputProjBiases("../weights/transformer.h." +
                                           std::to_string(i) +
                                           ".attn.c_proj.bias.txt");
        InputVectorFromFilePtr(oBiases, attnOutputProjBiases, ATTN_BIAS_SIZE);
      }

      BlockWeights &block = weights[i];
      block.ln1.gamma = copyToDevice(gammaAttention, D);
      block.ln1.beta = copyToDevice(betaAttention, D);
      block.ln2.gamma = copyToDevice(gammaMLP, D);
      block.ln2.beta = copyToDevice(betaMLP, D);

      block.attn.q = std::move(qWeights_device);
      block.attn.k = std::move(kWeights_device);
      block.attn.v = std::move(vWeights_device);
      block.attn.o = copyToDevice(oWeights, ATTN_WEIGHT_SIZE);
      block.attn.qb = copyToDevice(qBiases, ATTN_BIAS_SIZE);
      block.attn.kb = copyToDevice(kBiases, ATTN_BIAS_SIZE);
      block.attn.vb = copyToDevice(vBiases, ATTN_BIAS_SIZE);
      block.attn.ob = copyToDevice(oBiases, ATTN_BIAS_SIZE);

      block.mlp.fc = std::move(l1Weights_device);
      block.mlp.fcb = copyToDevice(l1Biases, HIDDEN);
      block.mlp.proj = std::move(l2Weights_device);
      block.mlp.projb = copyToDevice(l2Biases, D);

      delete[] gammaAttention;
      delete[] betaAttention;
      delete[] gammaMLP;
      delete[] betaMLP;
      delete[] l1Biases;
      delete[] l2Biases;
      delete[] qBiases;
      delete[] kBiases;
      delete[] vBiases;
      delete[] oWeights;
      delete[] oBiases;
    }

    std::cout << "weights loaded ... " << std::endl;
  }

  CudaBuffer<double> loop(const CudaBuffer<double> &embeddings,
                          int NUM_TOKENS) {
    const int cached = kvCacheLen;
    const int newCount = NUM_TOKENS - cached;
    assert(newCount > 0);
    assert(NUM_TOKENS <= N_CTX);

    auto newEmbeddings = embeddings.slice(
        static_cast<size_t>(cached) * N_EMBD,
        static_cast<size_t>(newCount) * N_EMBD);

    auto result = loopBlock(0, newEmbeddings, NUM_TOKENS);
    for (int i = 1; i < N_LAYER; i++) {
      result = loopBlock(i, result, NUM_TOKENS);
    }

    kvCacheLen = NUM_TOKENS;
    return result;
  }

  void resetKvCache() { kvCacheLen = 0; }
};

class Gpt2 {
  Gpt2Tokenizer tokenizer;
  Transformer transformer;

  LnWeights ln_f;
  CudaBuffer<double> wte;
  CudaBuffer<double> wpe;

  CudaBuffer<double> generateEmbeddings(const std::vector<int> &tokenIds) {
    const size_t countTokens = tokenIds.size();
    CudaBuffer<double> result(countTokens * N_EMBD);

    for (size_t i = 0; i < countTokens; i++) {
      auto tokenEmb = wte.slice(tokenIds[i] * N_EMBD, N_EMBD);
      auto posEmb = wpe.slice(i * N_EMBD, N_EMBD);
      auto embedding = CUDA::vectorCombine(tokenEmb, posEmb, addDoubles);
      result.copyFrom(embedding, i * N_EMBD);
    }

    return result;
  }

  CudaBuffer<double> embeddingFromTokenId(int tokenId, int position) {
    auto tokenEmb = wte.slice(static_cast<size_t>(tokenId) * N_EMBD, N_EMBD);
    auto posEmb = wpe.slice(static_cast<size_t>(position) * N_EMBD, N_EMBD);
    return CUDA::vectorCombine(tokenEmb, posEmb, addDoubles);
  }

  int generateLogic(const CudaBuffer<double> &embeddings) {
    int numEmbeddings = static_cast<int>(embeddings.n / N_EMBD);

    auto result = transformer.loop(embeddings, numEmbeddings);

    int resultTokens = static_cast<int>(result.n / N_EMBD);
    auto lastToken =
        result.slice(static_cast<size_t>(resultTokens - 1) * N_EMBD, N_EMBD);

    auto layerNormedResult =
        CUDA::LayerNorm(lastToken, ln_f.gamma, ln_f.beta, EPSILON);

    result = CudaBuffer<double>();

    // logits = wte @ lastToken  (VOCAB_SIZE x 1)
    auto logits = CUDA::MatMul<double>(wte, layerNormedResult, VOCAB_SIZE,
                                       N_EMBD, N_EMBD, 1);
    layerNormedResult = CudaBuffer<double>();

    CUDA::SoftMaxInPlace(logits);

    std::vector<double> hostProbs(VOCAB_SIZE);
    logits.copyToHost(hostProbs.data());

    int maxProbTokenId =
        static_cast<int>(std::max_element(hostProbs.begin(), hostProbs.end()) -
                         hostProbs.begin());

    return maxProbTokenId;
  }

public:
  Gpt2() {
    constexpr size_t D = N_EMBD;
    constexpr size_t FINAL_LN_SIZE = D;
    constexpr size_t WTE_SIZE = static_cast<size_t>(VOCAB_SIZE) * D;
    constexpr size_t WPE_SIZE = static_cast<size_t>(N_CTX) * D;

    double *finalLayerNormWeights = new double[FINAL_LN_SIZE];
    double *finalLayerNormBiases = new double[FINAL_LN_SIZE];
    double *wteHost = new double[WTE_SIZE];
    double *wpeHost = new double[WPE_SIZE];

    {
      std::ifstream lnWeights("../weights/transformer.ln_f.weight.txt");
      InputVectorFromFilePtr(finalLayerNormWeights, lnWeights, FINAL_LN_SIZE);
    }
    {
      std::ifstream lnBiases("../weights/transformer.ln_f.bias.txt");
      InputVectorFromFilePtr(finalLayerNormBiases, lnBiases, FINAL_LN_SIZE);
    }
    {
      std::ifstream wteWeightsFile("../weights/transformer.wte.weight.txt");
      InputVectorFromFilePtr(wteHost, wteWeightsFile, WTE_SIZE);
    }
    {
      std::ifstream wpeWeightsFile("../weights/transformer.wpe.weight.txt");
      InputVectorFromFilePtr(wpeHost, wpeWeightsFile, WPE_SIZE);
    }

    ln_f.gamma = CudaBuffer<double>(finalLayerNormWeights, FINAL_LN_SIZE);
    ln_f.beta = CudaBuffer<double>(finalLayerNormBiases, FINAL_LN_SIZE);
    wte = CudaBuffer<double>(wteHost, WTE_SIZE);
    wpe = CudaBuffer<double>(wpeHost, WPE_SIZE);

    delete[] finalLayerNormWeights;
    delete[] finalLayerNormBiases;
    delete[] wteHost;
    delete[] wpeHost;
  }

  void generate(std::string input, int n) {
    transformer.resetKvCache();

    auto tokenIds = tokenizer.encode(input);
    size_t numTokens = tokenIds.size();
    auto promptEmbeddings = generateEmbeddings(tokenIds);

    CudaBuffer<double> embeddings;
    embeddings.reserve(static_cast<size_t>(N_CTX) * N_EMBD);
    embeddings.resize(numTokens * N_EMBD);
    embeddings.copyFrom(promptEmbeddings);
    promptEmbeddings = CudaBuffer<double>();

    int tokenToGenerate = n;
    while (tokenToGenerate--) {
      int nextToken = generateLogic(embeddings);

      if (tokenToGenerate) {
        auto nextTokenEmbedding =
            embeddingFromTokenId(nextToken, static_cast<int>(numTokens));

        embeddings.resize((numTokens + 1) * N_EMBD);
        embeddings.copyFrom(nextTokenEmbedding, numTokens * N_EMBD);
        numTokens++;
      }

      std::cout << tokenizer.decode(nextToken);
      std::cout.flush();
    }
  }
};
