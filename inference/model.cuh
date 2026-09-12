#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "buffer.cuh"
#include "constants.hh"
#include "ops.cuh"
#include "tokenizer.cuh"

// Functor (not a function pointer) so the kernel gets a real __device__ call.
struct AddFloats {
  __host__ __device__ float operator()(const float &x,
                                        const float &y) const {
    return x + y;
  }
};

struct AttnWeights {
  // qkv: (embed x 3*embed) HF layout — embeddings @ qkv, no runtime transpose
  CudaBuffer<float> qkv, qkvb, o, ob;
};
struct MlpWeights {
  CudaBuffer<float> fc, proj, fcb, projb;
};
struct LnWeights {
  CudaBuffer<float> gamma, beta;
};
struct BlockWeights {
  LnWeights ln1, ln2;
  AttnWeights attn;
  MlpWeights mlp;
};

inline void InputVectorFromFilePtr(float *ptr, std::ifstream &stream,
                                   size_t count) {
  float inp;
  size_t idx = 0;
  while (idx < count && stream >> inp) {
    ptr[idx++] = inp;
  }
}

struct Transformer {
  std::vector<BlockWeights> weights;
  // kvCache: per layer, head-major [head][pos][headDim]
  CudaBuffer<float> kCache[N_LAYER], vCache[N_LAYER];
  int kvCacheLen = 0;

  // qkv: (newCount x 3*embed) rows [Q|K|V]. Reads head h in-place via stride.
  CudaBuffer<float> Attention(const CudaBuffer<float> &qkv, int blockIdx,
                               int headIdx, int totalTokens) {
    constexpr int embedDim = N_EMBD;
    constexpr int headDim = N_EMBD / N_HEAD;
    constexpr int lda = 3 * embedDim;

    const int cached = kvCacheLen;
    const int newCount = totalTokens - cached;
    assert(newCount > 0);
    assert(qkv.n == static_cast<size_t>(newCount) * lda);

    const int qCol0 = headIdx * headDim;
    const int kCol0 = embedDim + headIdx * headDim;
    const int vCol0 = 2 * embedDim + headIdx * headDim;

    const size_t headBase =
        static_cast<size_t>(headIdx) * N_CTX * headDim;
    auto kNew = kCache[blockIdx].slice(
        headBase + static_cast<size_t>(cached) * headDim,
        static_cast<size_t>(newCount) * headDim);
    auto vNew = vCache[blockIdx].slice(
        headBase + static_cast<size_t>(cached) * headDim,
        static_cast<size_t>(newCount) * headDim);
    // KV cache stays contiguous; only this append copies the strided head.
    CUDA::copyStridedHead(qkv, kNew, newCount, headDim, lda, kCol0);
    CUDA::copyStridedHead(qkv, vNew, newCount, headDim, lda, vCol0);

    auto kAll = kCache[blockIdx].slice(
        headBase, static_cast<size_t>(totalTokens) * headDim);
    auto vAll = vCache[blockIdx].slice(
        headBase, static_cast<size_t>(totalTokens) * headDim);

    auto kTranspose = CUDA::Transpose(kAll, totalTokens, headDim);

    // Q head is not contiguous across tokens — MatMul uses lda/aCol0.
    auto qkTranspose = CUDA::MatMul<float>(
        qkv, kTranspose, newCount, headDim, headDim, totalTokens, lda, qCol0);

    float dimensionsRoot = sqrtf(static_cast<float>(headDim));

    CUDA::vectorMapInPlace(qkTranspose,
                           [dimensionsRoot] __device__ __host__(float &v) {
                             v /= dimensionsRoot;
                             return v;
                           });

    CUDA::causalMask(qkTranspose, newCount, totalTokens, cached);
    CUDA::SoftMaxRows(qkTranspose, newCount, totalTokens);

    return CUDA::MatMul<float>(qkTranspose, vAll, newCount, totalTokens,
                                totalTokens, headDim);
  }

  CudaBuffer<float> MultiHeadAttention(const CudaBuffer<float> &embeddings,
                                        int blockIdx, int totalTokens) {
    constexpr int embedDim = N_EMBD;
    constexpr int headDim = N_EMBD / N_HEAD;
    constexpr int qkvOut = 3 * embedDim;
    const auto &attn = weights[blockIdx].attn;
    const int newCount = totalTokens - kvCacheLen;
    assert(newCount > 0);
    assert(static_cast<size_t>(newCount) * embedDim == embeddings.n);

    auto qkv = CUDA::MatMul<float>(embeddings, attn.qkv, newCount, embedDim,
                                    embedDim, qkvOut);
    CUDA::addRowBias(qkv, attn.qkvb, newCount, qkvOut);

    CudaBuffer<float> packed(static_cast<size_t>(newCount) * embedDim);
    packed.zero();

    for (int h = 0; h < N_HEAD; h++) {
      auto curResult = Attention(qkv, blockIdx, h, totalTokens);
      CUDA::packHead(curResult, packed, newCount, headDim, embedDim, h);
    }

    auto projectionResult = CUDA::MatMul<float>(packed, attn.o, newCount,
                                                 embedDim, embedDim, embedDim);

    CUDA::addRowBias(projectionResult, attn.ob, newCount, embedDim);
    return projectionResult;
  }

  CudaBuffer<float> MLP(const CudaBuffer<float> &embeddings, int blockIdx,
                         int NUM_TOKENS) {
    constexpr int dimensions = N_EMBD;
    const auto &mlp = weights[blockIdx].mlp;
    CudaBuffer<float> result(static_cast<size_t>(NUM_TOKENS) * dimensions);

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
  CudaBuffer<float> loopBlock(int blockIdx,
                               const CudaBuffer<float> &embeddings,
                               int totalTokens) {
    constexpr int D = N_EMBD;
    const auto &input = weights[blockIdx];
    const int newCount = totalTokens - kvCacheLen;

    auto add = [] __host__ __device__(const float &x, const float &y)
        -> float { return x + y; };

    CudaBuffer<float> layerNormedEmbeddings(static_cast<size_t>(newCount) * D);

    for (int i = 0; i < newCount; i++) {
      auto token = embeddings.slice(static_cast<size_t>(i) * D, D);
      auto normed =
          CUDA::LayerNorm(token, input.ln1.gamma, input.ln1.beta, EPSILON);
      layerNormedEmbeddings.copyFrom(normed, static_cast<size_t>(i) * D);
    }

    auto attentionResult =
        MultiHeadAttention(layerNormedEmbeddings, blockIdx, totalTokens);

    layerNormedEmbeddings = CudaBuffer<float>();

    CudaBuffer<float> normedForMLP(static_cast<size_t>(newCount) * D);

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

    normedForMLP = CudaBuffer<float>();

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
    constexpr size_t QKV_WEIGHT_SIZE = D * (3 * D);
    constexpr size_t QKV_BIAS_SIZE = 3 * D;
    constexpr size_t L1_WEIGHT_SIZE = HIDDEN * D;
    constexpr size_t L2_WEIGHT_SIZE = D * HIDDEN;

    auto copyToDevice = [](const float *hostPtr,
                           size_t count) -> CudaBuffer<float> {
      return CudaBuffer<float>(hostPtr, count);
    };

    // Host buffer -> device -> transpose on GPU. Frees hostPtr.
    auto transposeHostToDevice = [&](float *hostPtr, size_t count, int n,
                                     int m) -> CudaBuffer<float> {
      CudaBuffer<float> dev(hostPtr, count);
      delete[] hostPtr;
      return CUDA::Transpose(dev, n, m);
    };

    for (size_t i = 0; i < static_cast<size_t>(N_LAYER); i++) {
      float *gammaAttention = new float[D];
      float *betaAttention = new float[D];
      float *gammaMLP = new float[D];
      float *betaMLP = new float[D];
      float *l1Weights = new float[L1_WEIGHT_SIZE];
      float *l1Biases = new float[HIDDEN];
      float *l2Weights = new float[L2_WEIGHT_SIZE];
      float *l2Biases = new float[D];
      float *qkvWeights = new float[QKV_WEIGHT_SIZE];
      float *qkvBiases = new float[QKV_BIAS_SIZE];
      float *oWeights = new float[ATTN_WEIGHT_SIZE];
      float *oBiases = new float[ATTN_BIAS_SIZE];

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
        InputVectorFromFilePtr(qkvWeights, qkvWeightsFile, QKV_WEIGHT_SIZE);

        std::ifstream qkvBiasesFile("../weights/transformer.h." +
                                    std::to_string(i) +
                                    ".attn.c_attn.bias.txt");
        InputVectorFromFilePtr(qkvBiases, qkvBiasesFile, QKV_BIAS_SIZE);
      }

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

      // Keep HF (in x 3*out) layout — used as embeddings @ qkv
      block.attn.qkv = copyToDevice(qkvWeights, QKV_WEIGHT_SIZE);
      block.attn.qkvb = copyToDevice(qkvBiases, QKV_BIAS_SIZE);
      block.attn.o = copyToDevice(oWeights, ATTN_WEIGHT_SIZE);
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
      delete[] qkvWeights;
      delete[] qkvBiases;
      delete[] oWeights;
      delete[] oBiases;
    }

    std::cout << "weights loaded ... " << std::endl;
  }

  CudaBuffer<float> loop(const CudaBuffer<float> &embeddings,
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
  std::mt19937 rng{std::random_device{}()};

  LnWeights ln_f;
  CudaBuffer<float> wte;
  CudaBuffer<float> wpe;

  CudaBuffer<float> generateEmbeddings(const std::vector<int> &tokenIds) {
    const size_t countTokens = tokenIds.size();
    CudaBuffer<float> result(countTokens * N_EMBD);

    for (size_t i = 0; i < countTokens; i++) {
      auto tokenEmb = wte.slice(tokenIds[i] * N_EMBD, N_EMBD);
      auto posEmb = wpe.slice(i * N_EMBD, N_EMBD);
      auto embedding = CUDA::vectorCombine(tokenEmb, posEmb, AddFloats{});
      result.copyFrom(embedding, i * N_EMBD);
    }

    return result;
  }

  CudaBuffer<float> embeddingFromTokenId(int tokenId, int position) {
    auto tokenEmb = wte.slice(static_cast<size_t>(tokenId) * N_EMBD, N_EMBD);
    auto posEmb = wpe.slice(static_cast<size_t>(position) * N_EMBD, N_EMBD);
    return CUDA::vectorCombine(tokenEmb, posEmb, AddFloats{});
  }

  int generateLogic(const CudaBuffer<float> &embeddings) {
    int numEmbeddings = static_cast<int>(embeddings.n / N_EMBD);

    auto result = transformer.loop(embeddings, numEmbeddings);

    int resultTokens = static_cast<int>(result.n / N_EMBD);
    auto lastToken =
        result.slice(static_cast<size_t>(resultTokens - 1) * N_EMBD, N_EMBD);

    auto layerNormedResult =
        CUDA::LayerNorm(lastToken, ln_f.gamma, ln_f.beta, EPSILON);

    result = CudaBuffer<float>();

    // logits = wte @ lastToken  (VOCAB_SIZE x 1)
    auto logits = CUDA::MatMul<float>(wte, layerNormedResult, VOCAB_SIZE,
                                       N_EMBD, N_EMBD, 1);
    layerNormedResult = CudaBuffer<float>();

    CUDA::SoftMaxInPlace(logits);

    std::vector<float> hostProbs(VOCAB_SIZE);
    logits.copyToHost(hostProbs.data());

    // Top-5 sampling: keep 5 highest probs, renormalize, sample.
    constexpr int kTop = 5;
    std::vector<int> idx(VOCAB_SIZE);
    for (int i = 0; i < VOCAB_SIZE; i++)
      idx[i] = i;
    std::partial_sort(
        idx.begin(), idx.begin() + kTop, idx.end(),
        [&](int a, int b) { return hostProbs[a] > hostProbs[b]; });

    float mass = 0.0f;
    for (int i = 0; i < kTop; i++)
      mass += hostProbs[idx[i]];

    std::uniform_real_distribution<float> dist(0.0f, mass);
    float r = dist(rng);
    float cum = 0.0f;
    for (int i = 0; i < kTop; i++) {
      cum += hostProbs[idx[i]];
      if (r <= cum)
        return idx[i];
    }
    return idx[kTop - 1];
  }

public:
  Gpt2() {
    constexpr size_t D = N_EMBD;
    constexpr size_t FINAL_LN_SIZE = D;
    constexpr size_t WTE_SIZE = static_cast<size_t>(VOCAB_SIZE) * D;
    constexpr size_t WPE_SIZE = static_cast<size_t>(N_CTX) * D;

    float *finalLayerNormWeights = new float[FINAL_LN_SIZE];
    float *finalLayerNormBiases = new float[FINAL_LN_SIZE];
    float *wteHost = new float[WTE_SIZE];
    float *wpeHost = new float[WPE_SIZE];

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

    ln_f.gamma = CudaBuffer<float>(finalLayerNormWeights, FINAL_LN_SIZE);
    ln_f.beta = CudaBuffer<float>(finalLayerNormBiases, FINAL_LN_SIZE);
    wte = CudaBuffer<float>(wteHost, WTE_SIZE);
    wpe = CudaBuffer<float>(wpeHost, WPE_SIZE);

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

    CudaBuffer<float> embeddings;
    embeddings.reserve(static_cast<size_t>(N_CTX) * N_EMBD);
    embeddings.resize(numTokens * N_EMBD);
    embeddings.copyFrom(promptEmbeddings);
    promptEmbeddings = CudaBuffer<float>();

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
