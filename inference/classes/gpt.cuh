#pragma once

#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../constants.hh"
#include "../include/json.hpp"
#include "../kernels/layerNorm.cuh"
#include "../kernels/matmul.cuh"
#include "../kernels/softmax.cuh"
#include "../kernels/vectorCombine.cuh"
#include "cudaBuffer.cuh"
#include "transformer.cuh"
#include "weights.cuh"
#include <algorithm>

class Gpt2Tokenizer {

  std::vector<std::string> vocab;
  std::unordered_map<std::string, int> gpt2TokenToTokenId;
  std::map<std::pair<std::string, std::string>, int> merges;

  CudaBuffer<double> wte;
  CudaBuffer<double> wpe;

  int GetTokenIdFromToken(std::string token) {
    auto it = gpt2TokenToTokenId.find(token);
    assert(it != gpt2TokenToTokenId.end());
    return it->second;
  }

  void ParseMerges(const nlohmann::json &mergesJson) {
    this->merges.clear();
    int priority = 0;
    for (const auto &merge : mergesJson) {
      this->merges[{merge[0].get<std::string>(), merge[1].get<std::string>()}] =
          priority++;
    }
  }

public:
  Gpt2Tokenizer() {

    std::cout << "loading vocab, and some more weights ... " << std::endl;

    using json = nlohmann::json;

    std::ifstream f("../weights/tokenizer/tokenizer.json");
    assert(f.is_open() && "failed to open tokenizer.json");

    const json data = json::parse(f);

    this->vocab.assign(VOCAB_SIZE, "");
    gpt2TokenToTokenId.clear();

    for (const auto &[token, tokenIdJson] : data["model"]["vocab"].items()) {
      const int tokenId = tokenIdJson.get<int>();
      assert(tokenId >= 0 && tokenId < VOCAB_SIZE);
      this->vocab[tokenId] = token;
      gpt2TokenToTokenId[token] = tokenId;
    }

    if (data.contains("added_tokens")) {
      for (const auto &added : data["added_tokens"]) {
        const int tokenId = added["id"].get<int>();
        const std::string content = added["content"].get<std::string>();
        if (tokenId >= static_cast<int>(this->vocab.size()))
          this->vocab.resize(tokenId + 1);
        this->vocab[tokenId] = content;
        gpt2TokenToTokenId[content] = tokenId;
      }
    }

    ParseMerges(data["model"]["merges"]);

    constexpr size_t D = N_EMBD;
    constexpr size_t WPE_SIZE = static_cast<size_t>(N_CTX) * D;
    constexpr size_t WTE_SIZE = static_cast<size_t>(VOCAB_SIZE) * D;

    double *wpeHost = new double[WPE_SIZE];
    double *wteHost = new double[WTE_SIZE];

    {
      std::ifstream wpeWeightsFile("../weights/transformer.wpe.weight.txt");
      InputVectorFromFilePtr(wpeHost, wpeWeightsFile, WPE_SIZE);
      std::ifstream wteWeightsFile("../weights/transformer.wte.weight.txt");
      InputVectorFromFilePtr(wteHost, wteWeightsFile, WTE_SIZE);
    }

    wpe = CudaBuffer<double>(wpeHost, WPE_SIZE);
    wte = CudaBuffer<double>(wteHost, WTE_SIZE);

    delete[] wpeHost;
    delete[] wteHost;
  }

  std::vector<int> encode(std::string inputStr) {

    // regex based chunks, then chunks split into chars
    std::regex rg(" ?[A-Za-z]+| ?[0-9]+| ?[^ A-Za-z0-9]+|\\s+");
    std::sregex_iterator it(inputStr.begin(), inputStr.end(), rg);
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
          if (this->merges.contains(std::make_pair(chunk[i], chunk[i + 1]))) {
            priority = this->merges[std::make_pair(chunk[i], chunk[i + 1])];
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

  std::string decode(int id) {
    assert(id >= 0 && id < static_cast<int>(vocab.size()));
    const std::string &token = vocab[id];

    std::string decoded;
    decoded.reserve(token.size());
    const std::string gpt2Space = "\u0120";   // Ġ
    const std::string gpt2Newline = "\u010A"; // Ċ
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

  CudaBuffer<double> generateEmbeddings(std::string input,
                                        size_t &countTokens) {

    auto tokenIds = this->encode(input);
    countTokens = tokenIds.size();

    CudaBuffer<double> result(countTokens * N_EMBD);

    auto add = [] __host__ __device__(const double &x, const double &y)
        -> double { return x + y; };

    for (size_t i = 0; i < countTokens; i++) {
      auto tokenEmb = wte.slice(tokenIds[i] * N_EMBD, N_EMBD);
      auto posEmb = wpe.slice(i * N_EMBD, N_EMBD);
      auto embedding = CUDA::vectorCombine(tokenEmb, posEmb, add);
      result.copyFrom(embedding, i * N_EMBD);
    }

    return result;
  }

  CudaBuffer<double> embeddingFromTokenId(int tokenId, int position) {
    auto add = [] __host__ __device__(const double &x, const double &y)
        -> double { return x + y; };
    auto tokenEmb = wte.slice(static_cast<size_t>(tokenId) * N_EMBD, N_EMBD);
    auto posEmb = wpe.slice(static_cast<size_t>(position) * N_EMBD, N_EMBD);
    return CUDA::vectorCombine(tokenEmb, posEmb, add);
  }
};

class Gpt2 {

  Gpt2Tokenizer tokenizer;

  Transformer transformer;

  LnWeights ln_f;
  CudaBuffer<double> wte;

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

    double *finalLayerNormWeights = new double[FINAL_LN_SIZE];
    double *finalLayerNormBiases = new double[FINAL_LN_SIZE];
    double *wteHost = new double[WTE_SIZE];

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

    ln_f.gamma = CudaBuffer<double>(finalLayerNormWeights, FINAL_LN_SIZE);
    ln_f.beta = CudaBuffer<double>(finalLayerNormBiases, FINAL_LN_SIZE);
    wte = CudaBuffer<double>(wteHost, WTE_SIZE);

    delete[] finalLayerNormWeights;
    delete[] finalLayerNormBiases;
    delete[] wteHost;
  }

  void generate(std::string input, int n) {

    transformer.resetKvCache();

    size_t numTokens = 0;
    auto promptEmbeddings =
        this->tokenizer.generateEmbeddings(input, numTokens);

    CudaBuffer<double> embeddings;
    embeddings.reserve(static_cast<size_t>(N_CTX) * N_EMBD);
    embeddings.resize(numTokens * N_EMBD);
    embeddings.copyFrom(promptEmbeddings);
    promptEmbeddings = CudaBuffer<double>();

    int tokenToGenerate = n;
    while (tokenToGenerate--) {

      int nextToken = this->generateLogic(embeddings);

      if (tokenToGenerate) {
        auto nextTokenEmbedding = tokenizer.embeddingFromTokenId(
            nextToken, static_cast<int>(numTokens));

        embeddings.resize((numTokens + 1) * N_EMBD);
        embeddings.copyFrom(nextTokenEmbedding, numTokens * N_EMBD);
        numTokens++;
      }

      std::cout << tokenizer.decode(nextToken);
      std::cout.flush();
    }
  }
};
