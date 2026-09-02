#include "constants.hh"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <regex>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "include/json.hpp"

double GeluNew(double x) {
  return .5 * x * (1 + tanh(sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x)));
}

std::vector<double>
LayerNorm(std::span<const double> originalEmbedding,
          std::span<const double> gamma, std::span<const double> beta,
          double epsilon) // originalEmbedding, gamma, beta: 1d
{
  int len = originalEmbedding.size();
  double variance = 0, mean = 0;

  for (auto i : originalEmbedding)
    mean += i;

  mean /= len;

  for (auto i : originalEmbedding)
    variance += (i - mean) * (i - mean);

  variance /= len;

  double modifiedStandardDeviation = sqrt(variance + epsilon);

  std::vector<double> output(len); // 1d
  for (int i = 0; i < len; i++) {
    double normalizedValue =
        ((originalEmbedding[i] - mean) / modifiedStandardDeviation);
    output[i] = normalizedValue * gamma[i] + beta[i];
  }

  return output;
}

std::vector<double> AddVectors(std::span<const double> a,
                               std::span<const double> b) // a, b: 1d
{
  std::vector<double> result(a.begin(), a.end());
  for (size_t i = 0; i < result.size(); i++)
    result[i] += b[i];
  return result;
}

std::vector<double> SoftMax(std::span<const double> input) // input: 1d
{
  int len = input.size();
  auto max = *std::max_element(input.begin(), input.end());

  std::vector<double> softmax(len); // 1d
  for (int i = 0; i < len; i++) {
    softmax[i] = exp(input[i] - max);
  }

  auto sum = std::accumulate(softmax.begin(), softmax.end(), 0.0);

  for (auto &i : softmax)
    i /= sum;

  return softmax;
}

double DotProduct(std::span<const double> a,
                  std::span<const double> b) // a, b: 1d
{
  int len = a.size();
  double output = 0;
  for (int i = 0; i < len; i++)
    output += a[i] * b[i];
  return output;
}

std::vector<double> Transpose(std::span<const double> a, int n,
                              int m) // a: 2d (n x m) flattened as 1d
{
  std::vector<double> result(m * n); // 2d (m x n) flattened as 1d
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      result[i * n + j] = a[j * m + i];
    }
  }
  return result;
}

std::vector<double> MatMul(std::span<const double> a, int aRows, int aCols,
                           // a: 2d (aRows x aCols) flattened as 1d
                           std::span<const double> b, int bRows,
                           int bCols) // b: 2d (bRows x bCols) flattened as 1d
{
  assert(aCols == bRows);

  std::vector<double> result(aRows * bCols,
                             0); // 2d (aRows x bCols) flattened as 1d

  for (int i = 0; i < aRows; i++) {
    for (int j = 0; j < bCols; j++) {
      for (int k = 0; k < bRows; k++) {
        result[i * bCols + j] += a[i * aCols + k] * b[k * bCols + j];
      }
    }
  }

  return result;
}

std::vector<double> Attention(
    std::span<const double> embeddings, int NUM_TOKENS,
    int embedDim, // embeddings: 2d (NUM_TOKENS x embedDim) flattened as 1d
    std::span<const double>
        qWeights, // 2d (headDim x embedDim) flattened as 1d, transposed
    std::span<const double>
        kWeights, // 2d (headDim x embedDim) flattened as 1d, transposed
    std::span<const double>
        vWeights, // 2d (headDim x embedDim) flattened as 1d, transposed
    std::span<const double> qBiases, // 1d, this head's slice (headDim)
    std::span<const double> kBiases, // 1d, this head's slice (headDim)
    std::span<const double> vBiases, // 1d, this head's slice (headDim)
    int headDim) {
  std::vector<double> qProjections(
      NUM_TOKENS * headDim); // 2d (NUM_TOKENS x headDim) flattened as 1d
  std::vector<double> kProjections(
      NUM_TOKENS * headDim); // 2d (NUM_TOKENS x headDim) flattened as 1d
  std::vector<double> vProjections(
      NUM_TOKENS * headDim); // 2d (NUM_TOKENS x headDim) flattened as 1d

  for (int t = 0; t < NUM_TOKENS; t++) {
    auto tokenEmb = embeddings.subspan(t * embedDim, embedDim); // 1d
    for (int d = 0; d < headDim; d++) {
      auto qRow =
          qWeights.subspan(d * embedDim, embedDim); // 1d, transposed weight row
      auto kRow =
          kWeights.subspan(d * embedDim, embedDim); // 1d, transposed weight row
      auto vRow =
          vWeights.subspan(d * embedDim, embedDim); // 1d, transposed weight row

      qProjections[t * headDim + d] = DotProduct(tokenEmb, qRow) + qBiases[d];
      kProjections[t * headDim + d] = DotProduct(tokenEmb, kRow) + kBiases[d];
      vProjections[t * headDim + d] = DotProduct(tokenEmb, vRow) + vBiases[d];
    }
  }

  auto kTranspose =
      Transpose(kProjections, NUM_TOKENS,
                headDim); // 2d (headDim x NUM_TOKENS) flattened as 1d

  auto qkTranspose =
      MatMul(qProjections, NUM_TOKENS, headDim, kTranspose, headDim,
             NUM_TOKENS); // 2d (NUM_TOKENS x NUM_TOKENS) flattened as 1d

  double dimensionsRoot = sqrt(headDim);
  for (auto &v : qkTranspose)
    v /= dimensionsRoot;

  // causal mask: token i cannot attend to future tokens j > i
  for (int i = 0; i < NUM_TOKENS; i++) {
    for (int j = i + 1; j < NUM_TOKENS; j++)
      qkTranspose[i * NUM_TOKENS + j] =
          -std::numeric_limits<double>::infinity();
  }

  for (int i = 0; i < NUM_TOKENS; i++) {
    auto row = std::span(qkTranspose).subspan(i * NUM_TOKENS, NUM_TOKENS);
    auto softRow = SoftMax(row); // 1d
    for (int j = 0; j < NUM_TOKENS; j++)
      qkTranspose[i * NUM_TOKENS + j] = softRow[j];
  }

  return MatMul(qkTranspose, NUM_TOKENS, NUM_TOKENS, vProjections, NUM_TOKENS,
                headDim); // 2d (NUM_TOKENS x headDim) flattened as 1d
}

std::vector<double> MultiHeadAttention(
    std::span<const double> embeddings, int NUM_TOKENS,
    int embedDim, // embeddings: 2d (NUM_TOKENS x embedDim) flattened as 1d
    std::span<const double>
        qWeights, // 3d (heads x headDim x embedDim) flattened as 1d, transposed
    std::span<const double>
        kWeights, // 3d (heads x headDim x embedDim) flattened as 1d, transposed
    std::span<const double>
        vWeights, // 3d (heads x headDim x embedDim) flattened as 1d, transposed
    std::span<const double>
        oWeights, // 2d (embedDim x embedDim) flattened as 1d
    int heads, int headDim,
    std::span<const double> oBiases, // embedDim 1d
    std::span<const double>
        qBiases, // 2d (heads x headDim) flattened as 1d, full
    std::span<const double>
        kBiases, // 2d (heads x headDim) flattened as 1d, full
    std::span<const double>
        vBiases) // 2d (heads x headDim) flattened as 1d, full
{
  std::vector<double> result(NUM_TOKENS * embedDim,
                             0); // 2d (NUM_TOKENS x embedDim) flattened as 1d

  int weightBlockSize = embedDim * headDim;

  for (int h = 0; h < heads; h++) {
    auto qHead = qWeights.subspan(h * weightBlockSize,
                                  weightBlockSize); // 2d (headDim x embedDim)
    auto kHead = kWeights.subspan(h * weightBlockSize,
                                  weightBlockSize); // 2d (headDim x embedDim)
    auto vHead = vWeights.subspan(h * weightBlockSize,
                                  weightBlockSize); // 2d (headDim x embedDim)

    auto qBiasHead =
        qBiases.subspan(h * headDim, headDim); // 1d, this head's bias slice
    auto kBiasHead =
        kBiases.subspan(h * headDim, headDim); // 1d, this head's bias slice
    auto vBiasHead =
        vBiases.subspan(h * headDim, headDim); // 1d, this head's bias slice

    auto curResult =
        Attention(embeddings, NUM_TOKENS, embedDim, qHead, kHead, vHead,
                  qBiasHead, kBiasHead, vBiasHead,
                  headDim); // 2d (NUM_TOKENS x headDim) flattened as 1d

    for (int j = 0; j < NUM_TOKENS; j++)
      for (int k = 0; k < headDim; k++)
        result[j * embedDim + h * headDim + k] = curResult[j * headDim + k];
  }

  auto projectionResult =
      MatMul(result, NUM_TOKENS, embedDim, oWeights, embedDim,
             embedDim); // 2d (NUM_TOKENS x embedDim) flattened as 1d

  for (int i = 0; i < NUM_TOKENS; i++) {
    for (int j = 0; j < embedDim; j++) {
      projectionResult[i * embedDim + j] += oBiases[j];
    }
  }

  return projectionResult;
}

std::vector<double>
ForwardPass(std::span<const double> weights, std::span<const double> biases,
            std::span<const double> inputs,
            bool gelu = false) // weights: 2d (neurons x inputSize) flattened as
                               // 1d, transposed convention, biases/inputs: 1d
{
  int countNeurons = biases.size();
  std::vector<double> output(biases.begin(), biases.end()); // 1d

  for (int i = 0; i < countNeurons; i++) {
    for (int j = 0; j < inputs.size(); j++)
      output[i] += weights[i * inputs.size() + j] * inputs[j];

    if (gelu)
      output[i] = GeluNew(output[i]);
  }
  return output;
}

std::vector<double>
MLP(std::span<const double> embeddings, int NUM_TOKENS,
    int dimensions, // embeddings: 2d (NUM_TOKENS x dimensions) flattened as 1d
    std::span<const double>
        l1Weights, // 2d (hidden x dimensions) flattened as 1d, transposed
    std::span<const double> l1Biases, // 1d
    std::span<const double>
        l2Weights, // 2d (dimensions x hidden) flattened as 1d, transposed
    std::span<const double> l2Biases) // 1d
{
  std::vector<double> result(
      NUM_TOKENS * dimensions); // 2d (NUM_TOKENS x dimensions) flattened as 1d

  for (int i = 0; i < NUM_TOKENS; i++) {
    auto tokenEmbedding = embeddings.subspan(i * dimensions, dimensions); // 1d
    auto hiddenOut =
        ForwardPass(l1Weights, l1Biases, tokenEmbedding, true); // 1d
    auto out = ForwardPass(l2Weights, l2Biases, hiddenOut);     // 1d
    for (int j = 0; j < dimensions; j++)
      result[i * dimensions + j] = out[j];
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

  const std::vector<double>
      qWeights; // 3D (heads x headDim x embedDim), flattened as 1D, transposed
  const std::vector<double>
      kWeights; // 3D (heads x headDim x embedDim), flattened as 1D, transposed
  const std::vector<double>
      vWeights; // 3D (heads x headDim x embedDim), flattened as 1D, transposed

  const std::vector<double> qBiases; // 2D (heads x headDim), flattened as 1D
  const std::vector<double> kBiases; // 2D (heads x headDim), flattened as 1D
  const std::vector<double> vBiases; // 2D (heads x headDim), flattened as 1D

  const std::vector<double>
      oWeights; // 2D (embedDim x embedDim), flattened as 1D
  const std::vector<double> oBiases;

  const std::vector<double>
      l1Weights; // 2D (hidden x embedDim), flattened as 1D, transposed
  const std::vector<double> l1Biases; // 1D

  const std::vector<double>
      l2Weights; // 2D (embedDim x hidden), flattened as 1D, transposed
  const std::vector<double> l2Biases; // 1D

  const std::vector<double> gammaAttention; // 1D
  const std::vector<double> gammaMLP;       // 1D

  const std::vector<double> betaAttention; // 1D
  const std::vector<double> betaMLP;       // 1D

  TransformerInput(std::vector<double> qWeights, std::vector<double> kWeights,
                   std::vector<double> vWeights, std::vector<double> qBiases,
                   std::vector<double> kBiases, std::vector<double> vBiases,
                   std::vector<double> oWeights, std::vector<double> oBiases,
                   std::vector<double> l1Weights, std::vector<double> l1Biases,
                   std::vector<double> l2Weights, std::vector<double> l2Biases,
                   std::vector<double> gammaAttention,
                   std::vector<double> gammaMLP,
                   std::vector<double> betaAttention,
                   std::vector<double> betaMLP)
      : qWeights(std::move(qWeights)), kWeights(std::move(kWeights)),
        vWeights(std::move(vWeights)), qBiases(std::move(qBiases)),
        kBiases(std::move(kBiases)), vBiases(std::move(vBiases)),
        oWeights(std::move(oWeights)), oBiases(std::move(oBiases)),
        l1Weights(std::move(l1Weights)), l1Biases(std::move(l1Biases)),
        l2Weights(std::move(l2Weights)), l2Biases(std::move(l2Biases)),
        gammaAttention(std::move(gammaAttention)),
        gammaMLP(std::move(gammaMLP)), betaAttention(std::move(betaAttention)),
        betaMLP(std::move(betaMLP)) {}
};

std::vector<double>
Transformer(const TransformerInput &input, const int NUM_TOKENS,
            const std::vector<double>
                &embeddings // 2D (NUM_TOKENS x embedDim), flattened as 1D

) {
  std::vector<double> layerNormedEmbeddings(
      NUM_TOKENS * TransformerInput::EMBEDDING_DIMENSION);
  // 2D (NUM_TOKENS x embedDim), flattened as 1D

  for (int i = 0; i < NUM_TOKENS; i++) {
    std::vector<double> tokenEmbedding(
        embeddings.begin() + i * TransformerInput::EMBEDDING_DIMENSION,
        embeddings.begin() + (i + 1) * TransformerInput::EMBEDDING_DIMENSION);
    // 1D

    auto normed = LayerNorm(tokenEmbedding,       // 1D
                            input.gammaAttention, // 1D
                            input.betaAttention,  // 1D
                            TransformerInput::EPSILON_ATTENTION);
    // 1D

    for (int j = 0; j < TransformerInput::EMBEDDING_DIMENSION; j++) {
      layerNormedEmbeddings[i * TransformerInput::EMBEDDING_DIMENSION + j] =
          normed[j];
    }
  }

  auto attentionResult = MultiHeadAttention(
      layerNormedEmbeddings, // 2D (NUM_TOKENS x embedDim), flattened as 1D

      NUM_TOKENS, TransformerInput::EMBEDDING_DIMENSION,

      input.qWeights, // 3D (heads x headDim x embedDim), flattened as 1D
      input.kWeights, // 3D (heads x headDim x embedDim), flattened as 1D
      input.vWeights, // 3D (heads x headDim x embedDim), flattened as 1D

      input.oWeights, // 2D (embedDim x embedDim), flattened as 1D

      TransformerInput::HEADS, TransformerInput::HEAD_DIMENSION,

      input.oBiases, // oBiases
      input.qBiases, // 2D (heads x headDim), flattened as 1D
      input.kBiases, // 2D (heads x headDim), flattened as 1D
      input.vBiases  // 2D (heads x headDim), flattened as 1D
  );
  // 2D (NUM_TOKENS x embedDim), flattened as 1D

  std::vector<double> normedForMLP(NUM_TOKENS *
                                   TransformerInput::EMBEDDING_DIMENSION);
  // 2D (NUM_TOKENS x embedDim), flattened as 1D

  for (int i = 0; i < NUM_TOKENS; i++) {
    std::vector<double> origToken(
        embeddings.begin() + i * TransformerInput::EMBEDDING_DIMENSION,
        embeddings.begin() + (i + 1) * TransformerInput::EMBEDDING_DIMENSION);
    // 1D

    std::vector<double> attnToken(
        attentionResult.begin() + i * TransformerInput::EMBEDDING_DIMENSION,
        attentionResult.begin() +
            (i + 1) * TransformerInput::EMBEDDING_DIMENSION);
    // 1D

    auto withResidual = AddVectors(origToken, // 1D
                                   attnToken  // 1D
    );
    // 1D

    for (int j = 0; j < TransformerInput::EMBEDDING_DIMENSION; j++) {
      attentionResult[i * TransformerInput::EMBEDDING_DIMENSION + j] =
          withResidual[j];
    }

    auto normed = LayerNorm(withResidual,   // 1D
                            input.gammaMLP, // 1D
                            input.betaMLP,  // 1D
                            TransformerInput::EPSILON_MLP);
    // 1D

    for (int j = 0; j < TransformerInput::EMBEDDING_DIMENSION; j++) {
      normedForMLP[i * TransformerInput::EMBEDDING_DIMENSION + j] = normed[j];
    }
  }

  auto MLPResult =
      MLP(normedForMLP, // 2D (NUM_TOKENS x embedDim), flattened as 1D

          NUM_TOKENS, TransformerInput::EMBEDDING_DIMENSION,

          input.l1Weights, // 2D (hidden x embedDim), flattened as 1D
          input.l1Biases,  // 1D

          input.l2Weights, // 2D (embedDim x hidden), flattened as 1D
          input.l2Biases   // 1D
      );
  // 2D (NUM_TOKENS x embedDim), flattened as 1D

  for (int i = 0; i < NUM_TOKENS; i++) {
    std::vector<double> attnToken(
        attentionResult.begin() + i * TransformerInput::EMBEDDING_DIMENSION,
        attentionResult.begin() +
            (i + 1) * TransformerInput::EMBEDDING_DIMENSION);
    // 1D

    std::vector<double> mlpToken(
        MLPResult.begin() + i * TransformerInput::EMBEDDING_DIMENSION,
        MLPResult.begin() + (i + 1) * TransformerInput::EMBEDDING_DIMENSION);
    // 1D

    auto withResidual = AddVectors(attnToken, // 1D
                                   mlpToken   // 1D
    );
    // 1D

    for (int j = 0; j < TransformerInput::EMBEDDING_DIMENSION; j++) {
      MLPResult[i * TransformerInput::EMBEDDING_DIMENSION + j] =
          withResidual[j];
    }
  }

  return MLPResult;
  // 2D (NUM_TOKENS x embedDim), flattened as 1D
}

struct GptWeights {
  const std::vector<TransformerInput> transformerWeights;
  const std::vector<double> finalLayerNormWeights;
  const std::vector<double> finalLayerNormBiases;
  const std::vector<double> wpeWeights;
  const std::vector<double> wteWeights;
};

GptWeights LoadWeights() {

  auto InputVectorFromFile = [](std::vector<double> &v, std::ifstream &stream,
                                std::optional<size_t> maxCount =
                                    std::nullopt) -> void {
    double inp;
    size_t count = 0;
    while (stream >> inp) {
      v.push_back(inp);
      count++;
      if (maxCount.has_value() && count >= maxCount.value())
        break;
    }
  };

  std::vector<TransformerInput> result;

  double tempInput;

  for (int i = 0; i < TransformerInput::LAYERS; i++) {

    std::vector<double>
        qWeights; // 3D (heads x embedDim x headDim), flattened as 1D
    std::vector<double>
        kWeights; // 3D (heads x embedDim x headDim), flattened as 1D
    std::vector<double>
        vWeights; // 3D (heads x embedDim x headDim), flattened as 1D

    std::vector<double> qBiases; // 2D (heads x headDim), flattened as 1D
    std::vector<double> kBiases; // 2D (heads x headDim), flattened as 1D
    std::vector<double> vBiases; // 2D (heads x headDim), flattened as 1D

    std::vector<double> oWeights; // 2D (embedDim x embedDim), flattened as 1D
    std::vector<double> oBiases;  // 2D (embedDim x embedDim), flattened as 1D

    std::vector<double> l1Weights; // 2D (embedDim x hidden), flattened as 1D
    std::vector<double> l1Biases;  // 1D

    std::vector<double> l2Weights; // 2D (hidden x embedDim), flattened as 1D
    std::vector<double> l2Biases;  // 1D

    std::vector<double> gammaAttention; // 1D
    std::vector<double> gammaMLP;       // 1D

    std::vector<double> betaAttention; // 1D
    std::vector<double> betaMLP;       // 1D

    // LayerNorm 1
    std::ifstream ln1Weights("../weights/transformer.h." + std::to_string(i) +
                             ".ln_1.weight.txt");
    InputVectorFromFile(gammaAttention, ln1Weights);

    std::ifstream ln1Biases("../weights/transformer.h." + std::to_string(i) +
                            ".ln_1.bias.txt");
    InputVectorFromFile(betaAttention, ln1Biases);

    // LayerNorm 2
    std::ifstream ln2Weights("../weights/transformer.h." + std::to_string(i) +
                             ".ln_2.weight.txt");
    InputVectorFromFile(gammaMLP, ln2Weights);
    std::ifstream ln2Biases("../weights/transformer.h." + std::to_string(i) +
                            ".ln_2.bias.txt");
    InputVectorFromFile(betaMLP, ln2Biases);

    // MLP Layer 1
    std::ifstream mlpL1Weights("../weights/transformer.h." + std::to_string(i) +
                               ".mlp.c_fc.weight.txt");
    InputVectorFromFile(l1Weights, mlpL1Weights);

    std::ifstream mlpL1Biases("../weights/transformer.h." + std::to_string(i) +
                              ".mlp.c_fc.bias.txt");
    InputVectorFromFile(l1Biases, mlpL1Biases);

    // MLP Layer 2
    std::ifstream mlpL2Weights("../weights/transformer.h." + std::to_string(i) +
                               ".mlp.c_proj.weight.txt");
    InputVectorFromFile(l2Weights, mlpL2Weights);

    std::ifstream mlpL2Biases("../weights/transformer.h." + std::to_string(i) +
                              ".mlp.c_proj.bias.txt");
    InputVectorFromFile(l2Biases, mlpL2Biases);

    // c_fc.weight is stored (in=768, out=hidden) row-major; ForwardPass
    // expects (neurons x inputSize) = (hidden x 768). Transpose.
    l1Weights = Transpose(l1Weights, N_EMBD, l1Biases.size());
    // c_proj.weight is stored (in=hidden, out=768) row-major; ForwardPass
    // expects (neurons x inputSize) = (768 x hidden). Transpose.
    l2Weights = Transpose(l2Weights, l1Biases.size(), N_EMBD);

    // Attention QKV weights and biases
    std::ifstream qkvWeights("../weights/transformer.h." + std::to_string(i) +
                             ".attn.c_attn.weight.txt");

    for (int i = 0; i < N_EMBD; i++) {
      InputVectorFromFile(qWeights, qkvWeights, N_EMBD);
      InputVectorFromFile(kWeights, qkvWeights, N_EMBD);
      InputVectorFromFile(vWeights, qkvWeights, N_EMBD);
    }

    // take care on using these  , these are transposed
    qWeights = Transpose(qWeights, N_EMBD, N_EMBD);
    kWeights = Transpose(kWeights, N_EMBD, N_EMBD);
    vWeights = Transpose(vWeights, N_EMBD, N_EMBD);

    std::ifstream qkvBiases("../weights/transformer.h." + std::to_string(i) +
                            ".attn.c_attn.bias.txt");

    InputVectorFromFile(qBiases, qkvBiases, N_EMBD);
    InputVectorFromFile(kBiases, qkvBiases, N_EMBD);
    InputVectorFromFile(vBiases, qkvBiases, N_EMBD);

    // Attention output projection
    std::ifstream attnOutputProjWeights("../weights/transformer.h." +
                                        std::to_string(i) +
                                        ".attn.c_proj.weight.txt");
    InputVectorFromFile(oWeights, attnOutputProjWeights);

    std::ifstream attnOutputProjBiases("../weights/transformer.h." +
                                       std::to_string(i) +
                                       ".attn.c_proj.bias.txt");
    InputVectorFromFile(oBiases, attnOutputProjBiases);

    result.emplace_back(qWeights, kWeights, vWeights, qBiases, kBiases, vBiases,
                        oWeights, oBiases, l1Weights, l1Biases, l2Weights,
                        l2Biases, gammaAttention, gammaMLP, betaAttention,
                        betaMLP);
  }

  std::vector<double> finalLayerNormWeights, finalLayerNormBiases,
      wpeWeightsVector, wteWeightsVector;

  std::ifstream lnWeights("../weights/transformer.ln_f.weight.txt");
  std::ifstream lnBiases("../weights/transformer.ln_f.bias.txt");

  std::ifstream wpeWeights("../weights/transformer.wpe.weight.txt");
  std::ifstream wteWeights("../weights/transformer.wte.weight.txt");

  InputVectorFromFile(finalLayerNormWeights, lnWeights);
  InputVectorFromFile(finalLayerNormBiases, lnBiases);

  InputVectorFromFile(wpeWeightsVector, wpeWeights);
  InputVectorFromFile(wteWeightsVector, wteWeights);

  return {result, finalLayerNormWeights, finalLayerNormBiases, wpeWeightsVector,
          wteWeightsVector};
}

// return next tokens id
int GPT(const GptWeights &weights, const int numEmbeddings,
        const std::vector<double> &embeddings) {
  //
  auto result =
      Transformer(weights.transformerWeights[0], numEmbeddings, embeddings);
  for (int i = 1; i < N_LAYER; i++)
    result = Transformer(weights.transformerWeights[i], numEmbeddings, result);

  //
  std::span<double> lastTokenEmbedding(result.end() - N_EMBD, result.end());

  auto layerNormedResult =
      LayerNorm(lastTokenEmbedding, weights.finalLayerNormWeights,
                weights.finalLayerNormBiases, EPSILON);

  std::vector<double> distribution(VOCAB_SIZE);
  for (int v = 0; v < VOCAB_SIZE; v++) {
    std::span<const double> vocabRow(weights.wteWeights.data() + v * N_EMBD,
                                     N_EMBD);
    distribution[v] = DotProduct(layerNormedResult, vocabRow);
  }

  auto probDistribution = SoftMax(distribution);

  int maxProbTokenId =
      std::max_element(probDistribution.begin(), probDistribution.end()) -
      probDistribution.begin();

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

std::vector<double>
GetEmbeddingFromTokenId(int tokenId, const std::vector<double> &wteWeights,
                        const std::vector<double> &wpeWeights,
                        const int position) {
  return AddVectors(
      std::vector<double>{wteWeights.begin() + tokenId * N_EMBD,
                          wteWeights.begin() + (tokenId + 1) * N_EMBD},
      std::vector<double>{wpeWeights.begin() + position * N_EMBD,
                          wpeWeights.begin() + (position + 1) * N_EMBD});
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

std::vector<double> GenerateEmbeddings(std::string input,
                                       const std::vector<double> &wteWeights,
                                       const std::vector<double> &wpeWeights) {
  auto tokenIds = Tokenize(input);

  std::vector<double> result;
  for (int i = 0; i < tokenIds.size(); i++) {
    auto embedding =
        GetEmbeddingFromTokenId(tokenIds[i], wteWeights, wpeWeights, i);
    std::copy(embedding.begin(), embedding.end(), std::back_inserter(result));
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

    //
    std::string input;
    std::cout << "\nEnter text: ";
    std::getline(std::cin, input); // Reads until Enter is pressed
    std::cout << "You entered: " << input << std::endl;

    auto embeddings =
        GenerateEmbeddings(input, weights.wteWeights, weights.wpeWeights);

    int tokenToGenerate = 20;
    while (tokenToGenerate--) {
      auto nextToken = GPT(weights, embeddings.size() / N_EMBD, embeddings);
      if (tokenToGenerate) {
        auto nextTokenEmbedding = GetEmbeddingFromTokenId(
            nextToken, weights.wteWeights, weights.wpeWeights,
            embeddings.size() / N_EMBD);
        std::copy(nextTokenEmbedding.begin(), nextTokenEmbedding.end(),
                  std::back_inserter(embeddings));
      }

      std::cout << GetPrintableToken(GetTokenFromTokenId(nextToken));
      std::cout.flush();

      //
    }
  }
}