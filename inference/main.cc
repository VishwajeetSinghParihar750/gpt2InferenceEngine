#include "constants.hh"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <iostream>
#include <vector>

#include "include/json.hpp"

double GeluNew(double x)
{
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
  for (int i = 0; i < len; i++)
  {
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
  for (int i = 0; i < len; i++)
  {
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
  for (int i = 0; i < m; i++)
  {
    for (int j = 0; j < n; j++)
    {
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

  for (int i = 0; i < aRows; i++)
  {
    for (int j = 0; j < bCols; j++)
    {
      for (int k = 0; k < bRows; k++)
      {
        result[i * bCols + j] += a[i * aCols + k] * b[k * bCols + j];
      }
    }
  }

  return result;
}

std::vector<double> Attention(
    std::span<const double> embeddings, int NUM_TOKENS,
    int embedDim,                     // embeddings: 2d (NUM_TOKENS x embedDim) flattened as 1d
    std::span<const double> qWeights, // 2d (embedDim x headDim) flattened as 1d
    std::span<const double> kWeights, // 2d (embedDim x headDim) flattened as 1d
    std::span<const double> vWeights, // 2d (embedDim x headDim) flattened as 1d
    std::span<const double> qBiases,  // 1d
    std::span<const double> kBiases,  // 1d
    std::span<const double> vBiases,  // 1d
    int headDim)
{

  auto qProjections =
      MatMul(embeddings, NUM_TOKENS, embedDim, qWeights, embedDim,
             headDim); // 2d (NUM_TOKENS x headDim) flattened as 1d

  auto kProjections =
      MatMul(embeddings, NUM_TOKENS, embedDim, kWeights, embedDim,
             headDim); // 2d (NUM_TOKENS x headDim) flattened as 1d
  auto vProjections =
      MatMul(embeddings, NUM_TOKENS, embedDim, vWeights, embedDim,
             headDim); // 2d (NUM_TOKENS x headDim) flattened as 1d

  for (int i = 0; i < NUM_TOKENS; i++)
  {
    for (int j = 0; j < headDim; j++)
    {
      qProjections[i * headDim + j] += qBiases[j];
      kProjections[i * headDim + j] += kBiases[j];
      vProjections[i * headDim + j] += vBiases[j];
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

  for (int i = 0; i < NUM_TOKENS; i++)
  {
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
        qWeights, // 3d (heads x embedDim x headDim) flattened as 1d
    std::span<const double>
        kWeights, // 3d (heads x embedDim x headDim) flattened as 1d
    std::span<const double>
        vWeights, // 3d (heads x embedDim x headDim) flattened as 1d
    std::span<const double>
        oWeights, // 2d (embedDim x embedDim) flattened as 1d
    int heads, int headDim,
    std::span<const double> oBiases, // embedDim 1d
    std::span<const double> qBiases, // headDim 1d
    std::span<const double> kBiases, // headDim 1d
    std::span<const double> vBiases) // headDim 1d
{
  std::vector<double> result(NUM_TOKENS * embedDim,
                             0); // 2d (NUM_TOKENS x embedDim) flattened as 1d

  int weightBlockSize = embedDim * headDim;

  for (int h = 0; h < heads; h++)
  {
    auto qHead = qWeights.subspan(h * weightBlockSize,
                                  weightBlockSize); // 2d (embedDim x headDim)
    auto kHead = kWeights.subspan(h * weightBlockSize,
                                  weightBlockSize); // 2d (embedDim x headDim)
    auto vHead = vWeights.subspan(h * weightBlockSize,
                                  weightBlockSize); // 2d (embedDim x headDim)

    auto curResult =
        Attention(embeddings, NUM_TOKENS, embedDim, qHead, kHead, vHead,
                  qBiases, kBiases, vBiases,
                  headDim); // 2d (NUM_TOKENS x headDim) flattened as 1d

    for (int j = 0; j < NUM_TOKENS; j++)
      for (int k = 0; k < headDim; k++)
        result[j * embedDim + h * headDim + k] = curResult[j * headDim + k];
  }

  auto projectionResult =
      MatMul(result, NUM_TOKENS, embedDim, oWeights, embedDim,
             embedDim); // 2d (NUM_TOKENS x embedDim) flattened as 1d

  if (!oBiases.empty())
  {
    for (int i = 0; i < NUM_TOKENS; i++)
    {
      for (int j = 0; j < embedDim; j++)
      {
        projectionResult[i * embedDim + j] += oBiases[j];
      }
    }
  }

  return projectionResult;
}

std::vector<double>
ForwardPass(std::span<const double> weights, std::span<const double> biases,
            std::span<const double> inputs,
            bool gelu = false) // weights: 2d flattened as 1d, biases/inputs: 1d
{
  int countNeurons = biases.size();
  std::vector<double> output(biases.begin(), biases.end()); // 1d

  for (int i = 0; i < countNeurons; i++)
  {
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
        l1Weights,                    // 2d (dimensions x hidden) flattened as 1d
    std::span<const double> l1Biases, // 1d
    std::span<const double>
        l2Weights,                    // 2d (hidden x dimensions) flattened as 1d
    std::span<const double> l2Biases) // 1d
{
  std::vector<double> result(
      NUM_TOKENS * dimensions); // 2d (NUM_TOKENS x dimensions) flattened as 1d

  for (int i = 0; i < NUM_TOKENS; i++)
  {
    auto tokenEmbedding = embeddings.subspan(i * dimensions, dimensions); // 1d
    auto hiddenOut =
        ForwardPass(l1Weights, l1Biases, tokenEmbedding, true); // 1d
    auto out = ForwardPass(l2Weights, l2Biases, hiddenOut);     // 1d
    for (int j = 0; j < dimensions; j++)
      result[i * dimensions + j] = out[j];
  }

  return result;
}

struct TransformerInput
{
  static constexpr int EMBEDDING_DIMENSION = 784;
  static constexpr int HEADS = 12;
  static constexpr int LAYERS = 12;

  static constexpr double EPSILON_ATTENTION = 1e-5;
  static constexpr double EPSILON_MLP = 1e-5;

  static constexpr int HEAD_DIMENSION = 64;

  const std::vector<double>
      &qWeights; // 3D (heads x embedDim x headDim), flattened as 1D
  const std::vector<double>
      &kWeights; // 3D (heads x embedDim x headDim), flattened as 1D
  const std::vector<double>
      &vWeights; // 3D (heads x embedDim x headDim), flattened as 1D

  const std::vector<double> &qBiases; // 2D (heads x headDim), flattened as 1D
  const std::vector<double> &kBiases; // 2D (heads x headDim), flattened as 1D
  const std::vector<double> &vBiases; // 2D (heads x headDim), flattened as 1D

  const std::vector<double>
      &oWeights; // 2D (embedDim x embedDim), flattened as 1D

  const std::vector<double>
      &l1Weights;                      // 2D (embedDim x hidden), flattened as 1D
  const std::vector<double> &l1Biases; // 1D

  const std::vector<double>
      &l2Weights;                      // 2D (hidden x embedDim), flattened as 1D
  const std::vector<double> &l2Biases; // 1D

  const std::vector<double> &gammaAttention; // 1D
  const std::vector<double> &gammaMLP;       // 1D

  const std::vector<double> &betaAttention; // 1D
  const std::vector<double> &betaMLP;       // 1D

  TransformerInput(
      const std::vector<double> &qWeights, const std::vector<double> &kWeights,
      const std::vector<double> &vWeights, const std::vector<double> &qBiases,
      const std::vector<double> &kBiases, const std::vector<double> &vBiases,
      const std::vector<double> &oWeights, const std::vector<double> &l1Weights,
      const std::vector<double> &l1Biases, const std::vector<double> &l2Weights,
      const std::vector<double> &l2Biases,
      const std::vector<double> &gammaAttention,
      const std::vector<double> &gammaMLP,
      const std::vector<double> &betaAttention,
      const std::vector<double> &betaMLP)
      : qWeights(qWeights), kWeights(kWeights), vWeights(vWeights),
        qBiases(qBiases), kBiases(kBiases), vBiases(vBiases),
        oWeights(oWeights), l1Weights(l1Weights), l1Biases(l1Biases),
        l2Weights(l2Weights), l2Biases(l2Biases),
        gammaAttention(gammaAttention), gammaMLP(gammaMLP),
        betaAttention(betaAttention), betaMLP(betaMLP) {}
};

std::vector<double>
Transformer(const TransformerInput &input, const int NUM_TOKENS,
            const std::vector<double>
                &embeddings // 2D (NUM_TOKENS x embedDim), flattened as 1D

)
{
  std::vector<double> layerNormedEmbeddings(
      NUM_TOKENS * TransformerInput::EMBEDDING_DIMENSION);
  // 2D (NUM_TOKENS x embedDim), flattened as 1D

  for (int i = 0; i < NUM_TOKENS; i++)
  {
    std::vector<double> tokenEmbedding(
        embeddings.begin() + i * TransformerInput::EMBEDDING_DIMENSION,
        embeddings.begin() + (i + 1) * TransformerInput::EMBEDDING_DIMENSION);
    // 1D

    auto normed = LayerNorm(tokenEmbedding,       // 1D
                            input.gammaAttention, // 1D
                            input.betaAttention,  // 1D
                            TransformerInput::EPSILON_ATTENTION);
    // 1D

    for (int j = 0; j < TransformerInput::EMBEDDING_DIMENSION; j++)
    {
      layerNormedEmbeddings[i * TransformerInput::EMBEDDING_DIMENSION + j] =
          normed[j];
    }
  }

  auto attentionResult = MultiHeadAttention(
      layerNormedEmbeddings, // 2D (NUM_TOKENS x embedDim), flattened as 1D

      NUM_TOKENS, TransformerInput::EMBEDDING_DIMENSION,

      input.qWeights, // 3D (heads x embedDim x headDim), flattened as 1D
      input.kWeights, // 3D (heads x embedDim x headDim), flattened as 1D
      input.vWeights, // 3D (heads x embedDim x headDim), flattened as 1D

      input.oWeights, // 2D (embedDim x embedDim), flattened as 1D

      TransformerInput::HEADS, TransformerInput::HEAD_DIMENSION,

      {},            // oBiases
      input.qBiases, // 2D (heads x headDim), flattened as 1D
      input.kBiases, // 2D (heads x headDim), flattened as 1D
      input.vBiases  // 2D (heads x headDim), flattened as 1D
  );
  // 2D (NUM_TOKENS x embedDim), flattened as 1D

  std::vector<double> normedForMLP(NUM_TOKENS *
                                   TransformerInput::EMBEDDING_DIMENSION);
  // 2D (NUM_TOKENS x embedDim), flattened as 1D

  for (int i = 0; i < NUM_TOKENS; i++)
  {
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

    for (int j = 0; j < TransformerInput::EMBEDDING_DIMENSION; j++)
    {
      attentionResult[i * TransformerInput::EMBEDDING_DIMENSION + j] =
          withResidual[j];
    }

    auto normed = LayerNorm(withResidual,   // 1D
                            input.gammaMLP, // 1D
                            input.betaMLP,  // 1D
                            TransformerInput::EPSILON_MLP);
    // 1D

    for (int j = 0; j < TransformerInput::EMBEDDING_DIMENSION; j++)
    {
      normedForMLP[i * TransformerInput::EMBEDDING_DIMENSION + j] = normed[j];
    }
  }

  auto MLPResult =
      MLP(normedForMLP, // 2D (NUM_TOKENS x embedDim), flattened as 1D

          NUM_TOKENS, TransformerInput::EMBEDDING_DIMENSION,

          input.l1Weights, // 2D (embedDim x hidden), flattened as 1D
          input.l1Biases,  // 1D

          input.l2Weights, // 2D (hidden x embedDim), flattened as 1D
          input.l2Biases   // 1D
      );
  // 2D (NUM_TOKENS x embedDim), flattened as 1D

  for (int i = 0; i < NUM_TOKENS; i++)
  {
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

    for (int j = 0; j < TransformerInput::EMBEDDING_DIMENSION; j++)
    {
      MLPResult[i * TransformerInput::EMBEDDING_DIMENSION + j] =
          withResidual[j];
    }
  }

  return MLPResult;
  // 2D (NUM_TOKENS x embedDim), flattened as 1D
}

struct GptWeights
{
  const std::vector<TransformerInput> transformerWeights;
  const std::vector<double> finalLayerNormWeights;
  const std::vector<double> finalLayerNormBiases;
  const std::vector<double> wpeWeights;
  const std::vector<double> wteWeights;
};

GptWeights LoadWeights()
{

  auto InputVectorFromFile = [](std::vector<double> &v, std::ifstream &stream,
                                std::optional<size_t> maxCount =
                                    std::nullopt) -> void
  {
    double inp;
    size_t count = 0;
    while (stream >> inp)
    {
      v.push_back(inp);
      count++;
      if (maxCount.has_value() && count >= maxCount.value())
        break;
    }
  };

  std::vector<TransformerInput> result;

  double tempInput;

  for (int i = 0; i < TransformerInput::LAYERS; i++)
  {

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

    // Attention QKV weights and biases
    std::ifstream qkvWeights("../weights/transformer.h." + std::to_string(i) +
                             ".attn.c_attn.weight.txt");

    for (int i = 0; i < N_EMBD; i++)
    {
      InputVectorFromFile(qWeights, qkvWeights, N_EMBD);
      InputVectorFromFile(kWeights, qkvWeights, N_EMBD);
      InputVectorFromFile(vWeights, qkvWeights, N_EMBD);
    }

    Transpose(qWeights, N_EMBD, N_EMBD); // _________________________________________________________________________ take care here , these are transposed
    Transpose(kWeights, N_EMBD, N_EMBD);
    Transpose(vWeights, N_EMBD, N_EMBD);

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
                        oWeights, l1Weights, l1Biases, l2Weights, l2Biases,
                        gammaAttention, gammaMLP, betaAttention, betaMLP);
  }

  std::vector<double> finalLayerNormWeights, finalLayerNormBiases, wpeWeightsVector, wteWeightsVector;

  std::ifstream lnWeights("../weights/transformer.ln_f.weight.txt");
  std::ifstream lnBiases("../weights/transformer.ln_f.bias.txt");

  std::ifstream wpeWeights("../weights/transformer.wpe.weight.txt");
  std::ifstream wteWeights("../weights/transformer.wte.weight.txt");

  InputVectorFromFile(finalLayerNormBiases, lnWeights);
  InputVectorFromFile(finalLayerNormBiases, lnBiases);

  InputVectorFromFile(wpeWeightsVector, wpeWeights);
  InputVectorFromFile(wteWeightsVector, wteWeights);

  return {result, finalLayerNormWeights, finalLayerNormBiases, wpeWeightsVector, wteWeightsVector};
}

// return next tokens id
int GPT(const GptWeights &weights, const int numEmbeddings, const std::vector<double> &embeddings)
{
  //
  auto result = Transformer(weights.transformerWeights[0], numEmbeddings, embeddings);
  for (int i = 1; i < N_HEAD; i++)
    result = Transformer(weights.transformerWeights[i], numEmbeddings, result);

  //
  std::span<double> lastTokenEmbedding(result.end() - N_EMBD, result.end());

  auto layerNormedResult = LayerNorm(lastTokenEmbedding, weights.finalLayerNormWeights, weights.finalLayerNormBiases, EPSILON);

  auto distribution = MatMul(layerNormedResult, 1, N_EMBD, weights.wteWeights, VOCAB_SIZE, N_EMBD); // need transpose here maybe +++++++++++++++++++++++++++++++++++++++++++++++++++=

  auto probDistribution = SoftMax(distribution);

  int maxProbTokenId = std::max_element(probDistribution.begin(), probDistribution.end()) - probDistribution.begin();

  return maxProbTokenId;
}

std::string GetTokenFromTokenId(int tokenId)
{
}
int GetTokenIdFromToken(std::string token)
{
}

std::vector<double> GetEmbeddingFromTokenId(int tokenId, const std::vector<double> &wteWeights)
{
  return {wteWeights.begin() + tokenId * N_EMBD, wteWeights.begin() + (tokenId + 1) * N_EMBD};
}

std::vector<int> Tokenize(std::string input)
{

  std::vector<int> result;
  return result;
}

std::vector<double> GenerateEmbeddings(std::string input, const std::vector<double> &wteWeights)
{
  auto tokenIds = Tokenize(input);

  std::vector<double>
      result;
  for (auto i : tokenIds)
  {
    auto embedding = GetEmbeddingFromTokenId(i, wteWeights);
    std::copy(embedding.begin(), embedding.end(), std::back_inserter(result));
  }

  return result;
}

std::pair<std::vector<std::string>, std::unordered_map<std::string, int>> LoadVocab()
{

  std::vector<std::string> tokens(VOCAB_SIZE);
  std::unordered_map<std::string, int> tokenToTokenId;

  using json = nlohmann::json;
  std::ifstream f("../weights/tokenizer/tokenizer.json");
  json data = json::parse(f);

  for (auto &[key, value] : data.items())
  {
    if (key == "model")
    {
      for (auto &[key, value] : value.items())
      {

        if (key == "vocab")
        {
          for (auto &[token, tokenId] : value.items())
          {
            tokens[tokenId] = token;
            tokenToTokenId[token] = tokenId;
          }
        }
        else if (key == "merges")
        {
          for (auto pairs : value.array())
          {
            std::string a = pairs[0], b = pairs[1];
          }
        }
      }
    }
  }
}

int main()
{
  auto weights = LoadWeights();

  auto [tokens, tokenToTokenId] = LoadVocab();

  //
  std::string input;
  std::cout << "Enter text: ";
  std::getline(std::cin, input); // Reads until Enter is pressed
  std::cout << "You entered: " << input << std::endl;

  auto embeddings = GenerateEmbeddings(input, weights.wteWeights);

  int tokenToGenerate = 100;
  while (tokenToGenerate--)
  {
    auto nextToken = GPT(weights, embeddings.size() / N_EMBD, embeddings);
    if (tokenToGenerate)
    {
      auto nextTokenEmbedding = GetEmbeddingFromTokenId(nextToken, weights.wteWeights);
      std::copy(nextTokenEmbedding.begin(), nextTokenEmbedding.end(), std::back_inserter(embeddings));
    }

    std::cout << GetTokenFromTokenId(nextToken);
    //
  }
}