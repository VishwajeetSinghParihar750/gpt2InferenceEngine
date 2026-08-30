#include "constants.hh"
#include "global.hh"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

double GeluNew(double x) {
  return .5 * x * (1 + tanh(sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x)));
}

std::vector<double>
LayerNorm(std::vector<double> originalEmbedding, std::vector<double> gamma,
          std::vector<double> beta,
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

std::vector<double> AddVectors(std::vector<double> a,
                               const std::vector<double> &b) // a, b: 1d
{
  for (int i = 0; i < a.size(); i++)
    a[i] += b[i];
  return a;
}

std::vector<double> SoftMax(std::vector<double> input) // input: 1d
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

double DotProduct(const std::vector<double> &a,
                  const std::vector<double> &b) // a, b: 1d
{
  int len = a.size();
  double output = 0;
  for (int i = 0; i < len; i++)
    output += a[i] * b[i];
  return output;
}

std::vector<double> Transpose(const std::vector<double> &a, int n,
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

std::vector<double> MatMul(const std::vector<double> &a, int aRows,
                           int aCols, // a: 2d (aRows x aCols) flattened as 1d
                           const std::vector<double> &b, int bRows,
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
    const std::vector<double> &embeddings, int NUM_TOKENS,
    int embedDim, // embeddings: 2d (NUM_TOKENS x embedDim) flattened as 1d
    const std::vector<double>
        &qWeights, // 2d (embedDim x headDim) flattened as 1d
    const std::vector<double>
        &kWeights, // 2d (embedDim x headDim) flattened as 1d
    const std::vector<double>
        &vWeights, // 2d (embedDim x headDim) flattened as 1d
    int headDim) {
  auto qProjections =
      MatMul(embeddings, NUM_TOKENS, embedDim, qWeights, embedDim,
             headDim); // 2d (NUM_TOKENS x headDim) flattened as 1d
  auto kProjections =
      MatMul(embeddings, NUM_TOKENS, embedDim, kWeights, embedDim,
             headDim); // 2d (NUM_TOKENS x headDim) flattened as 1d
  auto vProjections =
      MatMul(embeddings, NUM_TOKENS, embedDim, vWeights, embedDim,
             headDim); // 2d (NUM_TOKENS x headDim) flattened as 1d

  auto kTranspose =
      Transpose(kProjections, NUM_TOKENS,
                headDim); // 2d (headDim x NUM_TOKENS) flattened as 1d

  auto qkTranspose =
      MatMul(qProjections, NUM_TOKENS, headDim, kTranspose, headDim,
             NUM_TOKENS); // 2d (NUM_TOKENS x NUM_TOKENS) flattened as 1d

  double dimensionsRoot = sqrt(headDim);
  for (auto &v : qkTranspose)
    v /= dimensionsRoot;

  for (int i = 0; i < NUM_TOKENS; i++) {
    std::vector<double> row(qkTranspose.begin() + i * NUM_TOKENS,
                            qkTranspose.begin() + (i + 1) * NUM_TOKENS); // 1d
    auto softRow = SoftMax(row);                                         // 1d
    for (int j = 0; j < NUM_TOKENS; j++)
      qkTranspose[i * NUM_TOKENS + j] = softRow[j];
  }

  return MatMul(qkTranspose, NUM_TOKENS, NUM_TOKENS, vProjections, NUM_TOKENS,
                headDim); // 2d (NUM_TOKENS x headDim) flattened as 1d
}

std::vector<double> MultiHeadAttention(
    const std::vector<double> &embeddings, int NUM_TOKENS,
    int embedDim, // embeddings: 2d (NUM_TOKENS x embedDim) flattened as 1d
    const std::vector<double>
        &qWeights, // 3d (heads x embedDim x headDim) flattened as 1d
    const std::vector<double>
        &kWeights, // 3d (heads x embedDim x headDim) flattened as 1d
    const std::vector<double>
        &vWeights, // 3d (heads x embedDim x headDim) flattened as 1d
    const std::vector<double>
        &oWeights, // 2d (embedDim x embedDim) flattened as 1d
    int heads, int headDim) {
  std::vector<double> result(NUM_TOKENS * embedDim,
                             0); // 2d (NUM_TOKENS x embedDim) flattened as 1d

  int weightBlockSize = embedDim * headDim;

  for (int h = 0; h < heads; h++) {
    std::vector<double> qHead(
        qWeights.begin() + h * weightBlockSize,
        qWeights.begin() +
            (h + 1) *
                weightBlockSize); // 2d (embedDim x headDim) flattened as 1d
    std::vector<double> kHead(
        kWeights.begin() + h * weightBlockSize,
        kWeights.begin() +
            (h + 1) *
                weightBlockSize); // 2d (embedDim x headDim) flattened as 1d
    std::vector<double> vHead(
        vWeights.begin() + h * weightBlockSize,
        vWeights.begin() +
            (h + 1) *
                weightBlockSize); // 2d (embedDim x headDim) flattened as 1d

    auto curResult =
        Attention(embeddings, NUM_TOKENS, embedDim, qHead, kHead, vHead,
                  headDim); // 2d (NUM_TOKENS x headDim) flattened as 1d

    for (int j = 0; j < NUM_TOKENS; j++)
      for (int k = 0; k < headDim; k++)
        result[j * embedDim + h * headDim + k] = curResult[j * headDim + k];
  }

  return MatMul(result, NUM_TOKENS, embedDim, oWeights, embedDim,
                embedDim); // 2d (NUM_TOKENS x embedDim) flattened as 1d
}

std::vector<double>
ForwardPass(const std::vector<double> &weights,
            const std::vector<double> &biases,
            const std::vector<double> &inputs,
            bool gelu = false) // weights: 2d flattened as 1d, biases/inputs: 1d
{
  int countNeurons = biases.size();
  std::vector<double> output = biases; // 1d

  for (int i = 0; i < countNeurons; i++) {
    for (int j = 0; j < inputs.size(); j++)
      output[i] += weights[i * inputs.size() + j] * inputs[j];

    if (gelu)
      output[i] = GeluNew(output[i]);
  }
  return output;
}

std::vector<double>
MLP(const std::vector<double> &embeddings, int NUM_TOKENS,
    int dimensions, // embeddings: 2d (NUM_TOKENS x dimensions) flattened as 1d
    const std::vector<double>
        &l1Weights, // 2d (dimensions x hidden) flattened as 1d
    const std::vector<double> &l1Biases, // 1d
    const std::vector<double>
        &l2Weights, // 2d (hidden x dimensions) flattened as 1d
    const std::vector<double> &l2Biases) // 1d
{
  std::vector<double> result(
      NUM_TOKENS * dimensions); // 2d (NUM_TOKENS x dimensions) flattened as 1d

  for (int i = 0; i < NUM_TOKENS; i++) {
    std::vector<double> tokenEmbedding(embeddings.begin() + i * dimensions,
                                       embeddings.begin() +
                                           (i + 1) * dimensions); // 1d
    auto hiddenOut =
        ForwardPass(l1Weights, l1Biases, tokenEmbedding, true); // 1d
    auto out = ForwardPass(l2Weights, l2Biases, hiddenOut);     // 1d
    for (int j = 0; j < dimensions; j++)
      result[i * dimensions + j] = out[j];
  }

  return result;
}

struct TransformerInput {
  static constexpr int EMBEDDING_DIMENSION = 784;
  static constexpr int HEADS = 12;
  static constexpr int LAYERS = 12;

  static constexpr double EPSILON_ATTENTION = 1e-5;
  static constexpr double EPSILON_MLP = 1e-5;

  static constexpr int HEAD_DIMENSION = 64;

  const int NUM_TOKENS;

  const std::vector<double>
      &embeddings; // 2D (NUM_TOKENS x embedDim), flattened as 1D

  const std::vector<double>
      &qWeights; // 3D (heads x embedDim x headDim), flattened as 1D
  const std::vector<double>
      &kWeights; // 3D (heads x embedDim x headDim), flattened as 1D
  const std::vector<double>
      &vWeights; // 3D (heads x embedDim x headDim), flattened as 1D

  const std::vector<double>
      &oWeights; // 2D (embedDim x embedDim), flattened as 1D

  const std::vector<double>
      &l1Weights; // 2D (embedDim x hidden), flattened as 1D
  const std::vector<double> &l1Biases; // 1D

  const std::vector<double>
      &l2Weights; // 2D (hidden x embedDim), flattened as 1D
  const std::vector<double> &l2Biases; // 1D

  const std::vector<double> &gammaAttention; // 1D
  const std::vector<double> &gammaMLP;       // 1D

  const std::vector<double> &betaAttention; // 1D
  const std::vector<double> &betaMLP;       // 1D

  TransformerInput(
      int numTokens, const std::vector<double> &embeddings,
      const std::vector<double> &qWeights, const std::vector<double> &kWeights,
      const std::vector<double> &vWeights, const std::vector<double> &oWeights,
      const std::vector<double> &l1Weights, const std::vector<double> &l1Biases,
      const std::vector<double> &l2Weights, const std::vector<double> &l2Biases,
      const std::vector<double> &gammaAttention,
      const std::vector<double> &gammaMLP,
      const std::vector<double> &betaAttention,
      const std::vector<double> &betaMLP)
      : NUM_TOKENS(numTokens), embeddings(embeddings), qWeights(qWeights),
        kWeights(kWeights), vWeights(vWeights), oWeights(oWeights),
        l1Weights(l1Weights), l1Biases(l1Biases), l2Weights(l2Weights),
        l2Biases(l2Biases), gammaAttention(gammaAttention), gammaMLP(gammaMLP),
        betaAttention(betaAttention), betaMLP(betaMLP) {}
};

std::vector<double> Transformer(TransformerInput *input) {
  std::vector<double> layerNormedEmbeddings(
      input->NUM_TOKENS * TransformerInput::EMBEDDING_DIMENSION);
  // 2D (NUM_TOKENS x embedDim), flattened as 1D

  for (int i = 0; i < input->NUM_TOKENS; i++) {
    std::vector<double> tokenEmbedding(
        input->embeddings.begin() + i * TransformerInput::EMBEDDING_DIMENSION,
        input->embeddings.begin() +
            (i + 1) * TransformerInput::EMBEDDING_DIMENSION);
    // 1D

    auto normed = LayerNorm(tokenEmbedding,        // 1D
                            input->gammaAttention, // 1D
                            input->betaAttention,  // 1D
                            TransformerInput::EPSILON_ATTENTION);
    // 1D

    for (int j = 0; j < TransformerInput::EMBEDDING_DIMENSION; j++) {
      layerNormedEmbeddings[i * TransformerInput::EMBEDDING_DIMENSION + j] =
          normed[j];
    }
  }

  auto attentionResult = MultiHeadAttention(
      layerNormedEmbeddings, // 2D (NUM_TOKENS x embedDim), flattened as 1D

      input->NUM_TOKENS, TransformerInput::EMBEDDING_DIMENSION,

      input->qWeights, // 3D (heads x embedDim x headDim), flattened as 1D
      input->kWeights, // 3D (heads x embedDim x headDim), flattened as 1D
      input->vWeights, // 3D (heads x embedDim x headDim), flattened as 1D

      input->oWeights, // 2D (embedDim x embedDim), flattened as 1D

      TransformerInput::HEADS, TransformerInput::HEAD_DIMENSION);
  // 2D (NUM_TOKENS x embedDim), flattened as 1D

  std::vector<double> normedForMLP(input->NUM_TOKENS *
                                   TransformerInput::EMBEDDING_DIMENSION);
  // 2D (NUM_TOKENS x embedDim), flattened as 1D

  for (int i = 0; i < input->NUM_TOKENS; i++) {
    std::vector<double> origToken(
        input->embeddings.begin() + i * TransformerInput::EMBEDDING_DIMENSION,
        input->embeddings.begin() +
            (i + 1) * TransformerInput::EMBEDDING_DIMENSION);
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

    auto normed = LayerNorm(withResidual,    // 1D
                            input->gammaMLP, // 1D
                            input->betaMLP,  // 1D
                            TransformerInput::EPSILON_MLP);
    // 1D

    for (int j = 0; j < TransformerInput::EMBEDDING_DIMENSION; j++) {
      normedForMLP[i * TransformerInput::EMBEDDING_DIMENSION + j] = normed[j];
    }
  }

  auto MLPResult =
      MLP(normedForMLP, // 2D (NUM_TOKENS x embedDim), flattened as 1D

          input->NUM_TOKENS, TransformerInput::EMBEDDING_DIMENSION,

          input->l1Weights, // 2D (embedDim x hidden), flattened as 1D
          input->l1Biases,  // 1D

          input->l2Weights, // 2D (hidden x embedDim), flattened as 1D
          input->l2Biases   // 1D
      );
  // 2D (NUM_TOKENS x embedDim), flattened as 1D

  for (int i = 0; i < input->NUM_TOKENS; i++) {
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

std::vector<TransformerInput> LoadTransformerWeights() {

  auto InputVectorFromFile = [](std::vector<double> &v,
                                std::ifstream &stream) -> void {
    double inp;
    while (stream >> inp)
      v.push_back(inp);
  };

  std::vector<TransformerInput> result(TransformerInput::LAYERS);

  double tempInput;

  for (int i = 0; i < TransformerInput::LAYERS; i++) {

    std::vector<double>
        qWeights; // 3D (heads x embedDim x headDim), flattened as 1D
    std::vector<double>
        kWeights; // 3D (heads x embedDim x headDim), flattened as 1D
    std::vector<double>
        vWeights; // 3D (heads x embedDim x headDim), flattened as 1D

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
    InputVectorFromFile(qWeights, qkvWeights);
    InputVectorFromFile(kWeights, qkvWeights);
    InputVectorFromFile(vWeights, qkvWeights);

    std::ifstream qkvBiases("../weights/transformer.h." + std::to_string(i) +
                            ".attn.c_attn.bias.txt");

    // ===================================================================
    // TORODOOD

    // Attention output projection
    std::ifstream attnOutputProjWeights("../weights/transformer.h." +
                                        std::to_string(i) +
                                        ".attn.c_proj.weight.txt");
    InputVectorFromFile(oWeights, attnOutputProjWeights);

    std::ifstream attnOutputProjBiases("../weights/transformer.h." +
                                       std::to_string(i) +
                                       ".attn.c_proj.bias.txt");
    InputVectorFromFile(oBiases, attnOutputProjBiases);
  }
  return {};
}

std::vector<double> GPT() { return {}; }