#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cassert>

double GeluNew(double x)
{
    return .5 * x * (1 + tanh(sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x)));
}

std::vector<double> LayerNorm(std::vector<double> originalEmbedding, std::vector<double> gamma, std::vector<double> beta, double epsilon) // originalEmbedding, gamma, beta: 1d
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
        double normalizedValue = ((originalEmbedding[i] - mean) / modifiedStandardDeviation);
        output[i] = normalizedValue * gamma[i] + beta[i];
    }

    return output;
}

std::vector<double> AddVectors(std::vector<double> a, const std::vector<double> &b) // a, b: 1d
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
    for (int i = 0; i < len; i++)
    {
        softmax[i] = exp(input[i] - max);
    }

    auto sum = std::accumulate(softmax.begin(), softmax.end(), 0.0);

    for (auto &i : softmax)
        i /= sum;

    return softmax;
}

double DotProduct(const std::vector<double> &a, const std::vector<double> &b) // a, b: 1d
{
    int len = a.size();
    double output = 0;
    for (int i = 0; i < len; i++)
        output += a[i] * b[i];
    return output;
}

std::vector<double> Transpose(const std::vector<double> &a, int n, int m) // a: 2d (n x m) flattened as 1d
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

std::vector<double> MatMul(const std::vector<double> &a, int aRows, int aCols, // a: 2d (aRows x aCols) flattened as 1d
                           const std::vector<double> &b, int bRows, int bCols) // b: 2d (bRows x bCols) flattened as 1d
{
    assert(aCols == bRows);

    std::vector<double> result(aRows * bCols, 0); // 2d (aRows x bCols) flattened as 1d

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

std::vector<double> Attention(const std::vector<double> &embeddings, int numTokens, int embedDim, // embeddings: 2d (numTokens x embedDim) flattened as 1d
                              const std::vector<double> &qWeights,                                // 2d (embedDim x headDim) flattened as 1d
                              const std::vector<double> &kWeights,                                // 2d (embedDim x headDim) flattened as 1d
                              const std::vector<double> &vWeights,                                // 2d (embedDim x headDim) flattened as 1d
                              int headDim)
{
    auto qProjections = MatMul(embeddings, numTokens, embedDim, qWeights, embedDim, headDim); // 2d (numTokens x headDim) flattened as 1d
    auto kProjections = MatMul(embeddings, numTokens, embedDim, kWeights, embedDim, headDim); // 2d (numTokens x headDim) flattened as 1d
    auto vProjections = MatMul(embeddings, numTokens, embedDim, vWeights, embedDim, headDim); // 2d (numTokens x headDim) flattened as 1d

    auto kTranspose = Transpose(kProjections, numTokens, headDim); // 2d (headDim x numTokens) flattened as 1d

    auto qkTranspose = MatMul(qProjections, numTokens, headDim, kTranspose, headDim, numTokens); // 2d (numTokens x numTokens) flattened as 1d

    double dimensionsRoot = sqrt(headDim);
    for (auto &v : qkTranspose)
        v /= dimensionsRoot;

    for (int i = 0; i < numTokens; i++)
    {
        std::vector<double> row(qkTranspose.begin() + i * numTokens, qkTranspose.begin() + (i + 1) * numTokens); // 1d
        auto softRow = SoftMax(row);                                                                             // 1d
        for (int j = 0; j < numTokens; j++)
            qkTranspose[i * numTokens + j] = softRow[j];
    }

    return MatMul(qkTranspose, numTokens, numTokens, vProjections, numTokens, headDim); // 2d (numTokens x headDim) flattened as 1d
}

std::vector<double> MultiHeadAttention(const std::vector<double> &embeddings, int numTokens, int embedDim, // embeddings: 2d (numTokens x embedDim) flattened as 1d
                                       const std::vector<double> &qWeights,                                // 3d (heads x embedDim x headDim) flattened as 1d
                                       const std::vector<double> &kWeights,                                // 3d (heads x embedDim x headDim) flattened as 1d
                                       const std::vector<double> &vWeights,                                // 3d (heads x embedDim x headDim) flattened as 1d
                                       const std::vector<double> &oWeights,                                // 2d (embedDim x embedDim) flattened as 1d
                                       int heads, int headDim)
{
    std::vector<double> result(numTokens * embedDim, 0); // 2d (numTokens x embedDim) flattened as 1d

    int weightBlockSize = embedDim * headDim;

    for (int h = 0; h < heads; h++)
    {
        std::vector<double> qHead(qWeights.begin() + h * weightBlockSize, qWeights.begin() + (h + 1) * weightBlockSize); // 2d (embedDim x headDim) flattened as 1d
        std::vector<double> kHead(kWeights.begin() + h * weightBlockSize, kWeights.begin() + (h + 1) * weightBlockSize); // 2d (embedDim x headDim) flattened as 1d
        std::vector<double> vHead(vWeights.begin() + h * weightBlockSize, vWeights.begin() + (h + 1) * weightBlockSize); // 2d (embedDim x headDim) flattened as 1d

        auto curResult = Attention(embeddings, numTokens, embedDim, qHead, kHead, vHead, headDim); // 2d (numTokens x headDim) flattened as 1d

        for (int j = 0; j < numTokens; j++)
            for (int k = 0; k < headDim; k++)
                result[j * embedDim + h * headDim + k] = curResult[j * headDim + k];
    }

    return MatMul(result, numTokens, embedDim, oWeights, embedDim, embedDim); // 2d (numTokens x embedDim) flattened as 1d
}

std::vector<double> ForwardPass(const std::vector<double> &weights, const std::vector<double> &biases, const std::vector<double> &inputs, bool gelu = false) // weights: 2d flattened as 1d, biases/inputs: 1d
{
    int countNeurons = biases.size();
    std::vector<double> output = biases; // 1d

    for (int i = 0; i < countNeurons; i++)
    {
        for (int j = 0; j < inputs.size(); j++)
            output[i] += weights[i * inputs.size() + j] * inputs[j];

        if (gelu)
            output[i] = GeluNew(output[i]);
    }
    return output;
}

std::vector<double> MLP(const std::vector<double> &embeddings, int numTokens, int dimensions, // embeddings: 2d (numTokens x dimensions) flattened as 1d
                        const std::vector<double> &l1Weights,                                 // 2d (dimensions x hidden) flattened as 1d
                        const std::vector<double> &l1Biases,                                  // 1d
                        const std::vector<double> &l2Weights,                                 // 2d (hidden x dimensions) flattened as 1d
                        const std::vector<double> &l2Biases)                                  // 1d
{
    std::vector<double> result(numTokens * dimensions); // 2d (numTokens x dimensions) flattened as 1d

    for (int i = 0; i < numTokens; i++)
    {
        std::vector<double> tokenEmbedding(embeddings.begin() + i * dimensions, embeddings.begin() + (i + 1) * dimensions); // 1d
        auto hiddenOut = ForwardPass(l1Weights, l1Biases, tokenEmbedding, true);                                            // 1d
        auto out = ForwardPass(l2Weights, l2Biases, hiddenOut);                                                             // 1d
        for (int j = 0; j < dimensions; j++)
            result[i * dimensions + j] = out[j];
    }

    return result;
}

std::vector<double> Transformer(const std::vector<double> &embeddings, int numTokens, int embedDim, // embeddings: 2d (numTokens x embedDim) flattened as 1d
                                const std::vector<double> &qWeights,                                // 3d (heads x embedDim x headDim) flattened as 1d
                                const std::vector<double> &kWeights,                                // 3d (heads x embedDim x headDim) flattened as 1d
                                const std::vector<double> &vWeights,                                // 3d (heads x embedDim x headDim) flattened as 1d
                                const std::vector<double> &oWeights,                                // 2d (embedDim x embedDim) flattened as 1d
                                int heads, int headDim,
                                const std::vector<double> &l1Weights,      // 2d (embedDim x hidden) flattened as 1d
                                const std::vector<double> &l1Biases,       // 1d
                                const std::vector<double> &l2Weights,      // 2d (hidden x embedDim) flattened as 1d
                                const std::vector<double> &l2Biases,       // 1d
                                const std::vector<double> &gammaAttention, // 1d
                                const std::vector<double> &gammaMLP,       // 1d
                                const std::vector<double> &betaAttention,  // 1d
                                const std::vector<double> &betaMLP,        // 1d
                                double epsilonAttention,
                                double epsilonMLP)
{
    std::vector<double> layerNormedEmbeddings(numTokens * embedDim); // 2d (numTokens x embedDim) flattened as 1d

    for (int i = 0; i < numTokens; i++)
    {
        std::vector<double> tokenEmbedding(embeddings.begin() + i * embedDim, embeddings.begin() + (i + 1) * embedDim); // 1d
        auto normed = LayerNorm(tokenEmbedding, gammaAttention, betaAttention, epsilonAttention);                       // 1d
        for (int j = 0; j < embedDim; j++)
            layerNormedEmbeddings[i * embedDim + j] = normed[j];
    }

    auto attentionResult = MultiHeadAttention(layerNormedEmbeddings, numTokens, embedDim, qWeights, kWeights, vWeights, oWeights, heads, headDim); // 2d (numTokens x embedDim) flattened as 1d

    std::vector<double> normedForMLP(numTokens * embedDim); // 2d (numTokens x embedDim) flattened as 1d

    for (int i = 0; i < numTokens; i++)
    {
        std::vector<double> origToken(embeddings.begin() + i * embedDim, embeddings.begin() + (i + 1) * embedDim);           // 1d
        std::vector<double> attnToken(attentionResult.begin() + i * embedDim, attentionResult.begin() + (i + 1) * embedDim); // 1d

        auto withResidual = AddVectors(origToken, attnToken); // 1d
        for (int j = 0; j < embedDim; j++)
            attentionResult[i * embedDim + j] = withResidual[j];

        auto normed = LayerNorm(withResidual, gammaMLP, betaMLP, epsilonMLP); // 1d
        for (int j = 0; j < embedDim; j++)
            normedForMLP[i * embedDim + j] = normed[j];
    }

    auto MLPResult = MLP(normedForMLP, numTokens, embedDim, l1Weights, l1Biases, l2Weights, l2Biases); // 2d (numTokens x embedDim) flattened as 1d

    for (int i = 0; i < numTokens; i++)
    {
        std::vector<double> attnToken(attentionResult.begin() + i * embedDim, attentionResult.begin() + (i + 1) * embedDim); // 1d
        std::vector<double> mlpToken(MLPResult.begin() + i * embedDim, MLPResult.begin() + (i + 1) * embedDim);              // 1d
        auto withResidual = AddVectors(attnToken, mlpToken);                                                                 // 1d
        for (int j = 0; j < embedDim; j++)
            MLPResult[i * embedDim + j] = withResidual[j];
    }

    return MLPResult; // 2d (numTokens x embedDim) flattened as 1d
}

std::vector<double> GPT()
{
    //
}