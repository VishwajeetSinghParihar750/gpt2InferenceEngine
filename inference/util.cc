#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cassert>

double GeluNew(double x)
{
    return .5 * x * (1 + tanh(sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x)));
}

std::vector<double> LayerNorm(std::vector<double> originalEmbedding, std::vector<double> gamma, std::vector<double> beta, double epsilon)
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

    //
    std::vector<double> output(len);
    for (int i = 0; i < len; i++)
    {
        double normalizedValue = ((originalEmbedding[i] - mean) / modifiedStandardDeviation);
        output[i] = normalizedValue * gamma[i] + beta[i];
    }

    return output;
}

std::vector<double> AddVectors(std::vector<double> a, const std::vector<double> &b)
{
    for (int i = 0; i < a.size(); i++)
        a[i] += b[i];
    return a;
}

std::vector<double> SoftMax(std::vector<double> input)
{
    int len = input.size();
    auto max = *std::max_element(input.begin(), input.end());

    std::vector<double> softmax(len);
    for (int i = 0; i < len; i++)
    {
        softmax[i] = exp(input[i] - max);
    }

    auto sum = std::accumulate(softmax.begin(), softmax.end(), 0.0);

    for (auto &i : softmax)
        i /= sum;

    return softmax;
}

double DotProduct(const std::vector<double> &a, const std::vector<double> &b)
{
    int len = a.size();
    double output = 0;
    for (int i = 0; i < len; i++)
        output += a[i] * b[i];
    return output;
}

std::vector<std::vector<double>> Transpose(const std::vector<std::vector<double>> &a)
{
    int n = a.size(), m = a[0].size();
    std::vector<std::vector<double>> result(m, std::vector<double>(n));
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            result[i][j] = a[j][i];
        }
    }
    return result;
}

std::vector<std::vector<double>> MatMul(const std::vector<std::vector<double>> &a, const std::vector<std::vector<double>> &b)
{
    assert(a[0].size() == b.size());

    std::vector<std::vector<double>> result(a.size(), std::vector<double>(b[0].size(), 0));

    for (int i = 0; i < a.size(); i++)
    {
        for (int j = 0; j < b[0].size(); j++)
        {
            for (int k = 0; k < b.size(); k++)
            {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    return result;
}

std::vector<std::vector<double>> Attention(std::vector<std::vector<double>> embeddings,
                                           std::vector<std::vector<double>> qWeights,
                                           std::vector<std::vector<double>> kWeights,
                                           std::vector<std::vector<double>> vWeights)
{
    int dimensions = qWeights[0].size();

    int en = embeddings.size(), em = embeddings[0].size();
    int qn = qWeights.size(), qm = qWeights[0].size();
    int kn = qWeights.size(), km = qWeights[0].size();
    int vn = qWeights.size(), vm = qWeights[0].size();

    assert(em == qn && em == kn && em == vn);

    auto qProjections = MatMul(embeddings, qWeights);
    auto kProjections = MatMul(embeddings, kWeights);
    auto vProjections = MatMul(embeddings, vWeights);

    // qk
    auto qkTranspose = MatMul(qProjections, Transpose(kProjections));

    // scale it down
    double dimensionsRoot = sqrt(dimensions);
    for (auto &i : qkTranspose)
    {
        for (auto &j : i)
        {
            j /= dimensionsRoot;
        }
    }

    // convert to probability distibution with softmax
    for (int i = 0; i < en; i++)
        qkTranspose[i] = SoftMax(qkTranspose[i]);
    // qkv
    return MatMul(qkTranspose, vProjections);
}

std::vector<std::vector<double>> MultiHeadAttention(std::vector<std::vector<double>> embeddings,
                                                    std::vector<std::vector<std::vector<double>>> qWeights,
                                                    std::vector<std::vector<std::vector<double>>> kWeights,
                                                    std::vector<std::vector<std::vector<double>>> vWeights,
                                                    std::vector<std::vector<double>> oWeights // output Weights
)
{

    int heads = qWeights.size();

    assert(heads == kWeights.size() && heads == vWeights.size());
    assert(oWeights.size() == embeddings[0].size());

    std::vector<std::vector<double>> result(embeddings.size(), std::vector<double>(embeddings[0].size()));

    int offset = embeddings[0].size() / heads;

    for (int i = 0; i < heads; i++)
    {
        auto curResult = Attention(embeddings, qWeights[i], kWeights[i], vWeights[i]);
        for (int j = 0; j < curResult.size(); j++)
        {
            for (int k = 0; k < curResult[0].size(); k++)
            {
                result[j][offset * i + k] = curResult[j][k];
            }
        }
    }

    return MatMul(result, oWeights);
}

std::vector<double> ForwardPass(const std::vector<double> &weights, const std::vector<double> &biases, const std::vector<double> &inputs, bool gelu = false)
{
    //
    int countNeurons = biases.size();
    std::vector<double> output = biases;

    for (int i = 0; i < countNeurons; i++)
    {
        for (int j = 0; j < inputs.size(); j++)
            output[i] += weights[i * inputs.size() + j] * inputs[j];

        if (gelu)
            output[i] = GeluNew(output[i]);
    }
    return output;
}

std::vector<std::vector<double>> MLP(std::vector<std::vector<double>> embeddings,
                                     std::vector<double> l1Weights, // flattened wieghts
                                     std::vector<double> l1Biases,
                                     std::vector<double> l2Weights, // flattened weights
                                     std::vector<double> l2Biases)
{
    int n = embeddings.size();
    int dimensions = embeddings[0].size();

    assert(l1Weights.size() == dimensions * l1Biases.size());

    std::vector<std::vector<double>> result(n);

    for (int i = 0; i < n; i++)
    {
        result[i] = ForwardPass(l1Weights, l1Biases, embeddings[i], true);
        result[i] = ForwardPass(l2Weights, l2Biases, result[i]);
    }

    return result;
}

std::vector<std::vector<double>> Transformer(std::vector<std::vector<double>> embeddings,
                                             std::vector<std::vector<std::vector<double>>> qWeights,
                                             std::vector<std::vector<std::vector<double>>> kWeights,
                                             std::vector<std::vector<std::vector<double>>> vWeights,
                                             std::vector<std::vector<double>> oWeights, // output Weights
                                             std::vector<double> l1Weights,             // flattened wieghts
                                             std::vector<double> l1Biases,
                                             std::vector<double> l2Weights, // flattened weights
                                             std::vector<double> l2Biases,
                                             std::vector<double> gammaAttention,
                                             std::vector<double> gammaMLP,
                                             std::vector<double> betaAttention,
                                             std::vector<double> betaMLP,
                                             double epsilonAttention,
                                             double epsilonMLP)
{

    int n = embeddings.size();
    std::vector<std::vector<double>> layerNormedEmbeddings(n);

    // layernorm for attention
    for (int i = 0; i < n; i++)
        layerNormedEmbeddings[i] = LayerNorm(embeddings[i], gammaAttention, betaAttention, epsilonAttention);

    // attention
    auto attentionResult = MultiHeadAttention(layerNormedEmbeddings, qWeights, kWeights, vWeights, oWeights);

    for (int i = 0; i < n; i++)
    {

        // residual connections
        attentionResult[i] = AddVectors(embeddings[i], attentionResult[i]);

        // layernorm for mlp
        layerNormedEmbeddings[i] = LayerNorm(attentionResult[i], gammaMLP, betaMLP, epsilonMLP);
    }

    // mlp
    auto MLPResult = MLP(layerNormedEmbeddings, l1Weights, l1Biases, l2Weights, l2Biases);

    // residual connections
    for (int i = 0; i < n; i++)
        MLPResult[i] = AddVectors(attentionResult[i], MLPResult[i]);

    return MLPResult;
}

std::vector<double> GPT()
{
    //
}