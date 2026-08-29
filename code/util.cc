#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>

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