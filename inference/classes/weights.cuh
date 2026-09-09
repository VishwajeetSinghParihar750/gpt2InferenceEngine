#pragma once

#include "../constants.hh"
#include "../kernels/transpose.cuh"
#include "cudaBuffer.cuh"
#include <fstream>
#include <string>
#include <utility>
#include <vector>

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

struct TransformerWeights {

  std::vector<BlockWeights> weights;
  LnWeights ln_f;
  CudaBuffer<double> wte;

  TransformerWeights() : weights(N_LAYER) {

    std::cout << "Loading weights ... " << std::endl;

    constexpr size_t D = N_EMBD;
    constexpr size_t HIDDEN = MLP_HIDDEN;
    constexpr size_t FINAL_LN_SIZE = D;
    constexpr size_t WTE_SIZE = static_cast<size_t>(VOCAB_SIZE) * D;
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

    ln_f.gamma = copyToDevice(finalLayerNormWeights, FINAL_LN_SIZE);
    ln_f.beta = copyToDevice(finalLayerNormBiases, FINAL_LN_SIZE);
    wte = copyToDevice(wteHost, WTE_SIZE);

    delete[] finalLayerNormWeights;
    delete[] finalLayerNormBiases;
    delete[] wteHost;
  }
};
