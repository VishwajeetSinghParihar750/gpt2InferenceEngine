#pragma once

#include "../constants.hh"
#include "cudaBuffer.cuh"
#include <fstream>
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
