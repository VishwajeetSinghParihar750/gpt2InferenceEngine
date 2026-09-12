#pragma once

#include "cudaBuffer.cuh"
#include <cassert>

struct KvCache {
  CudaBuffer<double> k, v;
};

class gpt2KvCache {

  KvCache cache[12];

public:
  const KvCache &getKvCache(int transformerIndex) {

    assert(transformerIndex < 12);

    return cache[transformerIndex];
  }
};