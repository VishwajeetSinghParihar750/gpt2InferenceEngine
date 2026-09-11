#pragma once

#include "cudaBuffer.cuh"

struct KvCache {
  CudaBuffer<double> k, v;
};