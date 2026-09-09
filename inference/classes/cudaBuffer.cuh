#pragma once

#include <cassert>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>

template <typename T>
struct CudaBuffer { // device memory; owning or non-owning view
  T *ptr = nullptr;
  size_t n = 0;
  bool owns = true;

  CudaBuffer() = default;

  explicit CudaBuffer(size_t count) : n(count), owns(true) {
    if (count == 0)
      return;
    cudaError_t err = cudaMalloc(&ptr, count * sizeof(T));
    if (err != cudaSuccess) {
      std::cout << "cudaMalloc failed: " << cudaGetErrorString(err);
      std::exit(-1);
    }
  }

  // Allocate device buffer and copy host data into it.
  CudaBuffer(const T *host, size_t count) : CudaBuffer(count) {
    if (count == 0)
      return;
    cudaError_t err =
        cudaMemcpy(ptr, host, count * sizeof(T), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      std::cout << "cudaMemcpy H2D failed: " << cudaGetErrorString(err);
      std::exit(-1);
    }
  }

  ~CudaBuffer() {
    if (owns && ptr) {
      cudaFree(ptr);
    }
    ptr = nullptr;
    n = 0;
    owns = true;
  }

  CudaBuffer(CudaBuffer &&other) noexcept
      : ptr(other.ptr), n(other.n), owns(other.owns) {
    other.ptr = nullptr;
    other.n = 0;
    other.owns = true;
  }

  CudaBuffer &operator=(CudaBuffer &&other) noexcept {
    if (this != &other) {
      if (owns && ptr)
        cudaFree(ptr);
      ptr = other.ptr;
      n = other.n;
      owns = other.owns;
      other.ptr = nullptr;
      other.n = 0;
      other.owns = true;
    }
    return *this;
  }

  CudaBuffer(const CudaBuffer<T> &) = delete;
  CudaBuffer &operator=(const CudaBuffer<T> &) = delete;

  // Non-owning view into [offset, offset+count).
  CudaBuffer slice(size_t offset, size_t count) const {
    assert(offset + count <= n);
    CudaBuffer view;
    view.ptr = ptr + offset;
    view.n = count;
    view.owns = false;
    return view;
  }

  void zero() {
    if (ptr && n)
      cudaMemset(ptr, 0, n * sizeof(T));
  }

  void copyFrom(const CudaBuffer<T> &src, size_t dstOffset = 0) {
    assert(dstOffset + src.n <= n);
    cudaMemcpy(ptr + dstOffset, src.ptr, src.n * sizeof(T),
               cudaMemcpyDeviceToDevice);
  }

  void copyToHost(T *host) const {
    cudaMemcpy(host, ptr, n * sizeof(T), cudaMemcpyDeviceToHost);
  }
};
