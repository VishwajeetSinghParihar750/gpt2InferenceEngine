#pragma once

#include <cassert>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>

inline void cudaCheck(cudaError_t err, const char *what) {
  if (err != cudaSuccess) {
    std::cout << what << ": " << cudaGetErrorString(err);
    std::exit(-1);
  }
}

template <typename T>
struct CudaBuffer { // device vector; owning buffer or non-owning view
  T *ptr = nullptr;
  size_t n = 0;   // size
  size_t cap = 0; // capacity (meaningful when owns)
  bool owns = true;

  CudaBuffer() = default;

  explicit CudaBuffer(size_t count) : n(count), cap(count), owns(true) {
    if (count == 0)
      return;
    cudaCheck(cudaMalloc(&ptr, count * sizeof(T)), "cudaMalloc failed");
  }

  // Allocate device buffer and copy host data into it.
  CudaBuffer(const T *host, size_t count) : CudaBuffer(count) {
    if (count == 0)
      return;
    cudaCheck(cudaMemcpy(ptr, host, count * sizeof(T), cudaMemcpyHostToDevice),
              "cudaMemcpy H2D failed");
  }

  ~CudaBuffer() { release(); }

  CudaBuffer(CudaBuffer &&other) noexcept
      : ptr(other.ptr), n(other.n), cap(other.cap), owns(other.owns) {
    other.ptr = nullptr;
    other.n = 0;
    other.cap = 0;
    other.owns = true;
  }

  CudaBuffer &operator=(CudaBuffer &&other) noexcept {
    if (this != &other) {
      release();
      ptr = other.ptr;
      n = other.n;
      cap = other.cap;
      owns = other.owns;
      other.ptr = nullptr;
      other.n = 0;
      other.cap = 0;
      other.owns = true;
    }
    return *this;
  }

  CudaBuffer(const CudaBuffer<T> &) = delete;
  CudaBuffer &operator=(const CudaBuffer<T> &) = delete;

  size_t size() const { return n; }
  T *data() { return ptr; }
  const T *data() const { return ptr; }

  void reserve(size_t newCap) {
    assert(owns && "reserve on non-owning view");
    if (newCap <= cap)
      return;
    T *newPtr = nullptr;
    cudaCheck(cudaMalloc(&newPtr, newCap * sizeof(T)), "cudaMalloc failed");
    if (ptr) {
      if (n) {
        cudaCheck(
            cudaMemcpy(newPtr, ptr, n * sizeof(T), cudaMemcpyDeviceToDevice),
            "cudaMemcpy D2D failed");
      }
      cudaFree(ptr);
    }
    ptr = newPtr;
    cap = newCap;
  }

  void resize(size_t newSize) {
    assert(owns && "resize on non-owning view");
    if (newSize > cap)
      reserve(growCapacity(newSize));
    n = newSize;
  }

  void clear() {
    assert(owns && "clear on non-owning view");
    n = 0;
  }

  // Non-owning view into [offset, offset+count).
  CudaBuffer slice(size_t offset, size_t count) const {
    assert(offset + count <= n);
    CudaBuffer view;
    view.ptr = ptr + offset;
    view.n = count;
    view.cap = count;
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

private:
  void release() {
    if (owns && ptr)
      cudaFree(ptr);
    ptr = nullptr;
    n = 0;
    cap = 0;
    owns = true;
  }

  size_t growCapacity(size_t minCap) const {
    size_t newCap = cap ? cap * 2 : 4;
    if (newCap < minCap)
      newCap = minCap;
    return newCap;
  }
};
