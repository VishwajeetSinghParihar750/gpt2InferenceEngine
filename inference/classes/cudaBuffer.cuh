#pragma once

#include <cassert>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

namespace detail {

inline void cudaCheck(cudaError_t err, const char *what) {
  if (err != cudaSuccess) {
    std::cout << what << ": " << cudaGetErrorString(err);
    std::exit(-1);
  }
}

} // namespace detail

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
    detail::cudaCheck(cudaMalloc(&ptr, count * sizeof(T)), "cudaMalloc failed");
  }

  // Allocate device buffer and copy host data into it.
  CudaBuffer(const T *host, size_t count) : CudaBuffer(count) {
    if (count == 0)
      return;
    detail::cudaCheck(
        cudaMemcpy(ptr, host, count * sizeof(T), cudaMemcpyHostToDevice),
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
  size_t capacity() const { return owns ? cap : n; }
  bool empty() const { return n == 0; }
  T *data() { return ptr; }
  const T *data() const { return ptr; }

  void reserve(size_t newCap) {
    assert(owns && "reserve on non-owning view");
    if (newCap <= cap)
      return;
    T *newPtr = nullptr;
    detail::cudaCheck(cudaMalloc(&newPtr, newCap * sizeof(T)),
                      "cudaMalloc failed");
    if (ptr) {
      if (n) {
        detail::cudaCheck(
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

  void resize(size_t newSize, const T &value) {
    assert(owns && "resize on non-owning view");
    size_t oldSize = n;
    resize(newSize);
    if (newSize > oldSize)
      fillDevice(ptr + oldSize, newSize - oldSize, value);
  }

  void clear() {
    assert(owns && "clear on non-owning view");
    n = 0;
  }

  void shrink_to_fit() {
    assert(owns && "shrink_to_fit on non-owning view");
    if (n == cap)
      return;
    if (n == 0) {
      if (ptr)
        cudaFree(ptr);
      ptr = nullptr;
      cap = 0;
      return;
    }
    T *newPtr = nullptr;
    detail::cudaCheck(cudaMalloc(&newPtr, n * sizeof(T)), "cudaMalloc failed");
    detail::cudaCheck(
        cudaMemcpy(newPtr, ptr, n * sizeof(T), cudaMemcpyDeviceToDevice),
        "cudaMemcpy D2D failed");
    cudaFree(ptr);
    ptr = newPtr;
    cap = n;
  }

  void push_back(const T &value) {
    assert(owns && "push_back on non-owning view");
    if (n + 1 > cap)
      reserve(growCapacity(n + 1));
    detail::cudaCheck(
        cudaMemcpy(ptr + n, &value, sizeof(T), cudaMemcpyHostToDevice),
        "cudaMemcpy H2D failed");
    ++n;
  }

  // Fill with count copies of value.
  void assign(size_t count, const T &value) {
    assert(owns && "assign on non-owning view");
    resize(count);
    if (count)
      fillDevice(ptr, count, value);
  }

  // Replace contents with host array [host, host+count).
  void assign(const T *host, size_t count) {
    assert(owns && "assign on non-owning view");
    resize(count);
    if (count == 0)
      return;
    detail::cudaCheck(
        cudaMemcpy(ptr, host, count * sizeof(T), cudaMemcpyHostToDevice),
        "cudaMemcpy H2D failed");
  }

  // Device-to-device assign from another buffer (or view).
  void assign(const CudaBuffer<T> &src) {
    assert(owns && "assign on non-owning view");
    assert(this != &src);
    resize(src.n);
    if (src.n == 0)
      return;
    detail::cudaCheck(
        cudaMemcpy(ptr, src.ptr, src.n * sizeof(T), cudaMemcpyDeviceToDevice),
        "cudaMemcpy D2D failed");
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

  static void fillDevice(T *dst, size_t count, const T &value) {
    if (count == 0)
      return;
    std::vector<T> host(count, value);
    detail::cudaCheck(
        cudaMemcpy(dst, host.data(), count * sizeof(T), cudaMemcpyHostToDevice),
        "cudaMemcpy H2D failed");
  }
};
