// TQ8: randomized 8-bit scalar quantization with one-to-one distance
// support. Algorithm follows TurboQuant (https://arxiv.org/abs/2504.19874),
// restricted to the 8-bit base case and symmetric (encoded-vs-encoded)
// scoring:
//
//   1. Multiply each coordinate by a shared random sign vector.
//   2. Normalize to unit L2.
//   3. Apply an in-place Fast Walsh-Hadamard Transform on a power-of-2 padded
//      buffer (zero-padded input). The dot product is preserved through
//      padding + FWHT, so this stands in for the subspace-decomposition +
//      orthogonal-mixer rotation used by the reference implementation.
//   4. Adaptive int8 quantization: code[i] = round(127 * v[i] / max_j |v[j]|).
//
// Per-point storage is `[float scale][float sqr_norm][int8 code[padded_dim]]`,
// where `scale = ||x|| / sqrt(sum code[i]^2)` and `sqr_norm = ||x||^2`. The
// dot product of two original vectors is approximated by
// `scale_x * scale_y * <code_x, code_y>`, and the squared Euclidean distance
// by `sqr_norm_x + sqr_norm_y - 2 * scale_x * scale_y * <code_x, code_y>`.
//
// Two metric variants are provided by templating on `IsMips`:
//   `TQ8_Point`       — Euclidean: distance(x,y) = ||x-y||²
//   `TQ8_Mips_Point`  — MIPS:      distance(x,y) = -<x,y>
// `sqr_norm` is stored in both cases (unused by MIPS) to keep one layout.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <random>
#include <unordered_map>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <x86intrin.h>
#endif

namespace parlayANN {

namespace tq8_internal {

// Smallest power of 2 >= n (n >= 1).
inline int next_pow2(int n) {
  int p = 1;
  while (p < n) p <<= 1;
  return p;
}

// Process-static registry of random ±1 sign vectors keyed by padded
// dimension. The signs are deterministic from the fixed kSignsSeed below, so
// every parameters object with the same padded_dim shares the same backing
// storage. Returning a raw pointer (lifetime = process) lets `parameters` be a
// trivially-copyable POD; without this, the existing `PointRange::operator[]`
// would copy a shared_ptr per `Points[i]`, which collapses parallel build
// throughput on shared atomic refcount traffic.
inline const float* get_signs(int padded_dim, uint64_t seed) {
  static std::mutex mu;
  static std::unordered_map<uint64_t, std::vector<float>*> cache;
  uint64_t key = (static_cast<uint64_t>(padded_dim) << 32) ^ seed;
  std::lock_guard<std::mutex> lk(mu);
  auto it = cache.find(key);
  if (it != cache.end()) return it->second->data();
  auto* v = new std::vector<float>(padded_dim);
  std::mt19937_64 gen(seed);
  std::uniform_int_distribution<int> bit(0, 1);
  for (int i = 0; i < padded_dim; ++i) (*v)[i] = bit(gen) ? 1.0f : -1.0f;
  cache.emplace(key, v);
  return v->data();
}

// In-place Fast Walsh-Hadamard Transform; n must be a power of 2.
inline void fwht_inplace(float* data, int n) {
  for (int len = 1; len < n; len <<= 1) {
    for (int i = 0; i < n; i += 2 * len) {
      for (int j = 0; j < len; ++j) {
        float u = data[i + j];
        float v = data[i + j + len];
        data[i + j] = u + v;
        data[i + j + len] = u - v;
      }
    }
  }
}

// Sum a[i] * b[i] over [0, D) into int32. D should be a multiple of 32 in the
// common path; a scalar tail handles any remainder.
inline int32_t int8_dot(const int8_t* a, const int8_t* b, int D) {
#if defined(__AVX2__)
  __m256i acc = _mm256_setzero_si256();
  int i = 0;
  for (; i + 32 <= D; i += 32) {
    __m256i av = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
    __m256i bv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
    __m256i a_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(av));
    __m256i a_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(av, 1));
    __m256i b_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(bv));
    __m256i b_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bv, 1));
    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a_lo, b_lo));
    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(a_hi, b_hi));
  }
  __m128i lo = _mm256_castsi256_si128(acc);
  __m128i hi = _mm256_extracti128_si256(acc, 1);
  __m128i s = _mm_add_epi32(lo, hi);
  s = _mm_hadd_epi32(s, s);
  s = _mm_hadd_epi32(s, s);
  int32_t total = _mm_cvtsi128_si32(s);
  for (; i < D; ++i) total += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
  return total;
#else
  int32_t total = 0;
  for (int i = 0; i < D; ++i)
    total += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
  return total;
#endif
}

}  // namespace tq8_internal

template <bool IsMips>
struct TQ8_Point_T {
  using T = int8_t;
  using distanceType = float;
  using byte = uint8_t;

  static constexpr int header_bytes = 8;  // float scale + float sqr_norm
  static constexpr uint64_t kSignsSeed = 42;

  struct parameters {
    int dims = 0;        // original dimensionality
    int padded_dim = 0;  // smallest power of 2 >= dims (>= 1)
    // Raw pointer into a process-static sign registry. Both Points and
    // Query_Points share the same backing storage (keyed by padded_dim), so
    // copying `parameters` is trivial — important because every
    // `PointRange::operator[]` copies it into a new Point on the hot path.
    const float* signs = nullptr;

    int num_bytes() const { return header_bytes + padded_dim; }

    parameters() = default;
    explicit parameters(int dims_) : dims(dims_) {
      int n = std::max(1, dims_);
      padded_dim = tq8_internal::next_pow2(n);
      signs = tq8_internal::get_signs(padded_dim, kSignsSeed);
    }
  };

  static distanceType d_min() {
    return IsMips ? -std::numeric_limits<float>::max() : 0.0f;
  }
  static constexpr bool is_metric = !IsMips;

  TQ8_Point_T() : values(nullptr), id_(-1) {}
  TQ8_Point_T(byte* values, long id, const parameters& p)
      : values(values), id_(id), params(p) {}

  long id() const { return id_; }

  // Stub: the encoded bytes have no direct correspondence to original
  // coordinates, but returning the raw int8 lets other code that templates
  // over `p[i]` compile.
  int8_t operator[](long i) const {
    return *(reinterpret_cast<const int8_t*>(values + header_bytes) + i);
  }

  float scale() const { return load_float(values); }
  float sqr_norm() const { return load_float(values + 4); }
  const int8_t* code() const {
    return reinterpret_cast<const int8_t*>(values + header_bytes);
  }

  float distance(const TQ8_Point_T& x) const {
    const int D = params.padded_dim;
    const int32_t dot = tq8_internal::int8_dot(code(), x.code(), D);
    const float s = scale() * x.scale();
    const float dp = s * static_cast<float>(dot);  // estimate of <x, y>
    if constexpr (IsMips) {
      return -dp;
    } else {
      const float xx = sqr_norm();
      const float yy = x.sqr_norm();
      const float d = xx + yy - 2.0f * dp;
      return d < 0.0f ? 0.0f : d;
    }
  }

  float translate_distance(double r) const { return static_cast<float>(r); }

  void prefetch() const {
    int l = (params.num_bytes() - 1) / 64 + 1;
    for (int i = 0; i < l; ++i)
      __builtin_prefetch(reinterpret_cast<const char*>(values) + i * 64);
  }

  void normalize() {}

  bool same_as(const TQ8_Point_T& q) const { return values == q.values; }

  bool operator==(const TQ8_Point_T& q) const {
    if (params.padded_dim != q.params.padded_dim) return false;
    return std::memcmp(values, q.values,
                       header_bytes + params.padded_dim) == 0;
  }

  template <typename In_Point>
  static void translate_point(byte* out, const In_Point& p,
                              const parameters& params) {
    const int d = params.dims;
    const int D = params.padded_dim;

    thread_local std::vector<float> buf;
    buf.assign(D, 0.0f);

    double sqr_norm_d = 0.0;
    for (int j = 0; j < d; ++j) {
      float v = static_cast<float>(p[j]);
      buf[j] = v;
      sqr_norm_d += static_cast<double>(v) * v;
    }
    const float sqr_norm = static_cast<float>(sqr_norm_d);

    int8_t* code_out = reinterpret_cast<int8_t*>(out + header_bytes);

    if (sqr_norm == 0.0f) {
      store_float(out, 0.0f);
      store_float(out + 4, 0.0f);
      std::memset(code_out, 0, D);
      return;
    }

    const float inv_norm = 1.0f / std::sqrt(sqr_norm);
    const float* signs = params.signs;
    for (int j = 0; j < D; ++j) buf[j] = buf[j] * signs[j] * inv_norm;
    tq8_internal::fwht_inplace(buf.data(), D);

    float max_abs = 0.0f;
    for (int j = 0; j < D; ++j) max_abs = std::max(max_abs, std::abs(buf[j]));

    int64_t int8_sqr_norm = 0;
    if (max_abs == 0.0f) {
      std::memset(code_out, 0, D);
    } else {
      const float scale_factor = 127.0f / max_abs;
      for (int j = 0; j < D; ++j) {
        int v = static_cast<int>(std::lround(buf[j] * scale_factor));
        if (v < -127) v = -127;
        if (v > 127) v = 127;
        code_out[j] = static_cast<int8_t>(v);
        int8_sqr_norm += static_cast<int64_t>(v) * v;
      }
    }

    const float scale =
        int8_sqr_norm == 0
            ? 0.0f
            : std::sqrt(sqr_norm / static_cast<float>(int8_sqr_norm));
    store_float(out, scale);
    store_float(out + 4, sqr_norm);
  }

  template <typename PR>
  static parameters generate_parameters(const PR& pr) {
    return parameters(pr.dimension());
  }

  parameters params;

 private:
  byte* values;
  long id_;

  static float load_float(const byte* p) {
    float f;
    std::memcpy(&f, p, sizeof(float));
    return f;
  }
  static void store_float(byte* p, float f) {
    std::memcpy(p, &f, sizeof(float));
  }
};

using TQ8_Point = TQ8_Point_T<false>;
using TQ8_Mips_Point = TQ8_Point_T<true>;

}  // namespace parlayANN
