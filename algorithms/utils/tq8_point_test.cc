// Standalone correctness test for TQ8_Point. Builds a small set of random
// vectors, encodes them, and checks that the TQ8-estimated squared L2
// distance is a low-error estimator of the true squared L2 distance, plus
// a few sanity properties (self-distance, symmetry).
//
// Build:
//   g++ -O3 -march=native -std=c++17 -I ../../parlaylib/include -pthread \
//       algorithms/utils/tq8_point_test.cc -o /tmp/tq8_test
// Run:
//   /tmp/tq8_test

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "tq8_point.h"

namespace {

using parlayANN::TQ8_Mips_Point;
using parlayANN::TQ8_Point;

// Trivial wrapper that lets translate_point iterate over a raw float vector.
struct FloatVec {
  const float* data;
  int dims;
  float operator[](int j) const { return data[j]; }
};

float true_sqr_l2(const float* a, const float* b, int d) {
  double s = 0.0;
  for (int i = 0; i < d; ++i) {
    double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    s += diff * diff;
  }
  return static_cast<float>(s);
}

float true_neg_dot(const float* a, const float* b, int d) {
  double s = 0.0;
  for (int i = 0; i < d; ++i)
    s += static_cast<double>(a[i]) * static_cast<double>(b[i]);
  return -static_cast<float>(s);
}

// Encode `n` random vectors of dimension `dims` and return:
//   - raw float storage,
//   - encoded byte storage (one block per point, sized to params.num_bytes()),
//   - parameters object.
template <class Point>
struct Encoded {
  int dims;
  int n;
  std::vector<float> raw;
  std::vector<uint8_t> codes;
  typename Point::parameters params;
};

template <class Point>
Encoded<Point> make_random(int dims, int n, uint64_t seed) {
  Encoded<Point> e;
  e.dims = dims;
  e.n = n;
  e.raw.resize(static_cast<size_t>(n) * dims);
  e.params = typename Point::parameters(dims);
  size_t per_point = e.params.num_bytes();
  e.codes.assign(static_cast<size_t>(n) * per_point, 0);

  std::mt19937_64 gen(seed);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  for (size_t i = 0; i < e.raw.size(); ++i) e.raw[i] = dist(gen);

  for (int i = 0; i < n; ++i) {
    FloatVec fv{&e.raw[static_cast<size_t>(i) * dims], dims};
    Point::translate_point(&e.codes[static_cast<size_t>(i) * per_point], fv,
                           e.params);
  }
  return e;
}

template <class Point>
Point point_at(const Encoded<Point>& e, int i) {
  size_t per_point = e.params.num_bytes();
  return Point(const_cast<uint8_t*>(&e.codes[static_cast<size_t>(i) * per_point]),
               i, e.params);
}

struct Stats {
  double mean_rel_err = 0;
  double max_rel_err = 0;
  double mean_abs_err = 0;
  double mean_true = 0;
  double mean_est = 0;
};

// `truth_fn(a, b, d)` produces the reference scalar (sqL2 or negDot).
// For MIPS we normalize relative error by mean |truth| since values can be
// near zero and even cross zero.
template <class Point, class TruthFn>
Stats run_pairs(const Encoded<Point>& e, int num_pairs, uint64_t seed,
                TruthFn truth_fn) {
  Stats s;
  std::mt19937_64 gen(seed);
  std::uniform_int_distribution<int> pick(0, e.n - 1);
  double abs_truth = 0;
  int counted = 0;
  for (int t = 0; t < num_pairs; ++t) {
    int i = pick(gen), j = pick(gen);
    if (i == j) continue;
    const float* xa = &e.raw[static_cast<size_t>(i) * e.dims];
    const float* xb = &e.raw[static_cast<size_t>(j) * e.dims];
    float truth = truth_fn(xa, xb, e.dims);
    float est = point_at(e, i).distance(point_at(e, j));
    double abs_err = std::abs(static_cast<double>(est - truth));
    s.mean_abs_err += abs_err;
    s.mean_true += truth;
    s.mean_est += est;
    abs_truth += std::abs(static_cast<double>(truth));
    ++counted;
  }
  s.mean_abs_err /= counted;
  s.mean_true /= counted;
  s.mean_est /= counted;
  // Mean relative error against mean |truth| — robust when truth straddles 0
  // (MIPS) or is uniformly positive (sqL2).
  s.mean_rel_err = s.mean_abs_err / std::max(abs_truth / counted, 1e-9);
  s.max_rel_err = s.mean_rel_err;  // single aggregate, max not meaningful here
  return s;
}

template <class Point>
void check_self_distance(const Encoded<Point>& e) {
  // For Euclidean: d(x,x) reduces to sqr_norm + sqr_norm - 2*scale^2*<c,c>
  //   which is exactly zero in real arithmetic but can have a couple of ULPs
  //   of float rounding error.
  // For MIPS: d(x,x) = -<x,x> ≈ -||x||² and equals -sqr_norm to within
  //   estimator error. We check that the estimate matches truth, not zero.
  for (int i = 0; i < std::min(e.n, 100); ++i) {
    Point p = point_at(e, i);
    float d = p.distance(p);
    float sn = p.sqr_norm();
    float expected = Point::is_metric ? 0.0f : -sn;
    if (std::abs(d - expected) > 1e-3f * std::max(sn, 1.0f)) {
      std::fprintf(stderr,
                   "self-distance too far from expected for i=%d: got %g, "
                   "expected %g (sn=%g)\n",
                   i, d, expected, sn);
      std::abort();
    }
  }
  std::printf("  self-distance check passed\n");
}

template <class Point>
void check_symmetry(const Encoded<Point>& e, int num_pairs, uint64_t seed) {
  std::mt19937_64 gen(seed);
  std::uniform_int_distribution<int> pick(0, e.n - 1);
  for (int t = 0; t < num_pairs; ++t) {
    int i = pick(gen), j = pick(gen);
    float dij = point_at(e, i).distance(point_at(e, j));
    float dji = point_at(e, j).distance(point_at(e, i));
    if (std::abs(dij - dji) > 1e-3f * std::max(std::abs(dij), 1.0f)) {
      std::fprintf(stderr, "symmetry violated (%d,%d): %g vs %g\n", i, j, dij,
                   dji);
      std::abort();
    }
  }
  std::printf("  symmetry check passed (%d pairs)\n", num_pairs);
}

template <class Point, class TruthFn>
void run_for_dims(const char* tag, int dims, TruthFn truth_fn) {
  std::printf("=== %s dims=%d (padded_dim=%d) ===\n", tag, dims,
              parlayANN::tq8_internal::next_pow2(std::max(1, dims)));
  Encoded<Point> e = make_random<Point>(dims, /*n=*/1000, /*seed=*/123);
  check_self_distance(e);
  check_symmetry(e, /*num_pairs=*/200, /*seed=*/7);

  Stats s = run_pairs(e, /*num_pairs=*/5000, /*seed=*/99, truth_fn);
  std::printf("  mean truth=%.3f, mean est=%.3f, bias=%.3f%%\n", s.mean_true,
              s.mean_est,
              100.0 * (s.mean_est - s.mean_true) /
                  std::max(std::abs(s.mean_true), 1e-9));
  std::printf("  mean rel err=%.3f%% (vs mean |truth|)\n",
              100.0 * s.mean_rel_err);
  if (s.mean_rel_err > 0.10) {
    std::fprintf(stderr, "FAIL: mean rel err > 10%% for %s dims=%d\n", tag,
                 dims);
    std::abort();
  }
  std::printf("  PASS\n");
}

}  // namespace

int main() {
  // Cover representative dims, including non-power-of-2 (which exercises the
  // zero-padding path), for both metric variants.
  for (int d : {32, 64, 96, 100, 128, 192, 256, 384, 768}) {
    run_for_dims<TQ8_Point>("Euclidean", d, true_sqr_l2);
    run_for_dims<TQ8_Mips_Point>("MIPS    ", d, true_neg_dot);
  }
  std::printf("\nAll tests passed.\n");
  return 0;
}
