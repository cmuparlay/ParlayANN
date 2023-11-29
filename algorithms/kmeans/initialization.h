/* Initialization methods for k-means clustering. Deleted other initialization
 * methods for simplicity, see parlaykmeans if interested. */
#ifndef INITERS
#define INITERS

#include <string>
#include "parlay/parallel.h"
#include "parlay/primitives.h"

// Lazy start makes the first k points the first k centers
// Then assigns cyclically
template <typename T, typename CT, typename index_type>
struct Lazy {
  // ad is aligned dimension
  void operator()(T* v, size_t n, size_t d, size_t ad, size_t k, CT* c,
                  index_type* asg) {
    parlay::parallel_for(0, k, [&](size_t i) {
      for (size_t j = 0; j < d; j++) {
        c[i * ad + j] = v[i * ad + j];
      }
    });
    parlay::parallel_for(0, n, [&](size_t i) { asg[i] = i % k; });
  }

  std::string name() { return "Lazy"; }
};

#endif   // INITERS
