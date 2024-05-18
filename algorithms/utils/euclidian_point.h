// This code is part of the Problem Based Benchmark Suite (PBBS)
// Copyright (c) 2011 Guy Blelloch and the PBBS team
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include <algorithm>
#include <iostream>

#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/internal/file_map.h"
#include "../bench/parse_command_line.h"
#include "NSGDist.h"

#include "../bench/parse_command_line.h"
#include "types.h"
// #include "common/time_loop.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

float euclidian_distance_(const uint8_t *p, const uint8_t *q, unsigned d) {
  int result = 0;
  for (int i = 0; i < d; i++) {
    result += ((int32_t)((int16_t)q[i] - (int16_t)p[i])) *
      ((int32_t)((int16_t)q[i] - (int16_t)p[i]));
  }
  return (float)result;
}

float euclidian_distance(const uint8_t *p, const uint8_t *q, unsigned d) {
  int32_t result = 0;
  for (int i = 0; i < d; i++) {
    int32_t qi = (int32_t) p[i];
    int32_t pi = (int32_t) q[i];
    result += (qi - pi) * (qi - pi);
  }
  return (float)result;
}

float euclidian_distance(const uint16_t *p, const uint16_t *q, unsigned d) {
  int64_t result = 0;
  for (int i = 0; i < d; i++) {
    int32_t qi = (int32_t) p[i];
    int32_t pi = (int32_t) q[i];
    result += (qi - pi) * (qi - pi);
  }
  return (float) (result >> 8);
}

float euclidian_distance(const int8_t *p, const int8_t *q, unsigned d) {
  int result = 0;
  for (int i = 0; i < d; i++) {
    result += ((int32_t)((int16_t)q[i] - (int16_t)p[i])) *
      ((int32_t)((int16_t)q[i] - (int16_t)p[i]));
  }
  return (float)result;
}

float euclidian_distance(const float *p, const float *q, unsigned d) {
  efanna2e::DistanceL2 distfunc;
  return distfunc.compare(p, q, d);
}

template<typename T, long range=(1l << sizeof(T)*8) - 1>
struct Euclidian_Point {
  using distanceType = float;

  struct parameters {
    float slope;
    int32_t offset;
    int dims;
    parameters() : slope(0), offset(0), dims(0) {}
    parameters(int dims) : slope(0), offset(0), dims(dims) {}
    parameters(float min_val, float max_val, int dims)
      : slope(range / (max_val - min_val)),
        offset((int32_t) round(min_val * slope)),
        dims(dims) {}
  };

  static distanceType d_min() {return 0;}
  static bool is_metric() {return true;}
  T operator[](long i) const {return *(values + i);}

  float distance(const Euclidian_Point<T>& x) const {
    return euclidian_distance(this->values, x.values, params.dims);
  }

  void prefetch() const {
    int l = (params.dims * sizeof(T) - 1)/64 + 1;
    for (int i=0; i < l; i++)
      __builtin_prefetch((char*) values + i* 64);
  }

  long id() const {return id_;}

  Euclidian_Point() : values(nullptr), id_(-1), params(0) {}

  Euclidian_Point(T* values, long id, parameters params)
    : values(values), id_(id), params(params) {}

  bool operator==(const Euclidian_Point<T>& q) const {
    for (int i = 0; i < params.dims; i++) {
      if (values[i] != q.values[i]) {
        return false;
      }
    }
    return true;
  }

  bool same_as(const Euclidian_Point<T>& q){
    return values == q.values;
  }
  
  template <typename Point>
  static void translate_point(T* values, const Point& p, const parameters& params) {
    float slope = params.slope;
    int32_t offset = params.offset;
    for (int j = 0; j < params.dims; j++) {
      auto x = p[j];
      int64_t r = (int64_t) (std::round(x * slope)) - offset;
      if (r < 0 || r > range) {
        std::cout << "out of range: " << r << ", " << range << std::endl;
        abort();
      }
      values[j] = (T) r;
    }
  }
  
  template <typename PR>
  static parameters generate_parameters(const PR& pr) {
    long n = pr.size();
    int dims = pr.dimension();
    parlay::sequence<typename PR::T> mins(n);
    parlay::sequence<typename PR::T> maxs(n);
    parlay::parallel_for(0, n, [&] (long i) {
      mins[i] = 0.0;
      maxs[i] = 0.0;
      for (int j = 0; j < dims; j++) {
        mins[i]= std::min(mins[i], pr[i][j]);
        maxs[i]= std::max(maxs[i], pr[i][j]);}});
    float min_val = *parlay::min_element(mins);
    float max_val = *parlay::max_element(maxs);
    //std::cout << min_val << ", " << max_val << std::endl;
    return parameters(min_val, max_val, dims);
  }

  parameters params;

private:
  T* values;
  long id_;
};
