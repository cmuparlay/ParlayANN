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

float euclidian_distance(const uint8_t *p, const uint8_t *q, unsigned d) {
  int result = 0;
  for (int i = 0; i < d; i++) {
    result += ((int32_t)((int16_t)q[i] - (int16_t)p[i])) *
      ((int32_t)((int16_t)q[i] - (int16_t)p[i]));
  }
  return (float)result;
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

template<typename T>
struct Euclidian_Point {
  using distanceType = float;

  static distanceType d_min() {return 0;}
  static bool is_metric() {return true;}
  T operator[](long i) const {return *(values + i);}

  float distance(const Euclidian_Point &x) const {
    return euclidian_distance(this->values, x.values, d);
  }

  void prefetch() const {
    int l = (aligned_d * sizeof(T))/64;
    for (int i=0; i < l; i++)
      __builtin_prefetch((char*) values + i* 64);
  }

  long id() const {return id_;}

  Euclidian_Point()
    : values(nullptr), d(0), aligned_d(0), id_(-1) {}

  Euclidian_Point(const T* values, unsigned int d, unsigned int ad, long id)
    : values(values), d(d), aligned_d(ad), id_(id) {}

  bool operator==(const Euclidian_Point q) const {
    for (int i = 0; i < d; i++) {
      if (values[i] != q.values[i]) {
        return false;
      }
    }
    return true;
  }

private:
  const T* values;
  unsigned int d;
  unsigned int aligned_d;
  long id_;
};

