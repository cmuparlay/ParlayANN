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

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <immintrin.h>

  float mips_distance(const uint8_t *p, const uint8_t *q, unsigned d) {
    int result = 0;
    for (int i = 0; i < (int) d; i++) {
      result += ((int32_t)q[i]) * ((int32_t)p[i]);
    }
    return -((float)result);
  }

  float mips_distance(const int8_t *p, const int8_t *q, unsigned d) {
    int result = 0;
    for (int i = 0; i < (int) d; i++) {
      result += ((int32_t)q[i]) * ((int32_t)p[i]);
    }
    return -((float)result);
  }


#ifdef __AVX512F__
  float mips_distance(const float *p, const float *q, unsigned d) {
    __m512 sum = _mm512_setzero_ps();
    for (int i = 0; i < d - 15; i += 16) {
      __m512 p_vec = _mm512_loadu_ps(p + i);
      __m512 q_vec = _mm512_loadu_ps(q + i);
      sum = _mm512_add_ps(sum, _mm512_mul_ps(p_vec, q_vec));
    }

    if (d % 16 != 0) {
      __mmask16 mask = (1 << (d % 16)) - 1;
      __m512 p_vec = _mm512_maskz_loadu_ps(mask, p + d - d % 16);
      __m512 q_vec = _mm512_maskz_loadu_ps(mask, q + d - d % 16);
      sum = _mm512_maskz_add_ps(mask, sum, _mm512_maskz_mul_ps(mask, p_vec, q_vec));
    }

    float result[16];
    _mm512_storeu_ps(result, sum);

    return - (result[0] + result[1] + result[2] + result[3] + result[4] + result[5] + result[6] + result[7] + result[8] + result[9] + result[10] + result[11] + result[12] + result[13] + result[14] + result[15]);
  }
#else
  float mips_distance(const float *p, const float *q, unsigned d) {
    float result = 0;
    for (int i = 0; i < (int) d; i++) {
      result += (q[i]) * (p[i]);
    }
    return -result;
  }
#endif

template<typename T>
struct Mips_Point {
  using distanceType = float; 
  
  static bool is_metric() {return false;}

  float distance(Mips_Point<T> x) {
    return mips_distance(this->values, x.values, d);
  }

  void prefetch() {
    int l = (aligned_d * sizeof(T))/64;
    for (int i=0; i < l; i++)
      __builtin_prefetch((char*) values + i* 64);
  }

  long id() {return id_;}

  Mips_Point(const T* values, unsigned int d, unsigned int ad, long id)
    : values(values), d(d), aligned_d(ad), id_(id) {}

  bool operator==(Mips_Point<T> q){
    for (int i = 0; i < d; i++) {
      if (values[i] != q.values[i]) {
        return false;
      }
    }
    return true;
  }

  std::string to_string() {
    std::string s = "";
    for (int i = 0; i < d; i++) {
      s += std::to_string(values[i]) + " ";
    }
    return s;
  }

  T* get() {return const_cast<T*>(values);}

private:
  const T* values;
  unsigned int d;
  unsigned int aligned_d;
  long id_;
};