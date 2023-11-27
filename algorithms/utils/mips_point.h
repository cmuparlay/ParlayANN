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


  float mips_distance(const uint8_t *p, const uint8_t *q, unsigned d) {
    int result = 0;
    for (int i = 0; i < d; i++) {
      result += ((int32_t)q[i]) * ((int32_t)p[i]);
    }
    return -((float)result);
  }

  float mips_distance(const int8_t *p, const int8_t *q, unsigned d) {
    int result = 0;
    for (int i = 0; i < d; i++) {
      result += ((int32_t)q[i]) * ((int32_t)p[i]);
    }
    return -((float)result);
  }

  float mips_distance(const float *p, const float *q, unsigned d) {
    float result = 0;
    for (int i = 0; i < d; i++) {
      result += (q[i]) * (p[i]);
    }
    return -result;
  }

template<typename T>
struct Mips_Point {
  using distanceType = float; 
  template<class C> friend struct Quantized_Mips_Point;
  
  static distanceType d_min() {return -std::numeric_limits<float>::max();}
  static bool is_metric() {return false;}
  T operator [](long i) const {return *(values + i);}

  float distance(const Mips_Point &x) const {
    return mips_distance(this->values, x.values, d);
  }

  void prefetch() const {
    int l = (aligned_d * sizeof(T))/64;
    for (int i=0; i < l; i++)
      __builtin_prefetch((char*) values + i* 64);
  }

  long id() const {return id_;}

  Mips_Point()
    : values(nullptr), d(0), aligned_d(0), id_(-1) {}

  Mips_Point(const T* values, unsigned int d, unsigned int ad, long id)
    : values(values), d(d), aligned_d(ad), id_(id) {}

  bool operator==(const Mips_Point &q) const {
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

template<typename T>
float quantized_mips_distance(const float* q, const T* p, unsigned d, float max_coord, float min_coord){
  float result = 0;
  uint T_bits = sizeof(T)*8;
  float maxval = static_cast<float>(static_cast<T>((((size_t) 1) << T_bits)-1));
  float delta = max_coord - min_coord;
  float mult = delta/maxval;
  float dc;
  for(int i=0; i<d; i++){
    dc = static_cast<float>(p[i])*mult + min_coord;
    result += dc*q[i];
  }
  return result;
}

template<typename T>
float quantized_mips_distance(const T* q, const T* p, unsigned d, float max_coord, float min_coord){
  float result = 0;
  uint T_bits = sizeof(T)*8;
  float maxval = static_cast<float>(static_cast<T>((((size_t) 1) << T_bits)-1));
  float delta = max_coord - min_coord;
  float mult = delta/maxval;
  float dc;
  float dcq;
  for(int i=0; i<d; i++){
    dc = static_cast<float>(p[i])*mult + min_coord;
    dcq = static_cast<float>(q[i])*mult + min_coord;
    result += dc*dcq;
  }
  return result;
}


template<typename T>
struct Quantized_Mips_Point{
    using distanceType = float; 
  
  static distanceType d_min() {return -std::numeric_limits<float>::max();}
  static bool is_metric() {return false;}
  
  T operator [] (long j) const {if(j >= d) abort(); return *(values+j);}


  float distance(const Mips_Point<float> &x) const {return quantized_mips_distance(x.values, this->values, d, max_coord, min_coord);}

  float distance(const Quantized_Mips_Point &x) const {return quantized_mips_distance(x.values, this->values, d, max_coord, min_coord);}

  void prefetch() const {
    int l = (aligned_d * sizeof(T))/64;
    for (int i=0; i < l; i++)
      __builtin_prefetch((char*) values + i* 64);
  }

  long id() const {return id_;}

  Quantized_Mips_Point(const T* values, unsigned int d, unsigned int ad, long id, float max_coord, float min_coord)
    : values(values), d(d), aligned_d(ad), id_(id), max_coord(max_coord), min_coord(min_coord) {;
    }

  bool operator==(const Quantized_Mips_Point &q) const {
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
  float min_coord;
  float max_coord;
};
