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
#include "scalar_quantize.h"

#include "../bench/parse_command_line.h"
#include "types.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>


  float mips_distance(const uint8_t *p, const uint8_t *q, unsigned d) {
    int result = 0;
    for (int i = 0; i < (int) d; i++) {
      result += ((int64_t)q[i]) * ((int64_t)p[i]);
    }
    return -((float)result);
  }

  float mips_distance(const uint16_t *p, const uint16_t *q, unsigned d) {
    int result = 0;
    for (int i = 0; i < (int) d; i++) {
      result += ((int32_t)q[i]) * ((int32_t)p[i]);
    }
    return -(static_cast<float>(result));
  }

  float mips_distance(const int8_t *p, const int8_t *q, unsigned d) {
    int result = 0;
    for (int i = 0; i < (int) d; i++) {
      result += ((int32_t)q[i]) * ((int32_t)p[i]);
    }
    return -((float)result);
  }

  float mips_distance(const float *p, const float *q, unsigned d) {
    float result = 0;
    for (int i = 0; i < (int) d; i++) {
      result += (q[i]) * (p[i]);
    }
    return -result;
  }

template<typename T>
struct Mips_Point {
  using distanceType = float; 
  template<class C, class D> friend struct Quantized_Mips_Point;
  friend struct T2I_Point;
  
  static distanceType d_min() {return -std::numeric_limits<float>::max();}
  static bool is_metric() {return false;}
  T operator [] (long j) {if(j >= d) abort(); return *(values+j);}

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


private:
  const T* values;
  unsigned int d;
  unsigned int aligned_d;
  long id_;
};

template<typename T, typename U>
struct Quantized_Mips_Point{
  using distanceType = float; 
  
  static distanceType d_min() {return -std::numeric_limits<float>::max();}
  static bool is_metric() {return false;}
  
  T operator [] (long j) {if(j >= quantized_d) abort(); return *(values+j);}


  float distance(Mips_Point<U> x) {
    return mips_distance(decode<T, U>(this->values, d, quantized_d, max_coord, min_coord, bits).begin(), x.values, d);
  }

  float distance(Quantized_Mips_Point<T, U> x){
    return mips_distance(decode<T, U>(this->values, d, quantized_d, max_coord, min_coord, bits).begin(), decode<T, U>(x.values, d, quantized_d, max_coord, min_coord, bits).begin(), d);
  }

  //hack for compatibility
  float distance(Euclidian_Point<uint8_t> x) {return 0;}
  float distance(Euclidian_Point<int8_t> x) {return 0;}
  float distance(Euclidian_Point<float> x) {return 0;}
  float distance(Mips_Point<uint8_t> x) {return 0;}
  float distance(Mips_Point<int8_t> x) {return 0;}

  void prefetch() {
    int l = (aligned_d * sizeof(T))/64;
    for (int i=0; i < l; i++)
      __builtin_prefetch((char*) values + i* 64);
  }

  long id() {return id_;}

  Quantized_Mips_Point(const T* values, unsigned int d, unsigned int qd, unsigned int ad, long id, float max_coord, float min_coord, int bits)
    : values(values), d(d), quantized_d(qd), aligned_d(ad), id_(id), max_coord(max_coord), min_coord(min_coord), bits(bits) {;
    }

  bool operator==(Quantized_Mips_Point<T, U> q){
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
  unsigned int quantized_d;
  long id_;
  float min_coord;
  float max_coord;
  int bits;
};



struct T2I_Point{
  using distanceType = float;

  
  static distanceType d_min() {return -std::numeric_limits<float>::max();}
  static bool is_metric() {return false;}
  
  uint16_t operator [] (long j) {if(j >= quantized_d) abort(); return *(values+j);}


  float distance(Mips_Point<float> x) {
    return wildly_optimized_mips_distance(this->values, x.values, max_coord, min_coord);
  }

  
  //hack for compatibility
  float distance(Mips_Point<uint8_t> x) {return 0;}
  float distance(Mips_Point<int8_t> x) {return 0;}
  float distance(Euclidian_Point<uint8_t> x) {return 0;}
  float distance(Euclidian_Point<int8_t> x) {return 0;}
  float distance(Euclidian_Point<float> x) {return 0;}
  float distance(T2I_Point x){return 0;}

  void prefetch() {
    int l = (aligned_d * sizeof(uint16_t))/64;
    for (int i=0; i < l; i++)
      __builtin_prefetch((char*) values + i* 64);
  }

  long id() {return id_;}

  T2I_Point(const uint16_t* values, unsigned int d, unsigned int qd, unsigned int ad, long id, float max_coord, float min_coord, int bits)
    : values(values), d(d), quantized_d(qd), aligned_d(ad), id_(id), max_coord(max_coord), min_coord(min_coord), bits(bits) {;
    }

  bool operator==(T2I_Point q){
    for (int i = 0; i < d; i++) {
      if (values[i] != q.values[i]) {
        return false;
      }
    }
    return true;
  }


private:
  const uint16_t* values;
  unsigned int d;
  unsigned int aligned_d;
  unsigned int quantized_d;
  long id_;
  float min_coord;
  float max_coord;
  int bits;
};