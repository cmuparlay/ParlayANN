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
#include <bitset>

#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/internal/file_map.h"

#include "types.h"
//#include "NSGDist.h"
// #include "common/time_loop.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace parlayANN {

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
  //efanna2e::DistanceL2 distfunc;
  //return distfunc.compare(p, q, d);
  float result = 0.0;
  for (int i = 0; i < d; i++)
    result += (q[i] - p[i]) * (q[i] - p[i]);
  return (float)result;
}

template<typename T_, long range=(1l << sizeof(T_)*8) - 1>
struct Euclidian_Point {
  using distanceType = float;
  using T = T_;
  using byte = uint8_t;

  struct parameters {
    float slope;
    int32_t offset;
    int dims;
    int num_bytes() const {return dims * sizeof(T);}
    parameters() : slope(0), offset(0), dims(0) {}
    parameters(int dims) : slope(1.0), offset(0), dims(dims) {}
    parameters(float min_val, float max_val, int dims)
      : slope(range / (max_val - min_val)),
        offset((int32_t) round(min_val * slope)),
        dims(dims) {}
  };

  static distanceType d_min() {return 0;}
  static bool is_metric() {return true;}
  T operator[](long i) const {return *(values + i);}

  float distance(const Euclidian_Point& x) const {
    return euclidian_distance(this->values, x.values, params.dims);
  }

  void normalize() {
    double norm = 0.0;
    for (int j = 0; j < params.dims; j++)
      norm += values[j] * values[j];
    norm = std::sqrt(norm);
    if (norm == 0) norm = 1.0;
    for (int j = 0; j < params.dims; j++)
      values[j] = values[j] / norm;
  }

  void prefetch() const {
    int l = (params.dims * sizeof(T) - 1)/64 + 1;
    for (int i=0; i < l; i++)
      __builtin_prefetch((char*) values + i* 64);
  }

  long id() const {return id_;}

  Euclidian_Point() : values(nullptr), id_(-1), params(0) {}

  Euclidian_Point(byte* values, long id, parameters params)
    : values((T*) values), id_(id), params(params) {}

  // template <typename Point>
  // Euclidian_Point(const Point& p, const parameters& params) : id_(-1), params(params) {
  //   float slope = params.slope;
  //   int32_t offset = params.offset;
  //   float min_val = std::floor(offset / slope);
  //   float max_val = std::ceil((range + offset) / slope);
  //   values = new T[params.dims];
  //   if (slope == 1 && offset == 0) {
  //     for (int j = 0; j < params.dims; j++)
  //       values[j] = (T) p[j];
  //   } else {
  //     for (int j = 0; j < params.dims; j++) {
  //       auto x = p[j];
  //       if (x < min_val || x > max_val) {
  //         std::cout << x << " is out of range: [" << min_val << "," << max_val << "]" << std::endl;
  //         abort();
  //       }
  //       int64_t r = (int64_t) (std::round(x * slope)) - offset;
  //       if (r < 0 || r > range) {
  //         std::cout << "out of range: " << r << ", " << range << ", " << x << ", " << std::round(x * slope) - offset << ", " << slope << ", " << offset << std::endl;
  //         abort();
  //       }
  //       values[j] = (T) r;
  //     }
  //   }
  // }

  bool operator==(const Euclidian_Point& q) const {
    for (int i = 0; i < params.dims; i++) {
      if (values[i] != q.values[i]) {
        return false;
      }
    }
    return true;
  }

  bool same_as(const Euclidian_Point& q) const {
    return values == q.values;
  }

  template <typename Point>
  static void translate_point(byte* byte_values, const Point& p, const parameters& params) {
    T* values = (T*) byte_values;
    float slope = params.slope;
    int32_t offset = params.offset;
    if (slope == 1.0 && offset == 00) 
      for (int j = 0; j < params.dims; j++)
        values[j] = p[j];
    else {
      //float min_val = std::floor(offset / slope);
      //float max_val = std::ceil((range + offset) / slope);
      for (int j = 0; j < params.dims; j++) {
        auto x = p[j];
        // if (x < min_val || x > max_val) {
        //   std::cout << x << " is out of range: [" << min_val << "," << max_val << "]" << std::endl;
        //   abort();
        // }
        int64_t r = (int64_t) (std::round(x * slope)) - offset;
        if (r < 0) r = 0;
        if (r > range) r = range;
        // if (r < 0 || r > range) {
        //   std::cout << "out of range: " << r << ", " << range << ", " << x << ", " << std::round(x * slope) - offset << ", " << slope << ", " << offset << std::endl;
        //   abort();
        // }
        values[j] = (T) r;
      }
    }
  }

  template <typename PR>
  static parameters generate_parameters(const PR& pr) {
    long n = pr.size();
    int dims = pr.dimension();
    using MT = float; // typename PR::Point::T;
    parlay::sequence<MT> mins(n, 0.0);
    parlay::sequence<MT> maxs(n, 0.0);
    parlay::sequence<bool> ni(n, true);
    parlay::parallel_for(0, n, [&] (long i) {
      for (int j = 0; j < dims; j++) {
        ni[i] = ni[i] && (pr[i][j] >= 0) && (pr[i][j] - (long) pr[i][j]) == 0;
        mins[i]= std::min<MT>(mins[i], pr[i][j]);
        maxs[i]= std::max<MT>(maxs[i], pr[i][j]);}});
    float min_val = *parlay::min_element(mins);
    float max_val = *parlay::max_element(maxs);
    bool all_ints = *parlay::min_element(ni);
    if (all_ints) {
      if (sizeof(T) == 1 && max_val < 256) max_val = 255;
      else if (sizeof(T) == 2 && max_val < 65536) max_val = 65536;
      min_val = 0;
    }
    std::cout << "scalar quantization: min value = " << min_val
              << ", max value = " << max_val << std::endl;
    return parameters(min_val, max_val, dims);
  }

  parameters params;

private:
  T* values;
  long id_;
};

template <int jl_dims>
struct Euclidean_JL_Sparse_Point {
  using distanceType = float;
  using Data = std::bitset<jl_dims>;
  using byte = uint8_t;
  constexpr static int nz = 6; // number of non_zeros per row
  
  struct parameters {
    std::vector<int> JL_indices;
    int source_dims;
    int num_bytes() const {return sizeof(Data);}
    parameters() : source_dims(0) {}
    parameters(int dims) : source_dims(dims) {}
    parameters(std::vector<int> const& JL_indices,
               int source_dims)
      : JL_indices(JL_indices), source_dims(source_dims) {
      std::cout << "JL sparse quantization, dims = " << jl_dims << std::endl;
    }
  };
  
  static bool is_metric() {return false;}
  
  int8_t operator [] (long j) const {
    Data* pbits = (Data*) values;
    return (*pbits)[j] ? 1 : -1;}

  float distance(const Euclidean_JL_Sparse_Point &q) const {
    Data* pbits = (Data*) values;
    Data* qbits = (Data*) q.values;
    return (*pbits ^ *qbits).count();
  }

  void prefetch() const {
    int l = (sizeof(Data) - 1)/64 + 1;
    for (int i=0; i < l; i++)
      __builtin_prefetch((char*) values + i* 64);
  }
    
  bool same_as(const Euclidean_JL_Sparse_Point& q){
    return &q == this;
  }

  long id() const {return id_;}

  Euclidean_JL_Sparse_Point(byte* values, long id, const parameters& p)
    : values(values), id_(id) {}

  bool operator==(const Euclidean_JL_Sparse_Point &q) const {
    Data* pbits = (Data*) values;
    Data* qbits = (Data*) q.values;
    return *pbits == *qbits; }

  void normalize() {
    std::cout << "can't normalize quantized point" << std::endl;
    abort();
  }

  template <typename In_Point>
  static void translate_point(byte* values, const In_Point& p, const parameters& params) {
    Data* bits = new (values) Data;
    const std::vector<int>& jli = params.JL_indices;
    for (int i = 0; i < jl_dims; i++) {
      double vv = 0.0;
      for (int j = 0; j < nz/2; j++) 
        vv += (float) p[jli[i * nz + j]];
      for (int j = nz/2; j < nz; j++) 
        vv -= (float) p[jli[i * nz + j]];
      (*bits)[i] = (vv > 0);
    }
  }

  template <typename PR>
  static parameters generate_parameters(const PR& pr) {
    int source_dims = pr.dimension();
    std::vector<int> JL_indices(jl_dims * nz);
    std::mt19937 rng;
    std::uniform_int_distribution<std::mt19937::result_type> dist_i(0,source_dims);
    for (int i = 0; i < jl_dims * nz; i++) {
      JL_indices[i] = dist_i(rng);
    }
    return parameters(JL_indices, source_dims);
  }

private:
  byte* values;
  long id_;
};

struct Euclidean_Bit_Point {
  using distanceType = float;
  using Data = std::bitset<64>;
  using byte = uint8_t;
  
  struct parameters {
    int dims;
    long median;
    int num_bytes() const {return ((dims - 1) / 64 + 1) * 8;}
    parameters() : dims(0) {}
    parameters(int dims, long median)
      : dims(dims), median(median) {
      std::cout << "single-bit quantization with median: " << median << std::endl;
    }
  };
  
  static bool is_metric() {return false;}
  
  int8_t operator [] (long j) const {
    Data* pbits = (Data*) values;
    return pbits[j/64][j%64];
  }

  float distance(const Euclidean_Bit_Point &q) const {
    int num_blocks = (params.dims - 1)/64 + 1;
    Data* pbits = (Data*) values;
    Data* qbits = (Data*) q.values;
    int cnt = 0;
    for (int i=0; i < num_blocks; i++)
      cnt +=(*pbits ^ *qbits).count();
    return cnt;
  }

  void prefetch() const {
    int l = (params.num_bytes() - 1)/64 + 1;
    for (int i=0; i < l; i++)
      __builtin_prefetch((char*) values + i* 64);
  }
    
  bool same_as(const Euclidean_Bit_Point& q){
    return &q == this;
  }

  long id() const {return id_;}

  Euclidean_Bit_Point(byte* values, long id, const parameters& params)
    : values(values), id_(id), params(params) {}

  bool operator==(const Euclidean_Bit_Point &q) const {
    int num_blocks = (params.dims - 1)/64 + 1;
    Data* pbits = (Data*) values;
    Data* qbits = (Data*) q.values;
    for (int i = 0; i < num_blocks; i++)
      if (pbits[i] != qbits[i]) return false;
    return true;
  }

  void normalize() {
    std::cout << "can't normalize quantized point" << std::endl;
    abort();
  }

  template <typename In_Point>
  static void translate_point(byte* values, const In_Point& p, const parameters& params) {
    Data* pbits = (Data*) values;
    for (int i = 0; i < params.dims; i++)
      pbits[i/64][i%64] = p[i] > params.median;
  }

  template <typename PR>
  static parameters generate_parameters(const PR& pr) {
    long n = pr.size();
    int dims = pr.dimension();
    long len = n * dims;
    parlay::sequence<typename PR::Point::T> vals(len);
    parlay::parallel_for(0, n, [&] (long i) {
      for (int j = 0; j < dims; j++) 
        vals[i * dims + j] = pr[i][j];
    });
    parlay::sort_inplace(vals);
    long median = vals[n*dims/2];
    return parameters(dims, median);
  }

private:
  byte* values;
  long id_;
  parameters params;
};

} // end namespace
