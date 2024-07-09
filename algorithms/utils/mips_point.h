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

template<typename T_>
struct Mips_Point {
  using T = T_;
  using distanceType = float;
  using byte = uint8_t;
  //template<typename C, typename range> friend struct Quantized_Mips_Point;

  struct parameters {
    int dims;
    int num_bytes() const {return dims * sizeof(T);}
    parameters() : dims(0) {}
    parameters(int dims) : dims(dims) {}
  };

  static distanceType d_min() {return -std::numeric_limits<float>::max();}
  static bool is_metric() {return false;}
  T operator [](long i) const {return *(values + i);}

  float distance(const Mips_Point<T>& x) const {
    return mips_distance(this->values, x.values, params.dims);
  }

  void prefetch() const {
    int l = (params.dims * sizeof(T) - 1)/64 + 1;
    for (int i=0; i < l; i++)
      __builtin_prefetch((char*) values + i* 64);
  }

  long id() const {return id_;}

  Mips_Point() : values(nullptr), id_(-1), params(0) {}

  Mips_Point(byte* values, long id, parameters params)
    : values((T*) values), id_(id), params(params) {}

  bool operator==(const Mips_Point<T>& q) const {
    for (int i = 0; i < params.dims; i++) {
      if (values[i] != q.values[i]) {
        return false;
      }
    }
    return true;
  }

  bool same_as(const Mips_Point<T>& q) const {
    return values == q.values;
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

  template <typename Point>
  static void translate_point(byte* values, const Point& p, const parameters& params) {
    for (int j = 0; j < params.dims; j++) ((T*) values)[j] = (T) p[j];
  }

  template <typename PR>
  static parameters generate_parameters(const PR& pr) {
    return parameters(pr.dimension());}

private:
  T* values;
  long id_;
  parameters params;
};

// template<typename T_, bool trim = false, int range = (1 << sizeof(T_)*8) - 1>
// struct Quantized_Mips_Point{
//   using T = T_;
//   using distanceType = float;
//   using byte = uint8_t;
  
//   struct parameters {
//     float max_val;
//     int dims;
//     int num_bytes() const {return dims * sizeof(T);}
//     parameters() : max_val(1), dims(0) {}
//     parameters(int dims) : max_val(1), dims(dims) {}
//     parameters(float max_val, int dims)
//       : max_val(max_val), dims(dims) {}
//   };

//   static distanceType d_min() {return -std::numeric_limits<float>::max();}
//   static bool is_metric() {return false;}
  
//   //T& operator [] (long j) const {if (j >= d) abort(); return *(values+j);}
//   T operator [] (long i) const {return *(values + i);}

//   float distance(int8_t* p, int8_t* q) const {
//     int32_t result = 0;
//     for (int i = 0; i < params.dims; i++){
//       result += (int16_t) p[i] * (int16_t) q[i];
//     }
//     //return (float) (r * r - result);
//     return (float) -result;
//   }

//   float distance(int16_t* p, int16_t* q) const {
//     int64_t result = 0;
//     for (int i = 0; i < params.dims; i++){
//       result += (int32_t) p[i] * (int32_t) q[i];
//     }
//     return (float) -result;
//   }

//   float distance(const Quantized_Mips_Point &x) const {
//     return distance(this->values, x.values);
//   }

//   void prefetch() const {
//     int l = (params.dims * sizeof(T) - 1)/64 + 1;
//     for (int i=0; i < l; i++)
//       __builtin_prefetch(values + i * 64);
//   }

//   bool same_as(const Quantized_Mips_Point& q){
//     return values == q.values;
//   }

//   long id() const {return id_;}

//   Quantized_Mips_Point(byte* values, long id, parameters p)
//     : values((T*) values), id_(id), params(p)
//   {}

//   bool operator==(const Quantized_Mips_Point &q) const {
//     for (int i = 0; i < params.dims; i++) {
//       if (values[i] != q.values[i]) {
//         return false;
//       }
//     }
//     return true;
//   }

//   void normalize() {
//     std::cout << "can't normalize quantized point" << std::endl;
//     abort();
//   }

//   template <typename Point>
//   static void translate_point(byte* byte_values, const Point& p, const parameters& params) {
//     T* values = (T*) byte_values;
//     for (int j = 0; j < params.dims; j++) {
//       float mv = params.max_val;
//       float pj = p[j];
//       if (pj < -mv) values[j] = - range/2 - 1;
//       else if (pj > mv) values[j] = range/2;
//       else {
//         //if (pj < -mv || pj > mv) {
//         //std::cout << pj << " is out of range, should be in [" << -mv << ":" << mv << "] " << std::endl;
//         //abort();
//         //}
//         int32_t x = std::round(pj * (range/2) / mv);
//         values[j] = (T) x;
//       }
//     }
//   }

//   template <typename PR>
//   static parameters generate_parameters(const PR& pr) {
//     long n = pr.size();
//     int dims = pr.dimension();
//     long len = n * dims;
//     parlay::sequence<typename PR::T> vals(len);
//     parlay::parallel_for(0, n, [&] (long i) {
//       for (int j = 0; j < dims; j++) 
//         vals[i * dims + j] = pr[i][j];
//     });
//     parlay::sort_inplace(vals);
//     float min_val, max_val;
//     if (trim) {
//       float cutoff = .0001;
//       min_val = vals[(long) (cutoff * len)];
//       max_val = vals[(long) ((1.0-cutoff) * (len-1))];
//     } else {
//       min_val = vals[0];
//       max_val = vals[len-1];
//     }
//     float bound = std::max(max_val, -min_val);

//     // parlay::sequence<typename PR::T> mins(n);
//     // parlay::sequence<typename PR::T> maxs(n);
//     // parlay::parallel_for(0, n, [&] (long i) {
//     //   mins[i] = 0.0;
//     //   maxs[i] = 0.0;
//     //   for (int j = 0; j < dims; j++) {
//     //     mins[i]= std::min(mins[i], pr[i][j]);
//     //     maxs[i]= std::max(maxs[i], pr[i][j]);}});
//     // float min_val = *parlay::min_element(mins);
//     // float max_val = *parlay::max_element(maxs);
//     // float bound = std::max(max_val, -min_val);
    
    
//     // if (sizeof(T) == 1) {
//     //   auto x = parlay::flatten(parlay::tabulate(n, [&] (long i) {
//     //     return parlay::tabulate(dims, [&] (long j) {
//     //       return 128 + (int8_t) (std::round(pr[i][j] * (range/2) / bound));});}));
//     //   auto y = parlay::histogram_by_index(x, 256);
//     //   for (int i = 0; i < 256; i++)
//     //     std::cout << i - 128 << ":" << y[i] << ", ";
//     //   std::cout << std::endl;
//     // }
//     std::cout << "scalar quantization: min value = " << min_val
//               << ", max value = " << max_val << std::endl;
//     return parameters(bound, dims); // 1.7 for glove-100, 1.4 for nytimes, 1.5 for glove-25 but bad
//   }

// private:
//   T* values;
//   long id_;
//   parameters params;
// };

template<int bits, bool trim = false, int range = (1 << bits) - 1>
struct Quantized_Mips_Point{
  using distanceType = float;
  using byte = uint8_t;
  
  struct parameters {
    float max_val;
    int dims;
    int num_bytes() const {return (dims * bits - 1) / 8 + 1;}
    parameters() : max_val(1), dims(0) {}
    parameters(int dims) : max_val(1), dims(dims) {}
    parameters(float max_val, int dims)
      : max_val(max_val), dims(dims) {}
  };

  static bool is_metric() {return false;}
  
  int operator [] (long i) const {
    if constexpr (bits <= 4) {
      if (i & 1)
        return ((int8_t) (values[i/2] & 240)) >> 4;
      else
        return ((int8_t) (values[i/2] << 4)) >> 4;
    } else {
      if constexpr (bits <= 8) {
        return *(((int8_t*) values) + i);
      } else {
        return *(((int16_t*) values) + i);
      }
    }
  }


  float distance_16(byte* p_, byte* q_) const {
    int16_t* p = (int16_t*) p_;
    int16_t* q = (int16_t*) q_;
    int64_t result = 0;
    for (int i = 0; i < params.dims; i++){
      result += (int32_t) p[i] * (int32_t) q[i];
    }
    return (float) -result;
  }

  float distance_8(byte* p_, byte* q_) const {
    int8_t* p = (int8_t*) p_;
    int8_t* q = (int8_t*) q_;
    int32_t result = 0;
    for (int i = 0; i < params.dims; i++){
      result += (int16_t) p[i] * (int16_t) q[i];
    }
    return (float) -result;
  }

  float distance_4(byte* p_, byte* q_) const {
    int8_t* p = (int8_t*) p_;
    int8_t* q = (int8_t*) q_;
    int32_t result = 0;
    int8_t mask = 240;
    for (int i = 0; i < params.dims/2; i++) {
      result += (int16_t) ((int8_t) (p[i] << 4)) * (int16_t) ((int8_t) (q[i] << 4));
    }
    for (int i = 0; i < params.dims/2; i++){
      result += (int16_t) (p[i] & mask) * (int16_t) (q[i] & mask);
    }
    return (float) -result;
  }

  float distance(const Quantized_Mips_Point &x) const {
    if constexpr (bits <= 4) {
      return distance_4(this->values, x.values);
    } else {
      if constexpr (bits <= 8) {
        return distance_8(this->values, x.values);
      } else {
        return distance_16(this->values, x.values);
      }
    }
  }

  
  void prefetch() const {
    int l = (params.num_bytes() - 1)/64 + 1;
    __builtin_prefetch(values);
    for (int i=0; i < l; i++)
      __builtin_prefetch(values + i * 64);
  }

  bool same_as(const Quantized_Mips_Point& q){
    return values == q.values;
  }

  long id() const {return id_;}

  Quantized_Mips_Point(byte* values, long id, parameters p)
    : values(values), id_(id), params(p)
  {}

  bool operator==(const Quantized_Mips_Point &q) const {
    for (int i = 0; i < params.dims; i++) {
      if (values[i] != q.values[i]) {
        return false;
      }
    }
    return true;
  }

  void normalize() {
    std::cout << "can't normalize quantized point" << std::endl;
    abort();
  }

  static void assign(byte* values, int i, int v) {
    if constexpr (bits <= 4) {
      byte* p = values + i/2;
      if (i & 1) {
        *p = (*p & 15) | (v << 4);
      } else {
        *p = (*p & 240) | v;
      }
    } else {
      if constexpr (bits <= 8) {
        ((int8_t*) values)[i] = (int8_t) v;
      } else {
        ((int16_t*) values)[i] = (int16_t) v;
      }
    }
  }
  
  template <typename Point>
  static void translate_point(byte* byte_values, const Point& p, const parameters& params) {
    for (int j = 0; j < params.dims; j++) {
      float mv = params.max_val;
      float pj = p[j];
      if (pj < -mv) assign(byte_values, j, - range/2 - 1);
      else if (pj > mv) assign(byte_values, j, range/2);
      else {
        //if (pj < -mv || pj > mv) {
        //std::cout << pj << " is out of range, should be in [" << -mv << ":" << mv << "] " << std::endl;
        //abort();
        //}
        int32_t v = std::round(pj * (range/2) / mv);
        assign(byte_values, j, v);
      }
    }
  }

  
  template <typename PR>
  static parameters generate_parameters(const PR& pr) {
    long n = pr.size();
    int dims = pr.dimension();
    long len = n * dims;
    parlay::sequence<typename PR::T> vals(len);
    parlay::parallel_for(0, n, [&] (long i) {
      for (int j = 0; j < dims; j++) 
        vals[i * dims + j] = pr[i][j];
    });
    parlay::sort_inplace(vals);
    float min_val, max_val;
    if (trim) {
      float cutoff = .0001;
      min_val = vals[(long) (cutoff * len)];
      max_val = vals[(long) ((1.0-cutoff) * (len-1))];
      std::cout << "scalar quantization: min value = " << vals[0]
                << ", max value = " << vals[len-1]
                << ", trimmed to: min = " << min_val << ", max = " << max_val << std::endl;
    } else {
      min_val = vals[0];
      max_val = vals[len-1];
      std::cout << "scalar quantization: min value = " << min_val
                << ", max value = " << max_val << std::endl;
    }
    float bound = std::max(max_val, -min_val);

    // parlay::sequence<typename PR::T> mins(n);
    // parlay::sequence<typename PR::T> maxs(n);
    // parlay::parallel_for(0, n, [&] (long i) {
    //   mins[i] = 0.0;
    //   maxs[i] = 0.0;
    //   for (int j = 0; j < dims; j++) {
    //     mins[i]= std::min(mins[i], pr[i][j]);
    //     maxs[i]= std::max(maxs[i], pr[i][j]);}});
    // float min_val = *parlay::min_element(mins);
    // float max_val = *parlay::max_element(maxs);
    // float bound = std::max(max_val, -min_val);
    
    
    // if (sizeof(T) == 1) {
    //   auto x = parlay::flatten(parlay::tabulate(n, [&] (long i) {
    //     return parlay::tabulate(dims, [&] (long j) {
    //       return 128 + (int8_t) (std::round(pr[i][j] * (range/2) / bound));});}));
    //   auto y = parlay::histogram_by_index(x, 256);
    //   for (int i = 0; i < 256; i++)
    //     std::cout << i - 128 << ":" << y[i] << ", ";
    //   std::cout << std::endl;
    // }
    return parameters(bound, dims); // 1.7 for glove-100, 1.4 for nytimes, 1.5 for glove-25 but bad
  }

private:
  byte* values;
  long id_;
  parameters params;
};

