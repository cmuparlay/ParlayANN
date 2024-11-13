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
#include <bit>

#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/internal/file_map.h"
#include "types.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace parlayANN {

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
    float inv_norm = 1.0 / norm;
    for (int j = 0; j < params.dims; j++)
      values[j] = values[j] * inv_norm;
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
  using T = int16_t;
  using distanceType = int64_t; //float;
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


  distanceType distance_16(byte* p_, byte* q_) const {
    int16_t* p = (int16_t*) p_;
    int16_t* q = (int16_t*) q_;
    int64_t result = 0;
    for (int i = 0; i < params.dims; i++){
      result += (int32_t) p[i] * (int32_t) q[i];
    }
    return (distanceType) -result;
  }

  distanceType distance_8(byte* p_, byte* q_) const {
    int8_t* p = (int8_t*) p_;
    int8_t* q = (int8_t*) q_;
    int32_t result = 0;
    for (int i = 0; i < params.dims; i++){
      result += (int16_t) p[i] * (int16_t) q[i];
    }
    return (distanceType) -result;
  }

  distanceType distance_4(byte* p_, byte* q_) const {
    int8_t* p = (int8_t*) p_;
    int8_t* q = (int8_t*) q_;
    int32_t result = 0;
    int8_t mask = -16; // bit representation is 11110000, used as mask to extract high 4 bits
    for (int i = 0; i < params.dims/2; i++) {
      result += (int16_t) ((int8_t) (p[i] << 4)) * (int16_t) ((int8_t) (q[i] << 4));
    }
    for (int i = 0; i < params.dims/2; i++){
      result += (int16_t) (p[i] & mask) * (int16_t) (q[i] & mask);
    }
    return (distanceType) -result;
  }

  distanceType distance(const Quantized_Mips_Point &x) const {
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
      float scale = (range/2) / mv;
      float pj = p[j];
      // cap if underflow or overflow
      if (pj < -mv) assign(byte_values, j, - range/2); // - 1);
      else if (pj > mv) assign(byte_values, j, range/2);
      else {
        int32_t v = std::round(pj * scale); 
        assign(byte_values, j, v);
      }
    }
  }

  
  template <typename PR>
  static parameters generate_parameters(const PR& pr) {
    long n = pr.size();
    int dims = pr.dimension();
    long len = n * dims;
    using MT = float;
    parlay::sequence<MT> vals(len);
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
      std::cout << "mips scalar quantization to " << bits
                << " bits: min value = " << vals[0]
                << ", max value = " << vals[len-1]
                << ", trimmed to: min = " << min_val << ", max = " << max_val << std::endl;
    } else {
      min_val = vals[0];
      max_val = vals[len-1];
      std::cout << "mips scalar quantization to " << bits
                << " bits: min value = " << min_val
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


struct Mips_2Bit_Point {
  using distanceType = float;
  using byte = uint8_t;
  using word = std::bitset<64>;
  //using word = uint64_t; 
  using T = int8_t;

  static int pop_count(word x) {
    return x.count();
    //return __builtin_popcountl(x);
  }

  static void set_bit(word& x, int i, bool v) {
    x[i] = v;
    //x = (~(1ul << i) & x) | ((uint64_t) v << i);
  }
  
  struct parameters {
    float cut;
    int dims;
    int num_bytes() const {return ((dims - 1) / 64 + 1) * 8 * 2;}
    parameters() : cut(.25), dims(0) {}
    parameters(int dims) : cut(.25), dims(dims) {}
    parameters(float cut, int dims)
      : cut(cut), dims(dims) {
      std::cout << "3-value quantization with cut = " << cut << std::endl;
    }
  };

  static bool is_metric() {return false;}
  
  int operator [] (long i) const {
    abort();
  }

  float distance_8(byte* p_, byte* q_) const {
    word* p = (word*) p_;
    word* q = (word*) q_;
    int num_blocks = params.num_bytes() / 16;
    int16_t total = 0;
    for (int i = 0; i < num_blocks; i++) {
      word not_equal = p[2 * i] ^ q[2 * i];
      word not_zero = p[2 * i + 1] & q[2 * i + 1];
      int16_t num_neg = pop_count(not_equal & not_zero);
      int16_t num_not_zero = pop_count(not_zero);
      total += (2 * num_neg) - num_not_zero;
    }
    return total;
  }

  float distance(const Mips_2Bit_Point &x) const {
    return distance_8(this->values, x.values);
  }
  
  void prefetch() const {
    int l = (params.num_bytes() - 1)/64 + 1;
    for (int i=0; i < l; i++)
      __builtin_prefetch(values + i * 64);
  }

  bool same_as(const Mips_2Bit_Point& q){
    return values == q.values;
  }

  long id() const {return id_;}

  Mips_2Bit_Point(byte* values, long id, parameters p)
    : values(values), id_(id), params(p)
  {}

  bool operator==(const Mips_2Bit_Point &q) const {
    for (int i = 0; i < params.num_bytes(); i++) {
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
  
  template <typename Point>
  static void translate_point(byte* byte_values, const Point& p, const parameters& params) {
    // two words per block, one for -1, +1, the other to mark if non-zero
    int num_blocks = params.num_bytes() / 16;
    word* words = (word*) byte_values;
    float cv = params.cut;
    for (int i = 0; i < num_blocks; i++) {
      for (int j = 0; j < 64; j++) {
        if (j + i * 64 >= params.dims) {
          set_bit(words[2 * i + 1], j, false);
          return;
        }
        set_bit(words[2 * i + 1], j, true);
        float pj = p[j + i * 64];
        if (pj < -cv) set_bit(words[2 * i], j, false);
        else if (pj > cv) set_bit(words[2 * i], j, true);
        else set_bit(words[2 * i + 1], j, false);
      }
    }
  }
  
  template <typename PR>
  static parameters generate_parameters(const PR& pr) {
    long n = pr.size();
    int dims = pr.dimension();
    long len = n * dims;
    using MT = float;
    parlay::sequence<MT> vals(len);
    parlay::parallel_for(0, n, [&] (long i) {
      for (int j = 0; j < dims; j++) 
        vals[i * dims + j] = pr[i][j];
    });
    parlay::sort_inplace(vals);
    float cutoff = .3;
    float min_cut = vals[(long) (cutoff * len)];
    float max_cut = vals[(long) ((1.0-cutoff) * (len-1))];
    float cut = std::max(max_cut, -min_cut);
    return parameters(cut, dims); 
  }

private:
  byte* values;
  long id_;
  parameters params;
};

struct Mips_Bit_Point {
  using distanceType = float;
  using Data = std::bitset<64>;
  using byte = uint8_t;
  
  struct parameters {
    int dims;
    int num_bytes() const {return ((dims - 1) / 64 + 1) * 8;}
    parameters() : dims(0) {}
    parameters(int dims)
      : dims(dims) {
      std::cout << "single-bit quantization" << std::endl;
    }
  };
  
  static bool is_metric() {return false;}
  
  int8_t operator [] (long j) const {
    Data* pbits = (Data*) values;
    return pbits[j/64][j%64] ? 1 : -1;
  }

  float distance(const Mips_Bit_Point &q) const {
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
    
  bool same_as(const Mips_Bit_Point& q){
    return &q == this;
  }

  long id() const {return id_;}

  Mips_Bit_Point(byte* values, long id, const parameters& params)
    : values(values), id_(id), params(params) {}

  bool operator==(const Mips_Bit_Point &q) const {
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
      pbits[i/64][i%64] = (p[i] > 0);
  }

  template <typename PR>
  static parameters generate_parameters(const PR& pr) {
    return parameters(pr.dimension());
  }

private:
  byte* values;
  long id_;
  parameters params;
};



struct Mips_4Bit_Point {
  using distanceType = float;
  using byte = uint8_t;
  using word = std::bitset<64>;
  //using word = uint64_t; 
  using T = int8_t;

  static int pop_count(word x) {
    return x.count();
  }

  static void set_bit(word& x, int i, bool v) {
    x[i] = v;
  }
  
  struct parameters {
    float cut;
    int dims;
    int num_bytes() const {return ((dims - 1) / 64 + 1) * 8 * 4;}
    parameters() : cut(.25), dims(0) {}
    parameters(int dims) : cut(.25), dims(dims) {}
    parameters(float cut, int dims)
      : cut(cut), dims(dims) {
      std::cout << "3-value quantization with cut = " << cut << std::endl;
    }
  };

  static bool is_metric() {return false;}
  
  int operator [] (long i) const {
    abort();
  }

  static int16_t triple(word a, word b, word plus, word minus) {
    word x = a & b;
    return pop_count(x & plus) - pop_count(x & minus);
  }
      
  float distance(byte* p_, byte* q_) const {
    word* p = (word*) p_;
    word* q = (word*) q_;
    int num_blocks = params.num_bytes() / 16;
    int16_t total = 0;
    for (int i = 0; i < num_blocks; i++) {
      word minus = p[2 * i] ^ q[2 * i];
      word plus = ~minus;
      auto triple = [=] (word a, word b) -> int16_t {
        word x = a & b;
        return pop_count(x & plus) - pop_count(x & minus);
      };
      total += triple(p[2 * i + 1], q[2 * i + 1]);
      total += triple(p[2 * i + 1], q[2 * i + 2]) * 2;
      total += triple(p[2 * i + 1], q[2 * i + 3]) * 4;
      total += triple(p[2 * i + 2], q[2 * i + 1]) * 2;
      total += triple(p[2 * i + 2], q[2 * i + 2]) * 4;
      total += triple(p[2 * i + 2], q[2 * i + 3]) * 8;
      total += triple(p[2 * i + 3], q[2 * i + 1]) * 4;
      total += triple(p[2 * i + 3], q[2 * i + 2]) * 8;
      total += triple(p[2 * i + 3], q[2 * i + 3]) * 16;
    }
    return total;
  }

  float distance(const Mips_4Bit_Point &x) const {
    return distance(this->values, x.values);
  }
  
  void prefetch() const {
    int l = (params.num_bytes() - 1)/64 + 1;
    for (int i=0; i < l; i++)
      __builtin_prefetch(values + i * 64);
  }

  bool same_as(const Mips_4Bit_Point& q){
    return values == q.values;
  }

  long id() const {return id_;}

  Mips_4Bit_Point(byte* values, long id, parameters p)
    : values(values), id_(id), params(p)
  {}

  bool operator==(const Mips_4Bit_Point &q) const {
    for (int i = 0; i < params.num_bytes(); i++) {
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
  
  template <typename Point>
  static void translate_point(byte* byte_values, const Point& p, const parameters& params) {
    // two words per block, one for -1, +1, the other to mark if non-zero
    int num_blocks = params.num_bytes() / 16;
    word* words = (word*) byte_values;
    float cv = params.cut;
    for (int i = 0; i < num_blocks; i++) {
      for (int j = 0; j < 64; j++) {
        if (j + i * 64 >= params.dims) {
          set_bit(words[2 * i + 1], j, false);
          return;
        }
        set_bit(words[2 * i + 1], j, true);
        float pj = p[j + i * 64];
        if (pj < -cv) set_bit(words[2 * i], j, false);
        else if (pj > cv) set_bit(words[2 * i], j, true);
        else set_bit(words[2 * i + 1], j, false);
      }
    }
  }
  
  template <typename PR>
  static parameters generate_parameters(const PR& pr) {
    long n = pr.size();
    int dims = pr.dimension();
    long len = n * dims;
    using MT = float;
    parlay::sequence<MT> vals(len);
    parlay::parallel_for(0, n, [&] (long i) {
      for (int j = 0; j < dims; j++) 
        vals[i * dims + j] = pr[i][j];
    });
    parlay::sort_inplace(vals);
    float cutoff = .3;
    float min_cut = vals[(long) (cutoff * len)];
    float max_cut = vals[(long) ((1.0-cutoff) * (len-1))];
    float cut = std::max(max_cut, -min_cut);
    return parameters(cut, dims); 
  }

private:
  byte* values;
  long id_;
  parameters params;
};

} // end namespace
