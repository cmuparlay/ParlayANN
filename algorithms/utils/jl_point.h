#pragma once

#include <algorithm>
#include <iostream>
#include <bitset>

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

template <int jl_dims = 128>
struct Mips_JL_Point {
  using T = int8_t;
  using distanceType = float;
  using Point = Quantized_Mips_Point<8>;
  using Params = typename Point::parameters;
  using byte = uint8_t;
  
  struct parameters {
    std::vector<int8_t> JL_vects;
    Params mips_params;
    int dims;
    int num_bytes() const {return mips_params.dims;}
    parameters() : dims(0) {}
    parameters(int dims) : dims(dims) {}
    parameters(std::vector<int8_t> const& JL_vects, int dims, int d)
      // vectors are normalized so few values will be greater than .3
      : JL_vects(JL_vects), dims(dims), mips_params(.3, d) {}
  };

  static bool is_metric() {return false;}
  
  T operator [] (long j) const {return pt[j];}

  float distance(const Mips_JL_Point &q) const {
    return pt.distance(q.pt);
  }

  void prefetch() const { pt.prefetch(); }

  bool same_as(const Mips_JL_Point& q){
    return pt.same_as(q.pt);
  }

  long id() const {return pt.id();}

  Mips_JL_Point(byte* values, long id, const parameters& p) 
    : pt(Point(values, id, p.mips_params)), params(&p) {}

  bool operator==(const Mips_JL_Point &q) const {
    return pt == q.pt; }

  void normalize() {
    std::cout << "can't normalize quantized point" << std::endl;
    abort();
  }

  template <typename In_Point>
  static void translate_point(byte* values, const In_Point& p, const parameters& params) {
    int dims = params.dims;
    const std::vector<int8_t>& jlv = params.JL_vects;
    int d = params.mips_params.dims;
    std::vector<float> v(d);
    double nn = 0.0;
    for (int i = 0; i < d; i++) {
      double vv = 0.0;
      for (int j = 0; j < dims; j++) {
        vv += (float) p[j] * (float) jlv[i * dims + j];
      }
      v[i] = vv;
      nn += vv * vv;
    }
    double norm = 1.0 / sqrt(nn);
    for (int i = 0; i < d; i++) {
      v[i] = v[i] * norm;
    }
    
    Point::translate_point(values, v, params.mips_params);
  }

  template <typename PR>
  static parameters generate_parameters(const PR& pr) {
    long n = pr.size();
    int dims = pr.dimension();
    std::vector<int8_t> JL_vects(jl_dims * dims);
    std::mt19937 rng;
    std::uniform_int_distribution<std::mt19937::result_type> dist(0,1);
    for (int i = 0; i < jl_dims * dims; i++)
      JL_vects[i] = (dist(rng) == 0) ? -1 : 1;
    return parameters(std::move(JL_vects), dims, jl_dims);
  }

private:
  Point pt;
  const parameters* params;
};

template <int jl_dims>
struct Mips_JL_Bit_Point {
  using distanceType = float;
  using Data = std::bitset<jl_dims>;
  using byte = uint8_t;
  
  struct parameters {
    std::vector<int8_t> JL_vects;
    int source_dims;
    int num_bytes() const {return sizeof(Data);}
    parameters() : source_dims(0) {}
    parameters(int dims) : source_dims(dims) {}
    parameters(std::vector<int8_t> const& JL_vects, int source_dims)
      : JL_vects(JL_vects), source_dims(source_dims) {
      std::cout << "JL dense quantization, dims = " << jl_dims << std::endl;
    }
  };
  
  static bool is_metric() {return false;}
  
  int8_t operator [] (long j) const {
    Data* pbits = (Data*) values;
    return (*pbits)[j] ? 1 : -1;}

  float distance(const Mips_JL_Bit_Point &q) const {
    Data* pbits = (Data*) values;
    Data* qbits = (Data*) q.values;
    return (*pbits ^ *qbits).count();
  }

  void prefetch() const {
    int l = (sizeof(Data) - 1)/64 + 1;
    for (int i=0; i < l; i++)
      __builtin_prefetch((char*) values + i* 64);
  }
    
  bool same_as(const Mips_JL_Bit_Point& q){
    return &q == this;
  }

  long id() const {return id_;}

  Mips_JL_Bit_Point(byte* values, long id, const parameters& p)
    : values(values), id_(id) {}

  bool operator==(const Mips_JL_Bit_Point &q) const {
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
    const std::vector<int8_t>& jlv = params.JL_vects;
    double nn = 0.0;
    for (int i = 0; i < jl_dims; i++) {
      double vv = 0.0;
      for (int j = 0; j < params.source_dims; j++) {
        vv += (float) p[j] * (float) jlv[i * params.source_dims + j];
      }
      (*bits)[i] = (vv > 0);
    }
  }

  template <typename PR>
  static parameters generate_parameters(const PR& pr) {
    long n = pr.size();
    int source_dims = pr.dimension();
    std::vector<int8_t> JL_vects(jl_dims * source_dims);
    std::mt19937 rng;
    std::uniform_int_distribution<std::mt19937::result_type> dist(0,1);
    for (int i = 0; i < jl_dims * source_dims; i++)
      JL_vects[i] = (dist(rng) == 0) ? -1 : 1;
    return parameters(std::move(JL_vects), source_dims);
  }

private:
  byte* values;
  long id_;
};

template <int jl_dims>
struct Mips_JL_Sparse_Point {
  using distanceType = float;
  using Data = std::bitset<jl_dims>;
  using byte = uint8_t;
  constexpr static int nz = 5; // number of non_zeros per row
  
  struct parameters {
    std::vector<int8_t> JL_signs;
    std::vector<int> JL_indices;
    int source_dims;
    int num_bytes() const {return sizeof(Data);}
    parameters() : source_dims(0) {}
    parameters(int dims) : source_dims(dims) {}
    parameters(std::vector<int8_t> const& JL_signs,
               std::vector<int> const& JL_indices,
               int source_dims)
      : JL_signs(JL_signs), JL_indices(JL_indices), source_dims(source_dims) {
      std::cout << "JL sparse quantization, dims = " << jl_dims << std::endl;
    }
  };
  
  static bool is_metric() {return false;}
  
  int8_t operator [] (long j) const {
    Data* pbits = (Data*) values;
    return (*pbits)[j] ? 1 : -1;}

  float distance(const Mips_JL_Sparse_Point &q) const {
    Data* pbits = (Data*) values;
    Data* qbits = (Data*) q.values;
    return (*pbits ^ *qbits).count();
  }

  void prefetch() const {
    int l = (sizeof(Data) - 1)/64 + 1;
    for (int i=0; i < l; i++)
      __builtin_prefetch((char*) values + i* 64);
  }
    
  bool same_as(const Mips_JL_Sparse_Point& q){
    return &q == this;
  }

  long id() const {return id_;}

  Mips_JL_Sparse_Point(byte* values, long id, const parameters& p)
    : values(values), id_(id) {}

  bool operator==(const Mips_JL_Sparse_Point &q) const {
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
    const std::vector<int8_t>& jls = params.JL_signs;
    const std::vector<int>& jli = params.JL_indices;
    double nn = 0.0;
    for (int i = 0; i < jl_dims; i++) {
      double vv = 0.0;
      for (int j = 0; j < nz; j++) 
        vv += (float) p[jli[i * nz + j]] * jls[i * nz + j];
      (*bits)[i] = (vv > 0);
    }
  }

  template <typename PR>
  static parameters generate_parameters(const PR& pr) {
    long n = pr.size();
    int source_dims = pr.dimension();
    std::vector<int8_t> JL_signs(jl_dims * nz);
    std::vector<int> JL_indices(jl_dims * nz);
    std::mt19937 rng;
    std::uniform_int_distribution<std::mt19937::result_type> dist_s(0,1);
    std::uniform_int_distribution<std::mt19937::result_type> dist_i(0,source_dims);
    for (int i = 0; i < jl_dims * nz; i++) {
      JL_signs[i] = (dist_s(rng) == 0) ? -1 : 1;
      JL_indices[i] = dist_i(rng);
    }
    return parameters(JL_signs, JL_indices, source_dims);
  }

private:
  byte* values;
  long id_;
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
    parlay::sequence<typename PR::T> vals(len);
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
