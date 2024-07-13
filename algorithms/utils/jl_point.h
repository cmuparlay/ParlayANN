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
struct JL_Point {
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

  float distance(const JL_Point &q) const {
    return pt.distance(q.pt);
  }

  void prefetch() const { pt.prefetch(); }

  bool same_as(const JL_Point& q){
    return pt.same_as(q.pt);
  }

  long id() const {return pt.id();}

  JL_Point(byte* values, long id, const parameters& p) 
    : pt(Point(values, id, p.mips_params)), params(&p) {}

  bool operator==(const JL_Point &q) const {
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
struct JL_Point_Bits{
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
      std::cout << "creating params: " << sizeof(Data) << std::endl;
    }
  };
  
  static bool is_metric() {return false;}
  
  int8_t operator [] (long j) const {
    Data* pbits = (Data*) values;
    return pbits[j] ? 1 : -1;}

  float distance(const JL_Point_Bits &q) const {
    Data* pbits = (Data*) values;
    Data* qbits = (Data*) q.values;
    return (*pbits ^ *qbits).count();
  }

  void prefetch() const {
    int l = (sizeof(Data) - 1)/64 + 1;
    for (int i=0; i < l; i++)
      __builtin_prefetch((char*) values + i* 64);
  }
    
  bool same_as(const JL_Point_Bits& q){
    return &q == this;
  }

  long id() const {return id_;}

  JL_Point_Bits(byte* values, long id, const parameters& p)
    : values(values), id_(id) {}

  bool operator==(const JL_Point_Bits &q) const {
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
  //const parameters* params;
};

