#ifndef ALGORITHMS_ANN_TYPES_H_
#define ALGORITHMS_ANN_TYPES_H_

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

#ifndef TYPES
#define TYPES

#include <algorithm>
#include <fstream>

#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "mmap.h"

namespace parlayANN {

template<typename T>
struct groundTruth{
  parlay::slice<T*, T*> coords;
  parlay::slice<float*, float*> dists;
  long dim;
  size_t n;

  groundTruth() : coords(parlay::make_slice<T*, T*>(nullptr, nullptr)),
                  dists(parlay::make_slice<float*, float*>(nullptr, nullptr)){}

  groundTruth(char* gtFile) : coords(parlay::make_slice<T*, T*>(nullptr, nullptr)),
                              dists(parlay::make_slice<float*, float*>(nullptr, nullptr)){
    if(gtFile == NULL){
      n = 0;
      dim = 0;
    } else{
      auto [fileptr, length] = mmapStringFromFile(gtFile);

      int num_vectors = *((T*) fileptr);
      int d = *((T*) (fileptr + 4));

      
      std::cout << "Ground truth: detected " << num_vectors << " points with num results " << d << std::endl;

      T* start_coords = (T*)(fileptr+8);
      T* end_coords = start_coords + d*num_vectors;

      float* start_dists = (float*)(end_coords);
      float* end_dists = start_dists + d*num_vectors;

      n = num_vectors;
      dim = d;
      coords = parlay::make_slice(start_coords, end_coords);
      dists = parlay::make_slice(start_dists, end_dists);
    }
  }

  groundTruth(parlay::sequence<parlay::sequence<T>> gt) : coords(parlay::make_slice<T*, T*>(nullptr, nullptr)),
                                                          dists(parlay::make_slice<float*, float*>(nullptr, nullptr)){
    n = gt.size();
    dim = gt[0].size();
    auto flat_gt = parlay::flatten(gt);
    coords = parlay::make_slice(flat_gt.begin(), flat_gt.end());
    parlay::sequence<float> dummy_ds = parlay::sequence<float>(dim * n, 0.0);
    dists = parlay::make_slice(dummy_ds.begin(), dummy_ds.end());
  }

  //saves in binary format
  //assumes gt is not so big that it needs block saving
  void save(char* save_path) {
    std::cout << "Writing groundtruth for " << n << " points and num results " << dim
              << std::endl;
    parlay::sequence<T> preamble = {static_cast<T>(n), static_cast<T>(dim)};
    std::ofstream writer;
    writer.open(save_path, std::ios::binary | std::ios::out);
    writer.write((char*)preamble.begin(), 2 * sizeof(T));
    writer.write((char*)coords.begin(), dim*n*sizeof(T));
    writer.write((char*)dists.begin(), dim*n*sizeof(float));
    writer.close();
  }

  T coordinates(long i, long j) const {return *(coords.begin() + i * dim + j);}

  float distances(long i, long j) const {return *(dists.begin() + i * dim + j);}

  size_t size() const {return n;}

  long dimension() const {return dim;}

};

template<typename T>
struct RangeGroundTruth{
  T* coords;
  parlay::sequence<T> offsets;
  parlay::slice<T*, T*> sizes;
  size_t n;
  size_t num_matches;

  RangeGroundTruth() : sizes(parlay::make_slice<T*, T*>(nullptr, nullptr)){}

  RangeGroundTruth(char* gtFile) : sizes(parlay::make_slice<T*, T*>(nullptr, nullptr)){
    if(gtFile == NULL){
      n = 0;
      num_matches = 0;
    } else{
      auto [fileptr, length] = mmapStringFromFile(gtFile);

      n = *((T*) fileptr);
      num_matches = *((T*) (fileptr + sizeof(T)));

      T* sizes_begin = (T*)(fileptr + 2 * sizeof(T)) ;
      T* sizes_end = sizes_begin+n;
      sizes = parlay::make_slice(sizes_begin, sizes_end);

      auto [offsets0, total] = parlay::scan(sizes);
      offsets0.push_back(total);
      offsets = offsets0;

      std::cout << "Detected " << n << " points with num matches " << num_matches << std::endl;

      coords = sizes_end;
    }
  }

  parlay::slice<T*, T*> operator[] (long i){
    T* begin = coords + offsets[i];
    T* end = coords + offsets[i + 1];
    return parlay::make_slice(begin, end);
  }

  size_t size(){return n;}
  size_t matches(){return num_matches;}
};


struct BuildParams{
  long R; //vamana and pynnDescent
  long L; //vamana
  double m_l = 0; // HNSW
  double alpha; //vamana and pyNNDescent
  int num_passes; //vamana

  long num_clusters; // HCNNG and pyNNDescent
  long cluster_size; //HCNNG and pyNNDescent
  long MST_deg; //HCNNG

  double delta; //pyNNDescent
  
  bool verbose;

  int quantize = 0; // use quantization for build and query (0 = none, 1 = one-level, 2 = two-level)
  double radius; // for radius search
  double radius_2; // for radius search
  bool self;
  bool range;
  int single_batch; //vamana
  long Q = 0; //beam width to pass onto query (0 indicates none specified)
  double trim = 0.0; // for quantization
  double rerank_factor = 100; // for reranking, k * factor = to rerank

  std::string alg_type;

  BuildParams(long R, long L, double a, int num_passes, long nc, long cs, long mst, double de,
              bool verbose = false, int quantize = 0, double radius = 0.0, double radius_2 = 0.0,
              bool self = false, bool range = false, int single_batch = 0, long Q = 0, double trim = 0.0,
              int rerank_factor = 100)
    : R(R), L(L), alpha(a), num_passes(num_passes), num_clusters(nc), cluster_size(cs), MST_deg(mst), delta(de),
      verbose(verbose), quantize(quantize), radius(radius), radius_2(radius_2), self(self), range(range), single_batch(single_batch), Q(Q), trim(trim), rerank_factor(rerank_factor) {
    if(R != 0 && L != 0 && alpha != 0){alg_type = m_l>0? "HNSW": "Vamana";}
    else if(num_clusters != 0 && cluster_size != 0 && MST_deg != 0){alg_type = "HCNNG";}
    else if(R != 0 && alpha != 0 && num_clusters != 0 && cluster_size != 0 && delta != 0){alg_type = "pyNNDescent";}
  }

  BuildParams() {}

  BuildParams(long R, long L, double a, int num_passes, bool verbose = false)
    : R(R), L(L), alpha(a), num_passes(num_passes), verbose(verbose), single_batch(0)
  {alg_type = "Vamana";}

  BuildParams(long R, long L, double m_l, double a)
    : R(R), L(L), m_l(m_l), alpha(a), verbose(false)
  {alg_type = "HNSW";}

  BuildParams(long nc, long cs, long mst)
    : num_clusters(nc), cluster_size(cs), MST_deg(mst), verbose(false)
  {alg_type = "HCNNG";}

  BuildParams(long R, double a, long nc, long cs, double de)
    : R(R), alpha(a), num_clusters(nc), cluster_size(cs), delta(de), verbose(false)
  {alg_type = "pyNNDescent";}

  long max_degree(){
    if(alg_type == "HCNNG") return num_clusters*MST_deg;
    else if(alg_type == "HNSW")  return R*2;
    else return R;
  }
};


struct QueryParams{
  long k;
  long beamSize;
  double cut;
  long limit;
  long degree_limit;
  int rerank_factor = 100;
  float pad = 1.0;

  QueryParams(long k, long Q, double cut, long limit, long dg, double rerank_factor = 100) : k(k), beamSize(Q), cut(cut), limit(limit), degree_limit(dg), rerank_factor(rerank_factor) {}

  QueryParams() {}

};

struct RangeParams{
  double rad;
  long initial_beam;

  RangeParams(double rad, long ib) : rad(rad), initial_beam(ib) {}

  RangeParams() {}

  void print(){
    std::cout << "Beam: " << initial_beam;
  }

};


template<typename T, typename Point>
class Desc_HNSW{
public:
  typedef T type_elem;
  typedef Point type_point;
  static auto distance(const type_point &u, const type_point &v, uint32_t dim)
  {
    (void)dim;
    return u.distance(v);
  }

  static auto get_id(const type_point &u)
  {
    return u.id();
  }
};

#endif
} // end namespace

#endif // ALGORITHMS_ANN_TYPES_H_
