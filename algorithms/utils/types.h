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

#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "mmap.h"

template<typename T>
struct groundTruth{
  parlay::slice<T*, T*> coords;
  parlay::slice<float*, float*> dists;
  long dim;
  size_t n;

  groundTruth(char* gtFile) : coords(parlay::make_slice<T*, T*>(nullptr, nullptr)),
    dists(parlay::make_slice<float*, float*>(nullptr, nullptr)){
      if(gtFile == NULL){
        this->n = 0;
        this->dim = 0;
      } else{
        auto [fileptr, length] = mmapStringFromFile(gtFile);

        int num_vectors = *((T*) fileptr);
        int d = *((T*) (fileptr+4));

        std::cout << "Detected " << num_vectors << " points with num results " << d << std::endl;

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

  T coordinates(long i, long j){return *(coords.begin() + i*dim + j);}

  float distances(long i, long j){return *(dists.begin() + i*dim + j);}

  size_t size(){return n;}

  long dimension(){return dim;}

};


struct BuildParams{
  long L; //vamana
  long R; //vamana and pynnDescent
  double alpha; //vamana and pyNNDescent

  long num_clusters; // HCNNG and pyNNDescent
  long cluster_size; //HCNNG and pyNNDescent
  long MST_deg; //HCNNG

  double delta; //pyNNDescent

  std::string alg_type;

  BuildParams() {}

  BuildParams(long R, long L, double a) : R(R), L(L), alpha(a) {}

  BuildParams(long R, long L, double a, long nc, long cs, long mst, double de) : R(R), L(L), alpha(a), num_clusters(nc), cluster_size(cs), MST_deg(mst), delta(de) {
    if(R != 0 && L != 0 && alpha != 0){alg_type = "Vamana";}
    else if(num_clusters != 0 && cluster_size != 0 && MST_deg != 0){alg_type = "HCNNG";}
    else if(R != 0 && alpha != 0 && num_clusters != 0 && cluster_size != 0 && delta != 0){alg_type = "pyNNDescent";}
  }

  long max_degree(){
    if(alg_type == "HCNNG") return num_clusters*MST_deg;
    else return R;
  }

  std::string toStringVamana() {
    return "R_" + std::to_string(R) + "_L_" + std::to_string(L) + "_a_" + std::to_string(std::round(alpha*100)/100);
  }
};


struct QueryParams{
  long k;
  long beamSize; 
  double cut;
  long limit;
  long degree_limit;

  QueryParams(long k, long Q, double cut, long limit, long dg) : k(k), beamSize(Q), cut(cut), limit(limit), degree_limit(dg) {}

  QueryParams() {}

};



#endif
