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

        this->n = num_vectors;
        this->dim = d;
        this->coords = parlay::make_slice(start_coords, end_coords);
        this->dists = parlay::make_slice(start_dists, end_dists);
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

  BuildParams(long R, long L, double a, long nc, long cs, long mst, double de) : R(R), L(L), alpha(a), num_clusters(nc), cluster_size(cs), MST_deg(mst), delta(de) {}
};


//for a file in .fvecs or .bvecs format, but extendible to other types
template<typename T>
struct alignas(64) Tvec_point {
  int id;
  size_t visited;
  size_t dist_calls;
  int rounds;
  parlay::slice<T*, T*> coordinates;
  parlay::slice<int*, int*> out_nbh;
  parlay::slice<int*, int*> new_nbh;
  Tvec_point()
      : coordinates(parlay::make_slice<T*, T*>(nullptr, nullptr)),
        out_nbh(parlay::make_slice<int*, int*>(nullptr, nullptr)),
        new_nbh(parlay::make_slice<int*, int*>(nullptr, nullptr)) {}
  parlay::sequence<int> ngh = parlay::sequence<int>();
};

// for an ivec file, which contains the ground truth
// only info needed is the coordinates of the nearest neighbors of each point
struct ivec_point {
  int id;
  parlay::slice<int*, int*> coordinates;
  parlay::slice<float*, float*> distances;
  ivec_point()
      : coordinates(parlay::make_slice<int*, int*>(nullptr, nullptr)),
        distances(parlay::make_slice<float*, float*>(nullptr, nullptr)) {}
};

#endif
