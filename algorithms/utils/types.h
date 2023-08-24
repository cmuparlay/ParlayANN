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

// template<typename T>
// struct store{
// public:  
//   size_t size;
//   unsigned int d;
//   parlay::slice<T*, T*> coordinates;
//   T* start;

//   store(size_t n, unsigned int d, parlay::slice<T*, T*> coordinates) : size(n), d(d), coordinates(coordinates){
//     start = coordinates.begin();
//   }

//   T* get(int i){
//     return start + d*i; 
//   }

//   void prefetch(T* p) {
//     int l = (d * sizeof(T))/64;
//     for (int i=0; i < l; i++)
//       __builtin_prefetch((char*) p + i* 64);
//   }

// };

// template<typename T>
// struct exp_store : public store<T> {
//   public: 
//     Distance* D;
//     double alpha;

//     exp_store(size_t n, unsigned int d, Distance* D, parlay::slice<T*, T*> coordinates, double alpha = 1.0) : store<T>(n, d, coordinates), D(D), alpha(alpha) { }

//     float distance(int i, int j){return D->distance(start+i*d, start+j*d, d);}
//     float distance(int i, T* c){return D->distance(start+i*d, c, d);}
//     float distance(T* c, int i){return D->distance(start+i*d, c, d);}
//     float distance(T* a, T* b){return D->distance(a, b, d);}
// };

template<typename T>
struct data_store{
    Distance* D;
    double alpha;
    size_t size;
    unsigned int d;
    parlay::slice<T*, T*> coordinates;
    T* start;

    data_store(size_t n, unsigned int d, Distance* D, parlay::slice<T*, T*> coordinates, double alpha = 1.0) : size(n), d(d), coordinates(coordinates), D(D), alpha(alpha) {
      start = coordinates.begin();
    }

    T* get(int i){
      return start + d*i; 
    }

    void prefetch(T* p) {
      int l = (d * sizeof(T))/64;
      for (int i=0; i < l; i++)
        __builtin_prefetch((char*) p + i* 64);
    }

    float distance(int i, int j){return D->distance(start+i*d, start+j*d, d);}
    float distance(int i, T* c){return D->distance(start+i*d, c, d);}
    float distance(T* c, int i){return D->distance(start+i*d, c, d);}
    float distance(T* a, T* b){return D->distance(a, b, d);}
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
