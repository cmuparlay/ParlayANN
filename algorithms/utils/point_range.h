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

#include "../bench/parse_command_line.h"
#include "types.h"
// #include "common/time_loop.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

//tp_size must divide 64 evenly--no weird/large types!
long dim_round_up(long dim, long tp_size){
  long qt = (dim*tp_size)/64;
  long remainder = (dim*tp_size)%64;
  if(remainder == 0) return dim;
  else return ((qt+1)*64)/tp_size;
}

template<typename T, class Point, class PR>
struct SubsetPointRange;

template<typename T, class Point>
struct PointRange{

  long dimension() const {return dims;}
  long aligned_dimension() const {return aligned_dims;}

  PointRange(){}

  PointRange(const char* filename){
      if(filename == NULL) {
        n = 0;
        dims = 0;
        return;
      }
      std::ifstream reader(filename);
      assert(reader.is_open());

      //read num points and max degree
      unsigned int num_points;
      unsigned int d;
      reader.read((char*)(&num_points), sizeof(unsigned int));
      n = num_points;
      reader.read((char*)(&d), sizeof(unsigned int));
      dims = d;
      std::cout << "Detected " << num_points << " points with dimension " << d << std::endl;
      aligned_dims =  dim_round_up(dims, sizeof(T));
      if(aligned_dims != dims) std::cout << "Aligning dimension to " << aligned_dims << std::endl;
      values = (T*) aligned_alloc(64, n*aligned_dims*sizeof(T));
      size_t BLOCK_SIZE = 1000000;
      size_t index = 0;
      while(index < n){
          size_t floor = index;
          size_t ceiling = index+BLOCK_SIZE <= n ? index+BLOCK_SIZE : n;
          T* data_start = new T[(ceiling-floor)*dims];
          reader.read((char*)(data_start), sizeof(T)*(ceiling-floor)*dims);
          T* data_end = data_start + (ceiling-floor)*dims;
          parlay::slice<T*, T*> data = parlay::make_slice(data_start, data_end);
          int data_bytes = dims*sizeof(T);
          parlay::parallel_for(floor, ceiling, [&] (size_t i){
            std::memmove(values + i*aligned_dims, data.begin() + (i-floor)*dims, data_bytes);
          });
          delete[] data_start;
          index = ceiling;
      }
  }

  /* a constructor which does not assume points are being read from a file 
  
  I would make a constructor which takes a numpy array but I don't want a pybind dependency in this file
  */
  PointRange(T* values, size_t n, unsigned int dims){
    this->n = n;
    this->dims = dims;
    aligned_dims = dim_round_up(dims, sizeof(T));
    if(aligned_dims != dims) std::cout << "Aligning dimension to " << aligned_dims << std::endl;
    this->values = (T*) aligned_alloc(64, n*aligned_dims*sizeof(T));
    parlay::parallel_for(0, n, [&] (size_t i){
      std::memcpy(this->values + i*aligned_dims, values + i*dims, dims*sizeof(T));
    });
  }

  std::unique_ptr<SubsetPointRange<T, Point, PointRange<T, Point>>> make_subset(parlay::sequence<int32_t> subset) {
      return std::make_unique<SubsetPointRange<T, Point, PointRange<T, Point>>>(this, subset);
    }

  // PointRange(char* filename) {
  //   if(filename == NULL) {
  //     n = 0;
  //     dims = 0;
  //     return;
  //   }
  //   auto [fileptr, length] = mmapStringFromFile(filename);
  //   int num_vectors = *((int*) fileptr);
  //   int d = *((int*) (fileptr+4));
  //   n = num_vectors;
  //   dims = d;
  //   aligned_dims = dims;
  //   values = (T*)(fileptr+8);
  //   std::cout << "Detected " << n
	//       << " points with dimension " << dims << std::endl;
  // }

  size_t size() const { return n; }
  
  Point operator [] (long i) {
    return Point(values+i*aligned_dims, dims, aligned_dims, i);
  }

private:
  T* values;
  unsigned int dims;
  unsigned int aligned_dims;
  size_t n;
};

/* a wrapper around PointRange which uses only a subset of the points

  Note that when indexing into the subset, the indices are relative to the included points, not the actual indices of the points in the original PointRange
 */
template<typename T, class Point, class PR = PointRange<T, Point>>
struct SubsetPointRange {
    PR *pr = nullptr;
    parlay::sequence<int32_t> subset;
    std::unordered_map<int32_t, int32_t> real_to_subset;
    size_t n;
    unsigned int dims;
    unsigned int aligned_dims;

    // in dire circumstances, we will want to initialize a subset point range which is actually a normal point range. This is a hack to allow that. If only there was a feature of OOP which would obviate this...
    // the unique ptr just protects us from a memory leak
    std::unique_ptr<PointRange<T, Point>> heap_point_range = nullptr;

    SubsetPointRange() {}

    SubsetPointRange(PR &pr, parlay::sequence<int32_t> subset) : pr(&pr), subset(subset) {
      n = subset.size();
      dims = pr.dimension();
      aligned_dims = pr.aligned_dimension();

      real_to_subset = std::unordered_map<int32_t, int32_t>();
      real_to_subset.reserve(n);
      for(int32_t i=0; i<n; i++) {
        real_to_subset[subset[i]] = i;
      }
    }

    SubsetPointRange(PR *pr, parlay::sequence<int32_t> subset) : pr(pr), subset(subset) {
      n = subset.size();
      dims = pr->dimension();
      aligned_dims = pr->aligned_dimension();

      real_to_subset = std::unordered_map<int32_t, int32_t>();
      real_to_subset.reserve(n);
      for(int32_t i=0; i<n; i++) {
        real_to_subset[subset[i]] = i;
      }
    }

    /* constructor from a twisted parallel dimension where inheritance doesn't exist */
    SubsetPointRange(T* values, size_t n, unsigned int dims){
      heap_point_range = std::make_unique<PointRange<T, Point>>(values, n, dims);
      pr = heap_point_range.get();
      subset = parlay::tabulate(n, [&] (int32_t i) {return i;});
      this->n = n;
      this->dims = dims;
      aligned_dims = pr->aligned_dimension();
      
      real_to_subset = std::unordered_map<int32_t, int32_t>();
      real_to_subset.reserve(n);
      for(int32_t i=0; i<n; i++) {
        real_to_subset[subset[i]] = i;
      }
    }

    // Move constructor
    SubsetPointRange(SubsetPointRange&& other) noexcept
        : pr(std::exchange(other.pr, nullptr)),
          subset(std::move(other.subset)),
          real_to_subset(std::move(other.real_to_subset)),
          n(std::exchange(other.n, 0)),
          dims(std::exchange(other.dims, 0)),
          aligned_dims(std::exchange(other.aligned_dims, 0)),
          heap_point_range(std::move(other.heap_point_range)) {}

    // Move assignment operator
    SubsetPointRange& operator=(SubsetPointRange&& other) noexcept {
        if (this != &other) {
            pr = std::exchange(other.pr, nullptr);
            subset = std::move(other.subset);
            real_to_subset = std::move(other.real_to_subset);
            n = std::exchange(other.n, 0);
            dims = std::exchange(other.dims, 0);
            aligned_dims = std::exchange(other.aligned_dims, 0);
            heap_point_range = std::move(other.heap_point_range);
        }
        return *this;
    }

    // Copy constructor with shallow copy semantics for heap_point_range
    SubsetPointRange(const SubsetPointRange& other)
        : pr(other.pr),  // Copy the pointer directly (shallow copy)
          subset(other.subset),
          real_to_subset(other.real_to_subset),
          n(other.n),
          dims(other.dims),
          aligned_dims(other.aligned_dims),
          heap_point_range(nullptr) {  // Do not take ownership of the heap_point_range
        if (other.heap_point_range != nullptr) {
            // Instead of copying the unique_ptr, just copy the raw pointer to maintain a shallow copy.
            // This is a deliberate choice to avoid ownership transfer of heap_point_range.
            heap_point_range = std::make_unique<PointRange<T, Point>>(*other.heap_point_range.get());
        }
    }
  
    size_t size() const { return n; }
  
    Point operator [] (long i) {
      return (*pr)[subset[i]];
    }

    long dimension() const {return dims;}
    long aligned_dimension() const {return aligned_dims;}

    int32_t real_index(int32_t i) const {
      return subset[i];
    }

    int32_t subset_index(int32_t i) const {
      // return real_to_subset.at(i);
      return std::find(subset.begin(), subset.end(), i) - subset.begin();
    }

    /* creates a subset of this subset without causing a chain of redirects every access
    
    subset should be provided with indices relative to the full dataset */
    std::unique_ptr<SubsetPointRange<T, Point, PR>> make_subset(parlay::sequence<int32_t> subset) {
      // parlay::sequence<int32_t> nonlocal_subset = parlay::map(subset, [&] (int32_t i) {return this->subset[i];});
      return std::make_unique<SubsetPointRange<T, Point, PR>>(this->pr, subset);
    }

};