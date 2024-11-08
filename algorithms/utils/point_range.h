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

#include <sys/mman.h>
#include <algorithm>
#include <iostream>

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

template<class Point_>
struct PointRange{
  //using T = T_;
  using Point = Point_;
  using parameters = typename Point::parameters;
  using byte = uint8_t;

  long dimension() const {return params.dims;}
  //long aligned_dimension() const {return aligned_dims;}

  PointRange() : values(std::shared_ptr<byte[]>(nullptr, std::free)), n(0) {}

  template <typename PR>
  PointRange(const PR& pr, const parameters& p) : params(p)  {
    n = pr.size();
    int num_bytes = p.num_bytes();
    aligned_bytes = (num_bytes <= 32) ? 32 : 64 * ((num_bytes - 1)/64 + 1);
    long total_bytes = n * aligned_bytes;
    byte* ptr = (byte*) aligned_alloc(1l << 21, total_bytes);
    madvise(ptr, total_bytes, MADV_HUGEPAGE);
    values = std::shared_ptr<byte[]>(ptr, std::free);
    byte* vptr = values.get();
    parlay::parallel_for(0, n, [&] (long i) {
      Point::translate_point(vptr + i * aligned_bytes, pr[i], params);});
  }

  template <typename PR>
  PointRange (PR& pr) : PointRange(pr, Point::generate_parameters(pr)) { }

  template <typename PR>
  PointRange (PR& pr, int dims) : PointRange(pr, Point::generate_parameters(dims)) { }

  PointRange(char* filename) : values(std::shared_ptr<byte[]>(nullptr, std::free)){
      if(filename == NULL) {
        n = 0;
        return;
      }
      std::ifstream reader(filename);
      if (!reader.is_open()) {
        std::cout << "Data file " << filename << " not found" << std::endl;
        std::abort();
      }

      //read num points and max degree
      unsigned int num_points;
      unsigned int d;
      reader.read((char*)(&num_points), sizeof(unsigned int));
      n = num_points;
      reader.read((char*)(&d), sizeof(unsigned int));
      params = parameters(d);
      std::cout << "Data: detected " << num_points << " points with dimension " << d << std::endl;
      int num_bytes = params.num_bytes();
      aligned_bytes =  64 * ((num_bytes - 1)/64 + 1);
      if (aligned_bytes != num_bytes)
        std::cout << "Aligning bytes to " << aligned_bytes << std::endl;
      long total_bytes = n * aligned_bytes;
      byte* ptr = (byte*) aligned_alloc(1l << 21, total_bytes);
      madvise(ptr, total_bytes, MADV_HUGEPAGE);
      values = std::shared_ptr<byte[]>(ptr, std::free);
      size_t BLOCK_SIZE = 1000000;
      size_t index = 0;
      while(index < n) {
          size_t floor = index;
          size_t ceiling = index+BLOCK_SIZE <= n ? index+BLOCK_SIZE : n;
          long m = ceiling - floor;
          byte* data_start = new byte[m * num_bytes];
          reader.read((char*)(data_start), m * num_bytes);
          parlay::parallel_for(floor, ceiling, [&] (size_t i) {
            std::memmove(values.get() + i * aligned_bytes,
                         data_start + (i - floor) * num_bytes,
                         num_bytes);
          });
          delete[] data_start;
          index = ceiling;
      }
  }

  size_t size() const { return n; }

  unsigned int get_dims() const { return params.dims; }
  
  Point operator [] (long i) const {
    if (i > n) {
      std::cout << "ERROR: point index out of range: " << i << " from range " << n << ", " << std::endl;
      abort();
    }
    return Point(values.get()+i*aligned_bytes, i, params);
  }

  byte* location(long i) const {
    return values.get() + i * aligned_bytes;
  }
  
  parameters params;

private:
  std::shared_ptr<byte[]> values;
  long aligned_bytes;
  size_t n;
};

} // end namespace
