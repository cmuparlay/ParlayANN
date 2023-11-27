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
// #include "NSGDist.h"

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

  
template<typename T, class Point>
struct PointRange{

  long dimension(){return dims;}
  long aligned_dimension(){return aligned_dims;}

  PointRange() : values(std::shared_ptr<T[]>(nullptr, std::free)) {n=0;}

  PointRange(char* filename) : values(std::shared_ptr<T[]>(nullptr, std::free)){
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
      values = std::shared_ptr<T[]>((T*) aligned_alloc(64, n*aligned_dims*sizeof(T)), std::free);
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
            std::memmove(values.get() + i*aligned_dims, data.begin() + (i-floor)*dims, data_bytes);
          });
          delete[] data_start;
          index = ceiling;
      }
  }

  size_t size() { return n; }

  unsigned int get_dims() const { return dims; }
  
  Point operator [] (long i) {
    return Point(values.get()+i*aligned_dims, dims, aligned_dims, i);
  }

private:
  std::shared_ptr<T[]> values;
  unsigned int dims;
  unsigned int aligned_dims;
  size_t n;

};
