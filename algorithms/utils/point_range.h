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

template<typename T, class Point>
struct PointRange{

  long dimension(){return dims;}
  long aligned_dimension(){return aligned_dims;}

  PointRange(){}

  PointRange(char* filename){
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
      base_size = num_points;
      learn_size = 0;
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

    PointRange(char* filename, char* sFile){
      if(filename == NULL) {
        n = 0;
        dims = 0;
        return;
      } if(sFile == NULL){
        std::ifstream reader(filename);
        assert(reader.is_open());

        //read num points and max degree
        unsigned int num_points;
        unsigned int d;
        reader.read((char*)(&num_points), sizeof(unsigned int));
        n = num_points;
        base_size = num_points;
        learn_size = 0;
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
        return;
      }
      std::ifstream reader(filename);
      assert(reader.is_open());

      //read num points and max degree
      unsigned int num_points;
      unsigned int d;
      reader.read((char*)(&num_points), sizeof(unsigned int));
      base_size = num_points;
      reader.read((char*)(&d), sizeof(unsigned int));
      dims = d;
      std::cout << "Detected " << base_size << " base points with dimension " << d << std::endl;

      std::ifstream sample_reader(sFile);
      assert(sample_reader.is_open());

      //read num points and max degree
      unsigned int num_sample_points;
      unsigned int sample_d;
      sample_reader.read((char*)(&num_sample_points), sizeof(unsigned int));
      learn_size = num_sample_points;
      sample_reader.read((char*)(&sample_d), sizeof(unsigned int));
      assert(sample_d == d);
      n = base_size + learn_size; 
      std::cout << "Detected " << learn_size << " sample points with dimension " << sample_d << std::endl;


      aligned_dims =  dim_round_up(dims, sizeof(T));
      if(aligned_dims != dims) std::cout << "Aligning dimension to " << aligned_dims << std::endl;
      values = (T*) aligned_alloc(64, n*aligned_dims*sizeof(T));
      size_t BLOCK_SIZE = 1000000;
      size_t index = 0;
      while(index < base_size){
          size_t floor = index;
          size_t ceiling = index+BLOCK_SIZE <= base_size ? index+BLOCK_SIZE : base_size;
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

      index = 0;
      while(index < learn_size){
          size_t floor = index;
          size_t ceiling = index+BLOCK_SIZE <= learn_size ? index+BLOCK_SIZE : learn_size;
          T* data_start = new T[(ceiling-floor)*dims];
          sample_reader.read((char*)(data_start), sizeof(T)*(ceiling-floor)*dims);
          T* data_end = data_start + (ceiling-floor)*dims;
          parlay::slice<T*, T*> data = parlay::make_slice(data_start, data_end);
          int data_bytes = dims*sizeof(T);
          parlay::parallel_for(floor, ceiling, [&] (size_t i){
            std::memmove(values + (i+base_size)*aligned_dims, data.begin() + (i-floor)*dims, data_bytes);
          });
          delete[] data_start;
          index = ceiling;
      }
      std::cout << "Built point set with " << n << " total points" << std::endl;
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

  size_t size() { return n; }
  size_t b_size() {return base_size;}
  size_t l_size() {return learn_size;}

  bool is_learn(long i){
    if(i >= n || i < 0){
      std::cout << "Error: invalid id " << i << std::endl;
      abort();
    }
    return (i >= base_size);
  }
  
  Point operator [] (long i) {
    if(i >= base_size) return Point(values+i*aligned_dims, dims, aligned_dims, i, true);
    else return Point(values+i*aligned_dims, dims, aligned_dims, i, false);
  }

private:
  T* values;
  unsigned int dims;
  unsigned int aligned_dims;
  size_t n;
  size_t learn_size;
  size_t base_size;
};
