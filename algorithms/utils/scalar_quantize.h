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
#include "point_range.h"

//quantizes based on a global min and max
template<typename PointRange, typename Point, typename T>
std::pair<parlay::sequence<T>, std::pair<float, float>> scalar_quantize_float_coarse(PointRange &Points){
    float min_coord = std::numeric_limits<float>::max();
    float max_coord = -std::numeric_limits<float>::min();
    for(long i=0; i<Points.dimension(); i++){
        auto vals = parlay::tabulate(Points.size(), [&] (size_t j) {
            return (Points[j])[i];
        });
        parlay::sort_inplace(vals);
        if(vals[0] < min_coord) min_coord = vals[0];
        if(vals[vals.size()-1] > max_coord) max_coord = vals[vals.size()-1];
    }
    std::cout << "Maximum coord: " << max_coord << ", Min coord: " << min_coord << std::endl;
    T maxval = std::numeric_limits<T>::max();
    auto quantized_data = parlay::flatten(parlay::tabulate(Points.size(), [&] (size_t i){
        parlay::sequence<T> quantized_vals = parlay::tabulate(Points.dimension(), [&] (size_t j){
            return static_cast<T>(floor(maxval * (Points[i][j] - min_coord)/(max_coord-min_coord)));
        });
        return quantized_vals;
    }));
    return std::make_pair(quantized_data, std::make_pair(max_coord, min_coord));
}

template<typename T>
parlay::sequence<float> decode(const T* vals, unsigned int d, float max_coord, float min_coord){
    parlay::sequence<float> decoded(d);
    float maxval = static_cast<float>(std::numeric_limits<T>::max());
    float delta = max_coord - min_coord;
    float mult = delta/maxval;
    // std::cout << "Max coord at decode: " << max_coord << std::endl;
    // std::cout << "Min coord at decode: " << min_coord << std::endl;
    // std::cout << "Delta: " << delta << std::endl;
    // std::cout << "Multiplying by: " << mult << std::endl;
    for(int i=0; i<d; i++){
        decoded[i] = static_cast<float>(vals[i])*mult + min_coord;
        // std::cout << decoded[i] << std::endl;
    }
    return decoded;
}


template<typename Point, typename T>
struct QuantizedPointRange{

    long dimension(){return dims;}
    long aligned_dimension(){return aligned_dims;}
    size_t size(){return n;}
    bool is_learn(long i) {return false;}

    Point operator [] (long i) {
        // std::cout << "Max coord at point range: " << max_coord << std::endl;
        // std::cout << "Min coord at point range: " << min_coord << std::endl;
        return Point(values+i*aligned_dims, dims, aligned_dims, i, max_coord, min_coord);
    }

    QuantizedPointRange(){}

    template<typename PointRange>
    QuantizedPointRange(PointRange &Points){
        auto [quantized_data, quantization_vals] = scalar_quantize_float_coarse<PointRange, Point, T>(Points);
        auto [maxc, minc] = quantization_vals;
        max_coord = maxc;
        min_coord = minc;
        // std::cout << "Max coord at point range construction: " << max_coord << std::endl;
        // std::cout << "Min coord at point range construction: " << min_coord << std::endl;
        n = Points.size();
        dims = Points.dimension();
        std::cout << "Detected " << n << " points with dimension " << dims << std::endl;
        aligned_dims =  dim_round_up(dims, sizeof(T));
        if(aligned_dims != dims) std::cout << "Aligning dimension to " << aligned_dims << std::endl;
        values = (T*) aligned_alloc(64, n*aligned_dims*sizeof(T));
        size_t BLOCK_SIZE = 1000000;
        size_t index = 0;
        while(index < n){
            size_t floor = index;
            size_t ceiling = index+BLOCK_SIZE <= n ? index+BLOCK_SIZE : n;
            int data_bytes = dims*sizeof(T);
            parlay::parallel_for(floor, ceiling, [&] (size_t i){
                std::memmove(values + i*aligned_dims, quantized_data.begin() + i*dims, data_bytes);
            });
            index = ceiling;
        }
    }

    // void save(char* save_path){
    // std::cout << "Writing data with " << n << " points and dimension " << dims
    //                 << std::endl;
    //   parlay::sequence<T> preamble = {static_cast<T>(n), static_cast<T>(dims)};
    //   std::ofstream writer;
    //   writer.open(save_path, std::ios::binary | std::ios::out);
    //   writer.write((char*)preamble.begin(), 2 * sizeof(T));
    // }

    private:
        T* values;
        unsigned int dims;
        unsigned int aligned_dims;
        size_t n;
        float max_coord;
        float min_coord;
};

