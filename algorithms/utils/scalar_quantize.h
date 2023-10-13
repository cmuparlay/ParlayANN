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
#include <bitset>

#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "point_range.h"

template<typename T>
int compute_packed_dim(size_t qv, int bits){
    int total_bits = bits*qv;
    int packed_size;
    int qt = total_bits/(sizeof(T)*8);
    int remainder = total_bits % (sizeof(T)*8);
    if(remainder == 0) packed_size = qt;
    else packed_size = qt+1;
    std::cout << "Packing " << qv << " dimensions into " << packed_size << " dimensions" << std::endl;
    return packed_size;
}

template<typename T>
parlay::sequence<T> pack(parlay::sequence<T> to_pack, int bits, int packed_size){
    parlay::sequence<T> packed(packed_size);
    int T_bits = sizeof(T)*8;
    int packed_size_bits = packed_size*T_bits;
    int bits_to_pack = bits*to_pack.size();
    if(packed_size_bits < bits_to_pack) abort();
    int bits_packed = 0;
    for(int i=0; i<to_pack.size(); i++){
        int current_index = bits_packed/T_bits;
        int remaining_bits = T_bits - (bits_packed % (T_bits));
        int to_shift = T_bits-remaining_bits;
        //shift all into one integer
        if(remaining_bits >= bits){
            T shifted_element = (to_pack[i] << to_shift);
            packed[current_index] |= (to_pack[i] << to_shift);
        }else{ //split across two integers
            //mask first part and shift
            T pre_mask = static_cast<T>(((((size_t) 1) << remaining_bits)-1));
            T mask_first = pre_mask << (bits - remaining_bits);
            T first_shifted = to_pack[i] & mask_first;
            packed[current_index] |= (first_shifted << (T_bits-bits));
            //mask second part and shift into next integer
            T second_mask = static_cast<T>(((((size_t) 1) << (bits - remaining_bits))-1));
            packed[current_index+1] |= (to_pack[i] & second_mask);  
        }
        bits_packed += bits;
    }
    return packed;
}

template<typename T>
parlay::sequence<T> unpack(parlay::sequence<T> to_unpack, int bits, int d){
    int T_bits = sizeof(T)*8;
    int num_elts = to_unpack.size()*(T_bits)/bits;
    if(num_elts < d) abort();
    parlay::sequence<T> unpacked(num_elts);
    int bits_unpacked = 0;
    for(int i=0; i<d; i++){
        int current_index = bits_unpacked/T_bits;
        int remaining_bits = T_bits - (bits_unpacked % T_bits);
        int already_shifted = T_bits-remaining_bits;
        //unshift all from one index
        if(remaining_bits >= bits){
            T mask_to_shift = static_cast<T>(((((size_t) 1) << bits)-1));
            T mask = mask_to_shift << already_shifted;
            unpacked[i] = (to_unpack[current_index] & mask) >> (already_shifted);
        }else{ //unshift from multiple indices
            T first_mask_to_shift = static_cast<T>(((((size_t) 1) << remaining_bits)-1));
            T first_mask = first_mask_to_shift << already_shifted;
            T first_part = (to_unpack[current_index] & first_mask) >> (T_bits - bits);
            T second_mask = static_cast<T>(((((size_t) 1) << (bits-remaining_bits))-1));
            T second_part = to_unpack[current_index+1] & second_mask;
            unpacked[i] = first_part | second_part;
        }
        bits_unpacked += bits;
    }
    return unpacked;
}

//quantizes based on a global min and max
template<typename PointRange, typename Point, typename T>
std::pair<parlay::sequence<T>, std::pair<float, float>> scalar_quantize_float_coarse(PointRange &Points, int bits, int qd){
    if(std::ceil(bits/8) > sizeof(T)) abort();
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
    T maxval = static_cast<T>((((size_t) 1) << bits)-1);
    std::cout << "Max val: " << maxval << std::endl;
    int d = Points.dimension();
    auto quantized_data = parlay::flatten(parlay::tabulate(Points.size(), [&] (size_t i){
        parlay::sequence<T> quantized_vals = parlay::tabulate(Points.dimension(), [&] (size_t j){
            return static_cast<T>(floor(maxval * (Points[i][j] - min_coord)/(max_coord-min_coord)));
        });
        return pack<T>(quantized_vals, bits, qd);
    }));
    return std::make_pair(quantized_data, std::make_pair(max_coord, min_coord));
}

template<typename T>
parlay::sequence<float> decode(const T* vals, unsigned int d, unsigned int qd, float max_coord, float min_coord, int bits){
    parlay::sequence<T> unpacked = unpack<T>(parlay::tabulate(qd, [&] (size_t i){return vals[i];}), bits, d);
    parlay::sequence<float> decoded(d);
    float maxval = static_cast<float>(static_cast<T>((((size_t) 1) << bits)-1));
    float delta = max_coord - min_coord;
    float mult = delta/maxval;
    for(int i=0; i<d; i++){
        decoded[i] = static_cast<float>(unpacked[i])*mult + min_coord;
    }
    return decoded;
}


template<typename Point, typename T>
struct QuantizedPointRange{

    long dimension(){return dims;}
    long aligned_dimension(){return aligned_dims;}
    long quantized_dimension(){return quantized_dims;}
    size_t size(){return n;}
    bool is_learn(long i) {return false;}

    Point operator [] (long i) {
        return Point(values+i*aligned_dims, dims, quantized_dims, aligned_dims, i, max_coord, min_coord, bits);
    }

    QuantizedPointRange(){}

    template<typename PointRange>
    QuantizedPointRange(PointRange &Points, int bits) : bits(bits) {
        n = Points.size();
        dims = Points.dimension();
        std::cout << "Detected " << n << " points with dimension " << dims << std::endl;
        quantized_dims = compute_packed_dim<T>(dims, bits);
        auto [quantized_data, quantization_vals] = scalar_quantize_float_coarse<PointRange, Point, T>(Points, bits, quantized_dims);
        auto [maxc, minc] = quantization_vals;
        max_coord = maxc;
        min_coord = minc;
        aligned_dims =  dim_round_up(quantized_dims, sizeof(T));
        if(aligned_dims != quantized_dims) std::cout << "Aligning quantized dimension to " << aligned_dims << std::endl;
        values = (T*) aligned_alloc(64, n*aligned_dims*sizeof(T));
        size_t BLOCK_SIZE = 1000000;
        size_t index = 0;
        while(index < n){
            size_t floor = index;
            size_t ceiling = index+BLOCK_SIZE <= n ? index+BLOCK_SIZE : n;
            int data_bytes = quantized_dims*sizeof(T);
            parlay::parallel_for(floor, ceiling, [&] (size_t i){
                std::memmove(values + i*aligned_dims, quantized_data.begin() + i*quantized_dims, data_bytes);
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
        unsigned int quantized_dims;
        unsigned int aligned_dims;
        size_t n;
        float max_coord;
        float min_coord;
        int bits;
};

