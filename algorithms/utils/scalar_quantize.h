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

//packed size = length of array when coordinates are densely packed
template<typename T>
parlay::sequence<T> pack(parlay::sequence<T> to_pack, int bits, int packed_size){
    parlay::sequence<T> packed(packed_size);
    int T_bits = sizeof(T)*8;
    //sanity check
    int packed_size_bits = packed_size*T_bits;
    int bits_to_pack = bits*to_pack.size();
    if(packed_size_bits < bits_to_pack) abort();
    //packing work begins
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

//add a local buffer and do all operations using that buffer
//don't use sequences in function
//keep a running sum for dot product instead of using separate functions
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
    float min_coord = std::numeric_limits<T>::max();
    float max_coord = -std::numeric_limits<T>::min();
    for(long i=0; i<Points.dimension(); i++){
        auto vals = parlay::tabulate(Points.size(), [&] (size_t j) {
            return (Points[j])[i];
        });
        parlay::sort_inplace(vals);
        if(vals[0] < min_coord) min_coord = vals[0];
        if(vals[vals.size()-1] > max_coord) max_coord = vals[vals.size()-1];
    }
    std::cout << "Maximum coord: " << max_coord << ", Min coord: " << min_coord << std::endl;
    float maxval = static_cast<T>((((size_t) 1) << bits)-1);
    std::cout << "Max val: " << maxval << std::endl;
    int d = Points.dimension();
    parlay::sequence<T> quantized_data = parlay::sequence<T>(qd*Points.size());
    size_t BLOCK_SIZE = 1000000;
    size_t index = 0;
    size_t n = Points.size();
    while(index<n){
        size_t fl = index;
        size_t ceiling = index+BLOCK_SIZE <= n ? index+BLOCK_SIZE : n;
        parlay::parallel_for(fl, ceiling, [&] (size_t i){
            parlay::sequence<T> quantized_vals = parlay::tabulate(Points.dimension(), [&] (size_t j){
                T ex = static_cast<T>(floor(maxval * ((float) Points[i][j] - min_coord)/(max_coord-min_coord)));
                return ex;
            });
            parlay::sequence<T> packed_vals = pack<T>(quantized_vals, bits, qd);
            for(size_t j=0; j<packed_vals.size(); j++){
                quantized_data[i*qd+j] = packed_vals[j];
            }
        });
        index = ceiling;
    }
    std::cout << "Finished quantization" << std::endl;
    return std::make_pair(std::move(quantized_data), std::make_pair(max_coord, min_coord));
}

//T is type of quantized vals
//U is type of decoded vals
template<typename T, typename U>
parlay::sequence<float> decode(const T* vals, unsigned int d, unsigned int qd, float max_coord, float min_coord, int bits){
    parlay::sequence<T> unpacked = unpack<T>(parlay::tabulate(qd, [&] (size_t i){return vals[i];}), bits, d);
    parlay::sequence<float> decoded(d);
    float maxval = static_cast<float>(static_cast<T>((((size_t) 1) << bits)-1));
    float delta = max_coord - min_coord;
    float mult = delta/maxval;
    for(int i=0; i<d; i++){
        decoded[i] = static_cast<U>(static_cast<float>(unpacked[i])*mult + min_coord);
    }
    return decoded;
}

// if(remaining_bits >= bits){
//     T mask_to_shift = static_cast<T>(((((size_t) 1) << bits)-1));
//     T mask = mask_to_shift << already_shifted;
//     unpacked[i] = (to_unpack[current_index] & mask) >> (already_shifted);
// }else{ //unshift from multiple indices
//     T first_mask_to_shift = static_cast<T>(((((size_t) 1) << remaining_bits)-1));
//     T first_mask = first_mask_to_shift << already_shifted;
//     T first_part = (to_unpack[current_index] & first_mask) >> (T_bits - bits);
//     T second_mask = static_cast<T>(((((size_t) 1) << (bits-remaining_bits))-1));
//     T second_part = to_unpack[current_index+1] & second_mask;
//     unpacked[i] = first_part | second_part;
// }

//specifically optimized for uint16 coordinates, 10 bits, and 200 dimensions
//DON'T USE IT FOR ANYTHING ELSE, DO YOU HEAR ME?!?!
float wildly_optimized_mips_distance(const uint16_t* q, const float* p, float max_coord, float min_coord){
    int d = 200;
    int bits = 10;
    int type_bits = 16;
    int decode_at_once = 8; //lcd of 16 and 10 is 80
    int loop_iterations = 25; //200 divided by 8; each loop iteration decodes 8 words at once
    float inner_product_total = 0.0;
    float maxval = static_cast<float>(static_cast<uint16_t>((((size_t) 1) << bits)-1));
    float delta = max_coord - min_coord;
    float mult = delta/maxval;
    uint16_t i1, i2, i3, i4, i5, i6, i7, i8;
    uint16_t mask1, mask2, part1, part2;
    for(int i=0; i<loop_iterations; i++){
        int index = i*5;
        
        //first val: 10 bits from index 0
        mask1 = (uint16_t)(((((size_t) 1) << 10)-1));
        i1 = q[index] & mask1;
        //second val: 6 bits from 0, then 4 from 1
        mask1 = (uint16_t)(((((size_t) 1) << 6)-1)) << 10;
        part1 = (q[index] & mask1) >> 6;
        mask2 = (uint16_t)(((((size_t) 1) << 4)-1));
        part2 = q[index+1] & mask2;
        i2 = part1 | part2;
        //third val: 10 bits from 1 
        mask1 = (uint16_t)(((((size_t) 1) << 10)-1)) << 4;
        i3 = (q[index+1] & mask1) >> 4;
        //fourth val: 2 bits from 1, 8 bits from 2
        mask1 = (uint16_t)(((((size_t) 1) << 2)-1)) << 14;
        part1 = (q[index+1] & mask1) >> 6;
        mask2 = (uint16_t)(((((size_t) 1) << 8)-1));
        part2 = q[index+2] & mask2;
        i4 = part1 | part2;
        //fifth val: 8 bits from 2, 2 bits from 3
        mask1 = (uint16_t)(((((size_t) 1) << 8)-1)) << 8;
        part1 = (q[index+2] & mask1) >> 6;
        mask2 = (uint16_t)(((((size_t) 1) << 2)-1));
        part2 = q[index+3] & mask2;
        i5 = part1 | part2;
        //sixth val: 10 bits from 3
        mask1 = (uint16_t)(((((size_t) 1) << 10)-1)) << 2;
        i6 = (q[index+3] & mask1) >> 2;
        //seventh val: 4 bits from 3, 6 bits from 4
        mask1 = (uint16_t)(((((size_t) 1) << 4)-1)) << 12;
        part1 = (q[index+3] & mask1) >> 6;
        mask2 = (uint16_t)(((((size_t) 1) << 6)-1));
        part2 = q[index+4] & mask2;
        i7 = part1 | part2;
        //eighth val: 10 bits from 4
        mask1 = (uint16_t)(((((size_t) 1) << 10)-1)) << 6;
        i8 = (q[index+4] & mask1) >> 6;

        //calculate additions to inner product
        inner_product_total += (static_cast<float>(i1)*mult + min_coord)*p[i*8];
        inner_product_total += (static_cast<float>(i2)*mult + min_coord)*p[i*8+1];
        inner_product_total += (static_cast<float>(i3)*mult + min_coord)*p[i*8+2];
        inner_product_total += (static_cast<float>(i4)*mult + min_coord)*p[i*8+3];
        inner_product_total += (static_cast<float>(i5)*mult + min_coord)*p[i*8+4];
        inner_product_total += (static_cast<float>(i6)*mult + min_coord)*p[i*8+5];
        inner_product_total += (static_cast<float>(i7)*mult + min_coord)*p[i*8+6];
        inner_product_total += (static_cast<float>(i8)*mult + min_coord)*p[i*8+7];
    }
    return -inner_product_total;
}


template<typename Point, typename T>
struct QuantizedPointRange{

    long dimension(){return dims;}
    long aligned_dimension(){return aligned_dims;}
    long quantized_dimension(){return quantized_dims;}
    size_t size(){return n;}
    bool is_learn(long i) {return false;}
    float max(){return max_coord;}
    float min(){return min_coord;}

    Point operator [] (long i) {
        return Point(values+i*aligned_dims, dims, quantized_dims, aligned_dims, i, max_coord, min_coord, bits);
    }

    QuantizedPointRange(){n=0;}

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

    QuantizedPointRange(char* filename){
        if(filename == NULL) {
            n = 0;
            dims = 0;
            return;
        }
        std::ifstream reader(filename);
        assert(reader.is_open());

        //read three data values: num points, original dimension, quantized dimension
        //read three values needed for decoding: bits, max val, min val
        unsigned int num_points, d, qd, bts;
        float mx, mn;
        reader.read((char*)(&num_points), sizeof(unsigned int)); n = num_points;
        reader.read((char*)(&d), sizeof(unsigned int)); dims = d;
        reader.read((char*)(&qd), sizeof(unsigned int)); quantized_dims = qd;
        reader.read((char*)(&bts), sizeof(unsigned int)); bits = bts;
        reader.read((char*)(&mx), sizeof(float)); max_coord = mx;
        reader.read((char*)(&mn), sizeof(float)); min_coord = mn;
        std::cout << "Detected " << num_points << " points with dimension " << d << std::endl;
        std::cout << "Quantized dimension is " << quantized_dims << " and compressed to " << bits << " bits" << std::endl;
        std::cout << "Max coord: " << max_coord << ", Min coord: " << min_coord << std::endl;
        aligned_dims =  dim_round_up(quantized_dims, sizeof(T));
        if(aligned_dims != quantized_dims) std::cout << "Aligning dimension to " << aligned_dims << std::endl;
        values = (T*) aligned_alloc(64, n*aligned_dims*sizeof(T));
        size_t BLOCK_SIZE = 1000000;
        size_t index = 0;  
        while(index < n){
            size_t floor = index;
            size_t ceiling = index+BLOCK_SIZE <= n ? index+BLOCK_SIZE : n;
            T* data_start = new T[(ceiling-floor)*quantized_dims];
            reader.read((char*)(data_start), sizeof(T)*(ceiling-floor)*quantized_dims);
            T* data_end = data_start + (ceiling-floor)*quantized_dims;
            parlay::slice<T*, T*> data = parlay::make_slice(data_start, data_end);
            int data_bytes = quantized_dims*sizeof(T);
            parlay::parallel_for(floor, ceiling, [&] (size_t i){
                std::memmove(values + i*aligned_dims, data.begin() + (i-floor)*quantized_dims, data_bytes);
            });
            delete[] data_start;
            index = ceiling;
        }
    }

    void save(char* save_path){
        std::cout << "Writing compressed data with " << n << " points and dimension " << dims
                    << std::endl;
        parlay::sequence<unsigned int> preamble = {static_cast<unsigned int>(n), dims, quantized_dims, static_cast<unsigned int>(bits)};
        parlay::sequence<float> quantization_info = {max_coord, min_coord};
        auto vals = parlay::tabulate(n, [&] (size_t i){
            return parlay::tabulate(quantized_dims, [&] (size_t j){
                return values[aligned_dims*i+j];
            });
        });
        parlay::sequence<T> data = parlay::flatten(vals);
        std::ofstream writer;
        writer.open(save_path, std::ios::binary | std::ios::out);
        writer.write((char*)preamble.begin(), 4 * sizeof(unsigned int));
        writer.write((char*)quantization_info.begin(), 2 * sizeof(float));
        writer.write((char*)data.begin(), n*quantized_dims*sizeof(T));
        writer.close();
    }

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

