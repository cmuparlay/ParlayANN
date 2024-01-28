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

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>


#include <cpam/cpam.h>
#include <pam/get_time.h>
#include <pam/pam.h>

#include <mutex>

#include "../graphs/aspen/aspen.h"

template<typename indexType>
struct Aspen_Graph{
    struct empty_weight {};
    using Graph = aspen::symmetric_graph<empty_weight>;
    using vertex = typename Graph::vertex;
    using edge_tree = typename Graph::edge_tree;
    using vertex_tree = typename Graph::vertex_tree;
    using vertex_node = typename Graph::vertex_node;
    using version = typename aspen::versioned_graph::version;

    long max_degree() const {return maxDeg;}
    size_t size() const {return n;}

    Aspen_Graph(){}

    Aspen_Graph(long maxDeg, size_t n) : maxDeg(maxDeg), n(n) {}

    Graph(char* gFile){
        std::ifstream reader(gFile);
        assert(reader.is_open());

        //read num points and max degree
        indexType num_points;
        indexType max_deg;
        reader.read((char*)(&num_points), sizeof(indexType));
        n = num_points;
        reader.read((char*)(&max_deg), sizeof(indexType));
        maxDeg = max_deg;
        std::cout << "Detected " << num_points << " points with max degree " << max_deg << std::endl;

        //read degrees and perform scan to find offsets
        indexType* degrees_start = new indexType[n];
        reader.read((char*)(degrees_start), sizeof(indexType)*n);
        indexType* degrees_end = degrees_start + n;
        parlay::slice<indexType*, indexType*> degrees0 = parlay::make_slice(degrees_start, degrees_end);
        auto degrees = parlay::tabulate(degrees0.size(), [&] (size_t i){return static_cast<size_t>(degrees0[i]);});
        auto [offsets, total] = parlay::scan(degrees);
        std::cout << "Total: " << total << std::endl;
        offsets.push_back(total);

        //write 1000000 vertices at a time
        size_t BLOCK_SIZE=1000000;
        size_t index = 0;
        size_t total_size_read = 0;
        while(index < n){
            size_t g_floor = index;
            size_t g_ceiling = g_floor + BLOCK_SIZE <= n ? g_floor + BLOCK_SIZE : n;
            size_t total_size_to_read = offsets[g_ceiling]-offsets[g_floor];
            indexType* edges_start = new indexType[total_size_to_read];
            reader.read((char*)(edges_start), sizeof(indexType)*total_size_to_read);
            indexType* edges_end = edges_start + total_size_to_read;
            parlay::slice<indexType*, indexType*> edges = parlay::make_slice(edges_start, edges_end);
            //here create batches of edge_trees and insert them inplace

            // parlay::parallel_for(g_floor, g_ceiling, [&] (size_t i){
            //    graph[i*(maxDeg+1)] = degrees[i]; 
            //     for(size_t j=0; j<degrees[i]; j++){
            //         graph[i*(maxDeg+1)+1+j] = edges[offsets[i] - total_size_read + j];
            //     }
            // });
            total_size_read += total_size_to_read;
            index = g_ceiling; 
            delete[] edges_start;
        }
        delete[] degrees_start;
    }

    void save(char* oFile){
        std::cout << "Writing graph with " << n << " points and max degree " << maxDeg
                    << std::endl;
        parlay::sequence<indexType> preamble = {static_cast<indexType>(n), static_cast<indexType>(maxDeg)};
        // need to change the get_size function
        // parlay::sequence<indexType> sizes = parlay::tabulate(n, [&] (size_t i){return static_cast<indexType>((*this)[i].size());});
        std::ofstream writer;
        writer.open(oFile, std::ios::binary | std::ios::out);
        writer.write((char*)preamble.begin(), 2 * sizeof(indexType));
        writer.write((char*)sizes.begin(), sizes.size() * sizeof(indexType));
        size_t BLOCK_SIZE = 1000000;
        size_t index = 0;
        while(index < n){
            size_t floor = index;
            size_t ceiling = index+BLOCK_SIZE <= n ? index+BLOCK_SIZE : n;
            //change edge data collection type
            // parlay::sequence<parlay::sequence<indexType>> edge_data = parlay::tabulate(ceiling-floor, [&] (size_t i){
            //     return parlay::tabulate(sizes[i+floor], [&] (size_t j){return (*this)[i+floor][j];});
            // });
            parlay::sequence<indexType> data = parlay::flatten(edge_data);
            writer.write((char*)data.begin(), data.size() * sizeof(indexType));
            index = ceiling;
        }
        writer.close();
    }

    

    private:
        size_t n;
        long maxDeg;
        parlay::sequence<indexType> graph;
        aspen::versioned_graph<Graph> VG;
        version cur_version;
};