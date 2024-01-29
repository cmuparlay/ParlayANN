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
#include <pam/pam.h>

#include <mutex>

// #include "../graphs/aspen/aspen.h"

template<typename indexType>
struct Aspen_Vertex{
    using vertex = typename Graph::vertex;
    using edge_tree = typename Graph::edge_tree;
    using vertex_tree = typename Graph::vertex_tree;
    using vertex_node = typename Graph::vertex_node;

    size_t size(){return v.out_degree();}
    indexType id(){return v.id;}

    Aspen_Vertex(){}
    Aspen_Vertex(vertex v) : v(v) {}

    void append_neighbor()

    template<typename rangeType>
    void append_neighbors(){}
    
    template<typename rangeType>
    void update_neighbors(){}

    //TODO is reordering vertices possible here? probably not right?

    void prefetch(){}

    private:
        vertex v;

};

template<typename indexType>
struct Aspen_Graph{
    struct empty_weight {};
    using Graph = aspen::symmetric_graph<empty_weight>;
    using vertex = typename Graph::vertex;
    using edge_tree = typename Graph::edge_tree;
    using vertex_tree = typename Graph::vertex_tree;
    using vertex_node = typename Graph::vertex_node;
    using version = typename aspen::versioned_graph<Graph>::version;

    long max_degree() const {return maxDeg;}
    size_t size() const {return G.num_vertices();}

    Aspen_Graph(){}

    Aspen_Graph(long maxDeg) : maxDeg(maxDeg) {}

    Aspen_Graph(version V, long maxDeg) {}

    //TODO we really need the ability to do both functional and inplace updates
    //should we control this with a flag?
    //with the version controller?
    //eg could have function like "Get_Graph_To_Update" which creates a new graph object?
    void batch_update(){}

    void batch_delete(){}

    private:
        long maxDeg;
        Graph G;
        version V;
};

template<typename indexType>
struct Graph_Version_Control{
    struct empty_weight {};
    using Graph = aspen::symmetric_graph<empty_weight>;
    using vertex = typename Graph::vertex;
    using edge_tree = typename Graph::edge_tree;
    using vertex_tree = typename Graph::vertex_tree;
    using vertex_node = typename Graph::vertex_node;

    Graph_Version_Control(){}

    Graph_Version_Control(size_t md) : maxDeg(md){
        Graph Initial_Graph;
        VG = aspen::versioned_graph<Graph>(std::move(Initial_Graph));
    }

    Graph_Version_Control(char* gFile){}

    Aspen_Graph Get_Graph(){}

    //TODO does this work?
    void Release_Graph(Aspen_Graph G){}

    void Update_Graph(Aspen_Graph G){}

    void save(char* oFile){}

    private:
        aspen::versioned_graph<Graph> VG;
        size_t maxDeg;

};