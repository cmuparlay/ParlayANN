// This code is part of the Parlay Project
// Copyright (c) 2024 Guy Blelloch, Magdalen Dobson and the Parlay team
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
#include <fcntl.h>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/internal/file_map.h"
#include "../../parlaylib/examples/helper/graph_utils.h"

#include "types.h"

namespace parlayANN {
  
template<typename indexType>
struct edgeRange{
  using graphUtils = graph_utils<indexType>;
  using Edges = typename graph_utils<indexType>::vertices;
  
  size_t size() const {return (*edges).size();}
  indexType id() const {return id_;}

  edgeRange() : edges(nullptr), id_(0) {}

  edgeRange(Edges* ngh, indexType maxDeg, indexType i)
    : edges(ngh), maxDeg(maxDeg), id_(i) {}

  indexType operator [] (indexType j) const {
    if (j > size()) {
      std::cout << "ERROR: index exceeds degree while accessing neighbors" << std::endl;
      abort();
    } else return (*edges)[j];
  }

  void append_neighbor(indexType nbh){
    (*edges).push_back(nbh);
  }

  template<typename rangeType>
  void update_neighbors(const rangeType& r){
    (*edges).clear();
    for (int i = 0; i < r.size(); i++) 
      (*edges).push_back(r[i]);
  }

  template<typename rangeType>
  void append_neighbors(const rangeType& r){
    for (int i = 0; i < r.size(); i++) 
      (*edges).push_back(r[i]);
  }

  void clear_neighbors(){
    (*edges).clear();
  }

  void prefetch() const {
    int l = (size() * sizeof(indexType))/64;
    for (int i = 0; i < l; i++)
      __builtin_prefetch(((char*) (*edges).data()) + i *  64);
  }

  template<typename F>
  void sort(F&& less){
    std::sort((*edges).begin(), (*edges).end(), less);}

  indexType* begin() const {return (*edges).data();}

  indexType* end() const {return (*edges).data() + size();}

private:
  Edges* edges;
  long maxDeg;
  indexType id_;
};

template<typename indexType_>
struct Graph{
  using indexType = indexType_;
  using graphUtils = graph_utils<indexType>;
  using Edges = typename graph_utils<indexType>::vertices;
  using gtype = typename graphUtils::graph;
  
  long max_degree() const {return maxDeg;}
  size_t size() const {return graph.size();}

  Graph(){}
  Graph(long maxDeg, size_t n) : maxDeg(maxDeg), graph(gtype(n)) {}

  Graph(char* gFile){
    std::string fname = gFile;
    graph = graphUtils::read_graph_from_file(fname);
    maxDeg = reduce(map(graph, parlay::size_of()), parlay::maximum<size_t>());
  }
  
  void save(char* oFile) {
    std::cout << "Writing graph with " << graph.size()
              << " points and max degree " << maxDeg
              << " to " << oFile 
              << std::endl;
    std::string fname = oFile;
    graphUtils::write_graph_to_file(graph, fname);
  }

  edgeRange<indexType> operator [] (indexType i) const {
    if (i > graph.size()) {
      std::cout << "ERROR: graph index out of range: " << i << std::endl;
      abort();
    }
    return edgeRange<indexType>((Edges*) &graph[i], maxDeg, i);
  }
  ~Graph(){}

private:
  indexType maxDeg;
  gtype graph;
};

} // end namespace
