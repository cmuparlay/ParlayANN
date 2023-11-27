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

#include <limits>
#include "parlay/primitives.h"
#include "parlay/parallel.h"
#include "parlay/internal/get_time.h"
#include "parlay/internal/block_delayed.h"
#include "common/graph.h"

namespace delayed = parlay::block_delayed;

namespace ligra {
  
template<typename vertexId>
struct vertex_subset {
  using sparse_t = parlay::sequence<vertexId>;
  using dense_t = parlay::sequence<bool>;
  bool is_sparse;
  size_t n;
  size_t size() const {return n;}
  sparse_t sparse;
  dense_t dense;
  vertex_subset(sparse_t x) :
    sparse(std::move(x)), is_sparse(true), n(x.size()) {}
  vertex_subset(vertexId v) :
    sparse(sparse_t(1,v)), is_sparse(true), n(1) {}
  vertex_subset(dense_t x) :
    dense(std::move(x)), is_sparse(false),
    n(parlay::count(x,true)) {}
};

template<typename Graph, typename Fa, typename Cond> 
struct edge_map {
  using vertexId = typename Graph::vertexId;
  using vertex_subset_ = vertex_subset<vertexId>;
  using vertex_subset_sparse = parlay::sequence<vertexId>;
  using vertex_subset_dense = parlay::sequence<bool>;
  Fa fa;
  Cond cond;
  const Graph& G;
  bool dedup;
  bool verbose;
  parlay::sequence<vertexId> dup_seq;
  edge_map(Graph const &G, Fa fa, Cond cond, bool dedup=false,
	   bool verbose=false) :
    G(G), fa(fa), cond(cond), dedup(dedup), verbose(verbose) {
    dup_seq = parlay::sequence<vertexId>::uninitialized(G.numVertices());
  }

  auto edge_map_sparse(vertex_subset_sparse const &vtx_subset) {
    if (verbose) std::cout << "edge map sparse: " << vtx_subset.size() << std::endl;
    auto nested_edges = parlay::map(vtx_subset, [&] (vertexId v) {
	return parlay::delayed_tabulate(G[v].degree, [&, v] (size_t i) {
	    return std::pair(v, G[v].Neighbors[i]);});});
    auto edges = delayed::flatten(nested_edges);
    auto r = delayed::filter_map(edges,
				 [&] (auto x) {return cond(x.second) && fa(x.first, x.second);},
				 [] (auto x)  {return x.second;});
    if (dedup) {
      parlay::parallel_for(0,r.size(), [&] (size_t i) { dup_seq[r[i]] = i;});
      auto flags = parlay::tabulate(r.size(), [&] (size_t i) {return i==dup_seq[r[i]];});
      return vertex_subset_(parlay::pack(r, flags));
    }
    return vertex_subset_(std::move(r));
  }

  auto edge_map_dense(vertex_subset_dense const &vtx_subset) {
    if (verbose) std::cout << "edge map dense:  " << vtx_subset.size() << std::endl;
    auto r = parlay::tabulate(G.numVertices(), [&] (vertexId v) -> bool {
        bool result = false;
        if (cond(v)) {        
	  size_t block_size = 5000;
	  auto vtx = G[v];
	  auto d = vtx.degree;
	  auto ngh = vtx.Neighbors;
	  auto do_block = [&, vsub=vtx_subset.begin()] (size_t i) {
            size_t begin = block_size * i;
	    size_t end = std::min<size_t>(begin + block_size, d);
	    for (size_t j = begin; j < end; j++) {
	      if (!cond(v)) return;
	      vertexId u = ngh[j];
	      if (vsub[u]) {
		bool x = fa(u,v);
		if (!result && x) result = true;
	      }}};
	  size_t num_blocks = vtx.degree/block_size + 1;
	  if (num_blocks == 1) do_block(0);
	  else parlay::parallel_for(0, num_blocks, do_block, 1);
	}
	return result;});
    return vertex_subset_(std::move(r));
  }

  auto operator() (vertex_subset_ const &vtx_subset) {
    parlay::internal::timer t("edge_map", verbose);
    auto l = vtx_subset.size();
    auto n = G.numVertices();
    bool do_dense;
    if (vtx_subset.is_sparse) {
      auto out_degree = parlay::reduce(parlay::delayed_map(vtx_subset.sparse, [&] (size_t i) {
			   return G[i].degree;}));
      if ((l + out_degree) > G.m/20) {
	parlay::sequence<bool> d_vtx_subset(n, false);
	parlay::parallel_for(0, l, [&] (size_t i) {
          d_vtx_subset[vtx_subset.sparse[i]] = true;});
	t.next("convert");
	return edge_map_dense(d_vtx_subset);
      } else return edge_map_sparse(vtx_subset.sparse);
    } else {
      if (l > n/20) return edge_map_dense(vtx_subset.dense);
      else {
	auto s_vtx_subset = parlay::pack_index<vertexId>(vtx_subset.dense);
	return edge_map_sparse(s_vtx_subset);
      }
    }
  }
};
}
