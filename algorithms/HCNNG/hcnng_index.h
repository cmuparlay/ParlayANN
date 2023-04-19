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

#include <algorithm>
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"
#include "../utils/clusterEdge.h"
#include <random>
#include <set>
#include <math.h>
#include <queue>

extern bool report_stats;

template<typename T>
struct hcnng_index{
	int maxDeg;
	unsigned d;
	Distance* D;
	using tvec_point = Tvec_point<T>;
	using slice_tvec = decltype(make_slice(parlay::sequence<tvec_point*>()));
	using edge = std::pair<int, int>;
	using labelled_edge = std::pair<edge, float>;
	using pid = std::pair<int, float>;

	hcnng_index(int md, unsigned dim, Distance* DD) : maxDeg(md), d(dim), D(DD) {}

	void remove_edge_duplicates(tvec_point* p){
		parlay::sequence<int> points;
		for(int i=0; i<size_of(p->out_nbh); i++){
			points.push_back(p->out_nbh[i]);
		}
		auto np = parlay::remove_duplicates(points);
		add_out_nbh(np, p);
	}

	void remove_all_duplicates(parlay::sequence<tvec_point*> &v){
		parlay::parallel_for(0, v.size(), [&] (size_t i){
			remove_edge_duplicates(v[i]);
		});
	}


	void build_index(parlay::sequence<tvec_point*> &v, int cluster_rounds, size_t cluster_size){ 
		clear(v); 
		cluster<T> C(d, D);
		C.multiple_clustertrees(v, cluster_size, cluster_rounds, d, maxDeg);
		remove_all_duplicates(v);
	}
	
};