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
#include "indexTools.h"
#include <random>
#include <set>
#include <math.h>
#include <functional>
#include <queue>

std::pair<size_t, size_t> select_two_random(parlay::sequence<size_t>& active_indices,
	parlay::random& rnd) {
	size_t first_index = rnd.ith_rand(0) % active_indices.size(); 
	size_t second_index_unshifted = rnd.ith_rand(1) % (active_indices.size()-1);
	size_t second_index = (second_index_unshifted < first_index) ?
	second_index_unshifted : (second_index_unshifted + 1);

	return {active_indices[first_index], active_indices[second_index]};
}

struct cluster_params{
	int MSTDeg;
	int L;
	double alpha;
	Distance* D;
	cluster_params(){}
};

template<typename T>
struct cluster{
	unsigned d; 
	Distance* D;
	using tvec_point = Tvec_point<T>;
	using edge = std::pair<int, int>;
	using labelled_edge = std::pair<edge, float>;

	cluster(unsigned dim, Distance* m): d(dim), D(m) {}

	void remove_edge_duplicates(tvec_point* p){
		parlay::sequence<int> points;
		for(int i=0; i<size_of(p->out_nbh); i++){
			points.push_back(p->out_nbh[i]);
		}
		auto np = parlay::remove_duplicates(points);
		add_out_nbh(np, p);
	}

	int generate_index(int N, int i){
		return (N*(N-1) - (N-i)*(N-i-1))/2;
	}

	bool tvec_equal(tvec_point* a, tvec_point* b, unsigned d){
		for(int i=0; i<d; i++){
			if(a->coordinates[i] != b->coordinates[i]){
				return false;
			}
		}
		return true;
	}

	template<typename F>
	void recurse(parlay::sequence<tvec_point*> &v, parlay::sequence<size_t> &active_indices,
		parlay::random& rnd, size_t cluster_size, 
		F f, cluster_params P, tvec_point* first, tvec_point* second){
		// Split points based on which of the two points are closer.
		auto closer_first = parlay::filter(parlay::make_slice(active_indices), [&] (size_t ind) {
			tvec_point* p = v[ind];
			float dist_first = D->distance(p->coordinates.begin(), first->coordinates.begin(), d);
			float dist_second = D->distance(p->coordinates.begin(), second->coordinates.begin(), d);
			return dist_first <= dist_second;

		});

		auto closer_second = parlay::filter(parlay::make_slice(active_indices), [&] (size_t ind) {
			tvec_point* p = v[ind];
			float dist_first = D->distance(p->coordinates.begin(), first->coordinates.begin(), d);
			float dist_second = D->distance(p->coordinates.begin(), second->coordinates.begin(), d);
			return dist_second < dist_first;
		});

		auto left_rnd = rnd.fork(0);
		auto right_rnd = rnd.fork(1);

		if(closer_first.size() == 1) {
			random_clustering(v, active_indices, right_rnd, cluster_size, f, P);
		}
		else if(closer_second.size() == 1){
			random_clustering(v, active_indices, left_rnd, cluster_size, f, P);
		}
		else{
			parlay::par_do(
				[&] () {random_clustering(v, closer_first, left_rnd, cluster_size, f, P);}, 
				[&] () {random_clustering(v, closer_second, right_rnd, cluster_size, f, P);}
			);
		}
	}

	template<typename F>
	void random_clustering(parlay::sequence<tvec_point*> &v, parlay::sequence<size_t> &active_indices,
		parlay::random& rnd, size_t cluster_size, F g, cluster_params P){
		if(active_indices.size() < cluster_size) g(v, active_indices, P);
		else{
			auto [f, s] = select_two_random(active_indices, rnd);
    		tvec_point* first = v[f];
    		tvec_point* second = v[s];
			int dim = v[0]->coordinates.size();
			if(tvec_equal(first, second, dim)){
				parlay::sequence<size_t> closer_first;
				parlay::sequence<size_t> closer_second;
				for(int i=0; i<active_indices.size(); i++){
					if(i<active_indices.size()/2) closer_first.push_back(active_indices[i]);
					else closer_second.push_back(active_indices[i]);
				}
				auto left_rnd = rnd.fork(0);
				auto right_rnd = rnd.fork(1);
				parlay::par_do(
					[&] () {random_clustering(v, closer_first, left_rnd, cluster_size, g, P);}, 
					[&] () {random_clustering(v, closer_second, right_rnd, cluster_size, g, P);}
				);
			} else{
				recurse(v, active_indices, rnd, cluster_size, g, P, first, second);
			}
		}
	}

	template<typename F>
	void random_clustering_wrapper(parlay::sequence<tvec_point*> &v, size_t cluster_size, 
		F f, cluster_params P){
		std::random_device rd;    
  		std::mt19937 rng(rd());   
  		std::uniform_int_distribution<int> uni(0,v.size()); 
    	parlay::random rnd(uni(rng));
    	auto active_indices = parlay::tabulate(v.size(), [&] (size_t i) { return i; });
    	random_clustering(v, active_indices, rnd, cluster_size, f, P);
	}

	template<typename F>
	void multiple_clustertrees(parlay::sequence<tvec_point*> &v, size_t cluster_size, int num_clusters,
		F f, cluster_params P){
		for(int i=0; i<num_clusters; i++){
			std::cout << "Cluster " << i << std::endl;
			random_clustering_wrapper(v, cluster_size, f, P);
			std::cout << "Built cluster " << i << " of " << num_clusters << std::endl;
		}
	}
};