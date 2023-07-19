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

struct DisjointSet{
	parlay::sequence<int> parent;
	parlay::sequence<int> rank;
	size_t N; 

	DisjointSet(size_t size){
		N = size;
		parent = parlay::sequence<int>(N);
		rank = parlay::sequence<int>(N);
		parlay::parallel_for(0, N, [&] (size_t i) {
			parent[i]=i;
			rank[i] = 0;
		});		
	}

	void _union(int x, int y){
		int xroot = parent[x];
		int yroot = parent[y];
		int xrank = rank[x];
		int yrank = rank[y];
		if(xroot == yroot)
			return;
		else if(xrank < yrank)
			parent[xroot] = yroot;
		else{
			parent[yroot] = xroot;
			if(xrank == yrank)
				rank[xroot] = rank[xroot] + 1;
		}
	}

	int find(int x){
		if(parent[x] != x)
			parent[x] = find(parent[x]);
		return parent[x];
	}

	void flatten(){
		for(int i=0; i<N; i++) find(i);
	}

	bool is_full(){
		flatten();
		parlay::sequence<bool> truthvals(N);
		parlay::parallel_for(0, N, [&] (size_t i){
			truthvals[i] = (parent[i]==parent[0]);
		});
		auto ff = [&] (bool a) {return not a;};
		auto filtered = parlay::filter(truthvals, ff);
		if(filtered.size()==0) return true;
		return false;
	}

};

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

	static void remove_edge_duplicates(tvec_point* p){
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
	
	//inserts each edge after checking for duplicates
	static void process_edges(parlay::sequence<tvec_point*> &v, parlay::sequence<edge> edges){
		int maxDeg = v[1]->out_nbh.begin() - v[0]->out_nbh.begin();
		auto grouped = parlay::group_by_key(edges);
		for(auto pair : grouped){
			auto [index, candidates] = pair;
			for(auto c : candidates){
				if(size_of(v[index]->out_nbh) < maxDeg){
					add_nbh(c, v[index]);
				}else{
					remove_edge_duplicates(v[index]);
					add_nbh(c, v[index]);
				}
			}
		}
	}

	//parameters dim and K are just to interface with the cluster tree code
	static void MSTk(parlay::sequence<tvec_point*> &v, parlay::sequence<size_t> &active_indices, 
		cluster_params C){
		//preprocessing for Kruskal's
		int N = active_indices.size();
		int dim = v[0]->coordinates.size();
		DisjointSet *disjset = new DisjointSet(N);
		size_t m = 10;
		auto less = [&] (labelled_edge a, labelled_edge b) {return a.second < b.second;};
		parlay::sequence<parlay::sequence<labelled_edge>> pre_labelled(N);
		parlay::random_generator gen;
  		std::uniform_real_distribution<float> dis(0.0, 1.0);
		parlay::parallel_for(0, N, [&] (size_t i){
			std::priority_queue<labelled_edge, std::vector<labelled_edge>, decltype(less)> Q(less);
			for(int j=0; j<N; j++){
				// float ff = dis(gen);
				// std::cout << ff << std::endl;
				if(j!=i){
					float dist_ij = C.D->distance(v[active_indices[i]]->coordinates.begin(), v[active_indices[j]]->coordinates.begin(), dim);
					if(Q.size() >= m){
						float topdist = Q.top().second;
						if(dist_ij < topdist){
							labelled_edge e;
							if(i<j) e = std::make_pair(std::make_pair(i,j), dist_ij);
							else e = std::make_pair(std::make_pair(j, i), dist_ij);
							Q.pop();
							Q.push(e);
						}
					}else{
						labelled_edge e;
						if(i<j) e = std::make_pair(std::make_pair(i,j), dist_ij);
						else e = std::make_pair(std::make_pair(j, i), dist_ij);
						Q.push(e);
					}
				}
			}
			// if(Q.size() < m) std::cout << "small queue" << std::endl;
			int limit = std::min(Q.size(), m);
			parlay::sequence<labelled_edge> edges(limit);
			for(int j=0; j<limit; j++){edges[j] = Q.top(); Q.pop();}
			pre_labelled[i] = edges;
		});
		auto flat_edges = parlay::flatten(pre_labelled);
		// std::cout << flat_edges.size() << std::endl;
		auto less_dup = [&] (labelled_edge a, labelled_edge b){
			auto dist_a = a.second;
			auto dist_b = b.second;
			if(dist_a == dist_b){
				int i_a = a.first.first;
				int j_a = a.first.second;
				int i_b = b.first.first;
				int j_b = b.first.second;
				if((i_a==i_b) && (j_a==j_b)){
					return true;
				} else{
					if(i_a != i_b) return i_a < i_b;
					else return j_a < j_b;
				}
			}else return (dist_a < dist_b);
		};
		auto labelled_edges = parlay::remove_duplicates_ordered(flat_edges, less_dup);
		// parlay::sort_inplace(labelled_edges, less);
		auto degrees = parlay::tabulate(active_indices.size(), [&] (size_t i) {return 0;});
		parlay::sequence<edge> MST_edges = parlay::sequence<edge>();
		//modified Kruskal's algorithm
		for(int i=0; i<labelled_edges.size(); i++){
			labelled_edge e_l = labelled_edges[i];
			edge e = e_l.first;
			if((disjset->find(e.first) != disjset->find(e.second)) && (degrees[e.first]<C.MSTDeg) && (degrees[e.second]<C.MSTDeg)){
				MST_edges.push_back(std::make_pair(active_indices[e.first], active_indices[e.second]));
				MST_edges.push_back(std::make_pair(active_indices[e.second], active_indices[e.first]));
				degrees[e.first] += 1;
				degrees[e.second] += 1;
				disjset->_union(e.first, e.second);
			}
			if(i%N==0){
				if(disjset->is_full()){
					break;
				}
			}
		}
		delete disjset;
		process_edges(v, MST_edges);
	}

	//robustPrune routine as found in DiskANN paper, with the exception that the new candidate set
	//is added to the field new_nbhs instead of directly replacing the out_nbh of p
	void robustPrune(tvec_point* p, parlay::sequence<tvec_point*> &v, double alpha, int maxDeg) {
    // add out neighbors of p to the candidate set.
		parlay::sequence<pid> candidates;
		for (size_t i=0; i<size_of(p->out_nbh); i++) {
			candidates.push_back(std::make_pair(p->out_nbh[i],
				D->distance(v[p->out_nbh[i]]->coordinates.begin(), p->coordinates.begin(), d)));
		}
		

		// Sort the candidate set in reverse order according to distance from p.
		auto less = [&] (pid a, pid b) {return a.second < b.second;};
		parlay::sort_inplace(candidates, less);

		parlay::sequence<int> new_nbhs = parlay::sequence<int>();

		
    	size_t candidate_idx = 0;
		while (new_nbhs.size() < maxDeg && candidate_idx < candidates.size()) {
			// Don't need to do modifications.
			int p_star = candidates[candidate_idx].first;
			candidate_idx++;
			if (p_star == p->id || p_star == -1) continue;

      		new_nbhs.push_back(p_star);

			for (size_t i = candidate_idx; i < candidates.size(); i++) {
				int p_prime = candidates[i].first;
				if (p_prime != -1) {
					float dist_starprime = D->distance(v[p_star]->coordinates.begin(), v[p_prime]->coordinates.begin(), d);
					float dist_pprime = candidates[i].second;
					if (alpha * dist_starprime <= dist_pprime) candidates[i].first = -1;
				}
			}
		}
		add_out_nbh(new_nbhs, p);
	}




	void build_index(parlay::sequence<tvec_point*> &v, int cluster_rounds, size_t cluster_size){ 
		clear(v); 
		cluster<T> C(d, D);
		cluster_params P;
		P.MSTDeg = 3;
		P.D = D;
		int maxDeg = P.MSTDeg*cluster_rounds;
		C.multiple_clustertrees(v, cluster_size, cluster_rounds, MSTk, P);
		remove_all_duplicates(v);
		// parlay::parallel_for(0, v.size(), [&] (size_t i){robustPrune(v[i], v, 1.1, maxDeg);});
	}
	
};