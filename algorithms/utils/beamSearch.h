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

#ifndef BEAMSEARCH
#define BEAMSEARCH

#include <algorithm>
#include <set>
#include <unordered_set>
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"
#include "types.h"
#include "indexTools.h"
#include <functional>
#include <random>

extern bool report_stats;

using pid = std::pair<int, float>;

// returns true if F \setminus V = emptyset
bool intersect_nonempty(parlay::sequence<pid>& V, parlay::sequence<pid>& F) {
  for (int i = 0; i < F.size(); i++) {
    auto pred = [&](pid a) { return F[i].first == a.first; };
    if (parlay::find_if(V, pred) == V.end()) return true;
  }
  return false;
}

// will only be used when there is an element in F that is not in V
// hence the ``return 0" line will never be called
pid id_next(parlay::sequence<pid>& V, parlay::sequence<pid>& F) {
  for (int i = 0; i < F.size(); i++) {
    auto pred = [&](pid a) { return F[i].first == a.first; };
    if (parlay::find_if(V, pred) == V.end()) return F[i];
  }
  return std::make_pair(0, 0);
}

// for debugging
void print_seq(parlay::sequence<int> seq) {
  int fsize = seq.size();
  std::cout << "[";
  for (int i = 0; i < fsize; i++) {
    std::cout << seq[i] << ", ";
  }
  std::cout << "]" << std::endl;
}

// updated version by Guy
template <typename T>
std::pair<std::pair<parlay::sequence<pid>, parlay::sequence<pid>>, int> beam_search(
    Tvec_point<T>* p, parlay::sequence<Tvec_point<T>*>& v,
    Tvec_point<T>* starting_point, int beamSize, unsigned d, Distance* D, int k=0, float cut=1.14, int limit=-1) {
  
  parlay::sequence<Tvec_point<T>*> start_points;
  start_points.push_back(starting_point);
  return beam_search(p, v, start_points, beamSize, d, D, k, cut, limit);

}

// updated version by Guy
template <typename T>
std::pair<std::pair<parlay::sequence<pid>, parlay::sequence<pid>>, size_t> beam_search(
    Tvec_point<T>* p, parlay::sequence<Tvec_point<T>*>& v,
    parlay::sequence<Tvec_point<T>*> starting_points, int beamSize, unsigned d, Distance* D, int k=0, float cut=1.14, int limit=-1) {
  // initialize data structures
  if(limit==-1) limit=v.size();
  size_t dist_cmps = 0;
  auto vvc = v[0]->coordinates.begin();
  long stride = v[1]->coordinates.begin() - v[0]->coordinates.begin();
  std::vector<pid> visited;
  auto less = [&](pid a, pid b) {
      return a.second < b.second || (a.second == b.second && a.first < b.first); };
  auto make_pid = [&] (int q) {
      return std::pair{q, D->distance(vvc + q*stride, p->coordinates.begin(), d)};
  };
  int bits = std::ceil(std::log2(beamSize*beamSize))-2;
  parlay::sequence<int> hash_table(1 << bits, -1);

  auto pre_frontier = parlay::tabulate(starting_points.size(), [&] (size_t i) {
    return make_pid(starting_points[i]->id);
  });

  dist_cmps += starting_points.size();

  auto frontier = parlay::sort(pre_frontier, less);

  std::vector<pid> unvisited_frontier(beamSize);
  parlay::sequence<pid> new_frontier(beamSize + v[0]->out_nbh.size());
  unvisited_frontier[0] = frontier[0];
  int remain = 1;
  int num_visited = 0;

  // terminate beam search when the entire frontier has been visited
  while (remain > 0 && num_visited<limit) {
    // the next node to visit is the unvisited frontier node that is closest to p
    pid currentPid = unvisited_frontier[0];
    Tvec_point<T>* current = v[currentPid.first];
    auto nbh = current->out_nbh.cut(0, size_of(current->out_nbh));
    auto candidates = parlay::filter(nbh,  [&] (int a) {
	     int loc = parlay::hash64_2(a) & ((1 << bits) - 1);
	     if (a == p->id || hash_table[loc] == a) return false;
	     hash_table[loc] = a;
	     return true;});
    auto pairCandidates = parlay::map(candidates, [&] (long c) {return make_pid(c);}, 1000);
    dist_cmps += candidates.size();
    auto sortedCandidates = parlay::sort(pairCandidates, less);
    auto f_iter = std::set_union(frontier.begin(), frontier.end(),
				 sortedCandidates.begin(), sortedCandidates.end(),
				 new_frontier.begin(), less);
    size_t f_size = std::min<size_t>(beamSize, f_iter - new_frontier.begin());
    if (k > 0 && f_size > k) {
      if(D->id() == "mips"){
        f_size = (std::upper_bound(new_frontier.begin(), new_frontier.begin() + f_size,
				std::pair{0, -cut * new_frontier[k].second}, less)
		- new_frontier.begin());
      }
      else{f_size = (std::upper_bound(new_frontier.begin(), new_frontier.begin() + f_size,
				std::pair{0, cut * new_frontier[k].second}, less)
		- new_frontier.begin());}
    }
    frontier = parlay::tabulate(f_size, [&] (long i) {return new_frontier[i];});
    visited.insert(std::upper_bound(visited.begin(), visited.end(), currentPid, less), currentPid);
    auto uf_iter = std::set_difference(frontier.begin(), frontier.end(),
				 visited.begin(), visited.end(),
				 unvisited_frontier.begin(), less);
    remain = uf_iter - unvisited_frontier.begin();
    num_visited++;
  }
  return std::make_pair(std::make_pair(frontier, parlay::to_sequence(visited)), dist_cmps);
}


// searches every element in q starting from a randomly selected point
template <typename T>
void beamSearchRandom(parlay::sequence<Tvec_point<T>*>& q,
                      parlay::sequence<Tvec_point<T>*>& v, int beamSizeQ, int k,
                      unsigned d, Distance* D, double cut = 1.14, int limit=-1) {
  if ((k + 1) > beamSizeQ) {
    std::cout << "Error: beam search parameter Q = " << beamSizeQ
              << " same size or smaller than k = " << k << std::endl;
    abort();
  }
  // use a random shuffle to generate random starting points for each query
  size_t n = v.size();
  // auto indices = parlay::random_permutation<int>(static_cast<int>(n), time(NULL));

  parlay::random_generator gen;
  std::uniform_int_distribution<long> dis(0, n-1);
  auto indices = parlay::tabulate(q.size(), [&](size_t i) {
    auto r = gen[i];
    return dis(r);
  });

  parlay::parallel_for(0, q.size(), [&](size_t i) {
    parlay::sequence<int> neighbors = parlay::sequence<int>(k);
    size_t index = indices[i];
    Tvec_point<T>* start = v[index];
    parlay::sequence<pid> beamElts;
    parlay::sequence<pid> visitedElts;
    auto [pairElts, dist_cmps] = beam_search(q[i], v, start, beamSizeQ, d, D, k, cut, limit);
    beamElts = pairElts.first;
    visitedElts = pairElts.second;
    for (int j = 0; j < k; j++) {
      neighbors[j] = beamElts[j].first;
    }
    q[i]->ngh = neighbors;
    if (report_stats) {q[i]->visited = visitedElts.size(); q[i]->dist_calls = dist_cmps; }
  });
}

template <typename T>
void searchAll(parlay::sequence<Tvec_point<T>*>& q,
                      parlay::sequence<Tvec_point<T>*>& v, int beamSizeQ, int k,
                      unsigned d, Tvec_point<T>* starting_point, Distance* D, float cut, int limit) {
    parlay::sequence<Tvec_point<T>*> start_points;
    start_points.push_back(starting_point);
    searchAll(q, v, beamSizeQ, k, d, start_points, D, cut, limit);
}

template <typename T>
void searchAll(parlay::sequence<Tvec_point<T>*>& q,
                      parlay::sequence<Tvec_point<T>*>& v, int beamSizeQ, int k,
                      unsigned d, parlay::sequence<Tvec_point<T>*> starting_points, Distance* D, float cut, int limit) {
  if ((k + 1) > beamSizeQ) {
    std::cout << "Error: beam search parameter Q = " << beamSizeQ
              << " same size or smaller than k = " << k << std::endl;
    abort();
  }
  parlay::parallel_for(0, q.size(), [&](size_t i) {
    parlay::sequence<int> neighbors = parlay::sequence<int>(k);
    auto [pairElts, dist_cmps] = beam_search(q[i], v, starting_points, beamSizeQ, d, D, k, cut, limit);
    auto [beamElts, visitedElts] = pairElts;
      for (int j = 0; j < k; j++) {
        neighbors[j] = beamElts[j].first;
      }
    q[i]->ngh = neighbors;
    q[i]->visited = visitedElts.size();
    q[i]->dist_calls = dist_cmps; 

  });
}

template<typename T>
void rangeSearchAll(parlay::sequence<Tvec_point<T>*> q, parlay::sequence<Tvec_point<T>*>& v, 
  int beamSize, unsigned d, Tvec_point<T>* start_point, double r, int k, double cut, double slack){
    parlay::parallel_for(0, q.size(), [&] (size_t i){
      auto in_range = range_search(q[i], v, beamSize, d, start_point, r, k, cut, slack);
      parlay::sequence<int> nbh;
      for(auto j : in_range) nbh.push_back(j);
      q[i]->ngh = nbh;
    });
}

template<typename T>
void rangeSearchRandom(parlay::sequence<Tvec_point<T>*> q, parlay::sequence<Tvec_point<T>*>& v, 
  int beamSize, unsigned d, double r, int k, double cut = 1.14, double slack = 1.0){
    size_t n = v.size();
    auto indices = parlay::random_permutation<int>(static_cast<int>(n), time(NULL));
    parlay::parallel_for(0, q.size(), [&] (size_t i){
      auto in_range = range_search(q[i], v, beamSize, d, v[indices[i]], r, k, cut, slack);
      parlay::sequence<int> nbh;
      for(auto j : in_range) nbh.push_back(j);
      q[i]->ngh = nbh;
    });
    
}

template<typename T>   
std::set<int> range_search(Tvec_point<T>* q, parlay::sequence<Tvec_point<T>*>& v, 
  int beamSize, unsigned d, Tvec_point<T>* start_point, double r, Distance* D, int k, float cut, double slack){
  
  double max_rad = 0;

  std::set<int> nbh;

  auto [pairElts, dist_cmps] = beam_search(q, v, start_point, beamSize, d, D, k, cut);
  auto [neighbors, visited] = pairElts;

  q->visited = visited.size();
  q->dist_calls = dist_cmps;
  
  for(auto p : visited){
    if((p.second <= r)) nbh.insert(p.first);
  }

  return nbh;

}
                      

#endif

