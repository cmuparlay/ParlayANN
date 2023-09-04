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
#include <functional>
#include <random>
#include <set>
#include <unordered_set>

#include "parlay/io.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"
#include "types.h"
#include "graph.h"
#include "stats.h"

using uint = unsigned int;
using vertex_id = unsigned int;
using distance = float;
using id_dist = std::pair<vertex_id, distance>;
using GraphI = Graph<unsigned int>;

template<typename T, template<typename C> class Point, template<typename E, template<typename D> class P> class PointRange>
std::pair<std::pair<parlay::sequence<id_dist>, parlay::sequence<id_dist>>, uint>
beam_search(Point<T> p, GraphI &G, PointRange<T, Point> &Points,
	    uint starting_point, int beamSize,
	    int k=0, float cut=1.14, int limit=-1) {
  
  parlay::sequence<uint> start_points = {starting_point};
  return beam_search(p, G, Points, start_points, beamSize, k, cut, limit);
}

// main beam search
template<typename T, template<typename C> class Point, template<typename E, template<typename D> class P> class PointRange>
std::pair<std::pair<parlay::sequence<id_dist>, parlay::sequence<id_dist>>, size_t>
beam_search(Point<T> p, GraphI &G, PointRange<T, Point> &Points,
	      parlay::sequence<uint> starting_points, long beamSize,
	      long k=0, double cut=1.14, long max_visit=-1) {
  if(max_visit == -1) max_visit = G.size();
  // if (D->id() == "mips") cut = -cut;

  // compare two (node_id,distance) pairs, first by distance and then id if
  // equal
  auto less = [&](id_dist a, id_dist b) {
    return a.second < b.second || (a.second == b.second && a.first < b.first);
  };

  
  // calculate distance from q to p
  auto distance_from_p = [&] (vertex_id q) { 
      return Points[q].distance(p);};
  

  // used as a hash filter (can give false negative -- i.e. can say
  // not in table when it is)
  int bits = std::max<int>(10, std::ceil(std::log2(beamSize * beamSize)) - 2);
  std::vector<uint> hash_filter(1 << bits, -1);
  auto has_been_seen = [&](vertex_id a) -> bool {
    int loc = parlay::hash64_2(a) & ((1 << bits) - 1);
    if (hash_filter[loc] == a) return true;
    hash_filter[loc] = a;
    return false;
  };

  // Frontier maintains the closest points found so far and its size
  // is always at most beamSize.  Each entry is a (id,distance) pair.
  // Initialized with starting points and kept sorted by distance.
  std::vector<id_dist> frontier;
  frontier.reserve(beamSize);
  for (auto q : starting_points)
    frontier.push_back(id_dist(q, distance_from_p(q)));
  std::sort(frontier.begin(), frontier.end(), less);

  // The subset of the frontier that has not been visited
  // Use the first of these to pick next vertex to visit.
  std::vector<id_dist> unvisited_frontier(beamSize);
  unvisited_frontier[0] = frontier[0];

  // maintains sorted set of visited vertices (id-distance pairs)
  std::vector<id_dist> visited;
  visited.reserve(2 * beamSize);

  // counters
  size_t dist_cmps = starting_points.size();
  int remain = 1;
  int num_visited = 0;
  double total;

  // used as temporaries in the loop
  std::vector<id_dist> new_frontier(beamSize + G.max_degree());
  std::vector<id_dist> candidates;
  candidates.reserve(G.max_degree());
  std::vector<vertex_id> keep;
  keep.reserve(G.max_degree());

  // The main loop.  Terminate beam search when the entire frontier
  // has been visited or have reached max_visit.
  while (remain > 0 && num_visited < max_visit) {
    // the next node to visit is the unvisited frontier node that is closest to
    // p
    id_dist current = unvisited_frontier[0];
    G[current.first].prefetch();
    // add to visited set
    visited.insert(
        std::upper_bound(visited.begin(), visited.end(), current, less),
        current);
    num_visited++;

    // keep neighbors that have not been visited (using approximate
    // hash). Note that if a visited node is accidentally kept due to
    // approximate hash it will be removed below by the union or will
    // not bump anyone else.
    candidates.clear();
    keep.clear();
    for (uint i=0; i<G[current.first].size(); i++) {
      auto a = G[current.first][i];
      if (a == p.id() || has_been_seen(a)) continue;  // skip if already seen
      keep.push_back(a);
      Points[a].prefetch();
    }

    // Further filter on whether distance is greater than current
    // furthest distance in current frontier (if full).
    distance cutoff = ((frontier.size() < beamSize)
                           ? (distance)std::numeric_limits<int>::max()
                           : frontier[frontier.size() - 1].second);
    for (auto a : keep) {
      distance dist = distance_from_p(a);
      total += dist;
      dist_cmps++;
      // skip if frontier not full and distance too large
      if (dist >= cutoff) continue;
      candidates.push_back(std::pair{a, dist});
    }

    // sort the candidates by distance from p
    std::sort(candidates.begin(), candidates.end(), less);

    // union the frontier and candidates into new_frontier, both are sorted
    auto new_frontier_size =
        std::set_union(frontier.begin(), frontier.end(), candidates.begin(),
                       candidates.end(), new_frontier.begin(), less) -
        new_frontier.begin();

    // trim to at most beam size
    new_frontier_size = std::min<size_t>(beamSize, new_frontier_size);

    // if a k is given (i.e. k != 0) then trim off entries that have a
    // distance greater than cut * current-kth-smallest-distance.
    // Only used during query and not during build.
    if (k > 0 && new_frontier_size > k)
      new_frontier_size =
          (std::upper_bound(new_frontier.begin(),
                            new_frontier.begin() + new_frontier_size,
                            std::pair{0, cut * new_frontier[k].second}, less) -
           new_frontier.begin());

    // copy new_frontier back to the frontier
    frontier.clear();
    for (uint i = 0; i < new_frontier_size; i++)
      frontier.push_back(new_frontier[i]);

    // get the unvisited frontier (we only care about the first one)
    remain =
        std::set_difference(frontier.begin(), frontier.end(), visited.begin(),
                            visited.end(), unvisited_frontier.begin(), less) -
        unvisited_frontier.begin();
  }

  return std::make_pair(std::make_pair(parlay::to_sequence(frontier),
                                       parlay::to_sequence(visited)),
                        dist_cmps);
}

// // has same functionality as above but written differently (taken from HNSW)
// // not quite as fast and does not prune based on cut.
// template<typename T, template<typename C> class Point, template<typename E, template<typename D> class P> class PointRange>
// std::pair<std::pair<parlay::sequence<id_dist>, parlay::sequence<id_dist>>, size_t>
// beam_search_(Tvec_point<T>* p, parlay::sequence<Tvec_point<T>*>& v, PointRange<T, Point> &Points, data_store<T> &Data,
// 	      parlay::sequence<Tvec_point<T>*> starting_points, int beamSize,
// 	      int k=0, float cut=1.14, int max_visit=-1) {
//   if(max_visit == -1) max_visit = v.size();

//   // used as a hash filter (can give false negative -- i.e. can say
//   // not in table when it is)
//   int bits = std::ceil(std::log2(beamSize * beamSize)) - 2;
//   std::vector<int> hash_filter(1 << bits, -1);
//   auto has_been_seen = [&](vertex_id a) -> bool {
//     int loc = parlay::hash64_2(a) & ((1 << bits) - 1);
//     if (hash_filter[loc] == a) return true;
//     hash_filter[loc] = a;
//     return false;
//   };

//   // calculate distance from q to p
//   auto distance_from_p = [&] (vertex_id q) { 
//     //  auto coord_len = (v[1]->coordinates.begin() - v[0]->coordinates.begin());
//     //  auto q_ptr = v[0]->coordinates.begin() + q * coord_len;
//      return Data.distance(q, p->coordinates.begin());};

//   // compare two (node_id,distance) pairs, first by distance and then id if
//   // equal
//   struct less {
//     constexpr bool operator()(id_dist a, id_dist b) const {
//       return a.second < b.second || (a.second == b.second && a.first < b.first);
//     };
//   };

//   parlay::sequence<id_dist> W, visited;
//   W.reserve(beamSize);
//   std::make_heap(W.begin(), W.end(), less());

//   std::set<id_dist, less> C;
//   std::unordered_set<vertex_id> W_visited(10 * beamSize);

//   int dist_cmps = 0;
//   int num_visited = 0;

//   // initialize starting points
//   for (auto q : starting_points) {
//     vertex_id qid = q->id;
//     has_been_seen(qid);
//     const auto d = distance_from_p(qid);
//     dist_cmps++;
//     C.insert({qid, d});
//     W.push_back({qid, d});
//     W_visited.insert(qid);
//   }

//   while (C.size() > 0 && num_visited < max_visit) {
//     if (C.begin()->second > W[0].second) break;
//     id_dist current = *C.begin();
//     visited.push_back(current);
//     num_visited++;
//     C.erase(C.begin());
//     for (vertex_id q : v[current.first]->out_nbh) {
//       if (q == -1) break;
//       if (has_been_seen(q) || W_visited.count(q) > 0) continue;
//       distance d = distance_from_p(q);
//       dist_cmps++;
//       if (W.size() < beamSize || d < W[0].second) {
//         C.insert({q, d});
//         W.push_back({q, d});
//         W_visited.insert(q);
//         std::push_heap(W.begin(), W.end(), less());
//         if (W.size() > beamSize) {
//           W_visited.erase(W[0].first);
//           std::pop_heap(W.begin(), W.end(), less());
//           W.pop_back();
//         }
//         if (C.size() > beamSize) C.erase(std::prev(C.end()));
//       }
//     }
//   }
//   std::sort(visited.begin(), visited.end(), less());
//   std::sort(W.begin(), W.end(), less());
//   return std::make_pair(
//       std::make_pair(parlay::to_sequence(W), parlay::to_sequence(visited)),
//       dist_cmps);
// }

// searches every element in q starting from a randomly selected point
template<typename T, template<typename C> class Point, template<typename E, template<typename D> class P> class PointRange>
parlay::sequence<parlay::sequence<uint>> beamSearchRandom(PointRange<T, Point>& Query_Points,
                                         GraphI &G, PointRange<T, Point> &Base_Points, stats<uint> &QueryStats, long beamSizeQ, 
                                        long k, double cut = 1.14, long max_visit=-1) {
  if ((k + 1) > beamSizeQ) {
    std::cout << "Error: beam search parameter Q = " << beamSizeQ
              << " same size or smaller than k = " << k << std::endl;
    abort();
  }
  // use a random shuffle to generate random starting points for each query
  size_t n = G.size();

  parlay::sequence<parlay::sequence<uint>> all_neighbors(Query_Points.size());

  parlay::random_generator gen;
  std::uniform_int_distribution<long> dis(0, n - 1);
  auto indices = parlay::tabulate(Query_Points.size(), [&](size_t i) {
    auto r = gen[i];
    return dis(r);
  });

  parlay::parallel_for(0, Query_Points.size(), [&](size_t i) {
    parlay::sequence<uint> neighbors = parlay::sequence<uint>(k);
    uint start = indices[i];
    parlay::sequence<id_dist> beamElts;
    parlay::sequence<id_dist> visitedElts;
    auto [pairElts, dist_cmps] = 
        beam_search(Query_Points[i], G, Base_Points, start, beamSizeQ, k, cut, max_visit);
    beamElts = pairElts.first;
    visitedElts = pairElts.second;
    for (uint j = 0; j < k; j++) {
      neighbors[j] = beamElts[j].first;
    }
    all_neighbors[i] = neighbors;
    QueryStats.increment_visited(i, visitedElts.size());
    QueryStats.increment_dist(i, dist_cmps);
  });
  return all_neighbors;
}

template<typename T, template<typename C> class Point, template<typename E, template<typename D> class P> class PointRange>
parlay::sequence<parlay::sequence<uint>> searchAll(PointRange<T, Point>& Query_Points,
	                                       GraphI &G, PointRange<T, Point> &Base_Points, stats<uint> &QueryStats,
                                        long beamSizeQ, long k,
	                                      uint starting_point, double cut, long max_visit) {
    parlay::sequence<uint> start_points = {starting_point};
    return searchAll(Query_Points, G, Base_Points, QueryStats, beamSizeQ, k, start_points, cut, max_visit);
}

template<typename T, template<typename C> class Point, template<typename E, template<typename D> class P> class PointRange>
parlay::sequence<parlay::sequence<uint>> searchAll(PointRange<T, Point> &Query_Points,
	                                       GraphI &G, PointRange<T, Point> &Base_Points, stats<uint> &QueryStats, long beamSizeQ, 
                                        long k, parlay::sequence<uint> starting_points,
	                                      double cut, long max_visit) {
  if ((k + 1) > beamSizeQ) {
    std::cout << "Error: beam search parameter Q = " << beamSizeQ
              << " same size or smaller than k = " << k << std::endl;
    abort();
  }
  parlay::sequence<parlay::sequence<uint>> all_neighbors(Query_Points.size());
  parlay::parallel_for(0, Query_Points.size(), [&](size_t i) {
    parlay::sequence<uint> neighbors = parlay::sequence<uint>(k);
    auto [pairElts, dist_cmps] = beam_search(Query_Points[i], G, Base_Points, starting_points,
					     beamSizeQ, k, cut, max_visit);
    auto [beamElts, visitedElts] = pairElts;
    for (uint j = 0; j < k; j++) {
      neighbors[j] = beamElts[j].first;
    }
    all_neighbors[i] = neighbors;
    QueryStats.increment_visited(i, visitedElts.size());
    QueryStats.increment_dist(i, dist_cmps);
  });

  return all_neighbors;
}

#endif
