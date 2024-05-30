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
#include <functional>
#include <random>
#include <set>
#include <unordered_set>
#include <queue>

#include "parlay/io.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"
#include "types.h"
#include "graph.h"
#include "stats.h"

// main beam search
template<typename indexType, typename Point, typename PointRange, class GT>
std::pair<std::pair<parlay::sequence<std::pair<indexType, typename Point::distanceType>>, parlay::sequence<std::pair<indexType, typename Point::distanceType>>>, size_t>
beam_search_impl(Point p, GT &G, PointRange &Points,
        parlay::sequence<indexType> starting_points, QueryParams &QP);

template<typename Point, typename PointRange, typename indexType>
std::pair<std::pair<parlay::sequence<std::pair<indexType, typename Point::distanceType>>, parlay::sequence<std::pair<indexType, typename Point::distanceType>>>, indexType>
beam_search(Point p, Graph<indexType> &G, PointRange &Points,
	    indexType starting_point, QueryParams &QP) {

  parlay::sequence<indexType> start_points = {starting_point};
  return beam_search_impl<indexType>(p, G, Points, start_points, QP);
}

template<typename Point, typename PointRange, typename indexType>
std::pair<std::pair<parlay::sequence<std::pair<indexType, typename Point::distanceType>>, parlay::sequence<std::pair<indexType, typename Point::distanceType>>>, size_t>
beam_search(Point p, Graph<indexType> &G, PointRange &Points,
        parlay::sequence<indexType> starting_points, QueryParams &QP) {
  return beam_search_impl<indexType>(p, G, Points, starting_points, QP);
}

// main beam search
template<typename indexType, typename Point, typename PointRange, class GT>
std::pair<std::pair<parlay::sequence<std::pair<indexType, typename Point::distanceType>>, parlay::sequence<std::pair<indexType, typename Point::distanceType>>>, size_t>
beam_search_impl(Point p, GT &G, PointRange &Points,
                 parlay::sequence<indexType> starting_points, QueryParams &QP) {
  if (starting_points.size() == 0) {
    std::cout << "beam search expects at least one start point" << std::endl;
    abort();
  }

  // compare two (node_id,distance) pairs, first by distance and then id if
  // equal
  using distanceType = typename Point::distanceType;
  auto less = [&](std::pair<indexType, distanceType> a, std::pair<indexType, distanceType> b) {
    return a.second < b.second || (a.second == b.second && a.first < b.first);
  };


  // used as a hash filter (can give false negative -- i.e. can say
  // not in table when it is)
  int bits = std::max<int>(10, std::ceil(std::log2(QP.beamSize * QP.beamSize)) - 2);
  std::vector<indexType> hash_filter(1 << bits, -1);
  auto has_been_seen = [&](indexType a) -> bool {
    int loc = parlay::hash64_2(a) & ((1 << bits) - 1);
    if (hash_filter[loc] == a) return true;
    hash_filter[loc] = a;
    return false;
  };

  // Frontier maintains the closest points found so far and its size
  // is always at most beamSize.  Each entry is a (id,distance) pair.
  // Initialized with starting points and kept sorted by distance.
  std::vector<std::pair<indexType, distanceType>> frontier;
  frontier.reserve(QP.beamSize);
  for (auto q : starting_points)
    frontier.push_back(std::pair<indexType, distanceType>(q, Points[q].distance(p)));
  std::sort(frontier.begin(), frontier.end(), less);

  // The subset of the frontier that has not been visited
  // Use the first of these to pick next vertex to visit.
  std::vector<std::pair<indexType, distanceType>> unvisited_frontier(QP.beamSize);
  unvisited_frontier[0] = frontier[0];

  // maintains sorted set of visited vertices (id-distance pairs)
  std::vector<std::pair<indexType, distanceType>> visited;
  visited.reserve(2 * QP.beamSize);

  // counters
  size_t dist_cmps = starting_points.size();
  int remain = 1;
  int num_visited = 0;
  double total;

  // used as temporaries in the loop
  std::vector<std::pair<indexType, distanceType>> new_frontier(std::max<size_t>(QP.beamSize,starting_points.size()) + G.max_degree());
  std::vector<std::pair<indexType, distanceType>> candidates;
  candidates.reserve(G.max_degree());
  std::vector<indexType> keep;
  keep.reserve(G.max_degree());

  // The main loop.  Terminate beam search when the entire frontier
  // has been visited or have reached max_visit.
  while (remain > 0 && num_visited < QP.limit) {
    // the next node to visit is the unvisited frontier node that is closest to
    // p
    std::pair<indexType, distanceType> current = unvisited_frontier[0];
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
    long num_elts = std::min<long>(G[current.first].size(), QP.degree_limit);
    for (indexType i=0; i<num_elts; i++) {
      auto a = G[current.first][i];
      if (has_been_seen(a) || Points[a].same_as(p)) continue;  // skip if already seen
      keep.push_back(a);
      Points[a].prefetch();
    }

    // Further filter on whether distance is greater than current
    // furthest distance in current frontier (if full).
    distanceType cutoff = ((frontier.size() < QP.beamSize)
                           ? (distanceType)std::numeric_limits<int>::max()
                           : frontier[frontier.size() - 1].second);
    for (auto a : keep) {
      distanceType dist = Points[a].distance(p);
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
    new_frontier_size = std::min<size_t>(QP.beamSize, new_frontier_size);

    // if a k is given (i.e. k != 0) then trim off entries that have a
    // distance greater than cut * current-kth-smallest-distance.
    // Only used during query and not during build.
    if (QP.k > 0 && new_frontier_size > QP.k && Points[0].is_metric())
      new_frontier_size =
          (std::upper_bound(new_frontier.begin(),
                            new_frontier.begin() + new_frontier_size,
                            std::pair{0, QP.cut * new_frontier[QP.k].second}, less) -
           new_frontier.begin());

    // copy new_frontier back to the frontier
    frontier.clear();
    for (indexType i = 0; i < new_frontier_size; i++)
      frontier.push_back(new_frontier[i]);

    // get the unvisited frontier (we only care about the first one)
    remain =
      std::set_difference(frontier.begin(), frontier.end(),
                          visited.begin(), visited.end(),
                          unvisited_frontier.begin(), less) -
        unvisited_frontier.begin();
  }
  
  return std::make_pair(std::make_pair(parlay::to_sequence(frontier),
                                       parlay::to_sequence(visited)),
                        dist_cmps);
}

// a range search that first finds a close point using a beam search,
// and then uses BFS to find all points within the range
template<typename indexType, typename Point, typename PointRange, class GT>
std::pair<std::vector<indexType>, typename Point::distanceType>
range_search(Point p, GT &G, PointRange &Points,
             parlay::sequence<indexType> seeds,
             typename Point::distanceType radius,
             typename Point::distanceType radius_2,
             QueryParams &QP, bool use_existing = false) {
  // first search for a starting point within the radius

  std::vector<indexType> result;
  std::unordered_set<indexType> seen;
  std::vector<indexType> starting_points;
  long distance_comparisons = 0;

  if (use_existing) {
    for (indexType i=0; i<G[p.id()].size(); i++)
      starting_points.push_back(G[p.id()][i]);
  } else {
    auto [beam_visited, dist_cmps] = beam_search(p, G, Points, seeds, QP);
    auto [beam, visited] = beam_visited;
    distance_comparisons = dist_cmps;
    for (auto x : beam)
      starting_points.push_back(x.first);
  }

  for (auto v : starting_points) {
    if (seen.count(v) > 0 || Points[v].same_as(p)) continue;
    distance_comparisons++;
    if (p.distance(Points[v]) > radius_2 ) continue;
    result.push_back(v);
    seen.insert(v);
  }

  // now do a BFS over all vertices with distance less than radius
  long position = 0;
  while (position < result.size()) {
    indexType next = result[position++];
    std::vector<indexType> unseen;
    for (long i = 0; i < G[next].size(); i++) {
      auto v = G[next][i];
      if (seen.count(v) > 0 || Points[v].same_as(p))
        continue;  // skip if already seen
      unseen.push_back(v);
      seen.insert(v);
      Points[v].prefetch();
    }
    for (auto v : unseen) {
      distance_comparisons++;
      if (Points[v].distance(p) <= radius_2)
        result.push_back(v);
    }
  }


  std::vector<indexType> result1;
  for (auto v : result) {
    if (p.distance(Points[v]) > radius ) continue;
    result1.push_back(v);
  }

  return std::pair(result1, distance_comparisons);
}

// searches every element in q starting from a randomly selected point
template<typename Point, typename PointRange, typename indexType>
parlay::sequence<parlay::sequence<indexType>> beamSearchRandom(PointRange& Query_Points,
                                         Graph<indexType> &G, PointRange &Base_Points, stats<indexType> &QueryStats,
                                         QueryParams &QP) {
  if (QP.k > QP.beamSize) {
    std::cout << "Error: beam search parameter Q = " << QP.beamSize
              << " same size or smaller than k = " << QP.k << std::endl;
    abort();
  }
  // use a random shuffle to generate random starting points for each query
  size_t n = G.size();

  parlay::sequence<parlay::sequence<indexType>> all_neighbors(Query_Points.size());

  parlay::random_generator gen;
  std::uniform_int_distribution<long> dis(0, n - 1);
  auto indices = parlay::tabulate(Query_Points.size(), [&](size_t i) {
    auto r = gen[i];
    return dis(r);
  });

  parlay::parallel_for(0, Query_Points.size(), [&](size_t i) {
    parlay::sequence<indexType> neighbors = parlay::sequence<indexType>(QP.k);
    indexType start = indices[i];
    parlay::sequence<std::pair<indexType, typename Point::distanceType>> beamElts;
    parlay::sequence<std::pair<indexType, typename Point::distanceType>> visitedElts;
    auto [pairElts, dist_cmps] =
        beam_search(Query_Points[i], G, Base_Points, start, QP);
    beamElts = pairElts.first;
    visitedElts = pairElts.second;
    for (indexType j = 0; j < QP.k; j++) {
      neighbors[j] = beamElts[j].first;
    }
    all_neighbors[i] = neighbors;
    QueryStats.increment_visited(i, visitedElts.size());
    QueryStats.increment_dist(i, dist_cmps);
  });
  return all_neighbors;
}

template<typename Point, typename PointRange, typename indexType>
parlay::sequence<parlay::sequence<indexType>> searchAll(PointRange& Query_Points,
	                                       Graph<indexType> &G, PointRange &Base_Points, stats<indexType> &QueryStats,
	                                      indexType starting_point, QueryParams &QP) {
    parlay::sequence<indexType> start_points = {starting_point};
    return searchAll<Point, PointRange, indexType>(Query_Points, G, Base_Points, QueryStats, start_points, QP);
}

template<typename Point, typename PointRange, typename indexType>
parlay::sequence<parlay::sequence<indexType>> searchAll(PointRange &Query_Points,
	                                       Graph<indexType> &G, PointRange &Base_Points, stats<indexType> &QueryStats,
                                        parlay::sequence<indexType> starting_points,
	                                      QueryParams &QP) {
  if (QP.k > QP.beamSize) {
    std::cout << "Error: beam search parameter Q = " << QP.beamSize
              << " same size or smaller than k = " << QP.k << std::endl;
    abort();
  }
  parlay::sequence<parlay::sequence<indexType>> all_neighbors(Query_Points.size());
  parlay::parallel_for(0, Query_Points.size(), [&](size_t i) {
    parlay::sequence<indexType> neighbors = parlay::sequence<indexType>(QP.k);
    auto [pairElts, dist_cmps] = beam_search(Query_Points[i], G, Base_Points, starting_points, QP);
    auto [beamElts, visitedElts] = pairElts;
    for (indexType j = 0; j < QP.k; j++) {
      neighbors[j] = beamElts[j].first;
    }
    all_neighbors[i] = neighbors;
    QueryStats.increment_visited(i, visitedElts.size());
    QueryStats.increment_dist(i, dist_cmps);
  });

  return all_neighbors;
}

template<typename Point, typename QPoint, typename PointRange, typename QPointRange, typename indexType>
parlay::sequence<indexType>
beam_search_rerank(const Point &p,
                   const QPoint &pq,
                   Graph<indexType> &G,
                   PointRange &Base_Points,
                   QPointRange &Q_Base_Points,
                   stats<indexType> &QueryStats,
                   parlay::sequence<indexType> starting_points,
                   QueryParams &QP) {
  // beam search with quantized points
  auto [pairElts, dist_cmps] = beam_search(pq, G, Q_Base_Points, starting_points, QP);
  auto [beamElts, visitedElts] = pairElts;

  int exp_factor = 5;
  
  // recalculate distances with non-quantized points and sort
  std::vector<std::pair<indexType, typename Point::distanceType>> pts;
  for (auto [j, ignore] : parlay::tabulate(std::min<int>(QP.k*exp_factor,beamElts.size()), [&] (long i) {return beamElts[i];}))
    pts.push_back(std::pair(j, p.distance(Base_Points[j])));
  std::sort(pts.begin(), pts.end(), [] (auto a, auto b) {return a.second < b.second;});

  // strip off the distances and keep first k
  parlay::sequence<indexType> neighbors;
  for (indexType j = 0; j < QP.k; j++)
    neighbors.push_back(pts[j].first);
  QueryStats.increment_visited(p.id(), visitedElts.size());
  QueryStats.increment_dist(p.id(), dist_cmps + beamElts.size());
  return neighbors;
}


template<typename Point, typename PointRange, typename QPointRange, typename indexType>
parlay::sequence<parlay::sequence<indexType>> qsearchAll(PointRange& Query_Points,
                                                         QPointRange& Q_Query_Points,
                                                         Graph<indexType> &G,
                                                         PointRange &Base_Points,
                                                         QPointRange &Q_Base_Points,
                                                         stats<indexType> &QueryStats,
                                                         indexType starting_point, QueryParams &QP) {
    parlay::sequence<indexType> start_points = {starting_point};
    return qsearchAll<Point, PointRange, QPointRange, indexType>(Query_Points, Q_Query_Points, G, Base_Points, Q_Base_Points, QueryStats, start_points, QP);
}

template<typename Point, typename PointRange, typename QPointRange, typename indexType>
parlay::sequence<parlay::sequence<indexType>> qsearchAll(PointRange &Query_Points,
                                                         QPointRange &Q_Query_Points,
                                                         Graph<indexType> &G,
                                                         PointRange &Base_Points,
                                                         QPointRange &Q_Base_Points,
                                                         stats<indexType> &QueryStats,
                                                         parlay::sequence<indexType> starting_points,
                                                         QueryParams &QP) {
  if (QP.k > QP.beamSize) {
    std::cout << "Error: beam search parameter Q = " << QP.beamSize
              << " same size or smaller than k = " << QP.k << std::endl;
    abort();
  }
  parlay::sequence<parlay::sequence<indexType>> all_neighbors(Query_Points.size());
  parlay::parallel_for(0, Query_Points.size(), [&](size_t i) {
    all_neighbors[i] = beam_search_rerank(Query_Points[i], Q_Query_Points[i], G,
                                          Base_Points, Q_Base_Points,
                                          QueryStats, starting_points, QP);
  });

  return all_neighbors;
}

template<typename Point, typename PointRange, typename indexType>
parlay::sequence<parlay::sequence<indexType>> RangeSearch(PointRange& Query_Points,
	                                       Graph<indexType> &G, PointRange &Base_Points, stats<indexType> &QueryStats,
	                                      indexType starting_point, RangeParams &QP) {
    parlay::sequence<indexType> start_points = {starting_point};
    return RangeSearch<Point, PointRange, indexType>(Query_Points, G, Base_Points, QueryStats, start_points, QP);
}

template<typename Point, typename PointRange, typename indexType>
parlay::sequence<parlay::sequence<indexType>> RangeSearch(PointRange &Query_Points,
	                                       Graph<indexType> &G, PointRange &Base_Points, stats<indexType> &QueryStats,
                                        parlay::sequence<indexType> starting_points,
	                                      RangeParams &RP) {

  parlay::sequence<parlay::sequence<indexType>> all_neighbors(Query_Points.size());
  // parlay::sequence<int> second_round(Query_Points.size(), 0);
  parlay::parallel_for(0, Query_Points.size(), [&](size_t i) {
    parlay::sequence<indexType> neighbors;
    QueryParams QP(RP.initial_beam, RP.initial_beam, 0.0, G.size(), G.max_degree());
    auto [pairElts, dist_cmps] = beam_search(Query_Points[i], G, Base_Points, starting_points, QP);
    auto [beamElts, visitedElts] = pairElts;
    for (indexType j = 0; j < beamElts.size(); j++) {
      if(beamElts[j].second <= RP.rad) neighbors.push_back(beamElts[j].first);
    }
    all_neighbors[i] = neighbors;
    // if(neighbors.size() < RP.initial_beam){
    //   all_neighbors[i] = neighbors;
    // } else{
    //   auto [in_range, dist_cmps] = range_search(Query_Points[i], G, Base_Points, neighbors, RP);
    //   parlay::sequence<indexType> ans;
    //   for (auto r : in_range) ans.push_back(r.first);
    //   if(in_range.size() > neighbors.size()) std::cout << "Range search found additional candidates" << std::endl;
    //   all_neighbors[i] = ans;
    //   second_round[i] = 1;
    //   QueryStats.increment_visited(i, in_range.size());
    //   QueryStats.increment_dist(i, dist_cmps);
    // }

    QueryStats.increment_visited(i, visitedElts.size());
    QueryStats.increment_dist(i, dist_cmps);
  });

  // std::cout << parlay::reduce(second_round) << " elements advanced to round two" << std::endl;

  return all_neighbors;
}
