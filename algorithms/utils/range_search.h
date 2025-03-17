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
#include "parlay/worker_specific.h"
#include "types.h"
#include "graph.h"
#include "stats.h"



//a variant specialized for range searching
template<typename Point, typename PointRange, typename indexType>
std::pair<parlay::sequence<std::pair<indexType, typename Point::distanceType>>, size_t>
range_search(Point p, Graph<indexType> &G, PointRange &Points,
	      parlay::sequence<std::pair<indexType, typename Point::distanceType>> starting_points, RangeParams &RP,
        parlay::sequence<std::pair<indexType, typename Point::distanceType>> already_visited) {
  // compare two (node_id,distance) pairs, first by distance and then id if
  // equal
  using distanceType = typename Point::distanceType; 
  auto less = [&](std::pair<indexType, distanceType> a, std::pair<indexType, distanceType> b) {
    return a.second < b.second || (a.second == b.second && a.first < b.first);
  };
  

  //need to use an unordered map for a dynamically sized hash table
  std::unordered_set<indexType> has_been_seen;

  //Insert everything from visited list into has_been_seen
  for(auto v : already_visited){
    if(!has_been_seen.count(v.first) > 0) has_been_seen.insert(v.first);
  }

  // Frontier maintains the points within radius found so far 
  // Each entry is a (id,distance) pair.
  // Initialized with starting points 
  std::queue<std::pair<indexType, distanceType>> frontier;
  for (auto q : starting_points){
    if (!has_been_seen.count(q.first) > 0) has_been_seen.insert(q.first);
    frontier.push(q);
  }
  

  // maintains set of visited vertices (id-distance pairs)
  std::vector<std::pair<indexType, distanceType>> visited;

  // counters
  size_t dist_cmps = starting_points.size();
  int remain = 1;
  int num_visited = 0;
  double total;

  // used as temporaries in the loop
  std::vector<indexType> keep;
  keep.reserve(G.max_degree());

  // The main loop.  Terminate beam search when the entire frontier
  // has been visited or have reached max_visit.
  while (frontier.size() > 0) {
    // the next node to visit is the unvisited frontier node that is closest to
    // p
    std::pair<indexType, distanceType> current = frontier.front();
    frontier.pop();
    G[current.first].prefetch();
    // add to visited set
    visited.push_back(current);
    num_visited++;

    // keep neighbors that have not been visited (using approximate
    // hash). Note that if a visited node is accidentally kept due to
    // approximate hash it will be removed below by the union or will
    // not bump anyone else.
    keep.clear();
    for (indexType i=0; i<G[current.first].size(); i++) {
      auto a = G[current.first][i];
      //TODO this is a bug when searching for a point not in the graph???
      if (a == p.id() || has_been_seen.count(a) > 0) continue;  // skip if already seen
      keep.push_back(a);
      Points[a].prefetch();
      has_been_seen.insert(a);
    }

    for (auto a : keep) {
      distanceType dist = Points[a].distance(p);
      total += dist;
      dist_cmps++;
      // filter out if not within radius
      if (dist > RP.slack_factor*RP.rad) continue;
      frontier.push(std::pair{a, dist});
    }
  }

  return std::make_pair(parlay::to_sequence(visited), dist_cmps);    
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
  parlay::sequence<int> second_round(Query_Points.size(), 0);
  parlay::parallel_for(0, Query_Points.size(), [&](size_t i) {
    parlay::sequence<indexType> neighbors;
    parlay::sequence<std::pair<indexType, typename Point::distanceType>> neighbors_within_larger_ball;
    QueryParams QP(RP.initial_beam, RP.initial_beam, 0.0, G.size(), G.max_degree(), RP.early_stop, RP.early_stop_radius);
    auto [pairElts, dist_cmps] = beam_search(Query_Points[i], G, Base_Points, starting_points, QP);
    auto [beamElts, visitedElts] = pairElts;
    for (indexType j = 0; j < beamElts.size(); j++) {
      if(beamElts[j].second <= RP.rad) neighbors.push_back(beamElts[j].first);
      if(beamElts[j].second <= RP.slack_factor*RP.rad) neighbors_within_larger_ball.push_back(beamElts[j]);
    }
    if(neighbors_within_larger_ball.size() < RP.initial_beam || RP.second_round == false){
      all_neighbors[i] = neighbors;
    } else{
      auto [in_range, dist_cmps] = range_search(Query_Points[i], G, Base_Points, neighbors_within_larger_ball, RP, visitedElts);
      parlay::sequence<indexType> ans;
      for (auto r : in_range) {
        if(r.second <= RP.rad) ans.push_back(r.first);
      }
      all_neighbors[i] = ans;
      second_round[i] = 1;
      QueryStats.increment_visited(i, in_range.size());
      QueryStats.increment_dist(i, dist_cmps);
    }
    
    QueryStats.increment_visited(i, visitedElts.size());
    QueryStats.increment_dist(i, dist_cmps);
  });


  return all_neighbors;
}



template<typename Point, typename PointRange, typename indexType>
std::pair<std::pair<parlay::sequence<parlay::sequence<indexType>>, parlay::sequence<parlay::sequence<std::pair<indexType, typename Point::distanceType>>>>,std::pair<double,double>> 
RangeSearchOverSubset(PointRange &Query_Points,
	                                       Graph<indexType> &G, PointRange &Base_Points, stats<indexType> &QueryStats, 
                                        indexType starting_point,
	                                      RangeParams &RP, parlay::sequence<indexType> active_indices) {
    parlay::sequence<indexType> start_points = {starting_point};
    return RangeSearchOverSubset<Point, PointRange, indexType>(Query_Points, G, Base_Points, QueryStats, start_points, RP, active_indices);
}

template<typename Point, typename PointRange, typename indexType>
std::pair<std::pair<parlay::sequence<parlay::sequence<indexType>>, parlay::sequence<parlay::sequence<std::pair<indexType, typename Point::distanceType>>>>,std::pair<double,double>> 
RangeSearchOverSubset(PointRange &Query_Points,
	                                       Graph<indexType> &G, PointRange &Base_Points, stats<indexType> &QueryStats, 
                                        parlay::sequence<indexType> starting_points,
	                                      RangeParams &RP, parlay::sequence<indexType> active_indices) {
  parlay::sequence<parlay::sequence<indexType>> all_neighbors(active_indices.size());
  parlay::sequence<parlay::sequence<std::pair<indexType, typename Point::distanceType>>> visit_order(active_indices.size());
  parlay::WorkerSpecific<double> beam_time;
  parlay::WorkerSpecific<double> other_time;
  QueryParams QP(RP.initial_beam, RP.initial_beam, 0.0, G.size(), G.max_degree(), RP.early_stop, RP.early_stop_radius);
  parlay::sequence<int> second_round(active_indices.size(), 0);
  parlay::parallel_for(0, active_indices.size(), [&](size_t i) {
    parlay::internal::timer t_search_beam("beam search time");
    parlay::internal::timer t_search_other("other time");
    t_search_beam.stop();
    t_search_other.stop();
    // bool first_run = true;
    // if(first_run){
    //   t_search_first.start();
    // }else{
    //   t_search_other.start();
    // }
    parlay::sequence<indexType> neighbors;
    parlay::sequence<std::pair<indexType, typename Point::distanceType>> neighbors_within_larger_ball;
    t_search_beam.start();
    auto [tmp, visit_order_pt] = beam_search(Query_Points[active_indices[i]], G, Base_Points, starting_points, QP, RP.rad);
    t_search_beam.stop();
    auto [pairElts, dist_cmps] = tmp;
    auto [beamElts, visitedElts] = pairElts;
    visit_order[i] = visit_order_pt;
    for (indexType j = 0; j < beamElts.size(); j++) {
      if(beamElts[j].second <= RP.rad) neighbors.push_back(beamElts[j].first);
      if(beamElts[j].second <= RP.slack_factor*RP.rad) neighbors_within_larger_ball.push_back(beamElts[j]);
    }
    if(neighbors_within_larger_ball.size() < RP.initial_beam || RP.second_round == false){
      all_neighbors[i] = neighbors;
    } else{
      t_search_other.start();
      auto [in_range, dist_cmps] = range_search(Query_Points[active_indices[i]], G, Base_Points, neighbors_within_larger_ball, RP, visitedElts);
      parlay::sequence<indexType> ans;
      for (auto r : in_range) {
        if(r.second <= RP.rad) ans.push_back(r.first);
      }
      all_neighbors[i] = ans;
      second_round[i] = 1;
      QueryStats.increment_visited(i, in_range.size());
      QueryStats.increment_dist(i, dist_cmps);
      t_search_other.stop();
    }

    // if(first_run){
    //   first_run = false;
    //   t_search_first.stop();
      *beam_time += t_search_beam.total_time();
      
    // }else{
      // t_search_other.stop();
      *other_time += t_search_other.total_time();
    // }
    
    QueryStats.increment_visited(i, visitedElts.size());
    QueryStats.increment_dist(i, dist_cmps);
  });

  double total_time_beam = 0;
  double total_time_other = 0;
  for (auto x : beam_time) total_time_beam += x;
  for (auto y: other_time) total_time_other += y;

  return std::make_pair(std::make_pair(all_neighbors, visit_order),std::make_pair(total_time_beam,total_time_other));
}







