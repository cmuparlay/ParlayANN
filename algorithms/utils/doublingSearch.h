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

template<typename Point, typename PointRange, typename indexType>
std::pair<parlay::sequence<parlay::sequence<indexType>>,std::pair<double,double>> DoubleBeamRangeSearch(PointRange &Query_Points,
	                                       Graph<indexType> &G, PointRange &Base_Points, stats<indexType> &QueryStats, 
                                        parlay::sequence<indexType> starting_points,
	                                      RangeParams &RP, parlay::sequence<indexType> active_indices) {
  parlay::sequence<parlay::sequence<indexType>> all_neighbors(active_indices.size());
  //parlay::sequence<int> second_round(Query_Points.size(), 0);
  parlay::sequence<parlay::sequence<indexType>> starting_points_index(active_indices.size());
  parlay::WorkerSpecific<double> first_round_time;
  parlay::WorkerSpecific<double> second_round_time;
  
  parlay::parallel_for(0, active_indices.size(), [&](size_t i) {

    parlay::internal::timer t_search_first("first round time");
    parlay::internal::timer t_search_other("after first round");
    t_search_first.stop();
    t_search_other.stop();
    bool first_run = true;
    if(first_run){
      t_search_first.start();
    }else{
      t_search_other.start();
    }

    bool results_smaller_than_beam = false;
    size_t initial_beam = RP.initial_beam;
    // Initialize starting points
    for(size_t k; k< starting_points.size(); k++){
      starting_points_index[i].push_back(starting_points[k]);
    }

    
    while(!results_smaller_than_beam){
      parlay::sequence<indexType> neighbors;
      parlay::sequence<std::pair<indexType, typename Point::distanceType>> neighbors_within_larger_ball;

      QueryParams QP(initial_beam, initial_beam, 0.0, G.size(), G.max_degree(), RP.early_stop, RP.early_stop_radius);

      all_neighbors[i].clear();

      std::pair<std::pair<std::pair<parlay::sequence<std::pair<indexType, typename Point::distanceType>>, parlay::sequence<std::pair<indexType, typename Point::distanceType>>>, size_t>, parlay::sequence<std::pair<indexType, typename Point::distanceType>>> pair; 

      if(initial_beam <= 1000){
        pair = beam_search(Query_Points[active_indices[i]], G, Base_Points, starting_points_index[i], QP);
      } else{
        pair = beam_search_impl_with_set(Query_Points[active_indices[i]], G, Base_Points, starting_points_index[i], QP, 0);
      }
      
      
      auto [tmp, visit_order_pt] = pair;
      auto [pairElts, dist_cmps] = tmp;
      auto [beamElts, visitedElts] = pairElts;

      starting_points_index[i].clear();
      for(size_t l=0; l<visitedElts.size(); l++){
        starting_points_index[i].push_back(visitedElts[l].first);
      }


      neighbors.clear();

      for (indexType j = 0; j < beamElts.size(); j++) {
        if(beamElts[j].second <= RP.rad) neighbors.push_back(beamElts[j].first);
      }
      if(neighbors.size() < initial_beam){
        //Neighbors size is smaller than beam size
        results_smaller_than_beam = true;
      } 
      all_neighbors[i] = neighbors;

      QueryStats.increment_visited(i, visitedElts.size());
      QueryStats.increment_dist(i, dist_cmps);
      initial_beam *= 2;
      neighbors.clear();

      if(first_run){
        first_run = false;
        t_search_first.stop();
        *first_round_time += t_search_first.total_time();
        
      }else{
        t_search_other.stop();
        *second_round_time += t_search_other.total_time();
      }
    }
    
  });


  double total_time_first = 0;
  double total_time_second = 0;
  for (auto x : first_round_time) total_time_first += x;
  for (auto y: second_round_time) total_time_second += y;

  return std::make_pair(all_neighbors,std::make_pair(total_time_first,total_time_second));
}


template<typename Point, typename PointRange, typename indexType>
parlay::sequence<parlay::sequence<indexType>> DoubleBeamRangeSearchNoUpdate(PointRange &Query_Points,
	                                       Graph<indexType> &G, PointRange &Base_Points, stats<indexType> &QueryStats, 
                                        parlay::sequence<indexType> starting_points,
	                                      RangeParams &RP) {
  parlay::sequence<parlay::sequence<indexType>> all_neighbors(Query_Points.size());
  parlay::sequence<int> second_round(Query_Points.size(), 0);
  parlay::parallel_for(0, Query_Points.size(), [&](size_t i) {
    bool results_smaller_than_beam = false;
    size_t initial_beam = RP.initial_beam;
    // Initialize starting points
    
    //std::cout << "start double beam:" << starting_points_index.size() << std::endl;
    while(!results_smaller_than_beam){
    parlay::sequence<indexType> neighbors;
    parlay::sequence<std::pair<indexType, typename Point::distanceType>> neighbors_within_larger_ball;
    
    QueryParams QP(initial_beam, initial_beam, 0.0, G.size(), G.max_degree(), RP.early_stop, RP.early_stop_radius);
    //std::cout << "bs: " << initial_beam << std::endl;

    // auto [pairElts, dist_cmps] = beam_search(Query_Points[i], G, Base_Points, starting_points, QP);
    // auto [beamElts, visitedElts] = pairElts;

    
    auto [tmp, visit_order_pt] = beam_search(Query_Points[i], G, Base_Points, starting_points, QP);
    auto [pairElts, dist_cmps] = tmp;
    auto [beamElts, visitedElts] = pairElts;

    for (indexType j = 0; j < beamElts.size(); j++) {
      if(beamElts[j].second <= RP.rad) neighbors.push_back(beamElts[j].first);
      if(beamElts[j].second <= RP.slack_factor*RP.rad) neighbors_within_larger_ball.push_back(beamElts[j]);
    }
    if(neighbors_within_larger_ball.size() < initial_beam){
      //This is the case when we don't do the second round search
      //Or neighbors size is smaller than beam size
      //std::cout<< "beam size:" << initial_beam << std::endl;
      all_neighbors[i] = neighbors;
      results_smaller_than_beam = true;
    } else{
      // When we do second round search, do the range search inside the query points
      // TODO: change this range search by doubling the beam
      // auto [in_range, dist_cmps] = range_search(Query_Points[i], G, Base_Points, neighbors_within_larger_ball, RP, visitedElts);
      // parlay::sequence<indexType> ans;
      // for (auto r : in_range) {
      //   if(r.second <= RP.rad) ans.push_back(r.first);
      // }
      // all_neighbors[i] = ans;
      // second_round[i] = 1;
      // QueryStats.increment_visited(i, in_range.size());
      // QueryStats.increment_dist(i, dist_cmps);
    }
    
    QueryStats.increment_visited(i, visitedElts.size());
    QueryStats.increment_dist(i, dist_cmps);
    initial_beam *= 2;
    if(initial_beam> 1000 * RP.initial_beam){
      results_smaller_than_beam = true;
    }
    //std::cout<< "End beam: " << initial_beam << std::endl;
    }

    
  });

  //if(RP.second_round) std::cout << parlay::reduce(second_round) << " elements advanced to round two" << std::endl;

  return all_neighbors;
}
