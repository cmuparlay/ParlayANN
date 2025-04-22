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
#include "beamSearch.h"

namespace parlayANN{
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

      QueryParams QP(initial_beam, initial_beam, 0.0, G.size(), G.max_degree(), RP.early_stop, RP.early_stop_radius, false, true, RP.rad);

      all_neighbors[i].clear();

      // std::pair<std::pair<std::pair<parlay::sequence<std::pair<indexType, typename Point::distanceType>>, parlay::sequence<std::pair<indexType, typename Point::distanceType>>>, size_t>, parlay::sequence<std::pair<indexType, typename Point::distanceType>>> pair; 
      // parlay::sequence<std::pair<indexType, typename Point::distanceType>> beamElts;
      // parlay::sequence<std::pair<indexType, typename Point::distanceType>> visitedElts;
      // size_t dist_cmps;
      
      // if(initial_beam <= 1000){
      auto [pairElts, dist_cmps] = beam_search(Query_Points[active_indices[i]], G, Base_Points, starting_points_index[i], QP);
      auto [beamElts, visitedElts] = pairElts;
      // }else{
      //   pair = beam_search_impl_with_set(Query_Points[active_indices[i]], G, Base_Points, starting_points_index[i], QP, 0);
      //   auto [tmp, visit_order_pt] = pair;
      //   auto [pairElts, dist_cmps] = tmp;
      //   auto [beamElts, visitedElts] = pairElts;
      // }

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
}
