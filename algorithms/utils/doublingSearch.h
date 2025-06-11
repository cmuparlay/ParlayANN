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
#include "earlyStopping.h"

namespace parlayANN{
  template<typename PointRange,
           typename QPointRange,
           typename indexType>
std::pair<parlay::sequence<parlay::sequence<indexType>>,std::pair<double,double>>
DoubleBeamRangeSearch(Graph<indexType> &G,
                      PointRange &Query_Points, PointRange &Base_Points,
                      QPointRange &Q_Query_Points, QPointRange &Q_Base_Points,
                      stats<indexType> &QueryStats, 
                      parlay::sequence<indexType> starting_points,
                      QueryParams &QP, parlay::sequence<indexType> active_indices) {
  parlay::sequence<parlay::sequence<indexType>> all_neighbors(active_indices.size());
  parlay::WorkerSpecific<double> first_round_time;
  parlay::WorkerSpecific<double> second_round_time;
  
  parlay::parallel_for(0, active_indices.size(), [&](size_t i) {
    parlay::sequence<indexType> neighbors;
    parlay::internal::timer t_search_first("first round time");
    parlay::internal::timer t_search_other("after first round");
    t_search_first.stop();
    t_search_other.stop();
    bool first_run = true;
    if(first_run) t_search_first.start();
    else t_search_other.start();
    auto P = Query_Points[active_indices[i]];
    auto Q_P = Q_Query_Points[active_indices[i]];
    using dtype = typename decltype(Query_Points[0])::distanceType;
    using id_dist = std::pair<indexType, dtype>;
    QueryParams QP1(QP.beamSize, QP.beamSize, 0.0,
                    G.size(), G.max_degree(),
                    QP.is_early_stop, Q_P.translate_distance(QP.early_stopping_radius),
                    QP.early_stopping_count,
                    QP.range_query_type, QP.radius);

    auto [pairElts, dist_cmps] = filtered_beam_search(G, Q_P, Q_Base_Points, Q_P, Q_Base_Points,
                                                      starting_points, QP1, false,
                                                      early_stopping<std::vector<id_dist>>);
    auto [beamElts, visitedElts] = pairElts;

    QueryStats.increment_visited(i, visitedElts.size());
    QueryStats.increment_dist(i, dist_cmps);
    
    for (auto b : beamElts)
      if (P.distance(Base_Points[b.first]) <= QP.radius)
        neighbors.push_back(b.first);
    //for (auto b : beamElts) 
    //if(b.second <= QP.radius) neighbors.push_back(b.first);
    
    bool results_smaller_than_beam = false;
    if (neighbors.size() < QP.beamSize)
      results_smaller_than_beam = true;
    
    all_neighbors[i] = std::move(neighbors);

    size_t initial_beam = QP.beamSize * 2;
    // Initialize starting points
    parlay::sequence<indexType> starting_points_idx;
    for (auto s : beamElts) 
      starting_points_idx.push_back(s.first);
    
    while(!results_smaller_than_beam){
      parlay::sequence<indexType> neighbors;

      QueryParams QP2(initial_beam, initial_beam, 0.0, G.size(), G.max_degree());
      auto [pairElts, dist_cmps] = beam_search(Q_P, G, Q_Base_Points, starting_points_idx, QP2);
      auto [beamElts, visitedElts] = pairElts;

      starting_points_idx.clear();
      for (auto v : beamElts) 
        starting_points_idx.push_back(v.first);

      for (auto b : beamElts)
        if (Query_Points[i].distance(Base_Points[b.first]) <= QP.radius)
          neighbors.push_back(b.first);

      if (neighbors.size() < initial_beam)
        results_smaller_than_beam = true;

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
