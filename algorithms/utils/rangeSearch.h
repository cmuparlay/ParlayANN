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
#include "beamSearch.h"
#include "earlyStopping.h"
#include "types.h"
#include "graph.h"
#include "stats.h"

namespace parlayANN {
    template<typename Point, typename PointRange, typename indexType>
parlay::sequence<parlay::sequence<indexType>>
RangeSearch(PointRange& Query_Points,
            Graph<indexType> &G, PointRange &Base_Points, stats<indexType> &QueryStats,
            indexType starting_point, QueryParams &QP) {
  parlay::sequence<indexType> start_points = {starting_point};
  return RangeSearch<Point, PointRange, indexType>(Query_Points, G, Base_Points, QueryStats, start_points, QP);
}

template<typename Point, typename PointRange, typename indexType>
parlay::sequence<parlay::sequence<indexType>>
RangeSearch(PointRange &Query_Points,
            Graph<indexType> &G, PointRange &Base_Points, stats<indexType> &QueryStats,
            parlay::sequence<indexType> starting_points,
            QueryParams &QP) {

  parlay::sequence<parlay::sequence<indexType>> all_neighbors(Query_Points.size());
  // parlay::sequence<int> second_round(Query_Points.size(), 0);
  parlay::parallel_for(0, Query_Points.size(), [&](size_t i) {
    parlay::sequence<indexType> neighbors;
    parlay::sequence<std::pair<indexType, typename Point::distanceType>> neighbors_with_distance;
    //QueryParams QP(RP.initial_beam, RP.initial_beam, 0.0, G.size(), G.max_degree(), RP.early_stop, RP.early_stop_radius, 
    //             RP.is_early_stop, false, RP.rad);
    using dtype = typename Point::distanceType;
    using id_dist = std::pair<indexType, dtype>;

    auto [pairElts, dist_cmps] = beam_search_es(Query_Points[i], G, Base_Points, starting_points, QP,
                                                early_stopping<std::vector<id_dist>>);
    auto [beamElts, visitedElts] = pairElts;
    for (indexType j = 0; j < beamElts.size(); j++) {
      if(beamElts[j].second <= QP.radius) {
        neighbors.push_back(beamElts[j].first);
        neighbors_with_distance.push_back(beamElts[j]);
      }
    }
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
    if(neighbors.size() < QP.beamSize || QP.is_beam_search){
      all_neighbors[i] = neighbors;
    } else{
      auto [in_range, dist_cmps] = greedy_search(Query_Points[i], G, Base_Points, neighbors_with_distance, QP, visitedElts);

      parlay::sequence<indexType> ans;

      //#define EndWithBeam
#ifdef EndWithBeam
      int beamSize = in_range.size() * 1.2;
      QueryParams QP2(beamSize, beamSize, 0.0, G.size(), G.max_degree());
      auto [pairElts, dist_cmps2] = beam_search(Query_Points[i], G, Base_Points, in_range, QP2);
      for (auto r : pairElts.first) 
        if (r.second <= QP.radius)
          ans.push_back(r.first);
#else
      for (auto r : in_range)
        if (Query_Points[i].distance(Base_Points[r]) <= QP.radius)
          ans.push_back(r);
#endif

      all_neighbors[i] = ans;
      QueryStats.increment_visited(i, in_range.size());
      QueryStats.increment_dist(i, dist_cmps);
    }

    QueryStats.increment_visited(i, visitedElts.size());
    QueryStats.increment_dist(i, dist_cmps);
  });

  // std::cout << parlay::reduce(second_round) << " elements advanced to round two" << std::endl;

  return all_neighbors;
}

}
