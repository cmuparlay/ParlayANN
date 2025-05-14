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
#include "beamSearch.h"
#include "types.h"
#include "graph.h"
#include "stats.h"

namespace parlayANN{
  template<typename PointInfo, typename Point>
  bool early_stopping(const PointInfo& frontier, 
                      const PointInfo& unvisited_frontier,
                      const PointInfo& visited,
                      const QueryParams& QP){
    bool has_visited_enough = (visited.size() >= QP.early_stop);
    bool early_stop = (QP.early_stop > 0); 
    bool has_found_candidate = (frontier[0].second <= QP.radius);
    bool within_early_stop_rad = (unvisited_frontier[0].second <= QP.early_stopping_radius);
    return early_stop && has_visited_enough && !has_found_candidate && !within_early_stop_rad;
    }
}
