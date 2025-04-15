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
    template<typename Point, typename PointRange, typename indexType, typename distanceType> 
    bool early_stopping(Point q, std::vector<std::pair<indexType, distanceType>>& frontier, 
         std::vector<std::pair<indexType, distanceType>>& unvisited_frontier, double rad, long es, double esr, int num_visited){
            bool has_visited_enough = (num_visited >= es);
            bool early_stop = (es > 0); 
            bool has_found_candidate = (frontier[0].second <= rad);
            bool within_early_stop_rad = (unvisited_frontier[0].second <= esr);
            return early_stop && has_visited_enough && !has_found_candidate && !within_early_stop_rad;
         }


}
