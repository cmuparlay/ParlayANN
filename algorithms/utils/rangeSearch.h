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
#include "filtered_hashset.h"

namespace parlayANN {

  template<typename Point, typename PointRange, typename indexType>
  std::pair<std::vector<indexType>, long>
greedy_search(Point p, Graph<indexType> &G, PointRange &Points,
              std::vector<std::pair<indexType, typename Point::distanceType>> &starting_points,
              double radius) {
  std::vector<indexType> result;
  hashset<indexType> has_been_seen(2 * starting_points.size() * 64);
  long distance_comparisons = 0;

  for (auto [v,d] : starting_points) {
    if (has_been_seen(v) || d > radius) continue;
    result.push_back(v);
  }

  // now do a BFS over all vertices with distance less than radius
  long position = 0;
  std::vector<indexType> unseen;
  while (position < result.size()) {
    indexType next = result[position++];
    unseen.clear();
    for (long i = 0; i < G[next].size(); i++) {
      auto v = G[next][i];
      if (has_been_seen(v)) continue;
      unseen.push_back(v);
      Points[v].prefetch();
    }
    for (auto v : unseen) {
      distance_comparisons++;
      if (Points[v].distance(p) <= radius)
        result.push_back(v);
    }
  }

  return std::pair(std::move(result), distance_comparisons);
}

  // Does a priority-first search up to the radius given
  template<typename Point, typename PointRange, typename indexType>
  std::pair<std::vector<indexType>, long>
greedy_search_pq(Point p, Graph<indexType> &G, PointRange &Points,
                 std::vector<std::pair<indexType, typename Point::distanceType>> &starting_points,
                 double radius) {

  std::vector<indexType> result;
  hashset<indexType> has_been_seen(2 * starting_points.size() * 64);
  
  long distance_comparisons = 0;
  using did = std::pair<typename Point::distanceType, indexType>;
  auto cmp = [] (did a, did b) {return a.first > b.first;};
  std::priority_queue<did, std::vector<did>, decltype(cmp)> pq(cmp);

  for (auto [v,d] : starting_points) {
    if (has_been_seen(v)) continue;
    if (d > radius ) continue;
    pq.push(std::pair(d,v));
  }

  long position = 0;
  std::vector<indexType> unseen;
  while (pq.top().first <= radius) {
    auto nxt = pq.top().second;
    pq.pop();
    result.push_back(nxt);
    unseen.clear();
    for (long i = 0; i < G[nxt].size(); i++) {
      auto v = G[nxt][i];
      if (has_been_seen(v)) continue;
      unseen.push_back(v);
      Points[v].prefetch();
    }
    for (auto v : unseen) {
      distance_comparisons++;
      pq.push(std::pair(Points[v].distance(p), v));
    }
  }

  return std::pair(std::move(result), distance_comparisons);
}

  //a variant specialized for range searching
template<typename Point, typename PointRange, typename indexType>
std::pair<std::vector<indexType>, size_t>
greedy_search_old(Point p, Graph<indexType> &G, PointRange &Points,
                  parlay::sequence<std::pair<indexType, typename Point::distanceType>> &starting_points,
                  double radius,
                  parlay::sequence<std::pair<indexType, typename Point::distanceType>> &already_visited) {
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
  std::queue<indexType> frontier;
  for (auto q : starting_points){
    if (!has_been_seen.count(q.first) > 0) has_been_seen.insert(q.first);
    frontier.push(q.first);
  }
  

  // maintains set of visited vertices (id-distance pairs)
  std::vector<indexType> visited;

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
    indexType current = frontier.front();
    frontier.pop();
    G[current].prefetch();
    // add to visited set
    visited.push_back(current);
    num_visited++;

    // keep neighbors that have not been visited (using approximate
    // hash). Note that if a visited node is accidentally kept due to
    // approximate hash it will be removed below by the union or will
    // not bump anyone else.
    keep.clear();
    for (indexType i=0; i<G[current].size(); i++) {
      auto a = G[current][i];
      //TODO this is a bug when searching for a point not in the graph???
      if (a == p.id() || has_been_seen.count(a) > 0) continue;  // skip if already seen
      keep.push_back(a);
      Points[a].prefetch();
      has_been_seen.insert(a);
    }

    for (auto a : keep) {
      distanceType dist = Points[a].distance(p);
      dist_cmps++;
      // filter out if not within radius
      if (dist > radius) continue;
      frontier.push(a);
    }
  }

  return std::make_pair(visited, dist_cmps);    
}

  template<typename Point, typename PointRange, typename QPointRange, typename indexType>
  std::pair<parlay::sequence<std::vector<indexType>>,std::pair<double,double>> 
RangeSearch(Graph<indexType> &G,
            PointRange &Query_Points, PointRange &Base_Points,
            QPointRange& Q_Query_Points, QPointRange &Q_Base_Points,
            stats<indexType> &QueryStats,
            indexType starting_point,
            QueryParams &QP) {

  parlay::sequence<indexType> starting_points = {starting_point};
  parlay::sequence<std::vector<indexType>> all_neighbors(Query_Points.size());
  parlay::WorkerSpecific<double> beam_time;
  parlay::WorkerSpecific<double> other_time;
  bool use_rerank = (Base_Points.params.num_bytes() != Q_Base_Points.params.num_bytes());
  parlay::parallel_for(0, Query_Points.size(), [&](size_t i) {
    parlay::internal::timer t_search_beam("beam search time");
    parlay::internal::timer t_search_other("other time");
    t_search_beam.stop();
    t_search_other.stop();
    std::vector<indexType> neighbors;
    std::vector<std::pair<indexType, typename Point::distanceType>> neighbors_with_distance;
    t_search_beam.start();
    using dtype = typename Point::distanceType;
    using id_dist = std::pair<indexType, dtype>;
    QueryParams QP1(QP.beamSize, QP.beamSize, 0.0, G.size(), G.max_degree(),
                    QP.is_early_stop, Q_Query_Points[i].translate_distance(QP.early_stopping_radius),
                    QP.early_stopping_count,
                    QP.range_query_type, Q_Query_Points[i].translate_distance(QP.radius));

    auto [pairElts, dist_cmps_beam] =
      filtered_beam_search(G, Q_Query_Points[i], Q_Base_Points,
                           Q_Query_Points[i], Q_Base_Points,
                           starting_points, QP1, false,
                           early_stopping<std::vector<id_dist>>);
    t_search_beam.stop();
    auto [beamElts, visitedElts] = pairElts;
    for (auto b : beamElts) {
      double dist;
      if (use_rerank) {
        dist = Query_Points[i].distance(Base_Points[b.first]);
      } else {
        dist = b.second;
      }
      if (dist <= QP.radius) {
        neighbors.push_back(b.first);
        neighbors_with_distance.push_back(b);
      }
    }
    if (neighbors.size() < QP.beamSize || QP.range_query_type == Beam){
      all_neighbors[i] = std::move(neighbors);
    } else{
      // if using quantization then use slightly larger radius
      t_search_other.start();
      double pad_factor = (QP1.radius > 0) ? 1.05 : .975;
      double radius = use_rerank ? pad_factor * QP1.radius : QP1.radius;
      auto [in_range, dist_cmps_greedy] =
        greedy_search(Q_Query_Points[i], G, Q_Base_Points,
                      neighbors_with_distance, radius);

      std::vector<indexType> ans;

      //#define EndWithBeam
#ifdef EndWithBeam
      int beamSize = in_range.size() * 1.1;
      QueryParams QP2(beamSize, beamSize, 0.0, G.size(), G.max_degree());
      auto [pairElts, dist_cmps2] = beam_search(Q_Query_Points[i], G, Q_Base_Points, in_range, QP2);
      for (auto r : pairElts.first) 
        if (Query_Points[i].distance(Base_Points[r.first]) <= QP.radius)
          ans.push_back(r.first);
#else
      for (auto r : in_range)
        if (!use_rerank || Query_Points[i].distance(Base_Points[r]) <= QP.radius)
          ans.push_back(r);
#endif
      all_neighbors[i] = std::move(ans);
      QueryStats.increment_visited(i, in_range.size());
      QueryStats.increment_dist(i, dist_cmps_greedy);
      t_search_other.stop();
    }
    

    *beam_time += t_search_beam.total_time();
    *other_time += t_search_other.total_time();
    QueryStats.increment_visited(i, visitedElts.size());
    QueryStats.increment_dist(i, dist_cmps_beam);
    
  });

  double total_time_beam = 0;
  double total_time_other = 0;
  for (auto x : beam_time) total_time_beam += x;
  for (auto y: other_time) total_time_other += y;
  return std::make_pair(all_neighbors,std::make_pair(total_time_beam,total_time_other));
}

}