#ifndef ALGORITHMS_ANN_BEAM_SEARCH_H_
#define ALGORITHMS_ANN_BEAM_SEARCH_H_

#include <algorithm>
#include <functional>
#include <limits>
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
#include "hashset.h"

namespace parlayANN {

  struct EarlyStopping {
  template<typename PointInfo>
  bool operator () (const PointInfo& frontier, 
                    const PointInfo& unvisited_frontier,
                    const PointInfo& visited,
                    const QueryParams& QP) { return false;}
  };

  
  // main beam search
template<typename indexType, typename Point, typename PointRange,
         typename QPoint, typename QPointRange, typename GT, typename ES = EarlyStopping>
std::pair<std::pair<parlay::sequence<std::pair<indexType, typename Point::distanceType>>,
                    parlay::sequence<std::pair<indexType, typename Point::distanceType>>>,
          size_t>
filtered_beam_search(const GT &G,
                     const Point p,  const PointRange &Points,
                     const QPoint qp, const QPointRange &Q_Points,
                     const parlay::sequence<indexType> starting_points,
                     const QueryParams &QP,
                     bool use_filtering = false,
                     ES early_stop = ES{}
                     ) {
  using dtype = typename Point::distanceType;
  using id_dist = std::pair<indexType, dtype>;
  int beamSize = QP.beamSize;
  int max_degree = QP.degree_limit;

  if (starting_points.size() == 0) {
    std::cout << "beam search expects at least one start point" << std::endl;
    abort();
  } else if (starting_points.size() > beamSize) {
    std::cout << "beam search has more starting points than beam size" << std::endl;
    abort();
  }

  // compare two (node_id,distance) pairs, first by distance and then id if
  // equal
  using distanceType = typename Point::distanceType;
  auto less = [&](id_dist a, id_dist b) {
    return a.second < b.second || (a.second == b.second && a.first < b.first);
  };

  hashset<indexType> has_been_seen(2 * (10 + beamSize) * max_degree);
  
  // Frontier maintains the closest points found so far and its size
  // is always at most beamSize.  Each entry is a (id,distance) pair.
  // Initialized with starting points and kept sorted by distance.
  std::vector<id_dist> frontier;
  frontier.reserve(beamSize);
  for (auto q : starting_points) {
    frontier.push_back(id_dist(q, Points[q].distance(p)));
    has_been_seen(q);
  }
  std::sort(frontier.begin(), frontier.end(), less);

  // The subset of the frontier that has not been visited
  // Use the first of these to pick next vertex to visit.
  std::vector<id_dist> unvisited_frontier(beamSize);
  for (int i=0; i < frontier.size(); i++)
    unvisited_frontier[i] = frontier[i];

  // maintains sorted set of visited vertices (id-distance pairs)
  std::vector<id_dist> visited;
  visited.reserve(2 * beamSize);

  // counters
  size_t dist_cmps = starting_points.size();
  size_t full_dist_cmps = starting_points.size();
  int remain = frontier.size();
  int num_visited = 0;

  // used as temporaries in the loop
  std::vector<id_dist> new_frontier(2 * std::max<size_t>(beamSize,starting_points.size()) +
                                    G.max_degree());
  std::vector<id_dist> candidates;
  candidates.reserve(G.max_degree() + beamSize);
  std::vector<indexType> filtered;
  filtered.reserve(G.max_degree());
  std::vector<indexType> pruned;
  pruned.reserve(G.max_degree());

  dtype filter_threshold_sum = 0.0;
  int filter_threshold_count = 0;
  dtype filter_threshold;
  indexType filter_id;
  indexType filter_tail_mean = 0;

  // offset into the unvisited_frontier vector (unvisited_frontier[offset] is the next to visit)
  int offset = 0;

  // The main loop.  Terminate beam search when the entire frontier
  // has been visited or have reached max_visit.
  while (remain > offset && num_visited < QP.limit) {
    
    // the next node to visit is the unvisited frontier node that is closest to p
    id_dist current = unvisited_frontier[offset];
    if (early_stop(frontier, unvisited_frontier, visited, QP))
      break;
    
    G[current.first].prefetch();
    // add to visited set
    auto position = std::upper_bound(visited.begin(), visited.end(), current, less);
    visited.insert(position, current);
    num_visited++;
    bool frontier_full = frontier.size() == beamSize;

    // if using filtering based on lower quality distances measure, then maintain the average
    // of low quality distance to the last point in the frontier (if frontier is full)
    if (use_filtering && frontier_full) {
      //constexpr int width = 5;
      int width = frontier.size();
      indexType id = frontier.back().first;
      if (filter_threshold_count == 0 || filter_id != id) {
        filter_tail_mean = 0.0;
        for (int i = frontier.size() - width; i < frontier.size(); i ++) 
          filter_tail_mean += Q_Points[frontier[i].first].distance(qp);
        filter_tail_mean /= width;
        filter_id = id;
      }
      filter_threshold_sum += filter_tail_mean;
      filter_threshold_count++;
      filter_threshold = filter_threshold_sum / filter_threshold_count;
    }

    // keep neighbors that have not been visited (using approximate
    // hash). Note that if a visited node is accidentally kept due to
    // approximate hash it will be removed below by the union.
    pruned.clear();
    filtered.clear();
    long num_elts = std::min<long>(G[current.first].size(), QP.degree_limit);
    for (indexType i=0; i<num_elts; i++) {
      auto a = G[current.first][i];
      if (has_been_seen(a) || Points[a].same_as(p)) continue;  // skip if already seen
      Q_Points[a].prefetch();
      pruned.push_back(a);
    }
    dist_cmps += pruned.size();

    // filter using low-quality distance
    if (use_filtering && frontier_full) {
      for (auto a : pruned) {
        if (frontier_full && Q_Points[a].distance(qp) >= filter_threshold) continue;
        filtered.push_back(a);
        Points[a].prefetch();
      }
    } else std::swap(filtered, pruned);

    // Further remove if distance is greater than current
    // furthest distance in current frontier (if full).
    distanceType cutoff = (frontier_full
                           ? frontier[frontier.size() - 1].second
                           : (distanceType)std::numeric_limits<distanceType>::max());
    for (auto a : filtered) {
      distanceType dist = Points[a].distance(p);
      full_dist_cmps++;
      // skip if frontier not full and distance too large
      if (dist >= cutoff) continue;
      candidates.push_back(std::pair{a, dist});
    }
    // If candidates insufficently full then skip rest of step until sufficiently full.
    // This iproves performance for higher accuracies (e.g. beam sizes of 100+)
    if (candidates.size() == 0 || 
        (QP.limit >= 2 * beamSize &&
         //candidates.size() < beamSize/8 &&
         candidates.size() < QP.batch_factor * beamSize &&
         offset + 1 < remain)) {
      offset++;
      continue;
    }
    offset = 0;

    // sort the candidates by distance from p,
    // and remove any duplicates (to be robust for neighbor lists with duplicates)
    std::sort(candidates.begin(), candidates.end(), less);
    auto candidates_end = std::unique(candidates.begin(), candidates.end(),
                                      [] (auto a, auto b) {return a.first == b.first;});

    // union the frontier and candidates into new_frontier, both are sorted
    auto new_frontier_size =
      std::set_union(frontier.begin(), frontier.end(), candidates.begin(),
                     candidates_end, new_frontier.begin(), less) -
      new_frontier.begin();
    candidates.clear();
    
    // trim to at most beam size
    new_frontier_size = std::min<size_t>(beamSize, new_frontier_size);

    // copy new_frontier back to the frontier
    frontier.clear();
    for (indexType i = 0; i < new_frontier_size; i++)
      frontier.push_back(new_frontier[i]);

    // get the unvisited frontier
    remain = (std::set_difference(frontier.begin(),
                                  frontier.begin() + std::min<long>(frontier.size(), QP.beamSize),
                                  visited.begin(),
                                  visited.end(),
                                  unvisited_frontier.begin(), less) -
              unvisited_frontier.begin());
  }

  return std::make_pair(std::make_pair(parlay::to_sequence(frontier),
                                       parlay::to_sequence(visited)),
                        full_dist_cmps);
}

  // alternative experimental version
  // about equal performance
template<typename indexType, typename Point, typename PointRange,
         typename QPoint, typename QPointRange, typename GT, typename ES = EarlyStopping>
std::pair<std::pair<parlay::sequence<std::pair<indexType, typename Point::distanceType>>,
                    parlay::sequence<std::pair<indexType, typename Point::distanceType>>>,
          size_t>
filtered_beam_search_new(const GT &G,
                      const Point p,  const PointRange &Points,
                      const QPoint qp, const QPointRange &Q_Points,
                      const parlay::sequence<indexType> starting_points,
                      const QueryParams &QP,
                      bool use_filtering = false,
                      ES early_stop = ES{}
                      ) {
  using dtype = typename Point::distanceType;
  using id_dist = std::pair<indexType, dtype>;
  int beamSize = QP.beamSize;
  int max_degree = QP.degree_limit;

  if (starting_points.size() == 0) {
    std::cout << "beam search expects at least one start point" << std::endl;
    abort();
  } else if (starting_points.size() > beamSize) {
    std::cout << "beam search has more starting points than beam size" << std::endl;
    abort();
  }

  // compare two (node_id,distance) pairs, first by distance and then id if
  // equal
  using distanceType = typename Point::distanceType;
  auto less = [&](id_dist a, id_dist b) {
    return a.second < b.second || (a.second == b.second && a.first < b.first);
  };

  long set_size = 1.5 * (10 + beamSize) * max_degree;
  hashset<indexType> has_been_seen(set_size);
  
  // Frontier maintains the closest points found so far and its size
  // is always at most beamSize.  Each entry is a (id,distance) pair.
  // Initialized with starting points and kept sorted by distance.
  std::vector<id_dist> frontier;
  frontier.reserve(2*beamSize);
  for (auto q : starting_points) {
    frontier.push_back(id_dist(q, Points[q].distance(p)));
    has_been_seen(q);
  }
  std::sort(frontier.begin(), frontier.end(), less);
  std::vector<id_dist> new_frontier;
  
  // maintains sorted set of visited vertices (id-distance pairs)
  std::vector<id_dist> visited;
  visited.reserve(2 * beamSize);

  // counters
  size_t dist_cmps = starting_points.size();
  size_t full_dist_cmps = starting_points.size();
  int num_visited = 0;

  // used as temporaries in the loop
  std::vector<id_dist> candidates;
  candidates.reserve(G.max_degree() + beamSize);
  std::vector<indexType> filtered;
  filtered.reserve(G.max_degree());
  std::vector<indexType> pruned;
  pruned.reserve(G.max_degree());

  // offset into the unvisited_frontier vector (unvisited_frontier[offset] is the next to visit)
  int offset = 0;
  std::priority_queue<dtype> topQ;
  for (auto [v,d] : frontier)
    topQ.push(d);
  std::priority_queue<dtype> visitedQ;

  float filter_threshold = 0.0;
  int filter_threshold_cnt = 0;
  float round_sum = 0.0;

  // The main loop.  Terminate beam search when the entire frontier
  // has been visited or have reached max_visit.
  while (frontier.size() > 0 && num_visited < QP.limit) {
    // the next node to visit is the unvisited frontier node that is closest to p
    id_dist current = frontier[offset];
    if (visitedQ.size() == beamSize && visitedQ.top() <= current.second) break;
    visited.push_back(current);
    visitedQ.push(current.second);
    if (visitedQ.size() > beamSize)
      visitedQ.pop();

    //if (early_stop(frontier, unvisited_frontier, visited, QP))
    //  break;
    
    G[current.first].prefetch();
    num_visited++;
    bool has_full_beam = (topQ.size() >= beamSize);

    pruned.clear();
    filtered.clear();
    long num_elts = std::min<long>(G[current.first].size(), QP.degree_limit);
    for (indexType i=0; i<num_elts; i++) {
      auto a = G[current.first][i];
      if (has_been_seen(a) || Points[a].same_as(p)) continue;  // skip if already seen
      Q_Points[a].prefetch();
      pruned.push_back(a);
    }
    dist_cmps += pruned.size();
    
    // filter using low-quality distance
    if (use_filtering && has_full_beam) {
      for (auto a : pruned) {
        if (Q_Points[a].distance(qp) >= filter_threshold) continue;
        filtered.push_back(a);
        Points[a].prefetch();
      }
    } else std::swap(filtered, pruned);
    
    // Further remove if distance is greater than current
    // furthest distance in current frontier (if full).
    for (auto a : filtered) {
      distanceType dist = Points[a].distance(p);
      full_dist_cmps++;
      // skip if frontier not full and distance too large
      if (topQ.size() == beamSize && topQ.top() <= dist)
        continue;
      topQ.push(dist);
      if (topQ.size() > beamSize) topQ.pop();
      if (use_filtering)
        round_sum += Q_Points[a].distance(qp);
      candidates.push_back(std::pair{a, dist});
    }

    offset++;

    // If candidates insufficently full then skip rest of step until sufficiently full.
    // This iproves performance for higher accuracies (e.g. beam sizes of 100+)
    if (offset != frontier.size() &&
        (candidates.size() == 0 || 
         (QP.limit >= 2 * beamSize &&
          candidates.size() < QP.batch_factor * beamSize)) &&
        (visitedQ.size() != beamSize ||
         visitedQ.top() > frontier[offset].second))
      continue;

    if (use_filtering) {
      float round_average = round_sum/candidates.size();
      // We use a rolling average to keep the filter_threshold smooth
      // and always a bit bigger than distances we have seen recently
      // so we don't filter out too many points.
      if (filter_threshold_cnt == 0)
        filter_threshold = round_average;
      else filter_threshold = (filter_threshold * .85 + round_average * .15);
      round_sum = 0;
      filter_threshold_cnt++;
    }
    
    // sort the candidates by distance from p,
    std::sort(candidates.begin(), candidates.end(), less);
    
    // merge the frontier and candidates into new_frontier, both are sorted
    long merge_size = frontier.size() - offset + candidates.size();
        
    new_frontier.resize(merge_size);
    std::merge(frontier.begin()+offset, frontier.end(), candidates.begin(),
               candidates.end(), new_frontier.begin(), less);
    if (merge_size > beamSize) 
      new_frontier.resize(beamSize);
    candidates.clear();
    std::swap(frontier, new_frontier);
    offset = 0;
  }

  // sort all visited points and take the first beamSize of them
  std::sort(visited.begin(), visited.end(), less);
  if (visited.size() > beamSize)
    visited.resize(beamSize);

  return std::make_pair(std::make_pair(parlay::to_sequence(visited),
                                       parlay::to_sequence(visited)),
                        full_dist_cmps);
}

  struct EStop {
    template<typename PointInfo>
    bool operator () (const PointInfo& frontier, 
                      const PointInfo& unvisited_frontier,
                      const PointInfo& visited,
                      const QueryParams& QP) { return false;}
  };
  
// version without filtering
  template<typename Point, typename PointRange, typename indexType> // = EarlyStopping>
std::pair<std::pair<parlay::sequence<std::pair<indexType, typename Point::distanceType>>,
                    parlay::sequence<std::pair<indexType, typename Point::distanceType>>>, size_t>
beam_search(const Point p, const Graph<indexType> &G, const PointRange &Points,
            const parlay::sequence<indexType> starting_points, const QueryParams &QP
            ) {
    return filtered_beam_search(G,p, Points, p, Points, starting_points, QP, false); //early_stop);
}

// backward compatibility (for hnsw)
template<typename indexType, typename Point, typename PointRange, class GT>
std::pair<std::pair<parlay::sequence<std::pair<indexType, typename Point::distanceType>>, parlay::sequence<std::pair<indexType, typename Point::distanceType>>>, size_t>
beam_search_impl(Point p, GT &G, PointRange &Points,
                 parlay::sequence<indexType> starting_points, QueryParams &QP) {
  return filtered_beam_search(G, p, Points, p, Points, starting_points, QP, false);
}

// pass single start point
template<typename Point, typename PointRange, typename indexType>
std::pair<std::pair<parlay::sequence<std::pair<indexType, typename Point::distanceType>>,
                    parlay::sequence<std::pair<indexType, typename Point::distanceType>>>, indexType>
beam_search(const Point p, const Graph<indexType> &G, const PointRange &Points,
            const indexType starting_point, const QueryParams &QP) {
  parlay::sequence<indexType> start_points = {starting_point};
  return beam_search(p, G, Points, start_points, QP);
}

// searches every element in q starting from a randomly selected point
template<typename PointRange, typename indexType>
parlay::sequence<parlay::sequence<indexType>>
beamSearchRandom(const PointRange& Query_Points,
                 const Graph<indexType> &G,
                 const PointRange &Base_Points,
                 stats<indexType> &QueryStats,
                 const QueryParams &QP) {
  using Point = typename PointRange::Point;
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

template<typename PointRange, typename indexType>
parlay::sequence<parlay::sequence<indexType>>
searchAll(PointRange& Query_Points,
          Graph<indexType> &G, PointRange &Base_Points, stats<indexType> &QueryStats,
          indexType starting_point, QueryParams &QP) {
  parlay::sequence<indexType> start_points = {starting_point};
  return searchAll<PointRange, indexType>(Query_Points, G, Base_Points, QueryStats, start_points, QP);
}

template< typename PointRange, typename indexType>
parlay::sequence<parlay::sequence<indexType>>
searchAll(PointRange &Query_Points,
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

// Returns a sequence of nearest neighbors each with their distance
template<typename Point, typename QPoint, typename QQPoint,
         typename PointRange, typename QPointRange, typename QQPointRange,
         typename indexType>
parlay::sequence<std::pair<indexType, typename Point::distanceType>>
beam_search_rerank(const Point &p,
                   const QPoint &qp,
                   const QQPoint &qqp,
                   const Graph<indexType> &G,
                   const PointRange &Base_Points,
                   const QPointRange &Q_Base_Points,
                   const QQPointRange &QQ_Base_Points,
                   stats<indexType> &QueryStats,
                   const parlay::sequence<indexType> starting_points,
                   const QueryParams &QP,
                   bool stats = true) {
  using dtype = typename Point::distanceType;
  using id_dist = std::pair<indexType, dtype>;
  auto QPP = QP;

  bool use_rerank = (Base_Points.params.num_bytes() != Q_Base_Points.params.num_bytes());
  bool use_filtering = (Q_Base_Points.params.num_bytes() != QQ_Base_Points.params.num_bytes());
  std::pair<std::pair<parlay::sequence<id_dist>, parlay::sequence<id_dist>>, size_t> r;
  r = filtered_beam_search(G,
                            qp, Q_Base_Points,
                            qqp, QQ_Base_Points,
                            starting_points, QPP, use_filtering);
  auto [pairElts, dist_cmps] = r;
  auto [beamElts, visitedElts] = pairElts;
  if (beamElts.size() < QP.k) {
    std::cout << "Error: for point id " << p.id()
              << " beam search returned " << beamElts.size()
              << " elements, which is less than k = " << QP.k << std::endl;
    abort();
  }
  
  if (stats) {
    QueryStats.increment_visited(p.id(), visitedElts.size());
    QueryStats.increment_dist(p.id(), dist_cmps);
  }

  if (use_rerank) {
    // recalculate distances with non-quantized points and sort
    int num_check = std::min<int>(QP.k * QP.rerank_factor, beamElts.size());
    std::vector<id_dist> pts;
    for (int i=0; i < num_check; i++) {
      int j = beamElts[i].first;
      pts.push_back(id_dist(j, p.distance(Base_Points[j])));
    }
    auto less = [&] (id_dist a, id_dist b) {
      return a.second < b.second || (a.second == b.second && a.first < b.first);
    };
    std::sort(pts.begin(), pts.end(), less);

    // keep first k
    parlay::sequence<id_dist> results;
    for (int i= 0; i < QP.k; i++)
      results.push_back(pts[i]);

    return results;
  } else {
    //return beamElts;
    parlay::sequence<id_dist> results;
    for (int i= 0; i < QP.k; i++) {
      int j = beamElts[i].first;
      results.push_back(id_dist(j, p.distance(Base_Points[j])));
    }
    return results;
  }
}

template<typename PointRange, typename QPointRange, typename QQPointRange, typename indexType>
parlay::sequence<parlay::sequence<indexType>>
qsearchAll(const PointRange &Query_Points,
           const QPointRange &Q_Query_Points,
           const QQPointRange &QQ_Query_Points,
           const Graph<indexType> &G,
           const PointRange &Base_Points,
           const QPointRange &Q_Base_Points,
           const QQPointRange &QQ_Base_Points,
           stats<indexType> &QueryStats,
           const indexType starting_point,
           const QueryParams &QP,
           bool random = false) {
  if (QP.k > QP.beamSize) {
    std::cout << "Error: beam search parameter Q = " << QP.beamSize
              << " same size or smaller than k = " << QP.k << std::endl;
    abort();
  }
  parlay::sequence<parlay::sequence<indexType>> all_neighbors(Query_Points.size());
  if (random) {
    parlay::random_generator gen;
    std::uniform_int_distribution<long> dis(0, G.size() - 1);
    auto indices = parlay::tabulate(Query_Points.size(), [&](size_t i) -> indexType {
      auto r = gen[i];
      return dis(r);
    });

    parlay::parallel_for(0, Query_Points.size(), [&](size_t i) {
      parlay::sequence<indexType> starting_points = {indices[i]};
      auto ngh_dist = beam_search_rerank(Query_Points[i], Q_Query_Points[i], QQ_Query_Points[i],
                                         G,
                                         Base_Points, Q_Base_Points, QQ_Base_Points,
                                         QueryStats, starting_points, QP);
      all_neighbors[i] = parlay::map(ngh_dist, [] (auto& p) {return p.first;});
    });
  } else {
    parlay::sequence<indexType> starting_points = {starting_point};
    parlay::parallel_for(0, Query_Points.size(), [&](size_t i) {
      auto ngh_dist = beam_search_rerank(Query_Points[i], Q_Query_Points[i], QQ_Query_Points[i],
                                         G,
                                         Base_Points, Q_Base_Points, QQ_Base_Points,
                                         QueryStats, starting_points, QP);
      all_neighbors[i] = parlay::map(ngh_dist, [] (auto& p) {return p.first;});
    });
  }

  return all_neighbors;
}



} // end namespace

#endif // ALGORITHMS_ANN_BEAM_SEARCH_H_
