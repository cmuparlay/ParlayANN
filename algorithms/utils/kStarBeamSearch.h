#pragma once

#include <algorithm>
#include <functional>
#include <random>
#include <set>
#include <unordered_set>
#include <unordered_map>

#include "parlay/io.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"
#include "types.h"
#include "graph.h"
#include "stats.h"
#include "filters.h"

/* A minimal struct for tracking the number of occurences of each filter
 */
struct Counts {
  std::unordered_map<int, int> data;

  Counts() : data() {}

  void increment(int f) {
    if (data.find(f) == data.end()) {
      data[f] = 1;
    } else {
      data[f]++;
    }
  }

  void decrement(int f) {
    if (data.find(f) == data.end()) {
      std::cerr << "ERROR: tried to decrement a filter that doesn't exist" << std::endl;
      abort();
    } else {
      data[f]--;
      if (data[f] == 0) {
        data.erase(f);
      } else if (data[f] < 0) {
        std::cerr << "ERROR: tried to decrement a filter that has a count of 0" << std::endl;
        abort();
      }
    }
  }

  /* returns true if there exists an entry for a given filter */
  bool contains(int f) {
    return data.find(f) != data.end();
  }

  int count(int f) {
    if (data.find(f) == data.end()) {
      return 0;
    } else {
      return data[f];
    }
  }
  
};


template<typename Point, typename PointRange, typename indexType>
std::pair<std::pair<parlay::sequence<std::pair<indexType, typename Point::distanceType>>, parlay::sequence<std::pair<indexType, typename Point::distanceType>>>, size_t>
k_star_beam_search(Point p, size_t k_star, Graph<indexType> &G, PointRange &Points,
	      parlay::sequence<indexType> starting_points, QueryParams &QP, csr_filters &filters) {

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

  // track filter counts to enforce k_star constraint
  Counts filter_counts;

  // Frontier maintains the closest points found so far and its size
  // is always at most beamSize.  Each entry is a (id,distance) pair.
  // Initialized with starting points and kept sorted by distance.
  // we assume that the k_star condition is satisfied for the starting points
  // TODO: check this
  std::vector<std::pair<indexType, distanceType>> frontier;
  frontier.reserve(QP.beamSize);
  for (auto q : starting_points){
    frontier.push_back(std::pair<indexType, distanceType>(q, Points[q].distance(p)));
    filter_counts.increment(filters.first_label(q));
  }
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
  std::vector<std::pair<indexType, distanceType>> new_frontier(QP.beamSize + G.max_degree());
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
      if (a == p.id() || has_been_seen(a)) continue;  // skip if already seen
      bool matches = false; // this is "matches" from filtered beam search, we leave the name but repurpose it for "can be added considering k-star requirements"
      
      size_t label = filters.first_label(a);
      if (!filter_counts.count(label) < k_star) { // if there's room for another, it's fine
        matches = true;
      } else { // see if it beats the worst conspecific
        // we iterate backwards through the frontier to find the worst conspecific, which should be the first match we encounter
        for (int j = frontier.size() - 1; j >= 0; j--) {
          if (filters.first_label(frontier[j].first) == label) {
            if (frontier[j].second > Points[a].distance(p)) {
              matches = true;
              break;
            } else {
              break;
            }
          }
        }
      }
      if (!matches) continue;
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
      total += dist;
      dist_cmps++;
      // skip if frontier not full and distance too large
      if (dist >= cutoff) continue;
      candidates.push_back(std::pair{a, dist});
    }

    // sort the candidates by distance from p
    std::sort(candidates.begin(), candidates.end(), less);

    // add the candidates to the counter
    for (auto c : candidates) {
      // possibly quite slow and perhaps not necessary, but we should check that there are no duplicates between this and the frontier
      filter_counts.increment(filters.first_label(c.first));
    }

    // union the frontier and candidates into new_frontier, both are sorted
    auto new_frontier_size =
        std::set_union(frontier.begin(), frontier.end(), candidates.begin(),
                       candidates.end(), new_frontier.begin(), less) -
        new_frontier.begin();

    
    // iterate backwards through the frontier to find overrepresented elements
    for (int i = frontier.size() - 1; i >= 0; i--) {
      if (filter_counts.count(filters.first_label(frontier[i].first)) > k_star) {
        // remove the element from the frontier
        filter_counts.decrement(filters.first_label(frontier[i].first));
        new_frontier_size--; 
        new_frontier.erase(new_frontier.begin() + i);
      }
    }


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
        std::set_difference(frontier.begin(), frontier.end(), visited.begin(),
                            visited.end(), unvisited_frontier.begin(), less) -
        unvisited_frontier.begin();
  }

  return std::make_pair(std::make_pair(parlay::to_sequence(frontier),
                                       parlay::to_sequence(visited)),
                        dist_cmps);
}
