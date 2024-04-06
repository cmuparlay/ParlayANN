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

#ifndef CLUSTER_EDGE
#define CLUSTER_EDGE

#include <math.h>

#include <algorithm>
#include <functional>
#include <queue>
#include <random>
#include <set>

#include "../utils/graph.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"


template <typename indexType>
std::pair<indexType, indexType> select_two_random(
    parlay::sequence<indexType>& active_indices, parlay::random& rnd) {
  indexType first_index = rnd.ith_rand(0) % active_indices.size();
  indexType second_index_unshifted = rnd.ith_rand(1) % (active_indices.size() - 1);
  indexType second_index = (second_index_unshifted < first_index)
                            ? second_index_unshifted
                            : (second_index_unshifted + 1);

  return {active_indices[first_index], active_indices[second_index]};
}

template<typename Point, typename PointRange, typename indexType>
struct cluster {
  using distanceType = typename Point::distanceType;
  using edge = std::pair<indexType, indexType>;
  using labelled_edge = std::pair<edge, distanceType>;
  using GraphI = Graph<indexType>;
  using PR = PointRange;

  cluster(){}

  int generate_index(int N, int i) {
    return (N * (N - 1) - (N - i) * (N - i - 1)) / 2;
  }

  template <typename F>
  void recurse(GraphI &G, PR &Points,
               parlay::sequence<indexType>& active_indices, parlay::random& rnd,
               size_t cluster_size, F f, long MSTDeg, indexType first,
               indexType second) {
    // // Split points based on which of the two points are closer.
    // // does all the distance calculations twice ???
    // auto closer_first =
    //     parlay::filter(parlay::make_slice(active_indices), [&](size_t ind) {
    //       distanceType dist_first = Points[ind].distance(Points[first]);
    //       distanceType dist_second = Points[ind].distance(Points[second]);
    //       return dist_first <= dist_second;
    //     });

    // auto closer_second =
    //     parlay::filter(parlay::make_slice(active_indices), [&](size_t ind) {
    //       distanceType dist_first = Points[ind].distance(Points[first]);
    //       distanceType dist_second = Points[ind].distance(Points[second]);
    //       return dist_second < dist_first;
    //     });

    // computes all the distances once, sorts points by relative distance to the two points, and then splits the points in half
    auto ids_and_rel_distances = parlay::tabulate(active_indices.size(), [&](size_t i) {
      distanceType dist_first = Points[active_indices[i]].distance(Points[first]);
      distanceType dist_second = Points[active_indices[i]].distance(Points[second]);
      return std::make_pair(active_indices[i], dist_first - dist_second);
    });

    std::sort(ids_and_rel_distances.begin(), ids_and_rel_distances.end(), [](auto a, auto b) {
      return a.second < b.second;
    });

    auto closer_first = parlay::tabulate(active_indices.size() / 2, [&](size_t i) {
      return ids_and_rel_distances[i].first;
    });

    auto closer_second = parlay::tabulate(active_indices.size() - active_indices.size() / 2, [&](size_t i) {
      return ids_and_rel_distances[i + active_indices.size() / 2].first;
    });

    auto left_rnd = rnd.fork(0);
    auto right_rnd = rnd.fork(1);

    parlay::par_do(
        [&]() {
          random_clustering(G, Points, closer_first, left_rnd, cluster_size, f, MSTDeg);
        },
        [&]() {
          random_clustering(G, Points, closer_second, right_rnd, cluster_size, f, MSTDeg);
        });
  }

  template <typename F>
  void random_clustering(GraphI &G, PR &Points,
                         parlay::sequence<indexType>& active_indices,
                         parlay::random& rnd, size_t cluster_size, F g,
                         long MSTDeg) {
    // if (std::max_element(active_indices.begin(), active_indices.end())[0] >= Points.size())
    //   std::cout << "oversized index passed to random_clustering" << std::endl;

    if (active_indices.size() == 0) {
      std::cout << "random_clustering: active_indices.size() == 0" << std::endl;
      // abort();
    }

    if (active_indices.size() <= cluster_size)
      g(G, Points, active_indices, MSTDeg);
    else {
      auto [f, s] = select_two_random(active_indices, rnd);
      if (Points[f]==Points[s]) {
        parlay::sequence<indexType> closer_first;
        parlay::sequence<indexType> closer_second;
        for (int i = 0; i < active_indices.size(); i++) {
          if (i < active_indices.size() / 2) // random split if the two points are the same ???
            closer_first.push_back(active_indices[i]);
          else
            closer_second.push_back(active_indices[i]);
        }
        auto left_rnd = rnd.fork(0);
        auto right_rnd = rnd.fork(1);
        parlay::par_do(
            [&]() {
              random_clustering(G, Points, closer_first, left_rnd, cluster_size, g, MSTDeg);
            },
            [&]() {
              random_clustering(G, Points, closer_second, right_rnd, cluster_size, g, MSTDeg);
            });
      } else {
        recurse(G, Points, active_indices, rnd, cluster_size, g, MSTDeg, f, s);
      }
    }
  }

  template <typename F>
  void random_clustering_wrapper(GraphI &G, PR &Points,
                                 size_t cluster_size, F f, long MSTDeg) {
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> uni(0, Points.size());
    parlay::random rnd(uni(rng));
    auto active_indices =
        parlay::tabulate(Points.size(), [&](indexType i) { return i; });

    if (std::max_element(active_indices.begin(), active_indices.end())[0] !=
        Points.size() - 1)
      std::cout << "max element is not the last element" << std::endl;
      
    random_clustering(G, Points, active_indices, rnd, cluster_size, f, MSTDeg);
  }

  template <typename F>
  void active_indices_rcw(GraphI &G, PR &Points, parlay::sequence<indexType> active_indices,
                          size_t cluster_size, F f, long MSTDeg) {
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> uni(0, Points.size());
    parlay::random rnd(uni(rng));
    random_clustering(G, Points, active_indices, rnd, cluster_size, f, MSTDeg);
  }

  template <typename F>
  void multiple_clustertrees(GraphI &G, PR &Points,
                             long cluster_size, long num_clusters, F f,
                             long MSTDeg) {
    for (long i = 0; i < num_clusters; i++) {
      random_clustering_wrapper(G, Points, cluster_size, f, MSTDeg);
      std::cout << "Built cluster " << i << " of " << num_clusters << std::endl;
    }
  }
};

#endif // CLUSTER_EDGE