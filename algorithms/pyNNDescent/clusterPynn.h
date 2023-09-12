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

#include <math.h>

#include <algorithm>
#include <functional>
#include <queue>
#include <random>
#include <set>

#include "../HCNNG/clusterEdge.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"
#include "../utils/union.h"

template<typename Point, typename PointRange, typename indexType>
struct clusterPID {
  using distanceType = typename Point::distanceType;
	using PR = PointRange;
  using edge = std::pair<indexType , indexType >;
  using pid = std::pair<indexType , distanceType>;

  clusterPID() {}

  parlay::sequence<parlay::sequence<pid>> intermediate_edges;

  void naive_neighbors(PR &Points,
                       parlay::sequence<size_t>& active_indices,
                       long maxK) {
    size_t n = active_indices.size();
    parlay::parallel_for(0, n, [&](size_t i) {
      auto less = [&](pid a, pid b) { return a.second < b.second; };
      std::priority_queue<pid, std::vector<pid>, decltype(less)> Q(less);
      size_t index = active_indices[i];
      // tabulate all-pairs distances between the elements in the leaf
      for (indexType  j = 0; j < n; j++) {
        if (j != i) {
          distanceType dist = Points[index].distance(Points[active_indices[j]]);
          pid e = std::make_pair(active_indices[j], dist);
          if (Q.size() >= maxK) {
            distanceType topdist = Q.top().second;
            if (dist < topdist) {
              Q.pop();
              Q.push(e);
            }
          } else {
            Q.push(e);
          }
        }
      }
      size_t q = Q.size();
      parlay::sequence<pid> sorted_edges(q);
      for (indexType  j = 0; j < q; j++) {
        sorted_edges[j] = Q.top();
        Q.pop();
      }
      auto rev_edges = parlay::reverse(sorted_edges);
      auto [new_best, changed] =
          seq_union_bounded(intermediate_edges[index], rev_edges, maxK, less);
      intermediate_edges[index] = new_best;
    });
  }


  void random_clustering(PR &Points,
                         parlay::sequence<size_t>& active_indices,
                         parlay::random& rnd, long cluster_size,
                         long K) {
    if (active_indices.size() <= cluster_size)
      naive_neighbors(Points, active_indices, K);
    else {
      auto [f, s] = select_two_random(active_indices, rnd);

      auto left_rnd = rnd.fork(0);
      auto right_rnd = rnd.fork(1);

      if (Points[f]==Points[s]) {
        parlay::sequence<size_t> closer_first;
        parlay::sequence<size_t> closer_second;
        for (indexType i = 0; i < active_indices.size(); i++) {
          if (i < active_indices.size() / 2)
            closer_first.push_back(active_indices[i]);
          else
            closer_second.push_back(active_indices[i]);
        }
        auto left_rnd = rnd.fork(0);
        auto right_rnd = rnd.fork(1);
        parlay::par_do(
            [&]() {
              random_clustering(Points, closer_first, left_rnd, cluster_size,
                                K);
            },
            [&]() {
              random_clustering(Points, closer_second, right_rnd, cluster_size,
                                K);
            });
      } else {
        // Split points based on which of the two points are closer.
        auto closer_first =
            parlay::filter(parlay::make_slice(active_indices), [&](size_t ind) {
              distanceType dist_first = Points[ind].distance(Points[f]);
              distanceType dist_second = Points[ind].distance(Points[s]);
              return dist_first <= dist_second;
            });

        auto closer_second =
            parlay::filter(parlay::make_slice(active_indices), [&](size_t ind) {
              distanceType dist_first = Points[ind].distance(Points[f]);
              distanceType dist_second = Points[ind].distance(Points[s]);
              return dist_second < dist_first;
            });


        parlay::par_do(
            [&]() {
              random_clustering(Points, closer_first, left_rnd, cluster_size, 
                                K);
            },
            [&]() {
              random_clustering(Points, closer_second, right_rnd, cluster_size,
                                  K);
        });

      }
    }
  }

  void random_clustering_wrapper(PR &Points,
                                 long cluster_size, long K) {
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<indexType> uni(0, Points.size());
    parlay::random rnd(uni(rng));
    auto active_indices =
        parlay::tabulate(Points.size(), [&](size_t i) { return i; });
    random_clustering(Points, active_indices, rnd, cluster_size, K);
  }

  void multiple_clustertrees(PR &Points,
                             long cluster_size, long num_clusters,
                             long K,
                             parlay::sequence<parlay::sequence<pid>>& old_nbh) {
    intermediate_edges = parlay::sequence<parlay::sequence<pid>>(Points.size());
    for (long i = 0; i < num_clusters; i++) {
      random_clustering_wrapper(Points, cluster_size, K);
      std::cout << "Cluster " << i << std::endl; 
    }
    parlay::parallel_for(0, Points.size(),
                         [&](size_t i) { old_nbh[i] = intermediate_edges[i]; });
  }
};
