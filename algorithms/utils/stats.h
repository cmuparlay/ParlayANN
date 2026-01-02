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

#pragma once

#include <algorithm>
#include <queue>
#include <set>

#include "graph.h"

#include "parlay/io.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "../../parlaylib/include/parlay/io.h"
#include "../../parlaylib/examples/BFS.h"
#include "../../parlaylib/examples/helper/graph_utils.h"

namespace parlayANN {

inline std::pair<double, int> graph_stats_(Graph<unsigned int> &G) {
  auto od = parlay::delayed_seq<size_t>(
      G.size(), [&](size_t i) { return G[i].size(); });
  size_t j = parlay::max_element(od) - od.begin();
  int maxDegree = od[j];
  size_t sum1 = parlay::reduce(od);
  double avg_deg = sum1 / ((double)G.size());
  return std::make_pair(avg_deg, maxDegree);
}

  template <typename indexType>
void print_graph_statistics(Graph<indexType> &G, indexType start) {
  long n = G.size();

  // convert to right format for transpose
  auto GG = parlay::tabulate(n, [&] (indexType i) {
                                  parlay::sequence<indexType> out;
                                  for (auto x : G[i]) out.push_back(x);
                                  return out;
                                });
  // generate some statistics for the graph
  auto GTrans = graph_utils<indexType>::transpose(GG);
  auto inDegrees = parlay::map(GTrans, parlay::size_of());
  auto lowInDegrees = parlay::filter(parlay::iota(n), [&] (long i) {return inDegrees[i] < 4;});
  auto maxInDegree = *parlay::max_element(inDegrees);
  auto outDegrees = parlay::map(GG, parlay::size_of());
  auto maxOutDegree = *parlay::max_element(outDegrees);
  // for (auto u : lowDegrees) 
  //   for (auto v : G[u])
  //     G[v].append_neighbor(u);

  auto layers = BFS(start, G);
  auto sizes = parlay::map(layers, parlay::size_of());
  auto visited = flatten(layers);
  
  std::cout << "Graph statistics:" << std::endl;
  //std::cout << "  average degree = " << float(parlay::reduce(GG, parlay::size_of()))/n << std::endl;
  //std::cout << "  max out-degree = " << *minDegreeLoc << std::endl;
  std::cout << "  max in-degree = " << maxInDegree << std::endl;
  std::cout << "  number with low (< 4) in-degree = " << lowInDegrees.size() << std::endl;
  std::cout << "  unreachable from source = " << (n - visited.size()) << "/" << n << std::endl;
  std::cout << "  radius from source = " << layers.size() << std::endl;
  std::cout << "  BFS level sizes = " << parlay::to_chars(sizes) << std::endl;
  }
  
template<typename indexType>
struct stats{

  stats() {}
  
  stats(size_t n){
    visited = parlay::sequence<indexType>(n, 0);
    distances = parlay::sequence<indexType>(n, 0);
  }

  parlay::sequence<indexType> visited;
  parlay::sequence<indexType> distances;

  void increment_dist(indexType i, indexType j){
    distances[i]+=j;}
  void increment_visited(indexType i, indexType j){
    visited[i]+=j;}

  parlay::sequence<indexType> visited_stats(){return statistics(this->visited);}
  parlay::sequence<indexType> dist_stats(){return statistics(this->distances);}

  void clear(){
    size_t n = visited.size();
    visited = parlay::sequence<indexType>(n, 0);
    distances = parlay::sequence<indexType>(n, 0);
  }

  static parlay::sequence<indexType> statistics(parlay::sequence<indexType> s){
    auto sl = parlay::map(s, [] (long x) { return x;});
    indexType avg = (indexType) (parlay::reduce(sl) / s.size());
    indexType tail = parlay::sort(s)[.99 * ((float)s.size())];
    auto result = {avg, tail};
    return result;
  }

};



// template <typename T>
// auto query_stats(parlay::sequence<Tvec_point<T> *> &q) {
//   parlay::sequence<size_t> vs = visited_stats(q);
//   parlay::sequence<size_t> ds = distance_stats(q);
//   auto result = {ds, vs};
//   return parlay::flatten(result);
// }

// template <typename T>
// auto range_query_stats(parlay::sequence<Tvec_point<T> *> &q) {
//   auto pred = [&](Tvec_point<T> *p) { return (p->ngh.size() == 0); };
//   auto pred1 = [&](Tvec_point<T> *p) { return !pred(p); };
//   auto zero_queries = parlay::filter(q, pred);
//   auto nonzero_queries = parlay::filter(q, pred1);
//   parlay::sequence<int> vn = visited_stats(nonzero_queries);
//   parlay::sequence<int> dn = distance_stats(nonzero_queries);
//   parlay::sequence<int> rn = rounds_stats(nonzero_queries);
//   parlay::sequence<int> vz = visited_stats(zero_queries);
//   parlay::sequence<int> dz = distance_stats(zero_queries);
//   parlay::sequence<int> rz = rounds_stats(zero_queries);
//   auto result = {rn, dn, vn, rz, dz, vz};
//   return parlay::flatten(result);
// }

// template <typename T>
// parlay::sequence<size_t> visited_stats(parlay::sequence<Tvec_point<T> *> &q) {
//   auto visited_stats =
//       parlay::tabulate(q.size(), [&](size_t i) { return q[i]->visited; });
//   parlay::sort_inplace(visited_stats);
//   size_t avg_visited = (int)parlay::reduce(visited_stats) / ((double)q.size());
//   size_t tail_index = .99 * ((float)q.size());
//   size_t tail_visited = visited_stats[tail_index];
//   auto result = {avg_visited, tail_visited};
//   return result;
// }

// template <typename T>
// parlay::sequence<size_t> distance_stats(parlay::sequence<Tvec_point<T> *> &q) {
//   auto dist_stats =
//       parlay::tabulate(q.size(), [&](size_t i) { return q[i]->dist_calls; });
//   parlay::sort_inplace(dist_stats);
//   size_t avg_dist = (size_t)parlay::reduce(dist_stats) / ((double)q.size());
//   size_t tail_index = .99 * ((float)q.size());
//   size_t tail_dist = dist_stats[tail_index];
//   auto result = {avg_dist, tail_dist};
//   return result;
// }

// template <typename T>
// parlay::sequence<size_t> rounds_stats(parlay::sequence<Tvec_point<T> *> &q) {
//   auto exp_stats =
//       parlay::tabulate(q.size(), [&](size_t i) { return q[i]->rounds; });
//   parlay::sort_inplace(exp_stats);
//   size_t avg_exps = (size_t)parlay::reduce(exp_stats) / ((double)q.size());
//   size_t tail_index = .99 * ((float)q.size());
//   size_t tail_exps = exp_stats[tail_index];
//   auto result = {avg_exps, tail_exps, exp_stats[exp_stats.size() - 1]};
//   return result;
// }

// void range_gt_stats(parlay::sequence<ivec_point> groundTruth) {
//   auto sizes = parlay::tabulate(groundTruth.size(), [&](size_t i) {
//     return groundTruth[i].coordinates.size();
//   });
//   parlay::sort_inplace(sizes);
//   size_t first_nonzero_index = 0;
//   for (size_t i = 0; i < sizes.size(); i++) {
//     if (sizes[i] != 0) {
//       first_nonzero_index = i;
//       break;
//     }
//   }
//   auto nonzero_sizes = (sizes).cut(first_nonzero_index, sizes.size());
//   auto sizes_sum = parlay::reduce(nonzero_sizes);
//   float avg =
//       static_cast<float>(sizes_sum) / static_cast<float>(nonzero_sizes.size());
//   std::cout << "Among nonzero entries, the average number of matches is " << avg
//             << std::endl;
//   std::cout << "25th percentile: " << nonzero_sizes[.25 * nonzero_sizes.size()]
//             << std::endl;
//   std::cout << "75th percentile: " << nonzero_sizes[.75 * nonzero_sizes.size()]
//             << std::endl;
//   std::cout << "99th percentile: " << nonzero_sizes[.99 * nonzero_sizes.size()]
//             << std::endl;
//   std::cout << "Max: " << nonzero_sizes[nonzero_sizes.size() - 1] << std::endl;
// }

// template <typename T>
// int connected_components(parlay::sequence<Tvec_point<T> *> &v) {
//   parlay::sequence<bool> visited(v.size(), false);
//   int cc = 0;
//   for (int i = 0; i < v.size(); i++) {
//     if (!visited[i]) {
//       BFS(i, v, visited);
//       cc++;
//     }
//   }
//   return cc;
// }

// template <typename T>
// void BFS(int start, parlay::sequence<Tvec_point<T> *> &v,
//          parlay::sequence<bool> &visited) {
//   std::queue<int> frontier;
//   frontier.push(start);
//   while (frontier.size() != 0) {
//     int c = frontier.front();
//     frontier.pop();
//     visited[c] = true;
//     for (int l = 0; l < size_of(v[c]->out_nbh); l++) {
//       int j = v[c]->out_nbh[l];
//       if (!visited[j]) frontier.push(j);
//     }
//   }
// }

} // end namespace
