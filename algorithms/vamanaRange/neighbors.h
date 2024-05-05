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

#include <algorithm>

#include "../utils/NSGDist.h"
#include "../utils/beamSearch.h"
#include "../utils/check_nn_recall.h"
#include "../utils/parse_results.h"
#include "../utils/stats.h"
#include "../utils/types.h"
#include "../utils/graph.h"
#include "index.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"


template<typename Point, typename PointRange, typename indexType>
void ANN(Graph<indexType> &G, long k, BuildParams &BP,
         PointRange &Query_Points,
         groundTruth<indexType> GT, char *res_file,
         bool graph_built, PointRange &Points) {
  using findex = knn_index<Point, PointRange, indexType>;
  findex I(BP);
  double idx_time;
  stats<unsigned int> BuildStats(G.size());
  parlay::internal::timer t("ANN");
  if(graph_built){
    idx_time = 0;
  } else{
    I.build_index(G, Points, BuildStats, false);
    idx_time = t.next_time();
  }

  indexType start_point = I.get_start();
  std::string name = "Vamana";
  std::string params =
      "R = " + std::to_string(BP.R) + ", L = " + std::to_string(BP.L);
  auto [avg_deg, max_deg] = graph_stats_(G);
  auto vv = BuildStats.visited_stats();

  long build_num_distances = parlay::reduce(parlay::map(BuildStats.distances, [] (auto x) {return (long) x;}));

  Graph_ G_(name, params, G.size(), avg_deg, max_deg, idx_time);
  G_.print();

  parlay::internal::timer t_range("range search time");
  QueryParams QP;
  QP.limit = (long) G.size();
  QP.degree_limit = (long) G.max_degree();
  QP.cut = 1.35;
  QP.k = 10;
  QP.beamSize = BP.L;
  long n = Points.size();
  parlay::sequence<long> counts(n);
  parlay::sequence<long> distance_comps(n);
  parlay::parallel_for(0, G.size(), [&] (long i) {
    parlay::sequence<indexType> pts;
    pts.push_back(Points[i].id()); //Points[i].id());
    auto [r, dc] = range_search(Points[i], G, Points, pts, k, QP, true);
    counts[i] = r.size();
    distance_comps[i] = dc;});
  t_range.total();
  long range_num_distances = parlay::reduce(distance_comps);

  std::cout << "edges within range: " << parlay::reduce(counts) << std::endl;
  std::cout << "distance comparisons during build = " << build_num_distances << std::endl;
  std::cout << "distance comparisons during range = " << range_num_distances << std::endl;

  //std::cout << "Average visited: " << vv[0] << ", Tail visited: " << vv[1]
  // << std::endl;

  // brute force
  if (false) {
    parlay::parallel_for(0, G.size(), [&] (long i) {
      parlay::sequence<indexType> pts;
      long cnt = 0;
      for (long j=0; j <= i; j++) 
        if (Points[i].distance(Points[j]) < k) cnt++;
      counts[i] = cnt;});
    std::cout << "gt count: " << parlay::reduce(counts) << std::endl;
  }
}

