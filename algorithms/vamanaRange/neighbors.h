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
#include "../utils/mips_point.h"
#include "../utils/euclidian_point.h"
#include "../utils/stats.h"
#include "../utils/types.h"
#include "../utils/graph.h"
#include "index.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"


template<typename Point, typename PointRange_, typename indexType>
void ANN(Graph<indexType> &G, long k, BuildParams &BP,
         PointRange_ &Query_Points,
         groundTruth<indexType> GT, char *res_file,
         bool graph_built, PointRange_ &Points) {

  indexType start_point;
  double idx_time;
  stats<unsigned int> BuildStats(G.size());
  parlay::internal::timer t("ANN");
  if(graph_built){
    idx_time = 0;
  } else {
    if (sizeof(typename PointRange_::T) >= 4) {
      if (Points[0].is_metric()) {
        using QT = uint8_t;
        using QPoint = Euclidian_Point<QT>;
        using QPR = PointRange<QT, QPoint>;
        QPR pr(Points);
        using findex = knn_index<QPoint, QPR, indexType>;
        findex I(BP);
        I.build_index(G, pr, BuildStats, false);
        start_point = I.get_start();
      } else {
        using QT = uint8_t;
        using QPoint = Quantized_Mips_Point<QT>;
        using QPR = PointRange<QT, QPoint>;
        QPR pr(Points);
        using findex = knn_index<QPoint, QPR, indexType>;
        findex I(BP);
        I.build_index(G, pr, BuildStats, false);
        start_point = I.get_start();
      }
    } else {
      using findex = knn_index<Point, PointRange_, indexType>;
      findex I(BP);
      I.build_index(G, Points, BuildStats, false);
      start_point = I.get_start();
    }
    idx_time = t.next_time();
  }

  std::string name = "Vamana";
  std::string params =
      "R = " + std::to_string(BP.R) + ", L = " + std::to_string(BP.L);
  auto [avg_deg, max_deg] = graph_stats_(G);
  auto vv = BuildStats.visited_stats();

  long build_num_distances = parlay::reduce(parlay::map(BuildStats.distances, [] (auto x) {return (long) x;}));

  Graph_ G_(name, params, G.size(), avg_deg, max_deg, idx_time);
  G_.print();

  parlay::internal::timer t_range("range search time");
  double radius = BP.radius;
  double radius_2 = BP.radius_2;
  std::cout << "radius = " << radius << " radius_2 = " << radius_2 << std::endl;
  QueryParams QP;
  QP.limit = (long) G.size();
  QP.degree_limit = (long) G.max_degree();
  QP.cut = 1.535;
  QP.k = 0;
  QP.beamSize = 45;
  long n = Points.size();
  parlay::sequence<long> counts(n);
  parlay::sequence<long> distance_comps(n);
  parlay::parallel_for(0, G.size(), [&] (long i) {
    parlay::sequence<indexType> pts;
    pts.push_back(Points[i].id()); //Points[i].id());
    auto [r, dc] = range_search(Points[i], G, Points, pts, radius, radius_2, QP, true);
    counts[i] = r.size();
    distance_comps[i] = dc;});
  t_range.total();
  long range_num_distances = parlay::reduce(distance_comps);

  std::cout << "edges within range: " << parlay::reduce(counts) << std::endl;
  std::cout << "distance comparisons during build = " << build_num_distances << std::endl;
  std::cout << "distance comparisons during range = " << range_num_distances << std::endl;

  // brute force
  if (false) {
    parlay::sequence<parlay::sequence<indexType>> in_radius(G.size());
    parlay::parallel_for(0, G.size(), [&] (long i) {
      if (i % 10000 == 0) std::cout << "." << std::flush;
      parlay::sequence<indexType> pts;
      long cnt = 0;
      for (long j=0; j < i; j++) 
        if (Points[i].distance(Points[j]) <= radius) {
          in_radius[i].push_back(j);
          //in_radius[j].push_back(i);
        }
                                      }, 1);
    parlay::parallel_for (0, G.size(), [&] (long i) {
                                         //std::sort(in_radius[i].begin(), in_radius[i].end());
                                         counts[i] = in_radius[i].size();
                                       });
    
    std::cout << "gt count: " << parlay::reduce(counts) * 2 << std::endl;
  }
}

