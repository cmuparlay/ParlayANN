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

template<typename Point, typename PointRange, typename QPointRange, typename indexType>
void ANN_(Graph<indexType> &G, long k, BuildParams &BP,
          PointRange &Query_Points, QPointRange &Q_Query_Points,
          groundTruth<indexType> GT, char *res_file,
          bool graph_built, PointRange &Points, QPointRange &Q_Points) {
  parlay::internal::timer t("ANN");

  bool verbose = BP.verbose;
  using findex = knn_index<QPointRange, indexType>;
  findex I(BP);
  indexType start_point;
  double idx_time;
  stats<unsigned int> BuildStats(G.size());
  if(graph_built){
    idx_time = 0;
    start_point = 0;
  } else{
    I.build_index(G, Q_Points, BuildStats);
    start_point = I.get_start();
    idx_time = t.next_time();
  }
  std::cout << "start index = " << start_point << std::endl;

  std::string name = "Vamana";
  std::string params =
    "R = " + std::to_string(BP.R) + ", L = " + std::to_string(BP.L);
  auto [avg_deg, max_deg] = graph_stats_(G);
  auto vv = BuildStats.visited_stats();
  std::cout << "Average visited: " << vv[0] << ", Tail visited: " << vv[1]
            << std::endl;
  Graph_ G_(name, params, G.size(), avg_deg, max_deg, idx_time);
  G_.print();

  long build_num_distances = parlay::reduce(parlay::map(BuildStats.distances,
                                                        [] (auto x) {return (long) x;}));

  if(Query_Points.size() != 0) {
    search_and_parse<Point, PointRange, QPointRange, indexType>(G_, G, Points, Query_Points,
                                                                Q_Points, Q_Query_Points, GT,
                                                                res_file, k, false, start_point,
                                                                verbose);
  } else if (BP.self) {
    if (BP.range) {
      parlay::internal::timer t_range("range search time");
      double radius = BP.radius;
      double radius_2 = BP.radius_2;
      std::cout << "radius = " << radius << " radius_2 = " << radius_2 << std::endl;
      QueryParams QP;
      long n = Points.size();
      parlay::sequence<long> counts(n);
      parlay::sequence<long> distance_comps(n);
      parlay::parallel_for(0, G.size(), [&] (long i) {
        parlay::sequence<indexType> pts;
        pts.push_back(Points[i].id());
        auto [r, dc] = range_search(Points[i], G, Points, pts, radius, radius_2, QP, true);
        counts[i] = r.size();
        distance_comps[i] = dc;});
      t_range.total();
      long range_num_distances = parlay::reduce(distance_comps);

      std::cout << "edges within range: " << parlay::reduce(counts) << std::endl;
      std::cout << "distance comparisons during build = " << build_num_distances << std::endl;
      std::cout << "distance comparisons during range = " << range_num_distances << std::endl;
    }
  }
}

template<typename Point, typename PointRange_, typename indexType>
void ANN(Graph<indexType> &G, long k, BuildParams &BP,
         PointRange_ &Query_Points,
         groundTruth<indexType> GT, char *res_file,
         bool graph_built, PointRange_ &Points) {
  if (BP.quantize && sizeof(typename PointRange_::T) >= 2) {
    std::cout << "quantizing build and first pass of search to 1 byte" << std::endl;
    if (Point::is_metric()) {
      using QT = uint8_t;
      using QPoint = Euclidian_Point<QT>;
      using QPR = PointRange<QT, QPoint>;
      QPR Q_Points(Points);  // quantized to one byte
      QPR Q_Query_Points(Query_Points, Q_Points.params);
      ANN_<Point, PointRange_, QPR, indexType>(G, k, BP, Query_Points, Q_Query_Points, GT, res_file, graph_built, Points, Q_Points);
    } else {
      using QT = int8_t;
      using QPoint = Quantized_Mips_Point<QT>;
      using QPR = PointRange<QT, QPoint>;
      QPR Q_Points(Points);
      QPR Q_Query_Points(Query_Points, Q_Points.params);
      ANN_<Point, PointRange_, QPR, indexType>(G, k, BP, Query_Points, Q_Query_Points, GT, res_file, graph_built, Points, Q_Points);
    }
  } else {
    ANN_<Point, PointRange_, PointRange_, indexType>(G, k, BP, Query_Points, Query_Points, GT, res_file, graph_built, Points, Points);
  }
}
