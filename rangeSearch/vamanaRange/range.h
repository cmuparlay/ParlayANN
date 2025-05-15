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
#include "../utils/check_range_recall.h"
#include "../utils/beamSearch.h"
#include "../utils/check_nn_recall.h"
#include "../utils/parse_results.h"
#include "../utils/stats.h"
#include "../utils/types.h"
#include "../utils/graph.h"
#include "../utils/mips_point.h"
#include "../utils/euclidian_point.h"
#include "../../algorithms/vamana/index.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"

namespace parlayANN{

template<typename Point, typename PointRange_,  typename indexType>
void RNG(Graph<indexType> &G, double rad, double esr, BuildParams &BP,
         PointRange_ &Query_Points,
         RangeGroundTruth<indexType> GT,
         char* res_file, bool graph_built, PointRange_ &Points, bool is_early_stop, bool is_double_beam, bool is_beam_search) {
  parlay::internal::timer t("ANN");
  using findex = knn_index<PointRange_, PointRange_, indexType>;
  findex I(BP);
  double idx_time;
  indexType start_point;

  stats<unsigned int> BuildStats(G.size());
  if(graph_built){
    idx_time = 0;
    start_point = 1;
  } else{
    I.build_index(G, Points, Points, BuildStats);
    start_point = 1; //  I.get_start();
    idx_time = t.next_time();
  }

  
  std::string name = "Vamana";
  std::string params =
      "R = " + std::to_string(BP.R) + ", L = " + std::to_string(BP.L);
  auto [avg_deg, max_deg] = graph_stats_(G);
  auto vv = BuildStats.visited_stats();
  std::cout << "Average visited: " << vv[0] << ", Tail visited: " << vv[1]
            << std::endl;
  Graph_ G_(name, params, G.size(), avg_deg, max_deg, idx_time);
  G_.print();
    if(Query_Points.size() != 0) {
    if (BP.quantize != 0) {
      std::cout << "quantizing build and first pass of search to 1 byte" << std::endl;
      if (Point::is_metric()) {
        using QT = uint8_t;
        using QPoint = Euclidian_Point<QT>;
        using QPR = PointRange<QPoint>;
        QPR Q_Points(Points);  // quantized to one byte
        QPR Q_Query_Points(Query_Points, Q_Points.params);
        range_search_wrapper<Point>(G,
                                    Points, Query_Points,
                                    Q_Points, Q_Query_Points,
                                    GT, rad, start_point, is_early_stop,
                                    is_double_beam, is_beam_search, esr);
      } else {
        using QPoint = Quantized_Mips_Point<8,true,255>;
        using QPR = PointRange<QPoint>;
        QPR Q_Points(Points);
        QPR Q_Query_Points(Query_Points, Q_Points.params);
        range_search_wrapper<Point>(G,
                                    Points, Query_Points,
                                    Q_Points, Q_Query_Points,
                                    GT, rad, start_point, is_early_stop,
                                    is_double_beam, is_beam_search, esr);
      }
    } else {
      range_search_wrapper<Point>(G,
                                  Points, Query_Points,
                                  Points, Query_Points,
                                  GT, rad, start_point, is_early_stop,
                                  is_double_beam, is_beam_search, esr);
    }
  }
}
}
