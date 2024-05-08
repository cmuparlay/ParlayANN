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

#include "algorithms/utils/NSGDist.h"
#include "algorithms/utils/beamSearch.h"
#include "algorithms/utils/check_nn_recall.h"
#include "algorithms/utils/parse_results.h"
#include "algorithms/utils/stats.h"
#include "algorithms/utils/types.h"
#include "algorithms/utils/graph.h"
#include "index.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"


template<typename Point, typename PointRange, typename indexType>
void ANN(Graph<indexType> &G, long k, BuildParams &BP,
         PointRange &Query_Points,
         groundTruth<indexType> GT, char *res_file,
         bool graph_built, PointRange &Points) {
  parlay::internal::timer t("ANN");
  using findex = knn_index<Point, PointRange, indexType>;
  findex I(BP);
  double idx_time;
  stats<unsigned int> BuildStats(G.size());
  if(graph_built){
    idx_time = 0;
  } else{
    I.build_index(G, Points, BuildStats);
    idx_time = t.next_time();
  }

  

  // I.set_start();
  // parlay::sequence<indexType> inserts = parlay::tabulate(Points.size()/2, [&] (size_t i){
	// 				    return static_cast<indexType>(i);});
  // I.batch_insert(inserts, G, Points, BuildStats, BP.alpha, true, 2, .02);

  // std::cout << "built on " << inserts.size() << " points" << std::endl; 


  // size_t index = inserts[inserts.size()-1];
  // size_t st = inserts[inserts.size()-1];
  // parlay::sequence<int> changed(G.size(), 0);
  // size_t num_batches = 50;
  // size_t bs = 1000;
  // size_t count = 0;

  // while(count < num_batches){
  //   parlay::sequence<indexType> next_inserts = parlay::tabulate(bs, [&] (size_t i){
	// 				    return static_cast<indexType>(index+i);});
  //   I.batch_insert_with_stats_count(next_inserts, G, Points, BP.alpha, changed);
  //   count++;
  //   index += bs;
  //   size_t inserted_so_far = index - st;
  //   std::cout << "Elements changed after " << inserted_so_far << " inserts: " << parlay::reduce(changed) << std::endl;
    
  // }
  

  indexType start_point = I.get_start();
  std::string name = "Vamana";
  std::string params =
      "R = " + std::to_string(BP.R) + ", L = " + std::to_string(BP.L);
  auto [avg_deg, max_deg] = graph_stats_(G);
  auto vv = BuildStats.visited_stats();
  std::cout << "Average visited: " << vv[0] << ", Tail visited: " << vv[1]
            << std::endl;
  Graph_ G_(name, params, G.size(), avg_deg, max_deg, idx_time);
  G_.print();
  if(Query_Points.size() != 0) search_and_parse<Point, PointRange, indexType>(G_, G, Points, Query_Points, GT, res_file, k, false, start_point);
}



