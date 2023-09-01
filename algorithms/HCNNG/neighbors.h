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
#include <cmath>
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"
#include "../utils/NSGDist.h"  
#include "../utils/types.h"
#include "../utils/beamSearch.h"
#include "../utils/stats.h"
#include "../utils/parse_results.h"
#include "../utils/check_nn_recall.h"
#include "../utils/graph.h"
#include "hcnng_index.h"

template<typename T, template<typename C> class Point, template<typename E, template<typename D> class P> class PointRange>
void ANN(Graph<unsigned int> &G, int k, BuildParams &BP,
         PointRange<T, Point> &Query_Points,
         groundTruth<int> GT, char *res_file,
         bool graph_built, PointRange<T, Point> &Points) {

  parlay::internal::timer t("ANN"); 
  using findex = hcnng_index<T, Point, PointRange>;

  double idx_time;
  if(!graph_built){
    findex I;
    I.build_index(G, Points, BP.num_clusters, BP.cluster_size);
    idx_time = t.next_time();
  } else{idx_time=0;}
  std::string name = "HCNNG";
  std::string params = "Trees = " + std::to_string(BP.num_clusters);
  auto [avg_deg, max_deg] = graph_stats_(G);
  Graph_ G_(name, params, G.size(), avg_deg, max_deg, idx_time);
  G_.print();
  if(Query_Points.size() != 0) search_and_parse(G_, G, Points, Query_Points, GT, res_file);
}
