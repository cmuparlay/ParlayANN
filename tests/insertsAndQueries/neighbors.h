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
// #include "../utils/aspen_graph.h"
#include "../../algorithms/vamana/index.h"
#include "../utils/aspen_flat_graph.h"
#include "../utils/aspen_graph.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"


template<typename Point, typename PointRange, typename indexType, typename GraphType>
void ANN(GraphType &Graph, long k, BuildParams &BP,
         PointRange &Query_Points,
         groundTruth<indexType> GT, char *res_file,
         bool graph_built, PointRange &Points) {
  std::cout << "Size of dataset: " << Points.size() << std::endl;
  using findex = knn_index<Point, PointRange, indexType, GraphType>;
  findex I(BP);
  size_t n = Points.size();
  size_t update_batch_size = 50000;
  size_t query_batch_size = 10000;

  float update_frac = .8;
  float query_frac = 1 - update_frac;

  size_t num_updates = n * update_frac;
  size_t num_queries = n * query_frac;

  size_t num_update_batches = num_updates / update_batch_size;
  size_t num_query_batches = num_queries / query_batch_size;

  size_t query_start = n * update_frac;
  // std::cout << query_start << std::endl;

  stats<unsigned int> BuildStats(Points.size());

  parlay::sequence<indexType> indices = parlay::tabulate(update_batch_size, [&](indexType j) {
    return static_cast<indexType>(j);
  });
  I.insert(Graph, Points, BuildStats, indices);

  auto updater = [&]() {
    // timer update_t;
    for (size_t i = 1; i < num_update_batches; i++) {
      parlay::sequence<indexType> indices = parlay::tabulate(update_batch_size, [&](indexType j) {
        return static_cast<indexType>(i * update_batch_size + j);
      });
      std::cout << "Inserting indices " << indices[0] << " through "
                << indices[indices.size() - 1] << std::endl;
      I.insert(Graph, Points, BuildStats, indices);
      std::cout << "Finished inserting" << std::endl;
    }
  };

  parlay::sequence<indexType> start_points = {I.get_start()};
  auto queries = [&]() {
    // timer query_t;
    for (int i = 0; i < (int)num_query_batches; i++) {
      std::cout << "Querying elements " << query_start + (i * query_batch_size)
                << " through " << query_start + ((i + 1) * query_batch_size)
                << std::endl;
      using GraphI = typename GraphType::Graph;
      GraphI G = Graph.Get_Graph_Read_Only();
      QueryParams QP((long) 0, BP.L,  (double) 0.0, (long) Points.size(), (long) G.max_degree());
      parlay::parallel_for(0, query_batch_size, [&] (size_t j){
        beam_search(Points[query_start+i* query_batch_size + j], G, Points, start_points, QP);
      });
      Graph.Release_Graph(std::move(G));
      std::cout << "Finished query batch" << std::endl;
    }
  };

  size_t p = parlay::num_workers();


  parlay::par_do([&] {
    parlay::execute_with_scheduler(p/10, queries);}, 
    [&] {parlay::execute_with_scheduler((9*p)/10, updater);
  }); 

}