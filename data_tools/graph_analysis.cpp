/*
 * This utility helps analyze graphs built by
 * (1) proximity-graph construction algorithms like HCNNG, HNSW, and Vamana
 * (2) k-NN graphs built by brute-forcing or using an ANNS method
 *
 * The input graph is expected to be in the graph format used in utils/graph.h
 */

#include <algorithm>
#include <iostream>

#include "parlay/io.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "utils/beamSearch.h"
#include "utils/euclidian_point.h"
#include "utils/mips_point.h"
#include "utils/point_range.h"
#include "vamana/index.h"

using AdjGraph = parlay::sequence<parlay::sequence<unsigned int>>;

void OutputDegreeDistribution(commandLine& P) {
  char* gFile = P.getOptionValue("-graph_file");
  using indexType = uint32_t;
  Graph<indexType> G(gFile, true);
  parlay::sequence<indexType> degrees(G.size());
  for (size_t i=0; i<G.size(); ++i) {
    auto neighbors = G[i];
    for (size_t j=0; j < neighbors.size(); ++j) {
      degrees[neighbors[j]]++;
    }
  }
  auto h = parlay::histogram_by_key(degrees);
  parlay::sort_inplace(h);
  for (size_t i=0; i < h.size(); ++i) {
    std::cout << h[i].first << "," << h[i].second << std::endl;
  }
}

void DiffGraphs(commandLine& P) {
  char* gknn = P.getOptionValue("-knn_graph");
  char* gproxim = P.getOptionValue("-proximity_graph");
  size_t k = P.getOptionLongValue("-k", 10);
  std::cout << "In diff task" << std::endl;
  using indexType = uint32_t;
  std::cout << gknn << " " << gproxim << std::endl;
  std::cout << "Loading graph" << std::endl;
  Graph<indexType> GKnn(gknn, false);
  Graph<indexType> GProxim(gproxim, false);
  std::cout << "Loaded graph" << std::endl;

  parlay::sequence<uint32_t> contained(GKnn.size());
  parlay::parallel_for(0, GKnn.size(), [&] (size_t i) {
    auto neighbors_knn = GKnn[i];
    auto neighbors_proxim = GProxim[i];
    for (size_t j=0; j < neighbors_knn.size(); ++j) {
      auto neighbor = neighbors_knn[j];
      for (size_t k=0; k<neighbors_proxim.size(); ++k) {
        if (neighbor == neighbors_proxim[k]) {
          contained[i]++;
          break;
        }
      }
    }
  });

  auto h = parlay::histogram_by_key(contained);
  parlay::sort_inplace(h);
  for (size_t i=0; i < h.size(); ++i) {
    std::cout << h[i].first << "," << h[i].second << std::endl;
  }

}

int main(int argc, char* argv[]) {
  commandLine P(
      argc, argv,
      "[-base_path <b>] [-query_path <q>] "
      "[-data_type <d>] [-k <k> ] [-dist_func <d>] [-gt_path <outfile>]");

  char* tk = P.getOptionValue("-task");
  std::string task = std::string(tk);

  if (task == "diff") {
    DiffGraphs(P);
  } else if (task == "degrees") {
    OutputDegreeDistribution(P);
  } else {
    std::cerr << "Unknown task: " << task << (task == "degrees")  << std::endl;
    exit(0);
  } 
}
