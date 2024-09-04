/*
 * This utility helps build (approximate) k-NN graphs from a pointset.
 * An optional proximity graph (e.g., vamana or HCNNG graph) can be passed in,
 * and the k-NN graph will be constructed by beam-searching this proximity
 * graph.
 *
 * The output k-NN graph is written out in the same binary format as
 * utils/graph.h.
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

// #include "kaminpar-shm/kaminpar.h"
// #include "kaminpar_graphbuilder.h"

using AdjGraph = parlay::sequence<parlay::sequence<unsigned int>>;

template <typename PointRange>
AdjGraph build_from_proximity_graph(PointRange& Points, Graph<uint32_t>& G,
                                    size_t k) {
  using indexType = uint32_t;
  using findex = knn_index<PointRange, indexType>;
  using Point = typename PointRange::Point;
  indexType start_point = 0;
  parlay::internal::timer t("ANN");

  stats<indexType> QueryStats(Points.size());
  QueryStats.clear();
  QueryParams QP;
  QP.k = k;
  QP.beamSize = std::max(4 * k, size_t{100});
  QP.cut = 1.35;
  QP.limit = G.size();

  std::cout << "Starting beam search random!" << std::endl;
  std::cout << "Distance: " << Points[0].distance(Points[1]) << std::endl;
  auto points = searchAll<Point, PointRange, indexType>(Points, G, Points,
                                                        QueryStats, 0, QP);
  double bs_time = t.next_time();
  std::cout << "# Search time: " << bs_time << std::endl;
  std::cout << "points.size = " << points.size() << std::endl;
  // for (size_t i=0; i < 10000; ++i) {
  //   std::cout << QueryStats.visited[i] << " " << QueryStats.distances[i] <<
  //   " edge_size = " << points[i].size() << std::endl;
  // }

  size_t n = points.size();
  AdjGraph ret = parlay::tabulate(n, [&](size_t i) {
    parlay::sequence<unsigned int> inner;
    for (size_t j = 0; j < points[i].size(); ++j) {
      if (points[i][j] < n) inner.push_back(points[i][j]);
    }
    return inner;
  });

  return ret;
}

// Returns the adjacency list for every point.
template <typename PointRange>
AdjGraph brute_force_knngraph(PointRange& P, size_t k) {
  unsigned d = P.dimension();
  size_t n = P.size();
  size_t block_size = 10000;
  parlay::sequence<parlay::sequence<std::pair<int, float>>> points(n);
  parlay::sequence<std::pair<int, float>> point_data(n);

  parlay::parallel_for(
      0, n, [&](size_t i) { point_data[i] = std::make_pair(0, P[0].d_min()); });

  parlay::internal::timer t;
  t.start();
  size_t n_blocks = (n + block_size - 1) / block_size;
  std::cout << "nblocks = " << n_blocks << std::endl;
  for (size_t b = 0; b < n_blocks; ++b) {
    size_t start = b * block_size;
    size_t end = std::min((b + 1) * block_size, n);

    parlay::parallel_for(0, n, [&](size_t i) {
      auto& topk = points[i];

      int toppos = point_data[i].first;
      float topdist = point_data[i].second;

      for (size_t j = start; j < end; ++j) {
        if (i != j) {
          float dist = P[i].distance(P[j]);
          if (topk.size() < k) {
            if (dist > topdist) {
              topdist = dist;
              toppos = topk.size();
            }
            topk.push_back(std::make_pair((int)j, dist));
          } else if (dist < topdist) {
            float new_topdist = P[0].d_min();
            int new_toppos = 0;
            topk[toppos] = std::make_pair((int)j, dist);
            for (size_t l = 0; l < topk.size(); l++) {
              if (topk[l].second > new_topdist) {
                new_topdist = topk[l].second;
                new_toppos = (int)l;
              }
            }
            topdist = new_topdist;
            toppos = new_toppos;
          }
        }
      }

      point_data[i].first = toppos;
      point_data[i].second = topdist;
    });
    std::cout << "Finished block b = " << b << std::endl;
    t.next("block time");
  }

  // Collect some stats:
  auto dup_seq = parlay::delayed_tabulate(n, [&](size_t i) {
    size_t dups = 0;
    for (size_t j = 0; j < points[i].size(); ++j) {
      if (points[i][j].second == 0) {
        ++dups;
      }
    }
    return dups;
  });
  size_t n_dups = parlay::reduce(dup_seq);
  std::cout << "Num dups = " << n_dups << std::endl;

  auto edge_count_seq =
      parlay::delayed_tabulate(n, [&](size_t i) { return points[i].size(); });
  std::cout << "Num edges = " << parlay::reduce(edge_count_seq) << std::endl;
  parlay::parallel_for(0, n, [&](size_t i) {
    auto comp = [&](auto l, auto r) { return l.second < r.second; };
    parlay::sort_inplace(points[i], comp);
  });

  AdjGraph ret = parlay::tabulate(n, [&](size_t i) {
    return parlay::tabulate(points[i].size(), [&](size_t j) -> unsigned int {
      return points[i][j].first;
    });
  });
  return ret;
}

// Saves the graph in the same binary format as utils/graph.h
void save_graph(AdjGraph& edges, unsigned int maxDeg, char* oFile) {
  using indexType = unsigned int;
  indexType n = edges.size();
  std::cout << "Writing graph with " << n << " points and max degree " << maxDeg
            << std::endl;
  parlay::sequence<indexType> preamble = {static_cast<indexType>(n),
                                          static_cast<indexType>(maxDeg)};
  parlay::sequence<indexType> sizes = parlay::tabulate(
      n, [&](size_t i) { return static_cast<indexType>(edges[i].size()); });
  std::ofstream writer;
  writer.open(oFile, std::ios::binary | std::ios::out);
  writer.write((char*)preamble.begin(), 2 * sizeof(indexType));
  writer.write((char*)sizes.begin(), sizes.size() * sizeof(indexType));
  size_t BLOCK_SIZE = 1000000;
  size_t index = 0;
  while (index < n) {
    size_t floor = index;
    size_t ceiling = index + BLOCK_SIZE <= n ? index + BLOCK_SIZE : n;
    auto edge_data = parlay::tabulate(ceiling - floor, [&](size_t i) {
      return parlay::tabulate(sizes[i + floor],
                              [&](size_t j) { return edges[i + floor][j]; });
    });
    parlay::sequence<indexType> data = parlay::flatten(edge_data);
    writer.write((char*)data.begin(), data.size() * sizeof(indexType));
    index = ceiling;
  }
  writer.close();
  std::cout << "Finished writing!" << std::endl;
}

template <typename PointRange>
void build_knn_graph(commandLine& P, PointRange& Points) {
  char* oFile = P.getOptionValue("-graph_outfile");
  int k = P.getOptionIntValue("-k", 1000);

  char* gFile = P.getOptionValue("-graph_file");
  AdjGraph knn_graph;
  if (gFile) {
    // Load graph and query it for the desired k.
    std::cout << "Building graph from an existing proximity graph..."
              << std::endl;
    Graph<uint32_t> G(gFile);
    knn_graph = build_from_proximity_graph(Points, G, k);
  } else {
    // Brute-force.
    std::cout << "Brute-forcing knn graph..." << std::endl;
    knn_graph = brute_force_knngraph(Points, k);
  }
  save_graph(knn_graph, k, oFile);

  //  //auto csr = ConvertAdjGraphToCSR(edges);
  //  auto csr = ParallelSymmetrizeAndConvertToCSR(edges);
  //  auto partitions = PartitionGraphWithKaMinPar(csr, 40, 0.05, 60, false,
  //  false);

  std::cout << "OK!" << std::endl;
}

int main(int argc, char* argv[]) {
  commandLine P(
      argc, argv,
      "[-base_path <b>] [-query_path <q>] "
      "[-data_type <d>] [-k <k> ] [-dist_func <d>] [-gt_path <outfile>]");

  char* bFile = P.getOptionValue("-base_path");
  char* vectype = P.getOptionValue("-data_type");
  char* dfc = P.getOptionValue("-dist_func");

  std::string tp = std::string(vectype);
  if ((tp != "uint8") && (tp != "int8") && (tp != "float")) {
    std::cout << "Error: data type not specified correctly, specify int8, "
                 "uint8, or float"
              << std::endl;
    abort();
  }

  std::string df = std::string(dfc);
  if (df != "Euclidian" && df != "mips") {
    std::cout << "Error: invalid distance type: specify Euclidian or mips"
              << std::endl;
    abort();
  }

  if (tp == "float") {
    std::cout << "Detected float coordinates" << std::endl;
    using T = float;
    if (df == "Euclidian") {
      auto Points = PointRange<T, Euclidian_Point<T>>(bFile);
      build_knn_graph<PointRange<T, Euclidian_Point<T>>>(P, Points);
    } else if (df == "mips") {
      auto Points = PointRange<T, Mips_Point<T>>(bFile);
      build_knn_graph<PointRange<T, Mips_Point<T>>>(P, Points);
    }
  } else if (tp == "uint8") {
    using T = uint8_t;
    if (df == "Euclidian") {
      auto Points = PointRange<T, Euclidian_Point<T>>(bFile);
      build_knn_graph<PointRange<T, Euclidian_Point<T>>>(P, Points);
    } else if (df == "mips") {
      auto Points = PointRange<T, Mips_Point<T>>(bFile);
      build_knn_graph<PointRange<T, Mips_Point<T>>>(P, Points);
    }
  } else if (tp == "int8") {
    using T = int8_t;
    if (df == "Euclidian") {
      auto Points = PointRange<T, Euclidian_Point<T>>(bFile);
      build_knn_graph<PointRange<T, Euclidian_Point<T>>>(P, Points);
    } else if (df == "mips") {
      auto Points = PointRange<T, Mips_Point<T>>(bFile);
      build_knn_graph<PointRange<T, Mips_Point<T>>>(P, Points);
    }
  }

  // Partition the k-NN graph and compute statistics.
  std::cout << "OK!" << std::endl;
}
