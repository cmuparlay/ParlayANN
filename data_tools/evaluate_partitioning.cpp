#include <iostream>
#include <algorithm>
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/io.h"

#include "vamana/index.h"

#include "utils/beamSearch.h"
#include "utils/euclidian_point.h"
#include "utils/mips_point.h"
#include "utils/point_range.h"

#include "kaminpar-shm/kaminpar.h"
#include "kaminpar_graphbuilder.h"

using AdjGraph = parlay::sequence<parlay::sequence<unsigned int>>;
using Partition = std::vector<int>;

template <typename PointRange>
parlay::sequence<parlay::sequence<unsigned int>> vamana_knngraph(PointRange& P, size_t k) {
  using indexType = uint32_t;
  using findex = knn_index<PointRange, indexType>;
  using Point = typename PointRange::Point;
  BuildParams BP;
  indexType start_point = 0;
  double idx_time;
  parlay::internal::timer t("ANN");

  size_t maxDeg = 2*k;
  BP.R = maxDeg;
  BP.L = 2*maxDeg;
  BP.alpha = 1.15;  // Update to pass in all of these parameters.
  BP.num_passes = 1;
  BP.single_batch = 0;

  stats<indexType> BuildStats(P.size());
  Graph<indexType> G(maxDeg, P.size());

  findex I(BP);
  I.build_index(G, P, BuildStats);
  start_point = I.get_start();
  idx_time = t.next_time();
  std::cout << "# Index time: " << idx_time << std::endl;

  stats<indexType> QueryStats(P.size());
  QueryStats.clear();
  QueryParams QP;
  QP.k = k;
  QP.beamSize = 2*k;
  QP.cut = 1.35;  // disable cut optimization for now.
  auto points = beamSearchRandom<Point>(P, G, P, QueryStats, QP);
  double bs_time = t.next_time();
  std::cout << "# Search time: " << bs_time << std::endl;
  std::cout << "points.size = " << points.size() << std::endl;

  size_t n = points.size();
  auto ret = parlay::tabulate(n, [&] (size_t i) {
    parlay::sequence<unsigned int> inner;
    for (size_t j=0; j < points[i].size(); ++j) {
      if (points[i][j] < n) inner.push_back(points[i][j]);
    }
    return inner;
  });

  return ret;
}

// Returns the adjacency list for every point.
// Maybe it should also return the number of duplicate points it found.
template <typename PointRange>
parlay::sequence<parlay::sequence<unsigned int>> brute_force_knngraph(PointRange& P, size_t k) {
  unsigned d = P.dimension();
  size_t n = P.size();
  size_t block_size = 10000;
  parlay::sequence<parlay::sequence<std::pair<int, float>>> points(n);
  parlay::sequence<std::pair<int, float>> point_data(n);

  parlay::parallel_for(0, n, [&] (size_t i) {
    point_data[i] = std::make_pair(0, P[0].d_min());
  });

  size_t n_blocks = (n + block_size - 1) / block_size;
  std::cout << "nblocks = " << n_blocks << std::endl;
  for (size_t b=0; b < n_blocks; ++b) {
    size_t start = b*block_size;
    size_t end = std::min((b+1)*block_size, n);

    parlay::parallel_for(0, n, [&] (size_t i) {
      auto& topk = points[i];

      int toppos = point_data[i].first;
      float topdist = point_data[i].second;

      for (size_t j=start; j<end; ++j) {
        if (i != j) {
          float dist = P[i].distance(P[j]);
          if(topk.size() < k) {
            if(dist > topdist){
              topdist = dist;
              toppos = topk.size();
            }
            topk.push_back(std::make_pair((int) j, dist));
          }
          else if(dist < topdist) {
            float new_topdist=P[0].d_min();
            int new_toppos=0;
            topk[toppos] = std::make_pair((int) j, dist);
            for(size_t l=0; l<topk.size(); l++) {
              if(topk[l].second > new_topdist){
                new_topdist = topk[l].second;
                new_toppos = (int) l;
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
  }

  // Collect some stats:
  auto dup_seq = parlay::delayed_tabulate(n, [&] (size_t i) {
    size_t dups = 0;
    for (size_t j=0; j<points[i].size(); ++j) {
      if (points[i][j].second == 0) {
        ++dups;
      }
    }
    return dups;
  });
  size_t n_dups = parlay::reduce(dup_seq);
  std::cout << "Num dups = " << n_dups << std::endl;

  auto edge_count_seq = parlay::delayed_tabulate(n, [&] (size_t i) {
    return points[i].size();
  });
  std::cout << "Num edges = " << parlay::reduce(edge_count_seq) << std::endl;

  parlay::sequence<parlay::sequence<unsigned int>> ret = parlay::tabulate(n, [&] (size_t i) {
    return parlay::tabulate(points[i].size(), [&] (size_t j) {
      return points[i][j].first;
    });
  });
  return ret;
}

parlay::sequence<parlay::sequence<unsigned int>> symmetrize(parlay::sequence<parlay::sequence<unsigned int>> edges) {
  std::cout << "Generating pairs" << std::endl;
  auto pairs = parlay::flatten(parlay::delayed_tabulate(edges.size(), [&] (size_t i) {
    return parlay::tabulate(2*edges[i].size(), [&] (size_t j) {
      unsigned int ngh = edges[i][j/2];
      if (j % 2 == 0) {
        return std::make_pair((unsigned int)i, ngh);
      } else {
        return std::make_pair(ngh, (unsigned int)i);
      }
    });
  }));
  std::cout << "Generated pairs" << std::endl;
  parlay::sort_inplace(pairs);

  // find start for each vertex
  size_t n = edges.size();
  auto starts = parlay::pack_index(parlay::delayed_tabulate(pairs.size(), [&] (size_t i) {
    return i == 0 || (std::get<0>(pairs[i]) != std::get<0>(pairs[i-1]));
  }));
  parlay::sequence<parlay::sequence<unsigned int>> new_edges(n);
  std::cout << "n = " << n << " num starts = " << starts.size() << std::endl;
  parlay::parallel_for(0, starts.size(), [&] (size_t i) {
    auto start = starts[i];
    auto end = (i == starts.size()-1) ? pairs.size() : starts[i+1];
    if (start > end) {
      std::cout << "bad!" << std::endl;
    }
    size_t our_idx = std::get<0>(pairs[start]);
    for (size_t j=start; j<end; ++j) {
      if (j == start || std::get<1>(pairs[j]) != std::get<1>(pairs[j-1])) {
        new_edges[our_idx].push_back(std::get<1>(pairs[j]));
      }
    }
  });
  std::cout << "Finished compacting" << std::endl;
  for (size_t i=0; i<new_edges.size(); ++i) {
    for (auto neighbor : new_edges[i]) {
      std::cout << i << " " << neighbor << std::endl;
    }
  }
  exit(0);
  return new_edges;
}

auto build_kaminpar_graph(parlay::sequence<parlay::sequence<unsigned int>>& edges) {
  auto del_seq = parlay::delayed_tabulate(edges.size(), [&] (size_t i) { 
      return edges[i].size(); 
  });
  size_t total_edges = parlay::reduce(del_seq);

  std::cout << "total_edges = " << total_edges << std::endl;
  std::cout << "Initializing GB. " << std::endl;
  kaminpar::shm::utils::GraphBuilder GB(edges.size(), total_edges);
  std::cout << "Starting to add edges" << std::endl;
  // Builds in CSR format.
  for (size_t i=0; i<edges.size(); ++i) {
    //std::cout << "Added node: " << i << std::endl;
    for (size_t j=0; j<edges[i].size(); ++j) {
      //std::cout << "adding edges: " << edges[i][j] << std::endl;
      GB.new_edge(edges[i][j], 1);
    }
    GB.new_node(1);
  }
  std::cout << "Finished adding edges... building!" << std::endl;

  return GB.build();
}

struct CSR {
    CSR() : xadj(1, 0) {}
    parlay::sequence<kaminpar::shm::EdgeID> xadj;
    parlay::sequence<kaminpar::shm::NodeID> adjncy;
    parlay::sequence<kaminpar::shm::NodeWeight> node_weights;
};

CSR ConvertAdjGraphToCSR(const AdjGraph& graph) {
    CSR csr;
    size_t num_edges = 0;
    for (const auto& n : graph) {
        num_edges += n.size();
    }
    csr.xadj.reserve(graph.size() + 1);
    csr.adjncy.reserve(num_edges);
    for (const auto& n : graph) {
        for (const int neighbor : n) {
            csr.adjncy.push_back(neighbor);
        }
        csr.xadj.push_back(csr.adjncy.size());
    }
    return csr;
}

CSR ParallelSymmetrizeAndConvertToCSR(const AdjGraph& adj_graph) {
    auto nested_edges = parlay::tabulate(adj_graph.size(), [&](size_t i) {
        parlay::sequence<std::pair<uint32_t, uint32_t>> zipped;
        for (const int v : adj_graph[i]) {
            zipped.push_back(std::make_pair(v, i));
        }
        return zipped;
    });

    auto flat = parlay::flatten(nested_edges);

    auto rev = parlay::group_by_index(flat, adj_graph.size());

    auto degree = parlay::tabulate(adj_graph.size(), [&](size_t i) -> kaminpar::shm::EdgeID { return adj_graph[i].size() + rev[i].size(); });

    auto [xadj, num_edges] = parlay::scan(degree);
    xadj.push_back(num_edges);

    auto adjncy = parlay::sequence<kaminpar::shm::NodeID>::uninitialized(num_edges);
    parlay::parallel_for(0, adj_graph.size(), [&](size_t i) {
        size_t j = xadj[i];
        for (const auto v : adj_graph[i]) {
            adjncy[j++] = v;
        }
        for (const auto v : rev[i]) {
            adjncy[j++] = v;
        }
    });

    CSR csr;
    csr.xadj = std::move(xadj);
    csr.adjncy = std::move(adjncy);
    return csr;
}

Partition PartitionGraphWithKaMinPar(CSR& graph, int k, double epsilon, int num_threads, bool strong, bool quiet) {
    size_t num_nodes = graph.xadj.size() - 1;
    std::vector<kaminpar::shm::BlockID> kaminpar_partition(num_nodes, -1);
    auto context = kaminpar::shm::create_default_context();
    if (strong) {
        context = kaminpar::shm::create_strong_context();
    }
    context.partition.epsilon = epsilon;
    kaminpar::KaMinPar shm(num_threads, context);
    if (quiet) {
        shm.set_output_level(kaminpar::OutputLevel::QUIET);
    }
    shm.borrow_and_mutate_graph(num_nodes, graph.xadj.data(), graph.adjncy.data(),
                   /* vwgt = */ graph.node_weights.empty() ? nullptr : graph.node_weights.data(),
                   /* adjwgt = */ nullptr);
    shm.compute_partition(k, kaminpar_partition.data());
//    double time = timer.Stop();
//    if (!quiet) {
//        std::cout << "Partitioning with KaMinPar took " << time << " seconds" << std::endl;
//    }
    Partition partition(num_nodes);
    for (size_t i = 0; i < partition.size(); ++i) {
        partition[i] = kaminpar_partition[i]; // convert unsigned int partition ID to signed int partition ID
    }
    return partition;
}

template <typename PointRange>
void run_eval(PointRange& P, size_t k) {
//  auto edges = brute_force_knngraph(P, k);
  auto edges = vamana_knngraph(P, k);

  //auto csr = ConvertAdjGraphToCSR(edges);
  auto csr = ParallelSymmetrizeAndConvertToCSR(edges);
  auto partitions = PartitionGraphWithKaMinPar(csr, 100, 0.05, 60, false, false);

//  size_t n = P.size();
//
//  std::cout << "starting stats" << std::endl;
//  // Collect some stats:
//  auto dup_seq = parlay::delayed_tabulate(n, [&] (size_t i) {
//    size_t dups = 0;
//    for (size_t j=0; j<edges[i].size(); ++j) {
//      if (P[i].distance(P[edges[i][j]]) == 0) {
//        ++dups;
//      }
//    }
//    return dups;
//  });
//  size_t n_dups = parlay::reduce(dup_seq);
//  std::cout << "Num dups = " << n_dups << std::endl;
//
//  auto sym_edges = symmetrize(std::move(edges));
//  auto G = build_kaminpar_graph(sym_edges);

//  // Call the shared-memory partitioner:
//  kaminpar::KaMinPar shm(72, kaminpar::shm::create_default_context());
//  shm.set_graph(std::move(G));
//
//  std::vector<kaminpar::shm::BlockID> partition{};
//  shm.compute_partition(100, partition.data());

  std::cout << "OK!" << std::endl;
}

int main(int argc, char* argv[]) {
  commandLine P(argc,argv,
  "[-base_path <b>] [-query_path <q>] "
      "[-data_type <d>] [-k <k> ] [-dist_func <d>] [-gt_path <outfile>]");

  char* gFile = P.getOptionValue("-gt_path");
  char* bFile = P.getOptionValue("-base_path");
  char* vectype = P.getOptionValue("-data_type");
  char* dfc = P.getOptionValue("-dist_func");
  int k = P.getOptionIntValue("-k", 10);

  std::string tp = std::string(vectype);
  if((tp != "uint8") && (tp != "int8") && (tp != "float")){
    std::cout << "Error: data type not specified correctly, specify int8, uint8, or float" << std::endl;
    abort();
  }

  std::string df = std::string(dfc);
  if(df != "Euclidian" && df != "mips"){
    std::cout << "Error: invalid distance type: specify Euclidian or mips" << std::endl;
    abort();
  }


  if(tp == "float") {
    std::cout << "Detected float coordinates" << std::endl;
    if(df == "Euclidian") {
      auto Points = PointRange<float, Euclidian_Point<float>>(bFile);
      run_eval<PointRange<float, Euclidian_Point<float>>>(Points, k);
    } else if(df == "mips") {
      auto Points = PointRange<float, Mips_Point<float>>(bFile);
      run_eval<PointRange<float, Mips_Point<float>>>(Points, k);
    }
  } else if (tp == "uint8") {
  } else if (tp == "int8") {
  }


  // Build k-NN graph

  // TODO: (optionally) write out the built k-NN graph.

  // Partition the k-NN graph and compute statistics.
  std::cout << "OK!" << std::endl;
}

