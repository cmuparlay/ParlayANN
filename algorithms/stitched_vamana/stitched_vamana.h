/* An implementation of the StitchedVamana index */

#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"
#include "parlay/sequence.h"
#include "parlay/monoid.h"
#include "parlay/internal/get_time.h"

#include "../utils/filters.h"
#include "../utils/graph.h"
#include "../utils/point_range.h"
#include "../utils/types.h"
#include "../utils/euclidian_point.h"
#include "../utils/mips_point.h"
#include "../utils/filteredBeamSearch.h"

#include "../vamana/index.h"

#include <algorithm>
#include <type_traits>
#include <unordered_map>
#include <limits>
#include <variant>
#include <vector>

#include "pybind11/numpy.h"
#include "pybind11/stl.h"

using index_type = int32_t;

namespace py = pybind11;
using NeighborsAndDistances =
   std::pair<py::array_t<unsigned int>, py::array_t<float>>;


// I fear we may want to do something more than just binary search for checking membership efficiently, so I'm attempting to get ahead of it here
// struct FilterMembership
// {
//     virtual bool is_member(int32_t index, int32_t filter) = 0;
// };

// struct CSR_BinMembership : public FilterMembership
// {
//     const csr_filters &filters;

//     CSR_BinMembership(const csr_filters &filters) : filters(filters)
//     {
//         if (filters.transposed)
//         {
//             std::cout << "Warning: CSR_BinMembership is not designed to work with transposed filters" << std::endl;
//         }
//     }

//     bool is_member(int32_t index, int32_t filter)
//     {
//         return filters.bin_match(index, filter);
//     }
// };

/* Implementation of StitchedVamana */
template <typename T, class Point>
struct StitchedVamana
{
    using pid = std::pair<index_type, float>;
    using PR = PointRange<T, Point>;
    using GraphI = Graph<index_type>;

    BuildParams build_params_small;
    BuildParams build_params_large;

    StitchedVamana(BuildParams build_params_small, BuildParams build_params_large) : build_params_small(build_params_small), build_params_large(build_params_large) {}

    StitchedVamana() : StitchedVamana(BuildParams(), BuildParams()) {}

    /* We expect the filter object to be transposed */
    parlay::sequence<index_type> build(PointRange<T, Point> &points, Graph<index_type> &G, csr_filters &filters) {
        auto untransposed_filters = filters.transpose();

        // pick starting points
        parlay::sequence<index_type> starting_points = parlay::tabulate(filters.n_points, [&](size_t i) {
            return filters.row_indices[filters.row_offsets[i]];
            });

        auto starting_points_rd = parlay::remove_duplicates(starting_points);
        std::cout << "Original size: " << starting_points.size() << ", after removing duplicates: " << starting_points_rd.size() << std::endl;

        // build the graphs for the filters
        auto filter_graphs = parlay::sequence<Graph<index_type>>::uninitialized(filters.n_points);
        auto subset_pr = parlay::sequence<SubsetPointRange<T, Point>>(filters.n_points) = parlay::tabulate(filters.n_points, [&](size_t i) {
            return SubsetPointRange<T, Point>(points, parlay::sequence<index_type>(filters.row_indices.get() + filters.row_offsets[i], filters.row_indices.get() + filters.row_offsets[i+1]));
        });

        // for (size_t i = 0; i < filters.n_points; i++) {
        parlay::parallel_for(0, filters.n_points, [&](size_t i) {
            filter_graphs[i] = Graph<index_type>(build_params_small.R, filters.row_offsets[i+1] - filters.row_offsets[i]);
            // subset_pr[i] = SubsetPointRange<T, Point>(points, parlay::sequence<index_type>(filters.row_indices.get() + filters.row_offsets[i], filters.row_indices.get() + filters.row_offsets[i+1]));

            std::cout << "Building graph for filter " << i << std::endl;

            knn_index<Point, SubsetPointRange<T, Point>, index_type> I(build_params_small);
            stats<index_type> BuildStats(points.size());

            I.build_index(filter_graphs[i], subset_pr[i], BuildStats);
        });

        std::cout << "Finished building graphs for filters" << std::endl;
        // }

        // parlay::parallel_for(0, points.size(), [&](size_t i) {
        //     parlay::sequence<index_type> edge_candidates;
        //     auto i_filters = untransposed_filters.point_filters(i);
        //     for (auto j : i_filters) {
                // auto neighbor_range = filter_graphs[j][subset_pr[j].subset_index(i)];

        //         std::cout << "Filter " << j << " has " << neighbor_range.size() << " neighbors" << std::endl;

        //         for (size_t k = 0; k < neighbor_range.size(); k++) {
        //             // pushing the real index of the k-th neighbor
        //             edge_candidates.push_back(subset_pr[j].real_index(neighbor_range[k]));
        //         }
        //     }
        //     edge_candidates = parlay::remove_duplicates(edge_candidates);
        //     auto edge_candidates_with_dist = parlay::tabulate(edge_candidates.size(), [&] (size_t j){
        //         float dist = points[edge_candidates[j]].distance(points[i]);        
        //         return std::make_pair(edge_candidates[j], dist);
        //     });
        //     auto edges = filteredRobustPrune(i, edge_candidates_with_dist, G, points, filters);
        //     G[i].update_neighbors(edges);
        // }, 100000000); // serial to debug

        // the above is broken, and it occurs to me that it's probably better to be operating over one graph at a time for cache reasons anyway
        // if we really care, this should use a threadlocal thing that gets merged at the end so we can iterate over the graphs in parallel
        parlay::sequence<parlay::sequence<index_type>> edge_candidates(points.size());

        for (size_t i = 0; i < untransposed_filters.n_filters; i++) {
            auto filter_graph = filter_graphs[i];
            auto filter_pr = subset_pr[i];

            // for each point in the filter, we add its neighbors to the edge candidates for that real index
            for (size_t j = 0; j < filter_pr.n; j++) {
                auto real_index = filter_pr.real_index(j);
                auto neighbor_range = filter_graph[j];

                for (size_t k = 0; k < neighbor_range.size(); k++) {
                    // pushing the real index of the k-th neighbor
                    edge_candidates[real_index].push_back(filter_pr.real_index(neighbor_range[k]));
                }
            }
        }

        std::cout << "Finished building edge candidates" << std::endl;

        // remove duplicates from the edge candidates, then run filteredRobustPrune
        parlay::parallel_for(0, points.size(), [&](size_t i) {
            edge_candidates[i] = parlay::remove_duplicates(edge_candidates[i]);
            auto edge_candidates_with_dist = parlay::tabulate(edge_candidates[i].size(), [&] (size_t j){
                float dist = points[edge_candidates[i][j]].distance(points[i]);        
                return std::make_pair(edge_candidates[i][j], dist);
            });
            auto edges = filteredRobustPrune(i, edge_candidates_with_dist, G, points, untransposed_filters);
            G[i].update_neighbors(edges);
        });

        return starting_points;
    }

    /* The FilteredRobustPrune from FilteredDiskANN.

        we have an added param filter_points, which is a slice of the points in the filter we're considering because we don't want to do the cases specific to the other filters that haven't been added yet

        operating over real indices, filters is NOT transposed
     */
    parlay::sequence<index_type> filteredRobustPrune(index_type p, parlay::sequence<pid>& cand,
                    GraphI &G, PR &Points, csr_filters &filters) {

    // if (filters.transposed) {
    //     std::cout << "Warning: filteredRobustPrune is not designed to work with transposed filters" << std::endl;
    //     abort();
    // }
    // add out neighbors of p to the candidate set.
    size_t out_size = G[p].size();
    std::vector<pid> candidates;
    for (auto x : cand) candidates.push_back(x);
 
    for(size_t i=0; i<out_size; i++) {
    // candidates.push_back(std::make_pair(v[p]->out_nbh[i], Points[v[p]->out_nbh[i]].distance(Points[p])));
    candidates.push_back(std::make_pair(G[p][i], Points[G[p][i]].distance(Points[p])));
    }
    

    // Sort the candidate set in reverse order according to distance from p.
    auto less = [&](pid a, pid b) { return a.second < b.second; };
    std::sort(candidates.begin(), candidates.end(), less);

    std::vector<index_type> new_nbhs;
    new_nbhs.reserve(build_params_large.R);

    size_t candidate_idx = 0;

    while (new_nbhs.size() < build_params_large.R && candidate_idx < candidates.size()) {
      // Don't need to do modifications.
      int p_star = candidates[candidate_idx].first;
      candidate_idx++;
      if (p_star == p || p_star == -1) {
        continue;
      }

      new_nbhs.push_back(p_star);

      for (size_t i = candidate_idx; i < candidates.size(); i++) {
        int p_prime = candidates[i].first;
        if (p_prime != -1) {
          float dist_starprime = Points[p_star].distance(Points[p_prime]);
          float dist_pprime = candidates[i].second;
          if (build_params_large.alpha * dist_starprime <= dist_pprime && !has_relevant_filter(p, p_star, p_prime, filters)) {
            candidates[i].first = -1;
          }
        }
      }
    }

    auto new_neighbors_seq = parlay::to_sequence(new_nbhs);
    return new_neighbors_seq;
  }

    bool has_relevant_filter(index_type p, index_type p_star, index_type p_prime, const csr_filters &filters) {
        auto p_filters = filters.point_filters(p);

        for (index_type i : p_filters) {
            if (filters.bin_match(p_star, i) && !filters.bin_match(p_prime, i)) {
                return true;
            }
        }
        return false;
    }

};

/* An index which uses the StitchedVamana index builder to create a graph and support queries */
template <typename T, class Point>
struct StitchedVamanaIndex {
    PointRange<T, Point> points; 
    csr_filters filters;
    parlay::sequence<index_type> starting_points;

    BuildParams build_params_small;
    BuildParams build_params_large;

    QueryParams query_params;

    Graph<index_type> G; // the graph to build and search

    StitchedVamanaIndex() {};

    StitchedVamanaIndex(BuildParams build_params_small, BuildParams build_params_large) : build_params_small(build_params_small), build_params_large(build_params_large) {}

    void fit(PointRange<T, Point> points, csr_filters& filters) {
        auto timer = parlay::internal::timer();
        timer.start();

        this->points = points;
        this->filters = filters;

        filters.transpose_inplace();

        this->G = Graph<index_type>(build_params_large.R, points.size());

        StitchedVamana<T, Point> builder(build_params_small, build_params_large);
        starting_points = builder.build(points, G, filters);

        std::cout << "Finished building StitchedVamana graph in " << timer.stop() << " seconds" << std::endl;
    }

    void fit_from_filename(std::string points_filename, std::string filters_filename) {
        PointRange<T, Point> points(points_filename.c_str());
        csr_filters filters(filters_filename.c_str());

        fit(points, filters);
    }

    NeighborsAndDistances batch_filter_search(
     py::array_t<T, py::array::c_style | py::array::forcecast>& queries,
     const std::vector<QueryFilter>& filters, uint64_t num_queries,
     uint64_t knn) {
        py::array_t<unsigned int> ids({num_queries, knn});
        py::array_t<float> dists({num_queries, knn});

        // TODO: we should probably also sort these by filter to be fair

        parlay::parallel_for(0, num_queries, [&](size_t i) {
            Point q = Point(queries.data(i), this->points.dimension(),
                      this->points.aligned_dimension(), i);
            parlay::sequence<index_type> query_filters = filters[i].get_sequence();

            auto [pairElts, dist_cmps] = filtered_beam_search(q, query_filters, G, points, starting_points, QueryParams(knn), this->filters);
        });
    }

    std::string get_index_name() {
        return "StitchedVamana_" + std::to_string(build_params_small.R) + "_" + std::to_string(build_params_small.L) + "_" + std::to_string(build_params_small.alpha) + "_" + std::to_string(build_params_large.R) + "_" + std::to_string(build_params_large.L) + "_" + std::to_string(build_params_large.alpha);
    }

    void save_graph(std::string prefix){
        std::string filename = prefix + get_index_name() + ".graph";
        G.save(filename.data());
    }

    void save_starting_points(std::string prefix) {
        std::string filename = prefix + get_index_name() + ".starting_points";
        // open the outfile
        std::ofstream outfile(filename.data(), std::ios::out | std::ios::binary);
        if (!outfile.is_open()) {
            std::cout << "Error opening file " << filename << std::endl;
            return;
        }

        // write the number of starting points
        size_t n_starting_points = starting_points.size();
        outfile.write((char*) &n_starting_points, sizeof(size_t));

        // write the starting points
        outfile.write((char*) starting_points.data(), sizeof(index_type) * n_starting_points);

        outfile.close();
    }

    void save(std::string prefix) {
        save_graph(prefix);
        save_starting_points(prefix);
    }

    void set_query_params(QueryParams query_params) {
        this->query_params = query_params;
    }

    void set_build_params_small(BuildParams build_params_small) {
        this->build_params_small = build_params_small;
    }

    void set_build_params_small(unsigned int R, unsigned int L, double alpha) {
        this->build_params_small = BuildParams(R, L, alpha);
    }

    void set_build_params_large(BuildParams build_params_large) {
        this->build_params_large = build_params_large;
    }

    void set_build_params_large(unsigned int R, unsigned int L, double alpha) {
        this->build_params_large = BuildParams(R, L, alpha);
    }


};