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
#include "../utils/threadlocal.h"
#include "../utils/stats.h"

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
template <typename T, class Point, class PR = PointRange<T, Point>>
struct StitchedVamana
{
    using pid = std::pair<index_type, float>;
    // using PR = PointRange<T, Point>;
    using GraphI = Graph<index_type>;

    BuildParams build_params_small;
    BuildParams build_params_large;

    StitchedVamana(BuildParams build_params_small, BuildParams build_params_large) : build_params_small(build_params_small), build_params_large(build_params_large) {}

    StitchedVamana() : StitchedVamana(BuildParams(), BuildParams()) {}

    /* We expect the filter object to be transposed */
    parlay::sequence<index_type> build(PR &points, Graph<index_type> &G, csr_filters &filters) {
        auto untransposed_filters = filters.transpose();

        // std::cout << "input (transposed) filters: " << std::endl;
        // filters.print_stats();

        // std::cout << "untransposed filters: " << std::endl;
        // untransposed_filters.print_stats();

        // pick starting points
        parlay::sequence<index_type> starting_points = parlay::tabulate(filters.n_points, [&](size_t i) {
            return filters.row_indices[filters.row_offsets[i] + parlay::hash64_2(i) % std::max(filters.row_offsets[i+1] - filters.row_offsets[i], static_cast<int64_t>(1))];
            });

        auto starting_points_rd = parlay::remove_duplicates(starting_points);
        std::cout << "Original size: " << starting_points.size() << ", after removing duplicates: " << starting_points_rd.size() << std::endl;

        // build the graphs for the filters
        auto filter_graphs = parlay::sequence<Graph<index_type>>::uninitialized(filters.n_points);
        auto subset_pr = parlay::sequence<SubsetPointRange<T, Point, PR>>(filters.n_points) = parlay::tabulate(filters.n_points, [&](size_t i) {
            return SubsetPointRange<T, Point, PR>(points, parlay::sequence<index_type>(filters.row_indices.get() + filters.row_offsets[i], filters.row_indices.get() + filters.row_offsets[i+1]));
        });

        for (size_t i = 0; i < filters.n_points; i++) {
        // parlay::parallel_for(0, filters.n_points, [&](size_t i) {
            filter_graphs[i] = Graph<index_type>(build_params_small.R, filters.row_offsets[i+1] - filters.row_offsets[i]);
            // subset_pr[i] = SubsetPointRange<T, Point>(points, parlay::sequence<index_type>(filters.row_indices.get() + filters.row_offsets[i], filters.row_indices.get() + filters.row_offsets[i+1]));

            std::cout << "Building graph for filter " << i << std::endl;

            knn_index<Point, SubsetPointRange<T, Point, PR>, index_type> I(build_params_small);
            stats<index_type> BuildStats(points.size());

            I.build_index(filter_graphs[i], subset_pr[i], BuildStats);

            auto [avgDegree, maxDegree] = graph_stats_(filter_graphs[i]);

            std::cout << "Finished building graph for filter " << i << " with avg degree " << avgDegree << " and max degree " << maxDegree << "(" << subset_pr[i].size() << " points)" <<  std::endl;
        // });
        }

        std::cout << "Finished building graphs for filters" << std::endl;
        // }

        parlay::internal::timer t;
        t.start();

        parlay::parallel_for(0, points.size(), [&](size_t i) {
        // for (size_t i = 0; i < points.size(); i++) {
            parlay::sequence<index_type> edge_candidates;
            parlay::sequence<int32_t> i_filters = untransposed_filters.point_filters(i);

            // std::cout << "Point " << i << " has " << i_filters.size() << " filters" << std::endl;
            if (i_filters.size() != 0) {
                for (auto j : i_filters) {
                    // std::cout << "just before 131" << std::endl;
                    auto neighbor_range = filter_graphs[j][subset_pr[j].subset_index(i)];

                    if (subset_pr[j].real_index(neighbor_range[0]) == i) {
                        printf("Warning: point %d is a neighbor of itself in filter %d\n", i, j);
                    }

                    // std::cout << "Filter " << j << " has " << neighbor_range.size() << " neighbors" << std::endl;

                    for (size_t k = 0; k < neighbor_range.size(); k++) {
                        // pushing the real index of the k-th neighbor
                        edge_candidates.push_back(subset_pr[j].real_index(neighbor_range[k]));
                    }
                }

                // std::cout << "Finished building edge candidates for point " << i << " with " << edge_candidates.size() << " candidates" << std::endl;

                edge_candidates = parlay::remove_duplicates(edge_candidates);

                // std::cout << "After removing duplicates, there are " << edge_candidates.size() << " candidates" << std::endl;

                if (edge_candidates.size() == 0) {
                    std::cout << "!!! Warning: point " << i << " has no edge candidates !!!" << std::endl;
                }

                auto edge_candidates_with_dist = parlay::tabulate(edge_candidates.size(), [&] (size_t j){
                    float dist = points[edge_candidates[j]].distance(points[i]);
                    return std::make_pair(edge_candidates[j], dist);
                });

                // std::cout << "Beginning prune for point " << i << std::endl;

                auto edges = filteredRobustPrune(i, edge_candidates_with_dist, G, points, untransposed_filters);

                // std::cout << "Finished prune for point " << i << " which now has " << edges.size() << " neighbor candidates" << std::endl;

                G[i].update_neighbors(edges);

                // std::cout << "Finished updating neighbors for point " << i << std::endl;
            }
        }); // serial to debug
        // }

        auto [avgDegree, maxDegree] = graph_stats_(G);

        std::cout << "Finished building graph with avg degree " << avgDegree << " and max degree " << maxDegree << " in " << t.next_time() << "s" << std::endl;

        // combining the graphs

        // the above is broken, and it occurs to me that it's probably better to be operating over one graph at a time for cache reasons anyway
        // if we really care, this should use a threadlocal thing that gets merged at the end so we can iterate over the graphs in parallel
        /* parlay::sequence<parlay::sequence<index_type>> edge_candidates(points.size());

        for (size_t i = 0; i < untransposed_filters.n_filters; i++) {
            auto filter_graph = filter_graphs[i];
            auto& filter_pr = subset_pr[i];

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
        }); */

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
            if (filters.std_match(p_star, i) && !filters.std_match(p_prime, i)) {
                return true;
            }
        }
        return false;
    }

};

/* An index which uses the StitchedVamana index builder to create a graph and support queries */
template <typename T, class Point, class PR = PointRange<T, Point>>
struct StitchedVamanaIndex {
    PR points; 
    csr_filters filters; // should be label-major
    csr_filters untransposed_filters; // should be point-major
    parlay::sequence<index_type> starting_points;

    BuildParams build_params_small;
    BuildParams build_params_large;

    QueryParams query_params;

    Graph<index_type> G; // the graph to build and search

    threadlocal::accumulator<size_t> dcmps; // the number of distance comparisons made during queries

    StitchedVamanaIndex() {};

    StitchedVamanaIndex(BuildParams build_params_small, BuildParams build_params_large) : build_params_small(build_params_small), build_params_large(build_params_large) {}
    
    /* filters here should be point-major */
    void fit(PR points, csr_filters filters) {
        auto timer = parlay::internal::timer();
        timer.start();

        this->points = points;
        this->untransposed_filters = filters;

        this->filters = this->untransposed_filters.transpose();

        this->validate_state();

        this->G = Graph<index_type>(build_params_large.R, points.size());

        StitchedVamana<T, Point, PR> builder(build_params_small, build_params_large);
        starting_points = builder.build(points, G, this->filters);

        std::cout << "Finished building StitchedVamana graph in " << timer.stop() << " seconds" << std::endl;
    }

    /* This will ofc not work if PR is not PointRange */
    void fit_from_filename(std::string points_filename, std::string filters_filename) {
        PointRange<T, Point> points(points_filename.c_str());
        csr_filters filters(filters_filename.c_str());



        fit(points, filters);
    }

    /* We assume here that filters is filters-major, not necessarily literally transposed but effectively. */
    void load(std::string prefix, PR points, csr_filters& filters) {
        this->points = points;
        this->filters = filters;
        this->untransposed_filters = this->filters.transpose();

        this->validate_state();

        // load the starting points
        std::string starting_points_filename = prefix + get_index_name() + ".starting_points";
        std::ifstream infile(starting_points_filename.data(), std::ios::in | std::ios::binary);
        std::cout << "Loading starting points from " << starting_points_filename << std::endl;
        if (!infile.is_open()) {
            std::cout << "Error opening file " << starting_points_filename << std::endl;
            return;
        }

        this->starting_points = parlay::sequence<index_type>(this->filters.n_points);

        // read the number of starting points, assert that it's the same as the number of filters
        size_t n_starting_points;
        infile.read((char*) &n_starting_points, sizeof(size_t));

        assert(n_starting_points == this->filters.n_points);

        // read the starting points
        infile.read((char*) this->starting_points.data(), sizeof(index_type) * n_starting_points);

        infile.close();

        // load the graph
        std::string graph_filename = prefix + get_index_name() + ".graph";
        this->G = Graph<index_type>(graph_filename.data());

        std::cout << "Finished loading StitchedVamana index" << std::endl;
    }

    void load_from_filename(std::string prefix, std::string points_filename, std::string filters_filename) {
        static_assert(std::is_same<PR, PointRange<T, Point>>::value, "Cannot load from filename if PR is not PointRange");

        PointRange<T, Point> points(points_filename.c_str());
        csr_filters filters(filters_filename.c_str());

        load(prefix, points, filters);
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

            parlay::sequence<index_type> start_from(query_filters.size());
            for (size_t j = 0; j < query_filters.size(); j++) {
                start_from[j] = starting_points[query_filters[j]];
            }

            auto [pairElts, dist_cmps] = filtered_beam_search(q, query_filters, G, points, start_from, this->query_params, this->untransposed_filters);

            auto frontier = pairElts.first;

            for (size_t j = 0; j < knn; j++) {
                ids.mutable_at(i, j) = frontier[j].first;
                dists.mutable_at(i, j) = frontier[j].second;
            }

            dcmps.add(dist_cmps);
        });

        return std::make_pair(std::move(ids), std::move(dists));
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

    size_t get_dist_comparisons() {
        return dcmps.total();
    }

    void validate_state() {
        if (this->untransposed_filters.n_points < this->untransposed_filters.n_filters) {
            throw std::invalid_argument("StitchedVamanaIndex: untransposed_filters should be label-major");
        }
        if (this->filters.n_points > this->filters.n_filters) {
            throw std::invalid_argument("StitchedVamanaIndex: filters should be point-major");
        }

        if (this->filters.n_points != this->untransposed_filters.n_filters) {
            throw std::invalid_argument("StitchedVamanaIndex: filters and untransposed_filters should have the same number of filters");
        }
        if (this->filters.n_filters != this->untransposed_filters.n_points) {
            throw std::invalid_argument("StitchedVamanaIndex: filters and untransposed_filters should have the same number of points");
        }

        if (this->filters.n_filters != this->points.size()) {
            throw std::invalid_argument("StitchedVamanaIndex: filters and points should have the same number of points");
        }
        if (this->untransposed_filters.n_points != this->points.size()) {
            throw std::invalid_argument("StitchedVamanaIndex: untransposed_filters and points should have the same number of points");
        }
    }

    void print_starting_points() {
        std::cout << "Starting points: ";
        for (size_t i = 0; i < starting_points.size(); i++) {
            std::cout << i << ":" << starting_points[i] << " ";
        }
        std::cout << std::endl;
    }

};

/* A stitched vamana index that only incorporates filters above a given size cutoff into the graph, doing exhaustive search for smaller filters
 
Internally, this is a wrapper around a StitchedVamanaIndex, which is used for the larger filters and uses the raw transposed csr_matrix for the smaller filters
*/
template <typename T, class Point>
struct HybridStitchedVamanaIndex {
    StitchedVamanaIndex<T, Point> stitched_vamana_index;
    PointRange<T, Point> points;
    csr_filters filters; // should be label-major

    // the cutoff for when to use the stitched vamana index
    size_t cutoff;
    
    // the below two map between real label indices and the subset label indices of the stitched vamana index
    std::unordered_map<index_type, index_type> real_to_subset_index; 
    parlay::sequence<index_type> subset_to_real_index; // not sure this is actually needed outside of fit

    
    HybridStitchedVamanaIndex() {};

    HybridStitchedVamanaIndex(size_t cutoff) : cutoff(cutoff) {};

    void fit(PointRange<T, Point> points, csr_filters filters) {
        this->points = points;
        this->filters = filters; // filters isn't a reference because we should own it

        if (filters.n_points < filters.n_filters) {
            throw std::invalid_argument("HybridStitchedVamanaIndex: filters arg should be row-major");
        }

        this->filters.transpose_inplace();

        // determine which filters are above the cutoff
        for (size_t i = 0; i < this->filters.n_points; i++) {
            if (this->filters.row_offsets[i+1] - this->filters.row_offsets[i] > cutoff) {
                subset_to_real_index.push_back(i);
                real_to_subset_index[i] = subset_to_real_index.size() - 1;
            }
        }

        std::cout << "Found " << subset_to_real_index.size() << " filters above the cutoff" << std::endl;

        // build the subset point range and filters
        // SubsetPointRange<T, Point> subset_points(points, subset_to_real_index);
        csr_filters subset_filters = this->filters.subset_rows(subset_to_real_index);

        subset_filters.transpose_inplace();

        // build the stitched vamana index
        stitched_vamana_index.fit(this->points, subset_filters);
    }

    void fit_from_filename(std::string points_filename, std::string filters_filename) {
        PointRange<T, Point> points(points_filename.c_str());
        csr_filters filters(filters_filename.c_str());

        fit(points, filters);
    }

    void save(std::string prefix) {
        stitched_vamana_index.save(prefix + "Hybrid_cutoff" + std::to_string(cutoff) + "_");
    }

    void load_from_filename(std::string prefix, std::string points_filename, std::string filters_filename) {
        this->points = PointRange<T, Point>(points_filename.c_str()); // original points
        this->filters = csr_filters(filters_filename.c_str()); // original filters 

        this->filters.transpose_inplace(); // transpose to conveniently get the subset filters

        // determine which filters are above the cutoff
        for (size_t i = 0; i < filters.n_points; i++) {
            if (filters.row_offsets[i+1] - filters.row_offsets[i] > cutoff) {
                subset_to_real_index.push_back(i);
                real_to_subset_index[i] = subset_to_real_index.size() - 1;
            }
        }

        // build the subset point range and filters
        // SubsetPointRange<T, Point> subset_points(points, subset_to_real_index);
        csr_filters subset_filters = filters.subset_rows(subset_to_real_index);

        stitched_vamana_index.load(prefix + "Hybrid_cutoff" + std::to_string(cutoff) + "_", this->points, subset_filters);

        auto [avgDegree, maxDegree] = graph_stats_(stitched_vamana_index.G);
        std::cout << "Finished loading StitchedVamana index with avg degree " << avgDegree << " and max degree " << maxDegree << std::endl;
    }

    // TODO: querying
    NeighborsAndDistances batch_filter_search(
     py::array_t<T, py::array::c_style | py::array::forcecast>& queries,
     const std::vector<QueryFilter>& filters, uint64_t num_queries,
     uint64_t knn) {
        py::array_t<unsigned int> ids({num_queries, knn});
        py::array_t<float> dists({num_queries, knn});

        // validating filters
        if (this->filters.n_points > this->filters.n_filters) {
            throw std::invalid_argument("HybridStitchedVamanaIndex: filters arg should be label-major");
        }

        parlay::parallel_for(0, num_queries, [&](size_t i) {
        // for (size_t i = 0; i < num_queries; i++) {
            Point q = Point(queries.data(i), this->points.dimension(),
                      this->points.aligned_dimension(), i);
            parlay::sequence<index_type> query_filters = filters[i].get_sequence();

            // determine which filters are above the cutoff
            parlay::sequence<index_type> graph_query_filters;
            parlay::sequence<index_type> exhaustive_query_filters;
            for (size_t j = 0; j < query_filters.size(); j++) {
                if (this->filters.row_offsets[query_filters[j]+1] - this->filters.row_offsets[query_filters[j]] > cutoff) {
                    graph_query_filters.push_back(real_to_subset_index[query_filters[j]]);
                } else {
                    exhaustive_query_filters.push_back(query_filters[j]);
                }
            }

            parlay::sequence<std::pair<index_type, float>> frontier;
            // if there are any filters above the cutoff, search graph
            if (graph_query_filters.size() > 0) {
                auto starting_points = parlay::tabulate(graph_query_filters.size(), [&](size_t j) {
                    return this->stitched_vamana_index.starting_points[graph_query_filters[j]];
                });
                auto [pairElts, dist_cmps] = filtered_beam_search(q, graph_query_filters, this->stitched_vamana_index.G, this->points, starting_points, this->stitched_vamana_index.query_params, this->stitched_vamana_index.untransposed_filters);

                frontier = pairElts.first;
            } else { // otherwise, initialize the frontier
                frontier = parlay::tabulate(knn, [&](size_t i) {
                    return std::make_pair(-1, std::numeric_limits<float>::infinity());
                });
            }

            // if there are any filters below the cutoff, do exhaustive search and update the frontier
            if (exhaustive_query_filters.size() > 0) {
                // this could be improved by taking the union of matches of each label and checking the output of that union
                // not really relevant to the current binary formulation though
                for (auto filter : exhaustive_query_filters) {
                    auto filter_points = this->filters.point_filters(filter); // should be all the points associated with said filter, seems to work

                    for (size_t k = 0; k < filter_points.size(); k++) {
                        auto filter_point = filter_points[k];
                        auto dist = q.distance(this->points[filter_point]);

                        if (dist >= frontier[knn-1].second) {
                            continue;
                        }

                        // update the frontier
                        // // this approach to updating the frontier may be better or worse than all that sorting
                        // for (size_t l = knn - 2; l >= 0; l--) { // iterate backwards so we can break early, ignore the last element
                        // // we know that the last element of the frontier will get booted and the current element will be inserted
                        // // the only question is where the current element will end up
                        //     if (dist >= frontier[l].second) { // s.t. the previous element is where we insert
                        //         // move all the subsequent elements back to make room, not including the last element
                        //         std::copy(frontier.begin() + l + 1, frontier.begin() + knn - 1, frontier.begin() + l + 2);
                        //         // insert the current element
                        //         frontier[l + 1] = std::make_pair(filter_point, dist); 
                        //         break;
                        //     }
                        // }

                        // pretty sure the above approach is terminally stupid, and at best overengineered.
                        // we will do all the normal sorting stuff
                        frontier[knn-1] = std::make_pair(filter_point, dist);
                        std::sort(frontier.begin(), frontier.end(), [](std::pair<index_type, float> a, std::pair<index_type, float> b) {
                            return a.second < b.second;
                        });
                    }
                }
            }

            // write the frontier to the output arrays
            for (size_t j = 0; j < knn; j++) {
                ids.mutable_at(i, j) = frontier[j].first;
                dists.mutable_at(i, j) = frontier[j].second;
            }

        });
        // }

        return std::make_pair(std::move(ids), std::move(dists));
     }

    std::string get_index_name() {
        return "Hybrid_cutoff" + std::to_string(cutoff) + "_" + stitched_vamana_index.get_index_name();
    }

    void set_query_params(QueryParams query_params) {
        this->stitched_vamana_index.set_query_params(query_params);
    }

    void set_build_params_small(BuildParams build_params_small) {
        this->stitched_vamana_index.set_build_params_small(build_params_small);
    }

    void set_build_params_small(unsigned int R, unsigned int L, double alpha) {
        this->stitched_vamana_index.set_build_params_small(R, L, alpha);
    }

    void set_build_params_large(BuildParams build_params_large) {
        this->stitched_vamana_index.set_build_params_large(build_params_large);
    }

    void set_build_params_large(unsigned int R, unsigned int L, double alpha) {
        this->stitched_vamana_index.set_build_params_large(R, L, alpha);
    }

    void set_cutoff(size_t cutoff) {
        this->cutoff = cutoff;
    }

};