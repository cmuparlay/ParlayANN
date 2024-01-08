#pragma once

#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"
#include "parlay/sequence.h"
#include "parlay/monoid.h"
#include "parlay/internal/get_time.h"
#include "parlay/internal/binary_search.h"

#include "../utils/filters.h"
#include "../utils/graph.h"
#include "../utils/point_range.h"
#include "../utils/types.h"
#include "../utils/euclidian_point.h"
#include "../utils/mips_point.h"
#include "../utils/filteredBeamSearch.h"
#include "../utils/threadlocal.h"

#include "../vamana/index.h"

#include <algorithm>
#include <type_traits>
#include <unordered_map>
#include <limits>
#include <variant>
#include <vector>

#include "pybind11/numpy.h"
#include "pybind11/stl.h"

#include "prefiltering.h"

using index_type = int32_t;

namespace py = pybind11;
using NeighborsAndDistances =
   std::pair<py::array_t<unsigned int>, py::array_t<float>>;

template <typename T, typename Point, class PR = PointRange<T, Point>, typename FilterType = float_t>
struct PostfilterVamanaIndex {
    using pid = std::pair<index_type, float>;

    std::unique_ptr<PR> pr;
    Graph<index_type> G;
    BuildParams BP;

    parlay::sequence<FilterType> filter_values;

    std::pair<FilterType, FilterType> range;

    PostfilterVamanaIndex(std::unique_ptr<PR>&& points, parlay::sequence<FilterType> filter_values, BuildParams BP = BuildParams(64, 500, 1.175), std::string& cache_path = "index_cache/")
        : pr(std::move(points)), filter_values(filter_values), BP(BP) {
        
    if (cache_path != "" &&
        std::filesystem::exists(this->graph_filename(cache_path))) {
      // std::cout << "Loading graph" << std::endl;

      std::string filename = this->graph_filename(cache_path);
      this->index_graph = Graph<index_type>(filename.data());
    } else {
      // std::cout << "Building graph" << std::endl;
      // this->start_point = indices[0];
      knn_index<Point, PR, index_type> I(BP);
      stats<index_type> BuildStats(this->points.size());

      // std::cout << "This filter has " << indices.size() << " points" <<
      // std::endl;

      this->index_graph = Graph<index_type>(BP.R, points.size());
      I.build_index(this->index_graph, points, BuildStats);

      if (cache_path != "") {
        this->save_graph(cache_path);
      }
    }

    std::cout << "Graph built, saved to " << graph_filename(cache_path) << std::endl;
    }

    PostfilterVamanaIndex(py::array_t<T> points, py::array_t<FilterType> filter_values) {
        py::buffer_info points_buf = points.request();
        if (points_buf.ndim != 2) {
            throw std::runtime_error("points numpy array must be 2-dimensional");
        }
        auto n = points_buf.shape[0]; // number of points
        auto dims = points_buf.shape[1]; // dimension of each point

        // avoiding this copy may have dire consequences from gc
        T* numpy_data = static_cast<T*>(points_buf.ptr);

        auto tmp_points = std::make_unique<PR>(numpy_data, n, dims);

        py::buffer_info filter_values_buf = filter_values.request();
        if (filter_values_buf.ndim != 1) {
            throw std::runtime_error("filter data numpy array must be 1-dimensional");
        }

        if (filter_values_buf.shape[0] != n) {
            throw std::runtime_error("filter data numpy array must have the same number of elements as the points array");
        }

        FilterType* filter_values_data = static_cast<FilterType*>(filter_values_buf.ptr);

        auto tmp_filter_values = parlay::sequence<FilterType>(filter_values_data, filter_values_data + n);


    }

    std::string graph_filename(std::string& cache_path) {
        return cache_path + "vamana_" + std::to_string(BP.L) + "_" + std::to_string(BP.R) + "_" + std::to_string(BP.alpha) + "_" + std::to_string(range.first) + "_" + std::to_string(range.second) + ".bin";
    }

    void save_graph(std::string filename_prefix) {
        std::string filename = this->graph_filename(filename_prefix);
        this->index_graph.save(filename.data());
    }

    
    parlay::sequence<pid> query(Point& q, std::pair<FilterType, FilterType> filter, QueryParams& QP = QueryParams(20, 100, 1.35, 10'000'000, 128)) {
        auto [pairElts, dist_cmps] = beam_search<Point, PR, index_type>(q, this->index_graph, this->pr, 0, QP);
        // auto [frontier, visited] = pairElts;
        auto frontier = pairElts.first;

        if constexpr (std::is_same<PR, PointRange<T, Point>>::value) {
            frontier = parlay::filter(frontier, [&](pid& p) {
            FilterType filter_value = filter_values[p.first];
            return filter_value >= filter.first && filter_value <= filter.second;
          });
        } else {
          // we actually want to filter and map to original coordinates at the same time
          frontier = parlay::map_maybe(frontier, [&](pid& p) {
            FilterType filter_value = filter_values[p.first];
            if (filter_value >= filter.first && filter_value <= filter.second) {
                return std::make_pair(pr->subset[p.first], p.second);
            } else {
                return std::optional<pid>();
            }
        });
        }

        return frontier;
    }

    NeighborsAndDistances batch_query(py::array_t<T, py::array::c_style | py::array::forcecast>& queries,
    const std::vector<std::pair<FilterType, FilterType>>& filters,
    uint64_t num_queries,
    uint64_t knn) {
        py::array_t<unsigned int> ids({num_queries, knn});
        py::array_t<float> dists({num_queries, knn});

        parlay::parallel_for(0, num_queries, [&](size_t i) {
            Point q = Point(queries.data(i));
            auto frontier = this->query(q, filters[i]);

            for (auto j = 0; j < knn; j++) {
                if (j < frontier.size()) {
                    ids.mutable_at(i, j) = frontier[j].first;
                    dists.mutable_at(i, j) = frontier[j].second;
                } else {
                    ids.mutable_at(i, j) = -1;
                    dists.mutable_at(i, j) = std::numeric_limits<float>::max();
                }
            }
        });

        return std::make_pair(ids, dists);
    }

    
};