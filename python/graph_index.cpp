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


#include "../algorithms/vamana/index.h"
#include "../algorithms/utils/types.h"
#include "../algorithms/utils/point_range.h"
#include "../algorithms/utils/graph.h"
#include "../algorithms/utils/euclidian_point.h"
#include "../algorithms/utils/mips_point.h"
#include "../algorithms/utils/stats.h"
#include "../algorithms/utils/beamSearch.h"
#include "../algorithms/HNSW/HNSW.hpp"
#include "pybind11/numpy.h"

#include "parlay/parallel.h"
#include "parlay/primitives.h"

#include <cstdio>
#include <utility>
#include <optional>

namespace py = pybind11;
using NeighborsAndDistances = std::pair<py::array_t<unsigned int>, py::array_t<float>>;

template<typename T, typename Point> 
struct GraphIndex{
    Graph<unsigned int> G;
    PointRange<T, Point> Points;
    std::optional<ANN::HNSW<Desc_HNSW<T, Point>>> HNSW_index;

    GraphIndex(std::string &data_path, std::string &index_path, size_t num_points, size_t dimensions, bool is_hnsw=false){
        Points = PointRange<T, Point>(data_path.data());
        assert(num_points == Points.size());
        assert(dimensions == Points.dimension());

        if(is_hnsw) {
            HNSW_index = ANN::HNSW<Desc_HNSW<T, Point>>(
                index_path,
                [&](unsigned int i/*indexType*/){
                    return Points[i];
                }
            );
        }
        else {
            G = Graph<unsigned int>(index_path.data());
        }
    }

    auto search_dispatch(const Point &q, QueryParams &QP)
    {
        if(HNSW_index) {
            using indexType = unsigned int; // be consistent with the type of G
            using std::pair;
            using seq_t = parlay::sequence<pair<indexType, typename Point::distanceType>>;

            indexType dist_cmps = 0;
            search_control ctrl{};
            if(QP.limit>0) {
                ctrl.limit_eval = QP.limit;
            }
            ctrl.count_cmps = &dist_cmps;

            seq_t frontier = HNSW_index->search(q, QP.k, QP.beamSize, ctrl);
            return pair(pair(std::move(frontier), seq_t{}), dist_cmps);
        }
        else return beam_search<Point, PointRange<T, Point>, unsigned int>(q, G, Points, 0, QP);
    }

    NeighborsAndDistances batch_search(py::array_t<T, py::array::c_style | py::array::forcecast> &queries, uint64_t num_queries, uint64_t knn,
                        uint64_t beam_width, int64_t visit_limit = -1){
        if(visit_limit == -1) visit_limit = HNSW_index? 0: G.size();
        QueryParams QP(knn, beam_width, 1.35, visit_limit, HNSW_index?0:G.max_degree());

        py::array_t<unsigned int> ids({num_queries, knn});
        py::array_t<float> dists({num_queries, knn});


        parlay::sequence<unsigned int> point_ids;
        parlay::sequence<float> point_distances;

        using parameters = typename Point::parameters;

        parlay::parallel_for(0, num_queries, [&] (size_t i){
          std::vector<T> v(Points.dimension());
          for (int j=0; j < v.size(); j++)
            v[j] = queries.data(i)[j];
          Point q = Point(v.data(), i, Points.params); 
            auto [pairElts, dist_cmps] = search_dispatch(q, QP);
            auto [frontier, visited] = pairElts;
            parlay::sequence<unsigned int> point_ids;
            parlay::sequence<float> point_distances;
            for(int j=0; j<knn; j++){
                ids.mutable_data(i)[j] = frontier[j].first;
                dists.mutable_data(i)[j] = frontier[j].second;
            }
        });
        return std::make_pair(std::move(ids), std::move(dists));
    }

    NeighborsAndDistances batch_search_from_string(std::string &queries, uint64_t num_queries, uint64_t knn,
                                    uint64_t beam_width){
        QueryParams QP(knn, beam_width, 1.35, HNSW_index?0:G.size(), HNSW_index?0:G.max_degree());
        PointRange<T, Point> QueryPoints = PointRange<T, Point>(queries.data());
        py::array_t<unsigned int> ids({num_queries, knn});
        py::array_t<float> dists({num_queries, knn});
        parlay::parallel_for(0, num_queries, [&] (size_t i){
            auto [pairElts, dist_cmps] = search_dispatch(QueryPoints[i], QP);
            auto [frontier, visited] = pairElts;
            parlay::sequence<unsigned int> point_ids;
            parlay::sequence<float> point_distances;
            for(int j=0; j<knn; j++){
                ids.mutable_data(i)[j] = frontier[j].first;
                dists.mutable_data(i)[j] = frontier[j].second;
            }
        });
        return std::make_pair(std::move(ids), std::move(dists));
    }

    void check_recall(std::string &gFile, py::array_t<unsigned int, py::array::c_style | py::array::forcecast> &neighbors, int k){
        groundTruth<unsigned int> GT = groundTruth<unsigned int>(gFile.data());

        size_t n = GT.size();
    
        int numCorrect = 0;
        for (unsigned int i = 0; i < n; i++) {
        parlay::sequence<int> results_with_ties;
            for (unsigned int l = 0; l < k; l++)
                results_with_ties.push_back(GT.coordinates(i,l));
            float last_dist = GT.distances(i, k-1);
            for (unsigned int l = k; l < GT.dimension(); l++) {
                if (GT.distances(i,l) == last_dist) {
                results_with_ties.push_back(GT.coordinates(i,l));
                }
            }
            std::set<int> reported_nbhs;
            for (unsigned int l = 0; l < k; l++) reported_nbhs.insert(neighbors.mutable_data(i)[l]);
            for (unsigned int l = 0; l < results_with_ties.size(); l++) {
                if (reported_nbhs.find(results_with_ties[l]) != reported_nbhs.end()) {
                numCorrect += 1;
                }
            }
        }
        float recall = static_cast<float>(numCorrect) / static_cast<float>(k * n);
        std::cout << "Recall: " << recall << std::endl;
    }

    /* Returns a neighbors and distances pair which actually represents the R
    out neighbors of each point and the length of their respective edges. 
    
    when a point has fewer than R edges, the distances will be float inf and 
    the indices will be unsigned int max. */
    NeighborsAndDistances edges_and_lengths() {
        size_t n = Points.size();
        size_t R = G.max_degree();
        py::array_t<unsigned int> ids({n, R});
        py::array_t<float> dists({n, R});
        parlay::parallel_for(0, n, [&] (size_t i){
            auto neighbors = G[i];
            
            for(int j=0; j<neighbors.size(); j++){
                ids.mutable_data(i)[j] = neighbors[j];
                dists.mutable_data(i)[j] = Points[i].distance(Points[neighbors[j]]);
            }
            for(int j=neighbors.size(); j<R; j++){
                ids.mutable_data(i)[j] = std::numeric_limits<unsigned int>::max();
                dists.mutable_data(i)[j] = std::numeric_limits<float>::infinity();
            }
        });
        return std::make_pair(std::move(ids), std::move(dists));
    }

};
