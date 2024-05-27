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
using RangeNeighborsAndDistances = std::pair<py::array_t<unsigned int>, NeighborsAndDistances>;

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


        parlay::parallel_for(0, num_queries, [&] (size_t i){
            Point q = Point(queries.data(i), Points.dimension(), Points.aligned_dimension(), i);
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

    // Range search results are represented like a sparse CSR matrix with 3 components:

    // lims, I, D

    // The results for query #q are:

    // I[lims[q]:lims[q + 1]] in int

    // And the corresponding distances are:

    // D[lims[q]:lims[q + 1]] in float

    // Thus, len(lims) = nq + 1, lims[i + 1] >= lims[i] forall i
    // and len(D) = len(I) = lims[-1].
    RangeNeighborsAndDistances batch_range_search(py::array_t<T, py::array::c_style | py::array::forcecast> &queries, uint64_t num_queries, double radius,
                        uint64_t beam_width){

        QueryParams QP(beam_width, beam_width, 1.35, G.size(), G.max_degree());

        parlay::sequence<parlay::sequence<std::pair<unsigned int, float>>> results(num_queries);


        parlay::parallel_for(0, num_queries, [&] (size_t i){
            Point q = Point(queries.data(i), Points.dimension(), Points.aligned_dimension(), i);
            auto [pairElts, dist_cmps] = beam_search<Point, PointRange<T, Point>, unsigned int>(q, G, Points, 0, QP);
            auto [frontier, visited] = pairElts;
            parlay::sequence<std::pair<unsigned int, float>> res;
            for(auto i : frontier){
                if(i.second <= radius) res.push_back(i);
            }
            results[i] = res;
        });

        return process_range_results(results);
        
    }

    RangeNeighborsAndDistances batch_range_search_from_string(std::string &queries, uint64_t num_queries, double radius,
                                    uint64_t beam_width){
         QueryParams QP(beam_width, beam_width, 1.35, G.size(), G.max_degree());
        PointRange<T, Point> QueryPoints = PointRange<T, Point>(queries.data());

        parlay::sequence<parlay::sequence<std::pair<unsigned int, float>>> results(num_queries);

        parlay::parallel_for(0, num_queries, [&] (size_t i){
            auto [pairElts, dist_cmps] = beam_search<Point, PointRange<T, Point>, unsigned int>(QueryPoints[i], G, Points, 0, QP);
            auto [frontier, visited] = pairElts;
            parlay::sequence<std::pair<unsigned int, float>> res;
            for(auto i : frontier){
                if(i.second <= radius) res.push_back(i);
            }
            results[i] = res;
        });
        return process_range_results(results);
    }

    //processes the output of range search into the format expected by the competition framework
    RangeNeighborsAndDistances process_range_results(parlay::sequence<parlay::sequence<std::pair<unsigned int, float>>> results){
        auto sizes = parlay::tabulate(results.size(), [&] (size_t i){
            return results[i].size();
        });

        auto [offsets, total] = parlay::scan(sizes);
        offsets.push_back(total);

        py::array_t<unsigned int> lims(offsets.size());
        py::array_t<unsigned int> ids(total);
        py::array_t<float> dists(total);

        parlay::parallel_for(0, offsets.size(), [&] (size_t i){
            lims.mutable_data()[i] = offsets[i];
        });

        parlay::parallel_for(0, results.size(), [&] (size_t i){
            size_t start = offsets[i];
            for(size_t j=0; j<results[i].size(); j++){
                ids.mutable_data()[start+j] = results[i][j].first;
                dists.mutable_data()[start+j] = results[i][j].second;
            }
        });

        return std::make_pair(std::move(lims), std::make_pair(std::move(ids), std::move(dists)));
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

    void check_range_recall(std::string &gFile, py::array_t<unsigned int, py::array::c_style | py::array::forcecast> &lims){

        RangeGroundTruth<unsigned int> RGT = RangeGroundTruth<unsigned int>(gFile.data());

        float pointwise_recall = 0.0;
        float reported_results = 0.0;
        float total_results = 0.0;
        float num_nonzero = 0.0;

        //since distances are exact, just have to cross-check number of results
        size_t n = lims.size()-1;
        int numCorrect = 0;
        for (size_t i = 0; i < n; i++) {
            float num_reported_results = lims.mutable_data()[i+1] - lims.mutable_data()[i];
            float num_actual_results = RGT[i].size();
            reported_results += num_reported_results;
            total_results += num_actual_results;
            if(num_actual_results != 0) {pointwise_recall += num_reported_results/num_actual_results; num_nonzero++;}
        }
        
        pointwise_recall /= num_nonzero;
        float cumulative_recall = reported_results/total_results;

        std::cout << "Pointwise Recall = " << pointwise_recall << ", Cumulative Recall = " << cumulative_recall << std::endl;
    }

    

};
