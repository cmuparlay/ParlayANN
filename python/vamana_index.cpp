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
#include "pybind11/numpy.h"

#include "parlay/parallel.h"
#include "parlay/primitives.h"

#include <stdio.h>

namespace py = pybind11;
using NeighborsAndDistances = std::pair<py::array_t<unsigned int>, py::array_t<float>>;

template<typename T, typename Point> 
struct VamanaIndex{
    Graph<unsigned int> G;
    Graph<unsigned int> G_S;
    PointRange<T, Point> Points;
    PointRange<T, Point> Sample_Points;
    groundTruth<unsigned int> Sample_GT;
    using QPR = QuantizedPointRange<T2I_Point, uint16_t>;
    QPR Quantized_Points;
    using pid = std::pair<unsigned int, float>;
    

    VamanaIndex(std::string &data_path, std::string &compressed_vectors_path, std::string& sample_path, 
                std::string &index_path, std::string& secondary_index_path, std::string& secondary_gt_path, 
                size_t num_points, size_t dimensions){
        G = Graph<unsigned int>(index_path.data());
        Points = PointRange<T, Point>(data_path.data());
        if(compressed_vectors_path != ""){ 
            std::cout << "Compressed path: " << compressed_vectors_path << std::endl;
            Quantized_Points = QPR(compressed_vectors_path.data());
        }
        if(sample_path != ""){
            std::cout << "Sample path: " << sample_path << std::endl;
            G_S = Graph<unsigned int>(secondary_index_path.data());
            Sample_Points = PointRange<T, Point>(sample_path.data());
            Sample_GT = groundTruth<unsigned int>(secondary_gt_path.data());
        }
        
    }

    parlay::sequence<unsigned int> generate_start_points(Point q){
        unsigned int start_point = 0;
        QueryParams QP = QueryParams(10, 10, 1.35, G_S.size(), G_S.max_degree());
        auto [pairElts, dist_cmps] = beam_search<Point, PointRange<T, Point>, unsigned int>(q, G_S, Sample_Points, 0, QP);
        auto [beamElts, visitedElts] = pairElts;
        parlay::sequence<unsigned int> closest_in_sample;
        //points to compute nn for 
        for(int j=0; j<10; j++) closest_in_sample.push_back(beamElts[j].first);
        parlay::sequence<unsigned int> starting_candidates;
        for(auto j : closest_in_sample){
            for(int l=0; l<10; l++){
                starting_candidates.push_back(Sample_GT.coordinates(j, l));
            }
        }
        auto less = [&] (pid a, pid b) {return a.second < b.second;};
        auto starting_candidates_rd = parlay::remove_duplicates(starting_candidates);
        auto starting_candidates_sorted = parlay::tabulate(starting_candidates_rd.size(), [&] (size_t j){
            unsigned int index = starting_candidates_rd[j];
            float dist = q.distance(Points[index]);
            return std::make_pair(index, dist);
        });
        std::sort(starting_candidates_sorted.begin(), starting_candidates_sorted.end(), less);
        //push back og start point
        parlay::sequence<unsigned int> s; 
        for(int j=0; j<10; j++) s.push_back(starting_candidates_sorted[j].first);
        s.push_back(start_point);
        return s;

    }

    NeighborsAndDistances batch_search(py::array_t<T, py::array::c_style | py::array::forcecast> &queries, uint64_t num_queries, uint64_t knn,
                        uint64_t beam_width){
        QueryParams QP(knn, beam_width, 1.35, G.size(), G.max_degree());

        py::array_t<unsigned int> ids({num_queries, knn});
        py::array_t<float> dists({num_queries, knn});



        parlay::parallel_for(0, num_queries, [&] (size_t i){
            Point q = Point(queries.data(i), Points.dimension(), Points.aligned_dimension(), i);
            parlay::sequence<unsigned int> start_points;
            if(Sample_Points.size() > 0){
                start_points = generate_start_points(q);
            } else start_points = {0};
            //search with quantized distances
            std::pair<parlay::sequence<pid>, parlay::sequence<pid>> pairElts;
            if(Quantized_Points.size() > 0){
                pairElts = (beam_search<Point, QPR, unsigned int>(q, G, Quantized_Points, start_points, QP)).first;
                //rerank
                auto [frontier, visited] = pairElts;
                auto less = [&] (pid a, pid b) {return a.second < b.second;};
                int sort_range = std::min<int>(2*QP.k, frontier.size());
                auto reranked_points = parlay::tabulate(sort_range, [&] (size_t j){
                    unsigned int index = frontier[j].first;
                    float dist = q.distance(Points[index]);
                    return std::make_pair(index, dist);
                });
                std::sort(reranked_points.begin(), reranked_points.end(), less);
                for(int j=0; j<knn; j++){
                    ids.mutable_data(i)[j] = reranked_points[j].first;
                    dists.mutable_data(i)[j] = reranked_points[j].second;
                }
            } else {
                pairElts = (beam_search<Point, PointRange<T, Point>, unsigned int>(q, G, Points, start_points, QP)).first;
                //rerank
                auto [frontier, visited] = pairElts;
                for(int j=0; j<knn; j++){
                    ids.mutable_data(i)[j] = frontier[j].first;
                    dists.mutable_data(i)[j] = frontier[j].second;
                }
            }
            
        });
        return std::make_pair(std::move(ids), std::move(dists));
    }

    NeighborsAndDistances batch_search_from_string(std::string &queries, uint64_t num_queries, uint64_t knn,
                                    uint64_t beam_width){
        QueryParams QP(knn, beam_width, 1.35, G.size(), G.max_degree());
        PointRange<T, Point> QueryPoints = PointRange<T, Point>(queries.data());
        py::array_t<unsigned int> ids({num_queries, knn});
        py::array_t<float> dists({num_queries, knn});
        parlay::parallel_for(0, num_queries, [&] (size_t i){
            auto start_points = generate_start_points(QueryPoints[i]);
            auto [pairElts, dist_cmps] = beam_search<Point, PointRange<T, Point>, unsigned int>(QueryPoints[i], G, Points, start_points, QP);
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

};