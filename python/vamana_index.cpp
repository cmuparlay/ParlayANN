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

namespace py = pybind11;

template<typename T, typename Point> 
struct VamanaIndex{
    Graph<unsigned int> G;
    PointRange<T, Point> Points;

    VamanaIndex(std::string &data_path, std::string &index_path, size_t num_points, size_t dimensions){
        G = Graph<unsigned int>(index_path.data());
        Points = PointRange<T, Point>(data_path.data());
        assert(num_points == Points.size());
        assert(dimensions == Points.dimension());
    }

    // void batch_search(py::array_t<T, py::array::c_style | py::array::forcecast> &queries, uint64_t num_queries, uint64_t knn,
    // uint64_t beam_width){
    //     QueryParams QP(knn, beam_width, 1.35, G.size(), G.max_degree());
    //     parlay::parallel_for(0, num_queries, [&] (size_t i){
    //         Point q = Point(queries[i].data(), Points.dimension(), Points.aligned_dimension(), i);
    //         auto [pairElts, dist_cmps] = beam_search(q, G, Points, 0, QP);
    //     });
    // }

    void batch_search_from_string(std::string &queries, uint64_t num_queries, uint64_t knn,
    uint64_t beam_width){
        QueryParams QP(knn, beam_width, 1.35, G.size(), G.max_degree());
        PointRange<T, Point> QueryPoints = PointRange<T, Point>(queries.data());
        parlay::parallel_for(0, num_queries, [&] (size_t i){
            auto [pairElts, dist_cmps] = beam_search<Point, PointRange<T, Point>, unsigned int>(QueryPoints[i], G, Points, 0, QP);
        });
    }

};