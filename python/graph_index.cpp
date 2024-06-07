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

  // euclidean quantized points
  using EQuantT = uint8_t;
  using EQuantPoint = Euclidian_Point<EQuantT>;
  using EQuantRange = PointRange<EQuantT, EQuantPoint>;
  EQuantRange EQuant_Points;

  // mips or angular quantized points
  using MQuantT = int8_t;
  using MQuantPoint = Quantized_Mips_Point<MQuantT>;
  using MQuantRange = PointRange<MQuantT, MQuantPoint>;
  MQuantRange MQuant_Points;
  
  bool use_quantization;

  std::optional<ANN::HNSW<Desc_HNSW<T, Point>>> HNSW_index;

  GraphIndex(std::string &data_path, std::string &index_path, bool is_hnsw=false)
    : use_quantization(false) {
    Points = PointRange<T, Point>(data_path.data());
    
    if (sizeof(T) > 1) {
      use_quantization = true;
      if (Point::is_metric()) {
        EQuant_Points = EQuantRange(Points);
      } else {
        for (int i=0; i < Points.size(); i++) 
          Points[i].normalize();
        MQuant_Points = MQuantRange(Points);
      }
    }

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
      if (G.size() != Points.size()) {
        std::cout << "graph size and point size do not match" << std::endl;
        abort();
      }
    }
  }

  auto search_dispatch(Point &q, QueryParams &QP, bool quant)
  {
    // if(HNSW_index) {
    //   using indexType = unsigned int; // be consistent with the type of G
    //   using std::pair;
    //   using seq_t = parlay::sequence<pair<indexType, typename Point::distanceType>>;

    //   indexType dist_cmps = 0;
    //   search_control ctrl{};
    //   if(QP.limit>0) {
    //     ctrl.limit_eval = QP.limit;
    //   }
    //   ctrl.count_cmps = &dist_cmps;

    //   seq_t frontier = HNSW_index->search(q, QP.k, QP.beamSize, ctrl);
    //   return pair(pair(std::move(frontier), seq_t{}), dist_cmps);
    // }
    //    else {
    using indexType = unsigned int;
    parlay::sequence<indexType> starts(1, 0);
    stats<indexType> Qstats(1);
    if (quant && use_quantization) {
      int dim = Points.params.dims;
      if (Point::is_metric()) {
        typename EQuantPoint::T buffer[dim];
        if (EQuant_Points.params.slope == 1) {
          for (int i=0; i < dim; i++)
            buffer[i] = q[i];
          EQuantPoint quant_q(buffer, 0, EQuant_Points.params);
          return beam_search(quant_q, G, EQuant_Points, starts, QP).first.first;
        } else {
          q.normalize();
          EQuantPoint::translate_point(buffer, q, EQuant_Points.params);
          EQuantPoint quant_q(buffer, 0, EQuant_Points.params);
          return beam_search_rerank(q, quant_q, G,
                                    Points, EQuant_Points,
                                    Qstats, starts, QP);
        }
      } else {
        typename MQuantPoint::T buffer[dim];
        MQuantPoint::translate_point(buffer, q, MQuant_Points.params);
        MQuantPoint quant_q(buffer, 0, MQuant_Points.params);
        return beam_search_rerank(q, quant_q, G,
                                  Points, MQuant_Points,
                                  Qstats, starts, QP);
      }
    } else {
      return beam_search(q, G, Points, starts, QP).first.first;
    }
  }

  NeighborsAndDistances batch_search(py::array_t<T, py::array::c_style | py::array::forcecast> &queries,
                                     //uint64_t num_queries_,
                                     uint64_t knn,
                                     uint64_t beam_width,
                                     bool quant = false,
                                     int64_t visit_limit = -1) {
    QueryParams QP(knn, beam_width, 1.35, visit_limit, std::min<int>(G.max_degree(), 3*visit_limit));

    uint64_t num_queries = queries.shape(0);
    py::array_t<unsigned int> ids({num_queries, knn});
    py::array_t<float> dists({num_queries, knn});

    parlay::parallel_for(0, num_queries, [&] (size_t i){
      std::vector<T> v(Points.dimension());
      for (int j=0; j < v.size(); j++)
        v[j] = queries.data(i)[j];
      Point q = Point(v.data(), 0, Points.params);
      auto frontier = search_dispatch(q, QP, quant);
      for(int j=0; j<knn; j++){
        ids.mutable_data(i)[j] = frontier[j].first;
        dists.mutable_data(i)[j] = frontier[j].second;
      }
    });
    return std::make_pair(std::move(ids), std::move(dists));
  }

  py::array_t<unsigned int>
  single_search(py::array_t<T>& q, uint64_t knn,
                uint64_t beam_width, bool quant,
                int64_t visit_limit) {
    QueryParams QP(knn, beam_width, 1.35, visit_limit, std::min<int>(G.max_degree(), 3*visit_limit));
    int dims = Points.dimension();

    py::array_t<unsigned int> ids({knn});
    //py::array_t<float> dists({knn});
    auto pp = q.mutable_unchecked();
    T v[dims];
    for (int j=0; j < dims; j++)
      v[j] = pp(j); //q.data()[j];
    Point p = Point(v, 0, Points.params);
    auto frontier = search_dispatch(p, QP, quant);
    for(int j=0; j<knn; j++) {
      ids.mutable_data()[j] = frontier[j].first;
      //dists.mutable_data()[j] = frontier[j].second;
    }
    return std::move(ids);
  }

  NeighborsAndDistances batch_search_from_string(std::string &queries,
                                                 //uint64_t num_queries_,
                                                 uint64_t knn,
                                                 uint64_t beam_width, bool quant = false,
                                                 int64_t visit_limit = -1) {
    QueryParams QP(knn, beam_width, 1.35, visit_limit, std::min<int>(G.max_degree(), 3*visit_limit));
    PointRange<T, Point> QueryPoints = PointRange<T, Point>(queries.data());
    uint64_t num_queries = QueryPoints.size();
    py::array_t<unsigned int> ids({num_queries, knn});
    py::array_t<float> dists({num_queries, knn});
    parlay::parallel_for(0, num_queries, [&] (size_t i){
      auto p = QueryPoints[i];
      auto frontier = search_dispatch(p, QP, quant);
      for(int j=0; j<knn; j++){
        ids.mutable_data(i)[j] = frontier[j].first;
        dists.mutable_data(i)[j] = frontier[j].second;
      }
    });
    return std::make_pair(std::move(ids), std::move(dists));
  }

  void check_recall(std::string &queries_file,
                    std::string &graph_file,
                    py::array_t<unsigned int, py::array::c_style | py::array::forcecast> &neighbors,
                    int k){
    groundTruth<unsigned int> GT = groundTruth<unsigned int>(graph_file.data());
    PointRange<T, Point> QueryPoints = PointRange<T, Point>(queries_file.data());

    size_t n = GT.size();
    
    int numCorrect = 0;
    for (unsigned int i = 0; i < n; i++) {
      parlay::sequence<int> results_with_ties;
      for (unsigned int l = 0; l < k; l++)
        results_with_ties.push_back(GT.coordinates(i,l));
      float last_dist = QueryPoints[i].distance(Points[GT.coordinates(i, k-1)]);
      for (unsigned int l = k; l < GT.dimension(); l++) {
        auto p = Points[GT.coordinates(i, l)];
        if (QueryPoints[i].distance(p) == last_dist) {
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

  // void check_recall(std::string &graph_file, py::array_t<unsigned int, py::array::c_style | py::array::forcecast> &neighbors, int k){
  //   groundTruth<unsigned int> GT = groundTruth<unsigned int>(graph_file.data());

  //   size_t n = GT.size();

  //   int numCorrect = 0;
  //   for (unsigned int i = 0; i < n; i++) {
  //     parlay::sequence<int> results_with_ties;
  //     for (unsigned int l = 0; l < k; l++)
  //       results_with_ties.push_back(GT.coordinates(i,l));
  //     std::cout << i << std::endl;
  //     float last_dist = GT.distances(i, k-1);
  //     for (unsigned int l = k; l < GT.dimension(); l++) {
  //       if (GT.distances(i,l) == last_dist) {
  //         results_with_ties.push_back(GT.coordinates(i,l));
  //       }
  //     }
  //     std::cout << "aa" << std::endl;
  //     std::set<int> reported_nbhs;
  //     for (unsigned int l = 0; l < k; l++) reported_nbhs.insert(neighbors.mutable_data(i)[l]);
  //     for (unsigned int l = 0; l < results_with_ties.size(); l++) {
  //       if (reported_nbhs.find(results_with_ties[l]) != reported_nbhs.end()) {
  //         numCorrect += 1;
  //       }
  //     }
  //   }
  //   float recall = static_cast<float>(numCorrect) / static_cast<float>(k * n);
  //   std::cout << "Recall: " << recall << std::endl;
  // }

};
