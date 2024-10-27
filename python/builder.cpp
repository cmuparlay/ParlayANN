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
#include "../algorithms/HCNNG/hcnng_index.h"
#include "../algorithms/pyNNDescent/pynn_index.h"
#include "../algorithms/HNSW/HNSW.hpp"
#include "../algorithms/utils/types.h"
#include "../algorithms/utils/point_range.h"
#include "../algorithms/utils/graph.h"
#include "../algorithms/utils/euclidian_point.h"
#include "../algorithms/utils/mips_point.h"
#include "../algorithms/utils/stats.h"

using namespace parlayANN;

template <typename T, typename Point>
void build_vamana_index(std::string metric, std::string &vector_bin_path,
                        std::string &index_output_path, uint32_t graph_degree, uint32_t beam_width,
                        float alpha, bool two_pass)
{
  //use file parsers to create Point object

  using Range = PointRange<Point>;
  Range* Points = new Range(vector_bin_path.data());
  if (!Point::is_metric()) { // normalize) {
    std::cout << "normalizing" << std::endl;
    for (int i=0; i < Points->size(); i++) 
      (*Points)[i].normalize();
    if (Points->dimension() <= 200) {
      if (Points->dimension() < 100)
        alpha = 1.0;
      else alpha = .98;
    }
  }

  //instantiate build params and stats objects
  BuildParams BP(graph_degree, beam_width, alpha, two_pass ? 2 : 1);
  stats<unsigned int> BuildStats(Points->size());

  if (sizeof(typename Range::Point::T) > 1) {
    if (Point::is_metric()) {
      using QuantT = uint8_t;
      using QuantPoint = Euclidian_Point<QuantT>;
      using QuantRange = PointRange<QuantPoint>;
      QuantRange Quant_Points(*Points);  // quantized to one byte
      delete Points; // remove original points
      Graph<unsigned int> G = Graph<unsigned int>(graph_degree, Points->size());

      //call the build function
      using index = knn_index<QuantRange, unsigned int>;
      index I(BP);
      I.build_index(G, Quant_Points, BuildStats);
      G.save(index_output_path.data());
    } else {
      using QuantT = int8_t;
      using QuantPoint = Quantized_Mips_Point<8, true>;
      using QuantRange = PointRange<QuantPoint>;
      QuantRange Quant_Points(*Points);  // quantized to one byte
      delete Points;  // remove original points
      Graph<unsigned int> G = Graph<unsigned int>(graph_degree, Points->size());

      //call the build function
      using index = knn_index<QuantRange, unsigned int>;
      index I(BP);
      I.build_index(G, Quant_Points, BuildStats);
      G.save(index_output_path.data());
    }
  } else {
    Graph<unsigned int> G = Graph<unsigned int>(graph_degree, Points->size());
    using index = knn_index<PointRange<Point>, unsigned int>;
    index I(BP);
    I.build_index(G, *Points, BuildStats);
    G.save(index_output_path.data());
  } 
}

template void build_vamana_index<float, Euclidian_Point<float>>(std::string , std::string &, std::string &, uint32_t, uint32_t,
                                        float, bool);                            
template void build_vamana_index<float, Mips_Point<float>>(std::string , std::string &, std::string &, uint32_t, uint32_t,
                                        float, bool);

template void build_vamana_index<int8_t, Euclidian_Point<int8_t>>(std::string , std::string &, std::string &, uint32_t, uint32_t,
                                         float, bool);
template void build_vamana_index<int8_t, Mips_Point<int8_t>>(std::string , std::string &, std::string &, uint32_t, uint32_t,
                                         float, bool);

template void build_vamana_index<uint8_t, Euclidian_Point<uint8_t>>(std::string , std::string &, std::string &, uint32_t, uint32_t,
                                          float, bool);
template void build_vamana_index<uint8_t, Mips_Point<uint8_t>>(std::string , std::string &, std::string &, uint32_t, uint32_t,
                                          float, bool);



template <typename T, typename Point>
void build_hcnng_index(std::string metric, std::string &vector_bin_path,
                         std::string &index_output_path, uint32_t mst_deg, uint32_t num_clusters,
                        uint32_t cluster_size)
{
    
    //instantiate build params object
    BuildParams BP(num_clusters, cluster_size, mst_deg);
    uint32_t graph_degree = BP.max_degree();

    //use file parsers to create Point object

    PointRange<Point> Points(vector_bin_path.data());
    //use max degree info to create Graph object
    Graph<unsigned int> G = Graph<unsigned int>(graph_degree, Points.size());

    //call the build function
    using index = hcnng_index<Point, PointRange<Point>, unsigned int>;
    index I;
    stats<unsigned int> BuildStats(G.size());
    I.build_index(G, Points, BP.num_clusters, BP.cluster_size, BP.MST_deg);

    //save the graph object
    G.save(index_output_path.data());

    
}

template void build_hcnng_index<float, Euclidian_Point<float>>(std::string , std::string &, std::string &, uint32_t, uint32_t,
                                        uint32_t);                            
template void build_hcnng_index<float, Mips_Point<float>>(std::string , std::string &, std::string &, uint32_t, uint32_t,
                                        uint32_t);

template void build_hcnng_index<int8_t, Euclidian_Point<int8_t>>(std::string , std::string &, std::string &, uint32_t, uint32_t,
                                         uint32_t);
template void build_hcnng_index<int8_t, Mips_Point<int8_t>>(std::string , std::string &, std::string &, uint32_t, uint32_t,
                                         uint32_t);

template void build_hcnng_index<uint8_t, Euclidian_Point<uint8_t>>(std::string , std::string &, std::string &, uint32_t, uint32_t,
                                          uint32_t);
template void build_hcnng_index<uint8_t, Mips_Point<uint8_t>>(std::string , std::string &, std::string &, uint32_t, uint32_t,
                                          uint32_t);


template <typename T, typename Point>
void build_pynndescent_index(std::string metric, std::string &vector_bin_path,
                         std::string &index_output_path, uint32_t max_deg, uint32_t num_clusters,
                        uint32_t cluster_size, double alpha, double delta)
{
    
    //instantiate build params object
    BuildParams BP(max_deg, alpha, num_clusters, cluster_size, delta);
    uint32_t graph_degree = BP.max_degree();

    //use file parsers to create Point object

    PointRange<Point> Points(vector_bin_path.data());
    //use max degree info to create Graph object
    Graph<unsigned int> G = Graph<unsigned int>(graph_degree, Points.size());

    //call the build function
    using index = pyNN_index<Point, PointRange<Point>, unsigned int>;
    index I(BP.R, BP.delta);
    stats<unsigned int> BuildStats(G.size());
    I.build_index(G, Points, BP.cluster_size, BP.num_clusters, BP.alpha);

    //save the graph object
    G.save(index_output_path.data());

    
}

template void build_pynndescent_index<float, Euclidian_Point<float>>(std::string , std::string &, std::string &, uint32_t, uint32_t,
                                        uint32_t, double, double);                            
template void build_pynndescent_index<float, Mips_Point<float>>(std::string , std::string &, std::string &, uint32_t, uint32_t,
                                        uint32_t, double, double);

template void build_pynndescent_index<int8_t, Euclidian_Point<int8_t>>(std::string , std::string &, std::string &, uint32_t, uint32_t,
                                         uint32_t, double, double);
template void build_pynndescent_index<int8_t, Mips_Point<int8_t>>(std::string , std::string &, std::string &, uint32_t, uint32_t,
                                         uint32_t, double, double);

template void build_pynndescent_index<uint8_t, Euclidian_Point<uint8_t>>(std::string , std::string &, std::string &, uint32_t, uint32_t,
                                          uint32_t, double, double);
template void build_pynndescent_index<uint8_t, Mips_Point<uint8_t>>(std::string , std::string &, std::string &, uint32_t, uint32_t,
                                          uint32_t, double, double);


template <typename T, typename Point>
void build_hnsw_index(std::string metric, std::string &vector_bin_path,
                         std::string &index_output_path, uint32_t graph_degree, uint32_t efc,
                        float m_l, float alpha)
{
    //instantiate build params object
    //BuildParams BP(graph_degree, efc, alpha);

    //use file parsers to create Point object
    PointRange<Point> Points(vector_bin_path.data());
    /*
    //use max degree info to create Graph object
    Graph<unsigned int> G = Graph<unsigned int>(graph_degree, Points.size());

    //call the build function
    using index = hnsw_index<Point, PointRange<T, Point>, unsigned int>;
    index I(BP);
    stats<unsigned int> BuildStats(G.size());
    I.build_index(G, Points, BuildStats);

    //save the graph object
    G.save(index_output_path.data());
    */
    using desc = Desc_HNSW<T, Point>;
    // using elem_t = typename desc::type_elem;

    // point_converter_default<elem_t> to_point;
    // auto [ps,dim] = load_point(vector_bin_path, to_point, cnt_points);
    auto ps = parlay::delayed_seq<Point>(
      Points.size(),
      [&](size_t i){return Points[i];}
    );
    const auto dim = Points.get_dims();
    auto G = ANN::HNSW<desc>(ps.begin(), ps.end(), dim, m_l, graph_degree, efc, alpha);
    G.save(index_output_path);
}

template void build_hnsw_index<float, Euclidian_Point<float>>(std::string , std::string &, std::string &, uint32_t, uint32_t,
                                        float, float);
template void build_hnsw_index<float, Mips_Point<float>>(std::string , std::string &, std::string &, uint32_t, uint32_t,
                                        float, float);

template void build_hnsw_index<int8_t, Euclidian_Point<int8_t>>(std::string , std::string &, std::string &, uint32_t, uint32_t,
                                         float, float);
template void build_hnsw_index<int8_t, Mips_Point<int8_t>>(std::string , std::string &, std::string &, uint32_t, uint32_t,
                                         float, float);

template void build_hnsw_index<uint8_t, Euclidian_Point<uint8_t>>(std::string , std::string &, std::string &, uint32_t, uint32_t,
                                          float, float);
template void build_hnsw_index<uint8_t, Mips_Point<uint8_t>>(std::string , std::string &, std::string &, uint32_t, uint32_t,
                                          float, float);
