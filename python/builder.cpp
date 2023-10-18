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


template <typename T, typename Point>
void build_vamana_index(std::string metric, std::string &vector_bin_path, std::string &sample_bin_path, std::string &compressed_vectors_path,
                        std::string &index_output_path, std::string &secondary_output_path, std::string& secondary_gt_path,
                        uint32_t graph_degree, uint32_t beam_width, float alpha, bool two_pass)
{
    
    // //instantiate build params object
    // BuildParams BP(graph_degree, beam_width, alpha, two_pass);

    // //use file parsers to create Point object
    // PointRange<T, Point> Points = PointRange<T, Point>(vector_bin_path.data());
    // PointRange<T, Point> Sample_Points = PointRange<T, Point>(sample_bin_path.data());
    // //use max degree info to create Graph object
    // Graph<unsigned int> G = Graph<unsigned int>(graph_degree, Points.size());

    // //call the build function
    // using index = knn_index<Point, PointRange<T, Point>, unsigned int>;
    // index I(BP);
    // stats<unsigned int> BuildStats(G.size());
    // I.build_index(G, Points, BuildStats);

    // Graph G_S = Graph<unsigned int>(BP.max_degree(), Sample_Points.size());
    // BuildParams BP_S(32, 500, 1.0, true);
    // index J(BP_S);
    // stats<unsigned int> Sample_Stats(Sample_Points.size());
    // J.build_index(G_S, Sample_Points, BuildStats);

    // parlay::sequence<parlay::sequence<unsigned int>> neighbors_in_G = parlay::tabulate(Sample_Points.size(), [&] (size_t i){
    //     QueryParams QP = QueryParams(50, 500, 1.35, G.size(), G.max_degree());
    //     auto [pairElts, dist_cmps] = beam_search(Sample_Points[i], G, Points, I.get_start(), QP);
    //     auto [beamElts, visitedElts] = pairElts;
    //     parlay::sequence<unsigned int> closest;
    //     //points to compute nn for 
    //     for(int j=0; j<10; j++) closest.push_back(beamElts[j].first);
    //     return closest;
    // });

    //compute quantization and save
    int bits = 10;
    using QPR = QuantizedPointRange<T2I_Point, uint16_t>;
    QPR Quantized_Points = QPR(Points, bits);
    Quantized_Points.save(compressed_vectors_path.data());

    // //save the graph object
    // G.save(index_output_path.data());
    // G_S.save(secondary_output_path.data());
    // groundTruth<unsigned int> GT(neighbors_in_G);
    // GT.save(secondary_gt_path.data());
    
}

template void build_vamana_index<float, Euclidian_Point<float>>(std::string, std::string &, std::string &, std::string &, std::string &, std::string &, std::string &, 
                                    uint32_t, uint32_t, float, bool);                            
template void build_vamana_index<float, Mips_Point<float>>(std::string , std::string &, std::string &, std::string &, std::string &, std::string &, std::string &, 
                                    uint32_t, uint32_t, float, bool);   

template void build_vamana_index<int8_t, Euclidian_Point<int8_t>>(std::string , std::string &, std::string &, std::string &, std::string &, std::string &, std::string &, 
                                    uint32_t, uint32_t, float, bool);   
template void build_vamana_index<int8_t, Mips_Point<int8_t>>(std::string , std::string &, std::string &, std::string &, std::string &, std::string &, std::string &, 
                                    uint32_t, uint32_t, float, bool);   

template void build_vamana_index<uint8_t, Euclidian_Point<uint8_t>>(std::string , std::string &, std::string &, std::string &, std::string &, std::string &, std::string &, 
                                    uint32_t, uint32_t, float, bool);   
template void build_vamana_index<uint8_t, Mips_Point<uint8_t>>(std::string , std::string &, std::string &, std::string &, std::string &, std::string &, std::string &, 
                                    uint32_t, uint32_t, float, bool);   