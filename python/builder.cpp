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
void build_vamana_index(std::string metric, std::string &vector_bin_path,
                         std::string &index_output_path, uint32_t graph_degree, uint32_t beam_width,
                        float alpha)
{
    
    //instantiate build params object
    BuildParams BP(graph_degree, beam_width, alpha);

    //use file parsers to create Point object

    PointRange<T, Point> Points = PointRange<T, Point>(vector_bin_path.data());
    //use max degree info to create Graph object
    Graph<unsigned int> G = Graph<unsigned int>(graph_degree, Points.size());

    //call the build function
    using index = knn_index<Point, PointRange<T, Point>, unsigned int>;
    index I(BP);
    stats<unsigned int> BuildStats(G.size());
    I.build_index(G, Points, BuildStats);

    //save the graph object
    G.save(index_output_path.data());

    
    
}

template void build_vamana_index<float, Euclidian_Point<float>>(std::string , std::string &, std::string &, uint32_t, uint32_t,
                                        float);                            
template void build_vamana_index<float, Mips_Point<float>>(std::string , std::string &, std::string &, uint32_t, uint32_t,
                                        float);

template void build_vamana_index<int8_t, Euclidian_Point<int8_t>>(std::string , std::string &, std::string &, uint32_t, uint32_t,
                                         float);
template void build_vamana_index<int8_t, Mips_Point<int8_t>>(std::string , std::string &, std::string &, uint32_t, uint32_t,
                                         float);

template void build_vamana_index<uint8_t, Euclidian_Point<uint8_t>>(std::string , std::string &, std::string &, uint32_t, uint32_t,
                                          float);
template void build_vamana_index<uint8_t, Mips_Point<uint8_t>>(std::string , std::string &, std::string &, uint32_t, uint32_t,
                                          float);