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
#include "../algorithms/utils.types.h"
#include "../algorithms/point_range.h"
#include "../algorithms/graph.h"
#include "../algorithms/euclidian_point.h"
#include "../algorithms/mips_point.h"


template <typename T>
void build_vamana_index(const std::string metric, const std::string &vector_bin_path,
                        const std::string &index_output_path, const uint32_t graph_degree, const uint32_t beam_width,
                        const float alpha)
{
    
    //instantiate build params object
    BuildParams BP(R, L, a);

    //use the metric string to infer the point type
    assert(metric == "Euclidian" | metric == "mips");

    //use file parsers to create Point object
    if(metric == "Euclidian"){
        PointRange<T, Euclidian_Point<T>> Points = PointRange<T, Euclidian_Point<T>>(vector_bin_path.c_str());
    }else if(metric == "mips"){
        PointRange<T, Mips_Point<T>> Points = PointRange<T, Mips_Point<T>>(vector_bin_path.c_str());
    }
    
    //use max degree info to create Graph object
    Graph<unsigned int> G = Graph<unsigned int>(graph_degree, Points.size());

    //call the build function
    using index = knn_index<Euclidian_Point<T>, PointRange<T, Euclidian_Point<T>>, unsigned int>;
    index I(BP);
    stats<unsigned int> BuildStats(G.size());
    I.build_index(G, Points, BuildStats);

    //save the graph object
    G.save(index_output_path.c_str());
}

template void build_vamana_index<float>(const std::string &, const std::string &, const std::string &, uint32_t, uint32_t,
                                        float);

template void build_vamana_index<int8_t>(const std::string &, const std::string &, const std::string &, uint32_t, uint32_t,
                                         float);

template void build_vamana_index<uint8_t>(const std::string &, const std::string &, const std::string &, uint32_t, uint32_t,
                                          float);
