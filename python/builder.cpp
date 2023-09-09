// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//TODO fill in correct includes
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

    //call the actual build function
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
