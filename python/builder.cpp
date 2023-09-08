// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

//TODO fill in correct includes
#include "../algorithms/vamana/index.h"
#include "../algorithms/utils.types.h"


template <typename T>
void build_vamana_index(const std::string metric, const std::string &vector_bin_path,
                        const std::string &index_output_path, const uint32_t graph_degree, const uint32_t beam_width,
                        const float alpha)
{
    BuildParams BP();
    //instantiate build params object
    //use the metric string to infer the point type
    //use file parsers to create Point object
    //use max degree info to create Graph object
    //call the actual build function
    //save the graph object

    diskann::IndexWriteParameters index_build_params = diskann::IndexWriteParametersBuilder(complexity, graph_degree)
                                                           .with_filter_list_size(filter_complexity)
                                                           .with_alpha(alpha)
                                                           .with_saturate_graph(false)
                                                           .with_num_threads(num_threads)
                                                           .build();
    diskann::IndexSearchParams index_search_params =
        diskann::IndexSearchParams(index_build_params.search_list_size, num_threads);
    size_t data_num, data_dim;
    diskann::get_bin_metadata(vector_bin_path, data_num, data_dim);

    diskann::Index<T, TagT, LabelT> index(metric, data_dim, data_num,
                                          std::make_shared<diskann::IndexWriteParameters>(index_build_params),
                                          std::make_shared<diskann::IndexSearchParams>(index_search_params), 0,
                                          use_tags, use_tags, false, use_pq_build, num_pq_bytes, use_opq);


    index.build(vector_bin_path.c_str(), data_num);

    index.save(index_output_path.c_str());
}

template void build_memory_index<float>(diskann::Metric, const std::string &, const std::string &, uint32_t, uint32_t,
                                        float, uint32_t, bool, size_t, bool, uint32_t, bool);

template void build_memory_index<int8_t>(diskann::Metric, const std::string &, const std::string &, uint32_t, uint32_t,
                                         float, uint32_t, bool, size_t, bool, uint32_t, bool);

template void build_memory_index<uint8_t>(diskann::Metric, const std::string &, const std::string &, uint32_t, uint32_t,
                                          float, uint32_t, bool, size_t, bool, uint32_t, bool);
