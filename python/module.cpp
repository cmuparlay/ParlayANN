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

#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pybind11/numpy.h"

#include "builder.cpp"
#include "graph_index.cpp"

PYBIND11_MAKE_OPAQUE(std::vector<uint32_t>);
PYBIND11_MAKE_OPAQUE(std::vector<float>);
PYBIND11_MAKE_OPAQUE(std::vector<int8_t>);
PYBIND11_MAKE_OPAQUE(std::vector<uint8_t>);

namespace py = pybind11;
using namespace pybind11::literals;

// using NeighborsAndDistances = std::pair<py::array_t<unsigned int, py::array::c_style | py::array::forcecast>, py::array_t<float, py::array::c_style | py::array::forcecast>>;

struct Variant
{
    std::string builder_name;
    std::string index_name;
};

const Variant FloatEuclidianVariant{"build_vamana_float_euclidian_index", "FloatEuclidianIndex"};
const Variant FloatMipsVariant{"build_vamana_float_mips_index", "FloatMipsIndex"};

const Variant UInt8EuclidianVariant{"build_vamana_uint8_euclidian_index", "UInt8EuclidianIndex"};
const Variant UInt8MipsVariant{"build_vamana_uint8_mips_index", "UInt8MipsIndex"};

const Variant Int8EuclidianVariant{"build_vamana_int8_euclidian_index", "Int8EuclidianIndex"};
const Variant Int8MipsVariant{"build_vamana_int8_mips_index", "Int8MipsIndex"};

template <typename T, typename Point> inline void add_variant(py::module_ &m, const Variant &variant)
{

    m.def(variant.builder_name.c_str(), build_vamana_index<T, Point>, "distance_metric"_a,
          "data_file_path"_a, "index_output_path"_a, "graph_degree"_a, "beam_width"_a, "alpha"_a, "two_pass"_a);

    py::class_<GraphIndex<T, Point>>(m, variant.index_name.c_str())
        .def(py::init<std::string &, std::string &, size_t, size_t, bool>(),
             "index_path"_a, "data_path"_a, "num_points"_a, "dimensions"_a, "hnsw"_a=false)
        //maybe "num_points" and "dimensions" are unnecessary?
        //do we want to add options like visited limit, or leave those as defaults?
        .def("batch_search", &GraphIndex<T, Point>::batch_search, "queries"_a, "num_queries"_a, "knn"_a,
             "beam_width"_a, "visit_limit"_a)
        .def("batch_search_from_string", &GraphIndex<T, Point>::batch_search_from_string, "queries"_a, "num_queries"_a, "knn"_a,
             "beam_width"_a)
        .def("check_recall", &GraphIndex<T, Point>::check_recall, "gFile"_a, "neighbors"_a, "k"_a);

   
}

const Variant FloatEuclidianHCNNGVariant{"build_hcnng_float_euclidian_index", "FloatEuclidianIndex"};
const Variant FloatMipsHCNNGVariant{"build_hcnng_float_mips_index", "FloatMipsIndex"};

const Variant UInt8EuclidianHCNNGVariant{"build_hcnng_uint8_euclidian_index", "UInt8EuclidianIndex"};
const Variant UInt8MipsHCNNGVariant{"build_hcnng_uint8_mips_index", "UInt8MipsIndex"};

const Variant Int8EuclidianHCNNGVariant{"build_hcnng_int8_euclidian_index", "Int8EuclidianIndex"};
const Variant Int8MipsHCNNGVariant{"build_hcnng_int8_mips_index", "Int8MipsIndex"};

template <typename T, typename Point> inline void add_hcnng_variant(py::module_ &m, const Variant &variant)
{

    m.def(variant.builder_name.c_str(), build_hcnng_index<T, Point>, "distance_metric"_a,
          "data_file_path"_a, "index_output_path"_a, "mst_deg"_a, "num_clusters"_a, "cluster_size"_a);

   
}

const Variant FloatEuclidianpyNNVariant{"build_pynndescent_float_euclidian_index", "FloatEuclidianIndex"};
const Variant FloatMipspyNNVariant{"build_pynndescent_float_mips_index", "FloatMipsIndex"};

const Variant UInt8EuclidianpyNNVariant{"build_pynndescent_uint8_euclidian_index", "UInt8EuclidianIndex"};
const Variant UInt8MipspyNNVariant{"build_pynndescent_uint8_mips_index", "UInt8MipsIndex"};

const Variant Int8EuclidianpyNNVariant{"build_pynndescent_int8_euclidian_index", "Int8EuclidianIndex"};
const Variant Int8MipspyNNVariant{"build_pynndescent_int8_mips_index", "Int8MipsIndex"};

template <typename T, typename Point> inline void add_pynndescent_variant(py::module_ &m, const Variant &variant)
{

    m.def(variant.builder_name.c_str(), build_pynndescent_index<T, Point>, "distance_metric"_a,
          "data_file_path"_a, "index_output_path"_a, "max_deg"_a, "num_clusters"_a, "cluster_size"_a, 
          "alpha"_a, "delta"_a);

   
}

const Variant FloatEuclidianHNSWVariant{"build_hnsw_float_euclidian_index", "FloatEuclidianIndex"};
const Variant FloatMipsHNSWVariant{"build_hnsw_float_mips_index", "FloatMipsIndex"};

const Variant UInt8EuclidianHNSWVariant{"build_hnsw_uint8_euclidian_index", "UInt8EuclidianIndex"};
const Variant UInt8MipsHNSWVariant{"build_hnsw_uint8_mips_index", "UInt8MipsIndex"};

const Variant Int8EuclidianHNSWVariant{"build_hnsw_int8_euclidian_index", "Int8EuclidianIndex"};
const Variant Int8MipsHNSWVariant{"build_hnsw_int8_mips_index", "Int8MipsIndex"};

template <typename T, typename Point> inline void add_hnsw_variant(py::module_ &m, const Variant &variant)
{

    m.def(variant.builder_name.c_str(), build_hnsw_index<T, Point>, "distance_metric"_a,
          "data_file_path"_a, "index_output_path"_a, "graph_degree"_a, "efc"_a, "m_l"_a, "alpha"_a);
}


PYBIND11_MODULE(_ParlayANNpy, m)
{
    m.doc() = "ParlayANN Python Bindings";
#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif

    // let's re-export our defaults
    py::module_ default_values = m.def_submodule(
        "defaults");

    default_values.attr("METRIC") = "Euclidian";
    default_values.attr("ALPHA") = 1.2;
    default_values.attr("GRAPH_DEGREE") = 64;
    default_values.attr("BEAMWIDTH") = 128;

    add_variant<float, Euclidian_Point<float>>(m, FloatEuclidianVariant);
    add_variant<float, Mips_Point<float>>(m, FloatMipsVariant);
    add_variant<uint8_t, Euclidian_Point<uint8_t>>(m, UInt8EuclidianVariant);
    add_variant<uint8_t, Mips_Point<uint8_t>>(m, UInt8MipsVariant);
    add_variant<int8_t, Euclidian_Point<int8_t>>(m, Int8EuclidianVariant);
    add_variant<int8_t, Mips_Point<int8_t>>(m, Int8MipsVariant);

    add_hcnng_variant<float, Euclidian_Point<float>>(m, FloatEuclidianHCNNGVariant);
    add_hcnng_variant<float, Mips_Point<float>>(m, FloatMipsHCNNGVariant);
    add_hcnng_variant<uint8_t, Euclidian_Point<uint8_t>>(m, UInt8EuclidianHCNNGVariant);
    add_hcnng_variant<uint8_t, Mips_Point<uint8_t>>(m, UInt8MipsHCNNGVariant);
    add_hcnng_variant<int8_t, Euclidian_Point<int8_t>>(m, Int8EuclidianHCNNGVariant);
    add_hcnng_variant<int8_t, Mips_Point<int8_t>>(m, Int8MipsHCNNGVariant);

    add_pynndescent_variant<float, Euclidian_Point<float>>(m, FloatEuclidianpyNNVariant);
    add_pynndescent_variant<float, Mips_Point<float>>(m, FloatMipspyNNVariant);
    add_pynndescent_variant<uint8_t, Euclidian_Point<uint8_t>>(m, UInt8EuclidianpyNNVariant);
    add_pynndescent_variant<uint8_t, Mips_Point<uint8_t>>(m, UInt8MipspyNNVariant);
    add_pynndescent_variant<int8_t, Euclidian_Point<int8_t>>(m, Int8EuclidianpyNNVariant);
    add_pynndescent_variant<int8_t, Mips_Point<int8_t>>(m, Int8MipspyNNVariant);

    add_hnsw_variant<float, Euclidian_Point<float>>(m, FloatEuclidianHNSWVariant);
    add_hnsw_variant<float, Mips_Point<float>>(m, FloatMipsHNSWVariant);
    add_hnsw_variant<uint8_t, Euclidian_Point<uint8_t>>(m, UInt8EuclidianHNSWVariant);
    add_hnsw_variant<uint8_t, Mips_Point<uint8_t>>(m, UInt8MipsHNSWVariant);
    add_hnsw_variant<int8_t, Euclidian_Point<int8_t>>(m, Int8EuclidianHNSWVariant);
    add_hnsw_variant<int8_t, Mips_Point<int8_t>>(m, Int8MipsHNSWVariant);
}
