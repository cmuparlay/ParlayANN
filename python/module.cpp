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
#include "vamana_index.cpp"
#include "../algorithms/sparse/IVF.h"
#include "../algorithms/sparse/posting_list.h"
#include "../algorithms/utils/filters.h"

PYBIND11_MAKE_OPAQUE(std::vector<uint32_t>);
PYBIND11_MAKE_OPAQUE(std::vector<float>);
PYBIND11_MAKE_OPAQUE(std::vector<int8_t>);
PYBIND11_MAKE_OPAQUE(std::vector<uint8_t>);

namespace py = pybind11;
using namespace pybind11::literals;

struct Variant
{
    std::string builder_name;
    std::string index_name;
    std::string ivf_name;
};

const Variant FloatEuclidianVariant{"build_vamana_float_euclidian_index", "VamanaFloatEuclidianIndex", "IVFFloatEuclidianIndex"};
const Variant FloatMipsVariant{"build_vamana_float_mips_index", "VamanaFloatMipsIndex", "IVFFloatMipsIndex"};

const Variant UInt8EuclidianVariant{"build_vamana_uint8_euclidian_index", "VamanaUInt8EuclidianIndex", "IVFUInt8EuclidianIndex"};
const Variant UInt8MipsVariant{"build_vamana_uint8_mips_index", "VamanaUInt8MipsIndex", "IVFUInt8MipsIndex"};

const Variant Int8EuclidianVariant{"build_vamana_int8_euclidian_index", "VamanaInt8EuclidianIndex", "IVFInt8EuclidianIndex"};
const Variant Int8MipsVariant{"build_vamana_int8_mips_index", "VamanaInt8MipsIndex", "IVFInt8MipsIndex"};

template <typename T, typename Point> inline void add_variant(py::module_ &m, const Variant &variant)
{

    m.def(variant.builder_name.c_str(), build_vamana_index<T, Point>, "distance_metric"_a,
          "data_file_path"_a, "index_output_path"_a, "graph_degree"_a, "beam_width"_a, "alpha"_a);

    py::class_<SparseVamana<T, Point>>(m, ("Squared" + variant.ivf_name).c_str())
        .def(py::init())
        .def("fit", &SparseVamana<T, Point>::fit, "points"_a, "filters"_a, "cutoff"_a, "cluster_size"_a) 
        .def("fit_from_filename", &SparseVamana<T, Point>::fit_from_filename, "filename"_a, "filter_filename"_a, "cutoff"_a, "cluster_size"_a)
        .def("batch_filter_search", &SparseVamana<T, Point>::batch_filter_search, "queries"_a, "filters"_a, "num_queries"_a, "knn"_a)
        .def("set_target_points", &SparseVamana<T, Point>::set_target_points, "target_points"_a)
        .def("set_tiny_cutoff", &SparseVamana<T, Point>::set_tiny_cutoff, "tiny_cutoff"_a)
        .def("set_max_iter", &SparseVamana<T, Point>::set_max_iter, "max_iter"_a);
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

    add_variant<float, Mips_Point<float>>(m, FloatMipsVariant);

//    py::class_<csr_filters>(m, "csr_filters")
//        .def(py::init<std::string &>())
//        .def("match", &csr_filters::match, "p"_a, "f"_a)
//        .def("first_label", &csr_filters::first_label, "p"_a)
//        .def("print_stats", &csr_filters::print_stats)
//        .def("filter_count", &csr_filters::filter_count, "f"_a)
//        .def("point_count", &csr_filters::point_count, "p"_a)
//        .def("transpose", &csr_filters::transpose)
//        .def("transpose_inplace", &csr_filters::transpose_inplace);

//    // should have initializers taking either one or two int32_t arguments
//    py::class_<QueryFilter>(m, "QueryFilter")
//        .def(py::init<int32_t>(), "a"_a)
//        .def(py::init<int32_t, int32_t>(), "a"_a, "b"_a)
//        .def("is_and", &QueryFilter::is_and)
//        .def("__repr__", [](const QueryFilter &f) {
//            return "<QueryFilter: " + std::to_string(f.a) + ", " + std::to_string(f.b) + ">";
//        })
//        .def("__str__", [](const QueryFilter &f) {
//            return "(" + std::to_string(f.a) + ", " + std::to_string(f.b) + ")";
//        });
}