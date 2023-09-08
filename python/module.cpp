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

#include "builder.h"
#include "vamana_index.h"


PYBIND11_MAKE_OPAQUE(std::vector<uint32_t>);
PYBIND11_MAKE_OPAQUE(std::vector<float>);
PYBIND11_MAKE_OPAQUE(std::vector<int8_t>);
PYBIND11_MAKE_OPAQUE(std::vector<uint8_t>);

namespace py = pybind11;
using namespace pybind11::literals;

struct Variant
{
    std::string vamana_builder_name;
    std::string vamana_index_name;
};

const Variant FloatVariant{"build_vamana_float_index", "VamanaFloatIndex"};

const Variant UInt8Variant{"build_vamana_uint8_index", "VamanaUint8Index"};

const Variant Int8Variant{"build_vamana_int8_index", "VamanaInt8Index"};

template <typename T> inline void add_variant(py::module_ &m, const Variant &variant)
{

    m.def(variant.builder_name.c_str(), build_vamana_index<T>, "distance_metric"_a,
          "data_file_path"_a, "index_output_path"_a, "graph_degree"_a, "beam_width"_a, "alpha"_a);

    py::class_<VamanaIndex<T>>(m, variant.vamana_index_name.c_str())
        .def(py::init<const std::string &, const std::string &, const size_t, const size_t>(),
             "distance_metric"_a, "index_path"_a, "num_points"_a, "dimensions"_a) //maybe these last two are unnecessary?
        .def("search", &VamanaIndex<T>::search, "query"_a, "knn"_a, "beam_width"_a) //do we want to add options like visited limit, or leave those as defaults?
        .def("batch_search", &VamanaIndex<T>::batch_search, "queries"_a, "num_queries"_a, "knn"_a,
             "beam_width"_a);

   
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
    default_values.attr("COMPLEXITY") = 128;

    add_variant<float>(m, FloatVariant);
    add_variant<uint8_t>(m, UInt8Variant);
    add_variant<int8_t>(m, Int8Variant);

}