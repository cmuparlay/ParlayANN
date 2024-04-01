/* A simple interface for a filtered dataset */
#ifndef FILTERED_DATASET_H
#define FILTERED_DATASET_H

#include "../algorithms/utils/filters.h"
#include "../algorithms/utils/point_range.h"
#include "../algorithms/utils/euclidian_point.h"
#include "../algorithms/utils/mips_point.h"

#include "parlay/sequence.h"

#include <vector>
#include <string>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "pybind11/numpy.h"
#include "pybind11/stl.h"


namespace py = pybind11;
using NeighborsAndDistances = std::pair<py::array_t<unsigned int>, py::array_t<float>>;

// using T = float;
// using Point = Mips_Point<T>;
using index_type = int32_t;


template <typename T = float, typename Point = Mips_Point<float>>
struct FilteredDataset {
    PointRange<T, Point> points;
    csr_filters filters;
    csr_filters transpose_filters;

    FilteredDataset(std::string points_filename, std::string filters_filename) {
        points = PointRange<T, Point>(points_filename.data());
        filters = csr_filters(filters_filename);
        transpose_filters = filters.transpose();
    }

    /* gives the euclidean distance between two points */
    float distance(index_type a, index_type b) {
        Point p1 = points[a];
        Point p2 = points[b];
        return p1.distance(p2);
    }

    size_t size() {
        return points.size();
    }

    size_t get_n_filters() {
        return filters.n_filters;
    }

    size_t get_filter_size(index_type filter_id) {
        return transpose_filters.point_count(filter_id);
    }

    size_t get_point_size(index_type point_id) {
        return filters.point_count(point_id);
    }

    size_t get_dim() {
        return points.dimension();
    }

    /* returns a numpy array of the ids of points associated with a filter */
    py::array_t<index_type> get_filter_points(index_type filter_id) {
        size_t filter_size = get_filter_size(filter_id);
        parlay::sequence<index_type> filter_points = transpose_filters.point_filters(filter_id);
        
        return py::array_t<index_type>(filter_size, filter_points.data());
    }

    /* returns a numpy array of the ids of filters associated with a point */
    py::array_t<index_type> get_point_filters(index_type point_id) {
        size_t point_size = get_point_size(point_id);
        parlay::sequence<index_type> filter_points = filters.point_filters(point_id);
        
        return py::array_t<index_type>(point_size, filter_points.data());
    }

    /* returns a numpy array of the ids of points present in the intersection of two filters */
    py::array_t<index_type> get_filter_intersection(index_type filter_id_1, index_type filter_id_2) {
        parlay::sequence<index_type> intersection = transpose_filters.point_intersection(filter_id_1, filter_id_2);
        
        return py::array_t<index_type>(intersection.size(), intersection.data());
    }

    /* returns a numpy array of the filters shared by two points */
    py::array_t<index_type> get_point_intersection(index_type point_id_1, index_type point_id_2) {
        parlay::sequence<index_type> intersection = filters.point_intersection(point_id_1, point_id_2);
        
        return py::array_t<index_type>(intersection.size(), intersection.data());
    }

    /* Writes the points to a file in fvec format

    Writes the points as int32_t because CAPS only supports int32_t and float32_t
    
    fvec format:
        <dim> <vector> <dim> <vector> ...
    */
    void write_fvec(std::string filename) {
        auto outfile = fopen(filename.data(), "w");
        int32_t dim = points.dimension();
        std::unique_ptr<int32_t[]> point_buffer(new int32_t[dim]);

        for (size_t i=0; i < points.size(); i++) {
            T* p = points[i].get();
            // write dim
            fwrite(&dim, sizeof(int32_t), 1, outfile);
            // cast vector to int32_t
            for (size_t j=0; j < dim; j++) {
                point_buffer[j] = (int32_t) p[j];
            }
            // write vector
            fwrite(point_buffer.get(), sizeof(int32_t), dim, outfile);
        }
    }

    /* Writes labels to the .txt format used by CAPS */
    void write_labels(std::string filename) {
        auto outfile = fopen(filename.data(), "w");
        for (size_t i=0; i < points.size(); i++) {
            fprintf(outfile, "%d ", points[i].id());
        }
    }

    /* Interprets the provided dataset as a set of queries, and returns a pair of 2d numpy arrays
    describing the indices of the k-NN for each point and their distances respectively */
    NeighborsAndDistances filtered_groundtruth(FilteredDataset &queries, size_t k = 100) {
        // construct the output arrays
        py::array_t<index_type> ids({queries.size(), k});
        py::array_t<float> dists({queries.size(), k});

        std::cout << queries.size() << std::endl;

        parlay::parallel_for(0, queries.size(), [&](size_t i) {
            // get the query point
            Point query = queries.points[i];
            // get the filter associated with the query point
            parlay::sequence<index_type> query_filters = queries.filters.point_filters(i);

            parlay::sequence<index_type> relevant_points;

            // get the points associated with the filter
            if (query_filters.size() == 0) {
                throw std::runtime_error("Query point has no associated filters");
            } else if (query_filters.size() == 1) {
                // if the query point has only one associated filter, use the filter to find the k-NN
                relevant_points = transpose_filters.point_filters(query_filters[0]);
            } else {
                // if the query point has multiple associated filters, use the intersection of the filters to find the k-NN
                relevant_points = transpose_filters.point_intersection(query_filters[0], query_filters[1]);
            }
            
            // get the k-NN
            parlay::sequence<std::pair<index_type, float>> knn = parlay::tabulate(k, [&](size_t j) {
                index_type point_id = relevant_points[j];
                float dist = query.distance(points[point_id]);
                return std::make_pair(point_id, dist);
            });

            // sort the k-NN by distance
            parlay::sort_inplace(knn, [](auto a, auto b) {
                return a.second < b.second;
            });

            // extract the indices and distances
            for (size_t j=0; j < k; j++) {
                ids.mutable_data(i)[j] = knn[j].first;
                dists.mutable_data(i)[j] = knn[j].second;
            }
        });

        return std::make_pair(std::move(ids), std::move(dists));
    }
};

#endif // FILTERED_DATASET_H