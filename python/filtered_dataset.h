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

using T = int8_t;
using Point = Euclidian_Point<T>;
using index_type = int32_t;

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
};

#endif // FILTERED_DATASET_H