#pragma once

#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"
#include "parlay/sequence.h"
#include "parlay/monoid.h"
#include "parlay/internal/get_time.h"
#include "parlay/internal/binary_search.h"

#include "../utils/filters.h"
#include "../utils/graph.h"
#include "../utils/point_range.h"
#include "../utils/types.h"
#include "../utils/euclidian_point.h"
#include "../utils/mips_point.h"
#include "../utils/filteredBeamSearch.h"
#include "../utils/threadlocal.h"

#include "../vamana/index.h"

#include <algorithm>
#include <type_traits>
#include <unordered_map>
#include <limits>
#include <variant>
#include <vector>

#include "pybind11/numpy.h"
#include "pybind11/stl.h"

using index_type = int32_t;

namespace py = pybind11;
using NeighborsAndDistances =
   std::pair<py::array_t<unsigned int>, py::array_t<float>>;

template <typename T, typename Point>
PointRange<T, Point> numpy_point_range(py::array_t<T> points) {
    py::buffer_info buf = points.request();
    if (buf.ndim != 2) {
      throw std::runtime_error("NumPy array must be 2-dimensional");
    }

    auto n = buf.shape[0]; // number of points
    auto dims = buf.shape[1]; // dimension of each point

    T* numpy_data = static_cast<T*>(buf.ptr);

    return std::move(PointRange<T, Point>(numpy_data, n, dims));
}


/* A very basic range filter index that does exhaustive search

Meant to be the bottom child of a RangeFilterTreeIndex below some cutoff size */
template <typename T, typename Point, class PR = PointRange<T, Point>, typename FilterType = float_t>
struct FlatRangeFilterIndex {
    using pid = std::pair<index_type, float>;
    
    // TODO: make points a unique_ptr to a PR. Currently stack allocating like a moron.
    std::unique_ptr<PR> points;
    
    parlay::sequence<index_type> indices; // indices of points wrt the original dataset, important for returning results
    parlay::sequence<FilterType> filter_values; // filter value for each point
    parlay::sequence<FilterType> sorted_filter_values; // sorted filter values
    parlay::sequence<index_type> sorted_filter_indices; // indices wrt the subset sorted by filter value

    std::pair<FilterType, FilterType> range; // min and max filter values
    FilterType median; // median filter value

    bool has_children = false;

    int32_t cutoff = 1000;

    // the below will have to be modified to be a variant if PR is not a subset PR at the top level
    // additionally, will have to consider how to handle the equivalent case in the real index, perhaps with a variant
    std::pair<std::unique_ptr<FlatRangeFilterIndex<T, Point, SubsetPointRange<T, Point>, FilterType>>, std::unique_ptr<FlatRangeFilterIndex<T, Point, SubsetPointRange<T, Point>, FilterType>>> children;

    // FlatRangeFilterIndex();

    /* an interface to support std::move-ing unique ptrs to the point range directly */
    FlatRangeFilterIndex(std::unique_ptr<PR> points, const parlay::sequence<FilterType>& filter_values, int32_t cutoff = 1000, bool recurse = true)
        : points(std::move(points)), filter_values(filter_values), cutoff(cutoff) {
        auto n = this->points->size();
        if constexpr (std::is_same_v<PR, PointRange<T, Point>>) {
            indices = parlay::tabulate(n, [](int32_t i) { return i; });
        } else {
            indices = parlay::sequence<index_type>(this->points->subset);
        }

        // std::cout<<"Building index with "<<n<<" points"<<std::endl;
        
        sorted_filter_values = parlay::sequence<FilterType>(n);
        sorted_filter_indices = parlay::tabulate(n, [](index_type i) { return i; });

        // argsort the filter values to get sorted indices
        parlay::sort_inplace(sorted_filter_indices, [&](auto i, auto j) {
            return filter_values[i] < filter_values[j];
        });

        // sort the filter values
        parlay::parallel_for(0, n, [&](auto i) {
            sorted_filter_values[i] = filter_values[sorted_filter_indices[i]];
        });

        // get the min and max filter values
        range = std::make_pair(sorted_filter_values[0], sorted_filter_values[n-1]);

        // get the median filter value
        median = sorted_filter_values[n/2];
        
        if (recurse) {
            build_children_recursive(cutoff);
            auto [a, b] = validate_children();
            std::cout << "Number of leaves: " << a << std::endl;
            std::cout << "Number of leaves with valid point ranges: " << b << std::endl;
        }

    }

    /* This constructor should be used internally by the C++ layer 
    */
    FlatRangeFilterIndex(const PR& points, const parlay::sequence<FilterType>& filter_values, int32_t cutoff = 1000, bool recurse = true) {
        auto unique_points = std::make_unique<PR>(points);
        *this = FlatRangeFilterIndex<T, Point, PR, FilterType>(std::move(unique_points), filter_values, cutoff, recurse);
    }

    /* This constructor should be used by the Python layer */
    FlatRangeFilterIndex(py::array_t<T> points, py::array_t<FilterType> filter_values, int32_t cutoff = 1000) {
        py::buffer_info points_buf = points.request();
        if (points_buf.ndim != 2) {
            throw std::runtime_error("points NumPy array must be 2-dimensional");
        }
        auto n = points_buf.shape[0]; // number of points
        auto dims = points_buf.shape[1]; // dimension of each point

        // avoiding this copy may have dire consequences from gc
        T* numpy_data = static_cast<T*>(points_buf.ptr);

        PointRange<T, Point> point_range = PointRange<T, Point>(numpy_data, n, dims);

        py::buffer_info filter_values_buf = filter_values.request();
        if (filter_values_buf.ndim != 1) {
            throw std::runtime_error("filter data NumPy array must be 1-dimensional");
        }

        if (filter_values_buf.shape[0] != n) {
            throw std::runtime_error("filter data NumPy array must have the same number of elements as the points array");
        }

        FilterType* filter_values_data = static_cast<FilterType*>(filter_values_buf.ptr);

        parlay::sequence<FilterType> filter_values_seq = parlay::sequence<FilterType>(filter_values_data, filter_values_data + n);

        *this = FlatRangeFilterIndex<T, Point, PR, FilterType>(point_range, filter_values_seq, cutoff, true);
    }

    /* the bounds here are inclusive */
    NeighborsAndDistances batch_filter_search(py::array_t<T, py::array::c_style | py::array::forcecast>& queries,
     const std::vector<std::pair<FilterType, FilterType>>& filters, uint64_t num_queries,
     uint64_t knn) {
        py::array_t<unsigned int> ids({num_queries, knn});
        py::array_t<float> dists({num_queries, knn});

        parlay::parallel_for(0, num_queries, [&](auto i) {
            Point q = Point(queries.data(i), this->points->dimension(), 
                this->points->aligned_dimension(), 
                i);
            std::pair<FilterType, FilterType> filter = filters[i];

            auto results = orig_serial_query(q, filter, knn);

            for (auto j = 0; j < knn; j++) {
                ids.mutable_at(i, j) = results[j].first;
                dists.mutable_at(i, j) = results[j].second;
            }
        });
        return std::make_pair(ids, dists);
     }
    
// private:
    /* This should do exhaustive search on every subrange contained entirely within the query range, and preprocess leaves. */
    parlay::sequence<pid> orig_serial_query(const Point& query, const std::pair<FilterType, FilterType>& range, uint64_t knn) {
        // if the query range is entirely outside the index range, return
        if (range.second < this->range.first || range.first > this->range.second) {
            std::cout << "Query range is entirely outside the index range" << std::endl;
            return parlay::sequence<pid>();
        }

        parlay::sequence<pid> frontier;

        // if there are no children, search the elements within the target range
        if (!has_children) {
            // identify the points that are within the filter range 
            size_t start = 0;
            size_t end = points->size();

            // if the start of the query range could fall within the index range, find the first point that is >= the start of the query range
            if (range.first > this->range.first) {
                size_t l = 0;
                size_t r = points->size();
                // should make sure this is correct
                while (l < r) {
                    size_t m = (l + r) / 2;
                    if (sorted_filter_values[m] < range.first) {
                        l = m + 1;
                    } else {
                        r = m;
                    }
                }
                start = l;
            }

            // ditto for the end of the query range
            if (range.second < this->range.second) {
                size_t l = start; // see no reason why this wouldn't be valid
                size_t r = points->size();
                // correctness of below is similarly suspect
                while (l < r) {
                    size_t m = (l + r) / 2;
                    if (sorted_filter_values[m] <= range.second) {
                        l = m + 1;
                    } else {
                        r = m;
                    }
                }
                end = l;
            }

            // if the start and end are the same, there are no points within the query range
            if (start == end) {
                return parlay::sequence<pid>();
            }

            // otherwise, return the k nearest points within the query range
            frontier = parlay::sequence<pid>(knn, std::make_pair(-1, std::numeric_limits<FilterType>::max()));
            // i here is indexing into the sorted values
            for (auto i = start; i < end; i++) {
                size_t j = sorted_filter_indices[i]; // local index of the point (wrt subset)
                
                auto d = (*points)[j].distance(query);
                if (d < frontier[knn-1].second) {
                    frontier[knn-1] = std::make_pair(this->indices[j], d);
                    parlay::sort_inplace(frontier, [&](auto a, auto b) {
                        return a.second < b.second;
                    });
                }
            }
        } else {
            // recurse on the children
            auto& [index1, index2] = this->children;
            parlay::sequence<pid> results1, results2;
            if (range.first <= median) {
                results1 = index1->orig_serial_query(query, range, knn);
            }
            if (range.second >= median) {
                results2 = index2->orig_serial_query(query, range, knn);
            }

            if (range.first > median) {
                frontier = results2;
            } else if (range.second < median) {
                frontier = results1;
            } else {
                // this is pretty lazy and inefficient
                // frontier = parlay::merge(results1, results2, [&](auto a, auto b) {
                //     return a.second < b.second;
                // });
                frontier = results1;
                for (pid p : results2) {
                    frontier.push_back(p);
                }
                parlay::sort_inplace(frontier, [&](auto a, auto b) {
                    return a.second < b.second;
                });

                if (frontier.size() > knn) {
                    // resize is probably the right thing here but not terribly well documented
                    frontier.pop_tail(frontier.size() - knn);
                }
            }
        }
        

        return frontier;
    }

    void build_children() {
        if (has_children) {
            throw std::runtime_error("Cannot build children of a node that already has children");
            return;
        }

        auto n = points->size();
        auto n1 = n/2;
        auto n2 = n - n1;

        std::unique_ptr<SubsetPointRange<T, Point>> points1 = points->make_subset(parlay::map(parlay::make_slice(sorted_filter_indices.begin(), sorted_filter_indices.begin() + n1), [&](auto i) { return indices[i]; }));
        std::unique_ptr<SubsetPointRange<T, Point>> points2 = points->make_subset(parlay::map(parlay::make_slice(sorted_filter_indices.begin() + n1, sorted_filter_indices.end()), [&](auto i) { return indices[i]; }));

        auto filter_values1 = parlay::sequence<FilterType>(sorted_filter_values.begin(), sorted_filter_values.begin() + n1);
        auto filter_values2 = parlay::sequence<FilterType>(sorted_filter_values.begin() + n1, sorted_filter_values.end());

        auto index1 = std::make_unique<FlatRangeFilterIndex<T, Point, SubsetPointRange<T, Point>, FilterType>>(std::move(points1), filter_values1, this->cutoff, false);
        auto index2 = std::make_unique<FlatRangeFilterIndex<T, Point, SubsetPointRange<T, Point>, FilterType>>(std::move(points2), filter_values2, this->cutoff, false);

        this->children = std::make_pair(
            std::move(index1),
            std::move(index2)
        );

        has_children = true;
    }

    /* intuitively, cutoff should be the size of the smallest possible index */
    void build_children_recursive(size_t cutoff) {
        if (has_children) {
            throw std::runtime_error("Cannot build children of a node that already has children");
            return;
        }

        if (points->size() <= cutoff * 2) {
            return;
        }

        build_children();

        parlay::par_do([&] {
            this->children.first->build_children_recursive(cutoff);
        }, [&] {
            this->children.second->build_children_recursive(cutoff);
        });
    }

    /* This method counts the leaves of the tree and how many of them have a subset point range which points to a point range with the same dimension (and is therefore probably not junk values) */
    std::pair<int, int> validate_children() {
        if (has_children){
            auto [a1, b1] = children.first->validate_children();
            auto [a2, b2] = children.second->validate_children();
            return std::make_pair(a1 + a2, b1 + b2);
        }
        
        if constexpr (std::is_same_v<PR, PointRange<T, Point>>) {
            return std::make_pair(1, 1);
        } else {
            if (points->dims == points->pr->dimension()) {
                return std::make_pair(1, 1);
            } else {
                return std::make_pair(1, 0);
            }
        }
    }
};


