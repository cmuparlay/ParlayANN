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
    
    PR points;
    parlay::sequence<index_type> indices; // indices of points wrt the original dataset, important for returning results
    parlay::sequence<FilterType> filter_values; // filter value for each point
    parlay::sequence<FilterType> sorted_filter_values; // sorted filter values
    parlay::sequence<index_type> sorted_filter_indices; // indices wrt the subset sorted by filter value

    std::pair<FilterType, FilterType> range; // min and max filter values
    FilterType median; // median filter value

    bool has_children = false;

    // the below will have to be modified to be a variant if PR is not a subset PR at the top level
    // additionally, will have to consider how to handle the equivalent case in the real index, perhaps with a variant
    std::pair<std::unique_ptr<FlatRangeFilterIndex<T, Point, PR, FilterType>>, std::unique_ptr<FlatRangeFilterIndex<T, Point, PR, FilterType>>> children;

    FlatRangeFilterIndex() = default;
    
    /* This constructor should be used internally by the C++ layer */
    FlatRangeFilterIndex(const PR& points, const parlay::sequence<FilterType>& filter_values, int32_t cutoff = 1000)
        : points(points), filter_values(filter_values) {
        auto n = points.size();
        indices = parlay::tabulate(n, [](auto i) { return i; });
        sorted_filter_values = parlay::sequence<FilterType>(n);
        sorted_filter_indices = parlay::tabulate(n, [](auto i) { return i; });

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
        
        build_children_recursive(cutoff);
    }

    /* This constructor should be used by the Python layer */
    FlatRangeFilterIndex(py::array_t<T> points, py::array_t<FilterType> filter_values, int32_t cutoff = 1000)
        : FlatRangeFilterIndex(numpy_point_range<T, Point>(points), parlay::sequence<FilterType>(filter_values)) {}

    /* the bounds here are inclusive */
    NeighborsAndDistances batch_filter_search(py::array_t<T, py::array::c_style | py::array::forcecast>& queries,
     const std::vector<std::pair<FilterType, FilterType>>& filters, uint64_t num_queries,
     uint64_t knn) {
        py::array_t<unsigned int> ids({num_queries, knn});
        py::array_t<float> dists({num_queries, knn});

        parlay::parallel_for(0, num_queries, [&](auto i) {
            Point q = Point(queries.data(i), this->points.dimension(), 
                this->points.aligned_dimension(), 
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
            size_t end = points.size();

            // if the start of the query range could fall within the index range, find the first point that is >= the start of the query range
            if (range.first >= this->range.first) {
                size_t l = 0;
                size_t r = points.size();
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
            if (range.second <= this->range.second) {
                size_t l = start; // see no reason why this wouldn't be valid
                size_t r = points.size();
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
                auto d = points[j].distance(query);
                if (d < frontier[knn-1].second) {
                    frontier[knn-1] = std::make_pair(this->indices[j], d);
                    parlay::sort_inplace(frontier, [&](auto a, auto b) {
                        return a.second < b.second;
                    });
                }
            }
        } else {
            // recurse on the children
            auto [index1, index2] = this->children;
            parlay::sequence<pid> results1, results2;
            if (range.first <= median) {
                results1 = index1->orig_serial_query(query, range, knn);
            }
            if (range.second >= median) {
                results2 = index2->orig_serial_query(query, range, knn);
            }

            if (results1.size() == 0) {
                frontier = results2;
            } else if (results2.size() == 0) {
                frontier = results1;
            } else {
                // this is pretty lazy and inefficient
                frontier = parlay::merge(results1, results2, [&](auto a, auto b) {
                    return a.second < b.second;
                });
                if (frontier.size() > knn) {
                    frontier.resize(knn);
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

        auto n = points.size();
        auto n1 = n/2;
        auto n2 = n - n1;

        auto points1 = points.make_subset(parlay::map(parlay::make_slice(sorted_filter_indices.begin(), sorted_filter_indices.begin() + n1), [&](auto i) { return indices[i]; }));
        auto points2 = points.make_subset(parlay::map(parlay::make_slice(sorted_filter_indices.begin() + n1, sorted_filter_indices.end()), [&](auto i) { return indices[i]; }));

        auto filter_values1 = parlay::sequence<FilterType>(filter_values.begin(), filter_values.begin() + n1);
        auto filter_values2 = parlay::sequence<FilterType>(filter_values.begin() + n1, filter_values.end());

        auto index1 = std::make_unique<FlatRangeFilterIndex<T, Point, PR, FilterType>>(points1, filter_values1);
        auto index2 = std::make_unique<FlatRangeFilterIndex<T, Point, PR, FilterType>>(points2, filter_values2);

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

        if (points.size() <= cutoff * 2) {
            return;
        }

        build_children();

        parlay::par_do([&] {
            this->children.first->build_children_recursive(cutoff);
        }, [&] {
            this->children.second->build_children_recursive(cutoff);
        });
    }
};


// /* Perhaps stupidly, this is a recursive struct */
// template <typename T, typename Point, class PR = PointRange<T, Point>, typename FilterType = float_t>
// struct RangeFilterTreeIndex {
//     using pid = std::pair<index_type, float>;
//     using GraphI = Graph<index_type>;

//     PR points;
//     parlay::sequence<FilterType> sorted_filter_values;
//     parlay::sequence<index_type> sorted_filter_indices;

//     RangeFilterTreeIndex() = default;

//     RangeFilterTreeIndex(const PR& points, const parlay::sequence<FilterType>& filter_values)
//         : points(points) {
//         auto n = points.size();
//         sorted_filter_values = parlay::sequence<FilterType>(n);
//         sorted_filter_indices = parlay::tabulate(n, [](auto i) { return i; });

//         // argsort the filter values to get sorted indices
//         parlay::sort_inplace(sorted_filter_indices, [&](auto i, auto j) {
//             return filter_values[i] < filter_values[j];
//         });

//         // sort the filter values
//         parlay::parallel_for(0, n, [&](auto i) {
//             sorted_filter_values[i] = filter_values[sorted_filter_indices[i]];
//         });
//     }
// }
