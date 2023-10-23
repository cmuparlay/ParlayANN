/* An IVF index */
#ifndef IVF_H
#define IVF_H

#include "parlay/internal/get_time.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"

#include "../HCNNG/clusterEdge.h"
#include "../utils/point_range.h"
#include "../utils/types.h"
#include "clustering.h"
#include "posting_list.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <utility>
#include <vector>

#include "pybind11/numpy.h"
#include "pybind11/stl.h"

namespace py = pybind11;
using NeighborsAndDistances =
   std::pair<py::array_t<unsigned int>, py::array_t<float>>;

/* A reasonably extensible ivf index */
template <typename T, typename Point, typename PostingList>
struct IVFIndex {
  PointRange<T, Point> points;
  parlay::sequence<PostingList> posting_lists = parlay::sequence<PostingList>();
  parlay::sequence<Point> centroids = parlay::sequence<Point>();
  size_t dim;
  size_t aligned_dim;

  IVFIndex() {
    std::cout << "===Running IVF Index" << std::endl;
  }

  // IVFIndex(PointRange<T, Point> points) : points(points) {}

  virtual void fit(PointRange<T, Point> points, size_t cluster_size = 1000) {
    auto timer = parlay::internal::timer();
    timer.start();

    // cluster the points
    auto clusterer =
       HCNNGClusterer<Point, PointRange<T, Point>, index_type>(cluster_size);

    std::cout << "clusterer initialized (" << timer.next_time() << "s)"
              << std::endl;

    parlay::sequence<parlay::sequence<index_type>> clusters =
       clusterer.cluster(points);

    std::cout << "clusters generated" << std::endl;

    // check if there are indices in the clusters that are too large
    // for (size_t i=0; i<clusters.size(); i++){
    //     for (size_t j=0; j<clusters[i].size(); j++){
    //         if (clusters[i][j] >= points.size()){
    //             std::cout << "IVFIndex::fit: clusters[" << i << "][" << j <<
    //             "] = " << clusters[i][j] << std::endl;
    //         }
    //     }
    // }

    // generate the posting lists
    posting_lists = parlay::tabulate(clusters.size(), [&](size_t i) {
      return PostingList(points, clusters[i]);
    });

    std::cout << "IVF: posting lists generated" << std::endl;

    // check if there are indices in the posting lists that are too large
    // for (size_t i=0; i<posting_lists.size(); i++){
    //     for (size_t j=0; j<posting_lists[i].indices.size(); j++){
    //         if (posting_lists[i].indices[j] >= points.size()){
    //             std::cout << "IVFIndex::fit: posting_lists[" << i <<
    //             "].indices[" << j << "] = " << posting_lists[i].indices[j] <<
    //             std::endl;
    //         }
    //     }
    // }

    centroids = parlay::map(posting_lists,
                            [&](PostingList pl) { return pl.centroid(); });
    // serially for debug
    // centroids = parlay::sequence<Point>();
    // for (size_t i=0; i<posting_lists.size(); i++){
    //     centroids.push_back(posting_lists[i].centroid());
    // }

    std::cout << "IVF: centroids generated" << std::endl;

    dim = points.dimension();
    aligned_dim = points.aligned_dimension();

    std::cout << "IVF: fit completed" << std::endl;
  }

  void fit_from_filename(std::string filename, size_t cluster_size = 1000) {
    PointRange<T, Point> points(filename.c_str());
    std::cout << "IVF: points loaded" << std::endl;
    this->fit(points, cluster_size);
  }

  /* A utility function to do nearest k centroids in linear time */
  parlay::sequence<unsigned int> nearest_centroids(Point query,
                                                   unsigned int n) {
    parlay::sequence<std::pair<unsigned int, float>> nearest_centroids =
       parlay::tabulate(n, [&](unsigned int i) {
         return std::make_pair(std::numeric_limits<unsigned int>().max(),
                               std::numeric_limits<float>().max());
       });
    for (unsigned int i = 0; i < n; i++) {
      float dist = query.distance(centroids[i]);
      if (dist < nearest_centroids[n - 1].second) {
        nearest_centroids[n - 1] = std::make_pair(i, dist);
        std::sort(nearest_centroids.begin(), nearest_centroids.end(),
                  [&](std::pair<unsigned int, float> a,
                      std::pair<unsigned int, float> b) {
                    return a.second < b.second;
                  });
      }
    }
    return parlay::map(
       nearest_centroids,
       [&](std::pair<unsigned int, float> p) { return p.first; });
  }

  NeighborsAndDistances batch_search(
     py::array_t<T, py::array::c_style | py::array::forcecast>& queries,
     uint64_t num_queries, uint64_t knn, uint64_t n_lists) {
    py::array_t<unsigned int> ids({num_queries, knn});
    py::array_t<float> dists({num_queries, knn});

    parlay::parallel_for(0, num_queries, [&](unsigned int i) {
      Point q = Point(queries.data(i), dim, dim, i);
      parlay::sequence<unsigned int> nearest_centroid_ids =
         nearest_centroids(q, n_lists);

      parlay::sequence<std::pair<unsigned int, float>> frontier =
         parlay::tabulate(knn, [&](unsigned int i) {
           return std::make_pair(std::numeric_limits<unsigned int>().max(),
                                 std::numeric_limits<float>().max());
         });

      for (unsigned int j = 0; j < nearest_centroid_ids.size(); j++) {
        posting_lists[nearest_centroid_ids[j]].query(q, knn, frontier);
      }

      // this sort should be redundant
      // std::sort(frontier.begin(), frontier.end(), [&] (std::pair<unsigned
      // int, float> a, std::pair<unsigned int, float> b) {
      //     return a.second < b.second;
      // });
      for (unsigned int j = 0; j < knn; j++) {
        ids.mutable_data(i)[j] = frontier[j].first;
        dists.mutable_data(i)[j] = frontier[j].second;
      }
    });

    // std::cout << "parfor done" << std::endl;

    return std::make_pair(std::move(ids), std::move(dists));
  }

  void print_stats() {
    size_t total = 0;
    size_t max = 0;
    size_t min = std::numeric_limits<size_t>().max();
    for (size_t i = 0; i < posting_lists.size(); i++) {
      size_t s = posting_lists[i].indices.size();
      total += s;
      if (s > max)
        max = s;
      if (s < min)
        min = s;
    }
    std::cout << "Total number of points: " << total << std::endl;
    std::cout << "Number of posting lists: " << posting_lists.size()
              << std::endl;
    std::cout << "Average number of points per list: "
              << total / posting_lists.size() << std::endl;
    std::cout << "Max number of points in a list: " << max << std::endl;
    std::cout << "Min number of points in a list: " << min << std::endl;
  }
};

template <typename T, typename Point, typename PostingList>
struct FilteredIVFIndex : IVFIndex<T, Point, PostingList> {
  csr_filters filters;

  FilteredIVFIndex() {
    std::cout << "===Running FilteredIVFIndex" << std::endl;
  }

  void fit(PointRange<T, Point> points, csr_filters& filters,
           size_t cluster_size = 1000) {
    auto timer = parlay::internal::timer();
    timer.start();

    this->filters = filters;

    this->filters.print_stats();

    std::cout << this->filters.first_label(42) << std::endl;

    // transpose the filters
    // this->filters.transpose_inplace();

    this->filters.print_stats();

    // std::cout << this->filters.first_label(6) << std::endl;

    // std::cout << "FilteredIVF: filters transposed (" << timer.next_time() <<
    // "s)" << std::endl;

    // cluster the points
    auto clusterer =
       HCNNGClusterer<Point, PointRange<T, Point>, index_type>(cluster_size);

    std::cout << "FilteredIVF: clusterer initialized (" << timer.next_time()
              << "s)" << std::endl;

    parlay::sequence<parlay::sequence<index_type>> clusters =
       clusterer.cluster(points);

    std::cout << "FilteredIVF: clusters generated (" << timer.next_time()
              << "s)" << std::endl;

    // we sort the clusters to facilitate the generation of subset filters
    parlay::parallel_for(0, clusters.size(), [&](size_t i) {
      std::sort(clusters[i].begin(), clusters[i].end());
    });

    std::cout << "FilteredIVF: clusters sorted (" << timer.next_time() << "s)"
              << std::endl;

    // generate the posting lists
    this->posting_lists = parlay::tabulate(clusters.size(), [&](size_t i) {
      return PostingList(points, clusters[i], this->filters);
    });

    std::cout << "FilteredIVF: posting lists generated (" << timer.next_time()
              << "s)" << std::endl;

    this->centroids = parlay::map(
       this->posting_lists, [&](PostingList pl) { return pl.centroid(); });

    std::cout << "FilteredIVF: centroids generated (" << timer.next_time()
              << "s)" << std::endl;

    this->dim = points.dimension();
    this->aligned_dim = points.aligned_dimension();

    this->points = points;
  }

  void fit_from_filename(std::string filename, std::string filter_filename,
                         size_t cluster_size = 1000) {
    PointRange<T, Point> points(filename.c_str());
    std::cout << "FilteredIVF: points loaded" << std::endl;
    csr_filters filters(filter_filename.c_str());
    std::cout << "FilteredIVF: filters loaded" << std::endl;
    this->fit(points, filters, cluster_size);
    std::cout << "FilteredIVF: fit completed" << std::endl;
  }

  /* The use of vector here is because that supposedly allows us to take python
  lists as input, although I'll believe it when I see it.

  This is incredibly easy, but might be slower than parsing the sparse array of
  filters in C++.  */
  NeighborsAndDistances batch_filter_search(
     py::array_t<T, py::array::c_style | py::array::forcecast>& queries,
     const std::vector<QueryFilter>& filters, uint64_t num_queries,
     uint64_t knn, uint64_t n_lists) {
    py::array_t<unsigned int> ids({num_queries, knn});
    py::array_t<float> dists({num_queries, knn});

    // TODO(blandrum): does using a granularity of 1 in the following
    // parallel-for help? It might if the number of queries in the
    // batch is small.
    parlay::parallel_for(0, num_queries, [&](unsigned int i) {
      Point q = Point(queries.data(i), this->dim, this->dim, i);
      const QueryFilter& filter = filters[i];
      parlay::sequence<unsigned int> nearest_centroid_ids =
         this->nearest_centroids(q, n_lists);

      // maybe should be sequential? a memcopy?
      parlay::sequence<std::pair<unsigned int, float>> frontier =
         parlay::tabulate(knn, [&](unsigned int i) {
           return std::make_pair(std::numeric_limits<unsigned int>().max(),
                                 std::numeric_limits<float>().max());
         });

      for (unsigned int j = 0; j < nearest_centroid_ids.size(); j++) {
        this->posting_lists[nearest_centroid_ids[j]].filtered_query(
           q, filter, knn, frontier);
      }

      for (unsigned int j = 0; j < knn; j++) {
        ids.mutable_data(i)[j] = frontier[j].first;
        dists.mutable_data(i)[j] = frontier[j].second;
      }
    });

    // std::cout << "parfor done" << std::endl;

    return std::make_pair(std::move(ids), std::move(dists));
  }
};

/* A wrapper around FilteredIVFIndex that intercepts queries with low frequency

    FilteredIVFIndex should probably be parameterized here, but I'm trying to
   get away from automatically making everything more extensible than it needs
   to be.
*/
template <typename T, typename Point, typename PostingList>
struct FilteredIVF2Stage {
  FilteredIVFIndex<T, Point, PostingList> index;
  csr_filters filters;
  std::unique_ptr<int32_t[]>
     filter_counts;   // can be cheaply computed with transposed filters

  FilteredIVF2Stage() {
    std::cout << "===Running FilteredIVF2Stage" << std::endl;
  }

  void fit(PointRange<T, Point> points, csr_filters& filters,
           size_t cluster_size = 1000) {
    this->filters = filters.transpose();

    filter_counts = std::make_unique<int32_t[]>(this->filters.n_points);
    parlay::parallel_for(0, this->filters.n_points, [&](size_t i) {
      this->filter_counts[i] = this->filters.point_count(i);
    });

    // should we even bother filtering out the low frequency filters?
    // They represent a pretty small fraction of the total number of
    // associations. at least for the time being I won't bother
    this->index.fit(points, filters, cluster_size);
  }

  void fit_from_filename(std::string filename, std::string filter_filename,
                         size_t cluster_size = 1000) {
    PointRange<T, Point> points(filename.c_str());
    std::cout << "FilteredIVF: points loaded" << std::endl;
    csr_filters filters(filter_filename.c_str());
    std::cout << "FilteredIVF: filters loaded" << std::endl;
    this->fit(points, filters, cluster_size);
    std::cout << "FilteredIVF: fit completed" << std::endl;
  }

  NeighborsAndDistances batch_filter_search(
     py::array_t<T, py::array::c_style | py::array::forcecast>& queries,
     const std::vector<QueryFilter>& filters, uint64_t num_queries,
     uint64_t knn, uint64_t n_lists, uint64_t threshold) {
    py::array_t<unsigned int> ids({num_queries, knn});
    py::array_t<float> dists({num_queries, knn});

#ifdef VERBOSE

    std::unique_ptr<double[]> times = std::make_unique<double[]>(num_queries);
    std::unique_ptr<double[]> n_matches =
       std::make_unique<double[]>(num_queries);

#endif

    parlay::parallel_for(0, num_queries, [&](unsigned int i) {
    // for (unsigned int i=0; i<num_queries; i++) {
#ifdef VERBOSE
      auto timer = parlay::internal::timer();
      timer.start();

#endif

      Point q =
         Point(queries.data(i), this->index.dim, this->index.aligned_dim, i);
      const QueryFilter& filter = filters[i];

      parlay::sequence<std::pair<unsigned int, float>> frontier =
         parlay::tabulate(knn, [&](unsigned int i) {
           return std::make_pair(std::numeric_limits<unsigned int>().max(),
                                 std::numeric_limits<float>().max());
         });

      // check if the filter is too small
      double proj_matches = static_cast<double>(this->filter_counts[filter.a]);
      if (filter.is_and()) {
        proj_matches *= static_cast<double>(this->filter_counts[filter.b]) /
                        static_cast<double>(this->filters.n_points);
      }

#ifdef VERBOSE

      n_matches[i] = proj_matches;

#endif

      // std::cout << "proj_matches: " << proj_matches << std::endl;

      // we may want to multiply the value by some constant to reflect that the
      // query pairs tend to have disproportionately high overlap empirically,
      // this constant seems to be around 1.2 for the training queries
      if (proj_matches <
          threshold) {   // if the filter is too small, just do a linear search
        parlay::sequence<int32_t> matches;
        if (filter.is_and()) {
          matches =
             join(this->filters.row_indices.get() +
                     this->filters.row_offsets[filter.a],
                  this->filters.row_offsets[filter.a + 1] -
                     this->filters.row_offsets[filter.a],
                  this->filters.row_indices.get() +
                     this->filters.row_offsets[filter.b],
                  this->filters.row_offsets[filter.b + 1] -
                     this->filters
                        .row_offsets[filter.b]);   // indices of matching points

          // std::cout << "join matches: ";
          // for (size_t j=0; j<10; j++){
          //     std::cout << matches[j] << " ";
          // }
          // std::cout << std::endl;
        } else {
          matches = parlay::to_sequence(
             parlay::make_slice(this->filters.row_indices.get() +
                                   this->filters.row_offsets[filter.a],
                                this->filters.row_indices.get() +
                                   this->filters.row_offsets[filter.a + 1]));

          // std::cout << "non-join matches: ";
          // for (size_t j=0; j<10; j++){
          //     std::cout << matches[j] << " ";
          // }
          // std::cout << std::endl;
        }

        // printing matches for debug
        // std::cout << "matches: " << matches.size() << std::endl;
        // for (size_t j=0; j<10; j++){
        //     std::cout << matches[j] << " ";
        // }
        // std::cout << std::endl;

        // std::cout << "max match: " << parlay::reduce(matches,
        // parlay::maxm<int32_t>()) << std::endl; std::cout << "min match: " <<
        // parlay::reduce(matches, parlay::minm<int32_t>()) << std::endl;

        for (unsigned int j = 0; j < matches.size(); j++) {   // for each match
          float dist = this->index.points[matches[j]].distance(
             q);   // compute the distance to query
          // these steps would be very slightly faster if reordered
          if (dist <
              frontier[knn - 1].second) {   // if it's closer than the furthest
                                            // point in the frontier
            frontier.push_back(
               std::make_pair(matches[j], dist));   // add it to the frontier
            std::sort(frontier.begin(), frontier.end(),
                      [&](std::pair<unsigned int, float> a,
                          std::pair<unsigned int, float> b) {
                        return a.second < b.second;
                      });          // sort the frontier
            frontier.pop_back();   // remove the furthest point
          }
        }
      } else {   // normal filtered ivf search
        parlay::sequence<unsigned int> nearest_centroid_ids =
           this->index.nearest_centroids(q, n_lists);

        for (unsigned int j = 0; j < nearest_centroid_ids.size(); j++) {
          this->index.posting_lists[nearest_centroid_ids[j]].filtered_query(
             q, filter, knn, frontier);
        }
      }

      for (unsigned int j = 0; j < knn; j++) {
        ids.mutable_data(i)[j] = frontier[j].first;
        dists.mutable_data(i)[j] = frontier[j].second;
      }

#ifdef VERBOSE

      double time = timer.next_time();   // doing this first in case false
                                         // sharing makes the below line slower
      times[i] = time;

#endif
      // }
    });

#ifdef VERBOSE

    int32_t subthreshold = parlay::count_if(
       parlay::make_slice(n_matches.get(), n_matches.get() + num_queries),
       [&](double x) { return x < threshold; });
    double total_subthreshold_time = 0;
    double total_time = 0;
    for (int32_t i = 0; i < num_queries; i++) {
      if (n_matches[i] < threshold) {
        total_subthreshold_time += times[i];
      }
      total_time += times[i];
    }

    double total_proj_matches = parlay::reduce(
       parlay::make_slice(n_matches.get(), n_matches.get() + num_queries));

    double fifth_percentile_super_time = parlay::kth_smallest_copy(
       parlay::filter(
          parlay::make_slice(times.get(), times.get() + num_queries),
          [&](double x) { return x >= threshold; }),
       num_queries / 20);
    double fifth_percentile_sub_time = parlay::kth_smallest_copy(
       parlay::filter(
          parlay::make_slice(times.get(), times.get() + num_queries),
          [&](double x) { return x < threshold; }),
       subthreshold / 20);

    double ninetyfifth_percentile_super_time = parlay::kth_smallest_copy(
       parlay::filter(
          parlay::make_slice(times.get(), times.get() + num_queries),
          [&](double x) { return x >= threshold; }),
       num_queries / 20, [](double a, double b) { return a > b; });
    double ninetyfifth_percentile_sub_time = parlay::kth_smallest_copy(
       parlay::filter(
          parlay::make_slice(times.get(), times.get() + num_queries),
          [&](double x) { return x < threshold; }),
       subthreshold / 20, [](double a, double b) { return a > b; });

    double median_freq = parlay::kth_smallest_copy(
       parlay::make_slice(n_matches.get(), n_matches.get() + num_queries),
       num_queries / 2);
    double median_time = parlay::kth_smallest_copy(
       parlay::make_slice(times.get(), times.get() + num_queries),
       num_queries / 2);

    std::cout << "Total time: " << total_time << std::endl;
    std::cout << "Queries with projected frequency below threshold: "
              << subthreshold << " ("
              << 100 * static_cast<double>(subthreshold) / num_queries << "%)"
              << std::endl;
    std::cout << "Total time spent on subthreshold queries: "
              << total_subthreshold_time << " ("
              << 100 * total_subthreshold_time / total_time << "%)"
              << std::endl;

    std::cout << "Mean time per query: " << total_time / num_queries
              << std::endl;
    std::cout << "Mean time per subthreshold query: "
              << total_subthreshold_time / subthreshold << std::endl;
    std::cout << "Mean time per superthreshold query: "
              << (total_time - total_subthreshold_time) /
                    (num_queries - subthreshold)
              << std::endl;
    std::cout << "Subthreshold queries are "
              << (total_subthreshold_time / subthreshold) /
                    ((total_time - total_subthreshold_time) /
                     (num_queries - subthreshold))
              << " times slower" << std::endl;

    std::cout << "5th percentile time for superthreshold queries: "
              << fifth_percentile_super_time << std::endl;
    std::cout << "5th percentile time for subthreshold queries: "
              << fifth_percentile_sub_time << std::endl;
    std::cout << "95th percentile time for superthreshold queries: "
              << ninetyfifth_percentile_super_time << std::endl;
    std::cout << "95th percentile time for subthreshold queries: "
              << ninetyfifth_percentile_sub_time << std::endl;

    std::cout << "Mean projected frequency: "
              << total_proj_matches / num_queries << std::endl;

    std::cout << "Median projected frequency: " << median_freq << std::endl;

    std::cout << "Min: "
              << parlay::reduce(
                    parlay::make_slice(n_matches.get(),
                                       n_matches.get() + num_queries),
                    parlay::minm<double>());
    std::cout << "\tMax: "
              << parlay::reduce(
                    parlay::make_slice(n_matches.get(),
                                       n_matches.get() + num_queries),
                    parlay::maxm<double>())
              << std::endl;

#endif

    // std::cout << "parfor done" << std::endl;

    return std::make_pair(std::move(ids), std::move(dists));
  }

  void print_stats() {
    this->index.print_stats();
  }
};

/* A posting list of sorts which has an index over its elements.

A virtual class for lists of points matching a filter with some index over the
points to query them.

--- The below was written when I didn't realize I would want an array of these
with mixed index types ---

Basically a wrapper over the index, which could be anything, and the points it
contains.

This is meant for the IVF^2 implementation to represent all the points
associated with a filter where depending on the size of the label, the points
contained inside could be indexed or stored as a list.

Index should implement a sorted_nearest method which takes a query point and
returns a sequence with indices in lexically sorted order (1, 2, 3, ...) which
are in theory the closest points to the query point.
 */
template <typename T, class Point>
struct MatchingPoints {
  // target points should probably be some multiple of the cutoff for
  // constructing a posting list
  virtual parlay::sequence<index_type> sorted_near(Point query) const = 0;
  virtual void set_n_target_points(size_t n) = 0;
};

/* An "index" which just stores an array of indices to matching points.

Meant to be used for the points associated with small filters */

template <typename T, class Point>
struct ArrayIndex : MatchingPoints<T, Point> {
  parlay::sequence<index_type> indices;

  ArrayIndex(const index_type* start, const index_type* end) {
    this->indices = parlay::sequence<index_type>(start, end);
  }

  parlay::sequence<index_type> sorted_near(Point query) const override {
    return this->indices;   // does/should this copy?
  }

  void set_n_target_points(size_t n) override {}
};

/* An extremely minimal IVF index which only returns full posting lists-worth of
   points

    For convenience, also does clustering on the points during initialization
   with a provided clusterer

    Might want to make another version which stores the sizes of the clusters to
   try and better control the number of points returned
*/
template <typename T, typename Point>
struct PostingListIndex : MatchingPoints<T, Point> {
  const PointRange<T, Point>* points;
  parlay::sequence<parlay::sequence<index_type>> clusters;
  parlay::sequence<Point> centroids;
  std::unique_ptr<T[]>
     centroid_data;   // those point objects store const pointers to their data,
                      // so we need to keep it around
  size_t dim;
  size_t aligned_dim;
  size_t n;   // n points in the index

  size_t n_target_points = 10000;   // number of points to try to return

  template <typename Clusterer>
  PostingListIndex(PointRange<T, Point>* points, const index_type* start,
                   const index_type* end, Clusterer clusterer) {
    auto indices = parlay::sequence<index_type>(start, end);

    this->points = points;
    this->dim = points->dimension();
    this->aligned_dim = points->aligned_dimension();

    this->n = indices.size();

    this->clusters = clusterer.cluster(*points, indices);

    this->centroid_data =
       std::make_unique<T[]>(this->clusters.size() * this->aligned_dim);

    if (this->clusters.size() == 0) {
      throw std::runtime_error("PostingListIndex: no clusters generated");
    }

    for (size_t i = 0; i < this->clusters.size(); i++) {
      size_t offset = i * this->aligned_dim;
      parlay::sequence<double> tmp_centroid(this->dim);
      for (size_t p = 0; p < this->clusters[i].size();
           p++) {   // for each point in this cluster
        T* data = (*points)[this->clusters[i][p]].get();
        for (size_t d = 0; d < this->dim; d++) {   // for each dimension
          tmp_centroid[d] += data[d];
        }
      }
      // divide by the number of points in the cluster and assign to centroid
      // data
      for (size_t d = 0; d < this->dim; d++) {   // for each dimension
        this->centroid_data[offset + d] = static_cast<T>(
           std::round(tmp_centroid[d] / this->clusters[i].size()));
      }

      this->centroids.push_back(Point(this->centroid_data.get() + offset,
                                      this->dim, this->aligned_dim, i));
    }
  }

  void set_n_target_points(size_t n) override { this->n_target_points = n; }

  parlay::sequence<index_type> sorted_near(Point query) const override {
    parlay::sequence<std::pair<index_type, float>> nearest_centroids =
       parlay::sequence<std::pair<index_type, float>>::uninitialized(
          this->clusters.size());
    for (size_t i = 0; i < this->clusters.size(); i++) {
      float dist = query.distance(this->centroids[i]);
      nearest_centroids[i] = std::make_pair(i, dist);
    }

    std::sort(
       nearest_centroids.begin(), nearest_centroids.end(),
       [&](std::pair<index_type, float> a, std::pair<index_type, float> b) {
         return a.second < b.second;
       });

    auto result = parlay::sequence<index_type>();
    size_t i = 0;
    do {
      result.append(this->clusters[nearest_centroids[i].first]);
      i++;
    } while (result.size() < this->n_target_points &&
             i < this->clusters.size());

    std::sort(result.begin(), result.end());

    return result;
  }
};

/* The IVF^2 index the above structs are for */
template <typename T, typename Point>
struct IVF_Squared {
  PointRange<T, Point> points;   // hosting here for posting lists to use
  // TODO: uncomment the below;
  // This could be a hash-table per CSR.
  // csr_filters filters; // probably not actually needed after construction
  parlay::sequence<std::unique_ptr<MatchingPoints<T, Point>>>
     posting_lists;   // array of posting lists where indices correspond to
                      // filters

  size_t target_points = 10000;   // number of points for each filter to return

  IVF_Squared() {
    std::cout << "===Running IVF_Squared" << std::endl;
  }

  /*
   * Creates a sequence of posting lists; MatchingPoints pointers.
   * - n_points here is the number of filters (need to refactor this).
   * This happens at build time, but it's important for queries.
   * If a filter is above the cutoff, it gets an array index,
   * otherwise it gets an array that's "global" (just stores all
   * points that match that filter).
   *
   * Later, in batch_filter_search 
   * */
  void fit(PointRange<T, Point> points, csr_filters& filters,
           size_t cutoff = 10000, size_t cluster_size = 1000) {
    // TODO: this->filters = filters
    filters.transpose_inplace();
    this->points = points;
    this->posting_lists =
       parlay::sequence<std::unique_ptr<MatchingPoints<T, Point>>>::
          uninitialized(filters.n_points);
    // auto clusterer = HCNNGClusterer<Point, PointRange<T, Point>,
    // index_type>(cluster_size);

    if (cluster_size <= 0) {
      throw std::runtime_error("IVF^2: cluster size must be positive");
    }

    parlay::parallel_for(0, filters.n_points, [&](size_t i) {
      if (filters.point_count(i) >
          cutoff) {   // The name of this method is so bad that it accidentally
                      // describes what it's doing
        this->posting_lists[i] = std::make_unique<PostingListIndex<T, Point>>(
           &points, filters.row_indices.get() + filters.row_offsets[i],
           filters.row_indices.get() + filters.row_offsets[i + 1],
           KMeansClusterer<T, Point, index_type>(filters.point_count(i) /
                                                 cluster_size));
      } else {
        this->posting_lists[i] = std::make_unique<ArrayIndex<T, Point>>(
           filters.row_indices.get() + filters.row_offsets[i],
           filters.row_indices.get() + filters.row_offsets[i + 1]);
      }
    });
  }

  void fit_from_filename(std::string filename, std::string filter_filename,
                         size_t cutoff = 10000, size_t cluster_size = 1000) {
    PointRange<T, Point> points(filename.c_str());
    std::cout << "IVF^2: points loaded" << std::endl;
    csr_filters filters(filter_filename.c_str());
    std::cout << "IVF^2: filters loaded" << std::endl;
    this->fit(points, filters, cutoff, cluster_size);
    std::cout << "IVF^2: fit completed" << std::endl;
  }

  void set_target_points(size_t n) {
    this->target_points = n;
    for (size_t i = 0; i < this->posting_lists.size(); i++) {
      this->posting_lists[i]->set_n_target_points(n);
    }
  }

  NeighborsAndDistances batch_filter_search(
     py::array_t<T, py::array::c_style | py::array::forcecast>& queries,
     const std::vector<QueryFilter>& filters, uint64_t num_queries,
     uint64_t knn) {
    py::array_t<unsigned int> ids({num_queries, knn});
    py::array_t<float> dists({num_queries, knn});

    parlay::parallel_for(0, num_queries, [&](size_t i) {
      Point q = Point(queries.data(i), this->points.dimension(),
                      this->points.aligned_dimension(), i);
      const QueryFilter& filter = filters[i];

      parlay::sequence<index_type> indices;

      // TODO: optimize for the case where one is very small, the
      // other requires checking the IVF. We can just brute-force
      // this case.
      // We can get perfect recall for this case, but we currently
      // don't, since it depends on the id popping up in the join,
      // which it could not.

      // We may want two different cutoffs for query / build.

      // Notice that this code doesn't care about the cutoff.
      // No distance comparisons yet other than looking at the
      // centroids.
      if (filter.is_and()) {
        indices = join(this->posting_lists[filter.a]->sorted_near(q),
                       this->posting_lists[filter.b]->sorted_near(q));
      } else {
        indices = this->posting_lists[filter.a]->sorted_near(q);
      }

      parlay::sequence<std::pair<index_type, float>> frontier =
         parlay::sequence<std::pair<index_type, float>>(
            knn, std::make_pair(std::numeric_limits<index_type>().max(),
                                std::numeric_limits<float>().max()));

      for (size_t j = 0; j < indices.size(); j++) {   // for each match
        float dist = this->points[indices[j]].distance(
           q);   // compute the distance to query
        // these steps would be very slightly faster if reordered
        if (dist <
            frontier[knn - 1].second) {   // if it's closer than the furthest
                                          // point in the frontier (maybe this
                                          // should be a variable we compare to)
          frontier.pop_back();            // remove the furthest point
          frontier.push_back(std::make_pair(
             indices[j], dist));   // add the new point to the frontier
          std::sort(frontier.begin(), frontier.end(),
                    [&](std::pair<index_type, float> a,
                        std::pair<index_type, float> b) {
                      return a.second < b.second;
                    });   // sort the frontier
        }
      }

      for (unsigned int j = 0; j < knn; j++) {
        ids.mutable_data(i)[j] = static_cast<unsigned int>(frontier[j].first);
        dists.mutable_data(i)[j] = frontier[j].second;
      }
    });

    return std::make_pair(std::move(ids), std::move(dists));
  }
};

#endif   // IVF_H