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

#ifdef COUNTERS
#include "../utils/threadlocal.h"
#endif


#define TINY_CASE_CUTOFF 5000 // when we do tiny x large instead of small x large (default value)

namespace py = pybind11;
using NeighborsAndDistances =
   std::pair<py::array_t<unsigned int>, py::array_t<float>>;

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
  csr_filters filters; 
  csr_filters filters_transpose;// probably not actually needed after construction
  parlay::sequence<std::unique_ptr<MatchingPoints<T, Point>>>
     posting_lists;   // array of posting lists where indices correspond to
                      // filters

  size_t cutoff;
  size_t target_points = 10000;   // number of points for each filter to return
  size_t tiny_cutoff = TINY_CASE_CUTOFF;   // cutoff below which we use the tiny case

  #ifdef COUNTERS
  // number of queries in each case
  threadlocal::accumulator<size_t> largexlarge{}; // most vexing parse???
  threadlocal::accumulator<size_t> largexsmall{};
  threadlocal::accumulator<size_t> smallxsmall{};
  threadlocal::accumulator<size_t> tinyxlarge{}; // also technically tinyxsmall
  threadlocal::accumulator<size_t> large{}; // these are for non-and queries
  threadlocal::accumulator<size_t> small{};

  // distance comparisons for each case
  // note that this is being done in a way that will not be robust to clever centroid bucketing
  threadlocal::accumulator<size_t> largexlarge_dcmp{};
  threadlocal::accumulator<size_t> largexsmall_dcmp{};
  threadlocal::accumulator<size_t> smallxsmall_dcmp{};
  threadlocal::accumulator<size_t> tinyxlarge_dcmp{};
  threadlocal::accumulator<size_t> large_dcmp{};
  threadlocal::accumulator<size_t> small_dcmp{};

  // time spent in each case
  threadlocal::accumulator<double> largexlarge_time{};
  threadlocal::accumulator<double> largexsmall_time{};
  threadlocal::accumulator<double> smallxsmall_time{};
  threadlocal::accumulator<double> tinyxlarge_time{};
  threadlocal::accumulator<double> large_time{};
  threadlocal::accumulator<double> small_time{};

  #endif

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
    this->filters = filters; // rn we will not bother storing the filter sizes because it's one cache miss and a subtraction to compute one at query time
    filters.transpose_inplace();
    this->filters_transpose = filters;
    this->points = points;
    this->posting_lists =
       parlay::sequence<std::unique_ptr<MatchingPoints<T, Point>>>::
          uninitialized(filters.n_points);
    // auto clusterer = HCNNGClusterer<Point, PointRange<T, Point>,
    // index_type>(cluster_size);
    this->cutoff = cutoff;

    if (cluster_size <= 0) {
      throw std::runtime_error("IVF^2: cluster size must be positive");
    }

    auto above_cutoff = parlay::delayed_seq<size_t>(filters.n_points, [&] (size_t i) {
      return filters.point_count(i) > cutoff;
    });

    size_t num_above_cutoff = parlay::reduce(above_cutoff);
    std::cout << "Num above cutoff = " << num_above_cutoff << " filters.n_points = " << filters.n_points << std::endl;
		// std::atomic<int> ctr = 0;

    parlay::parallel_for(0, filters.n_points, [&](size_t i) {
      if (filters.point_count(i) >
          cutoff) {   // The name of this method is so bad that it accidentally
                      // describes what it's doing
        this->posting_lists[i] = std::make_unique<PostingListIndex<T, Point>>(
           &points, filters.row_indices.get() + filters.row_offsets[i],
           filters.row_indices.get() + filters.row_offsets[i + 1],
           KMeansClusterer<T, Point, index_type>(filters.point_count(i) /
                                                 cluster_size));
				// ctr += 1;
				// std::cout << "Finished: " << ctr << " calls." << std::endl;
      } else {
        this->posting_lists[i] = std::make_unique<ArrayIndex<T, Point>>(
           filters.row_indices.get() + filters.row_offsets[i],
           filters.row_indices.get() + filters.row_offsets[i + 1]);
      }
    }, 100000000000); // sequentially for now.
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

      #ifdef COUNTERS

      parlay::internal::timer t;

      threadlocal::accumulator<double>* time_acc; // to add the time/distance to at the end
      threadlocal::accumulator<size_t>* dcmp_acc; 

      t.start();
      #endif

      // We may want two different cutoffs for query / build.

      // Notice that this code doesn't care about the cutoff.
      // No distance comparisons yet other than looking at the
      // centroids.
      if (filter.is_and()) {
        #ifdef COUNTERS
        size_t bigger = std::max(this->filters_transpose.point_count(filter.a), this->filters_transpose.point_count(filter.b));
        size_t smaller = std::min(this->filters_transpose.point_count(filter.a), this->filters_transpose.point_count(filter.b));
        if (smaller > this->cutoff){
          time_acc = &largexlarge_time;
          dcmp_acc = &largexlarge_dcmp;
          largexlarge.increment();
        } else if (bigger > this->cutoff){
            if (smaller > this->tiny_cutoff){
              time_acc = &largexsmall_time;
              dcmp_acc = &largexsmall_dcmp;
              largexsmall.increment();
            } else {
              time_acc = &tinyxlarge_time;
              dcmp_acc = &tinyxlarge_dcmp;
              tinyxlarge.increment();
            }
        } else {
          time_acc = &smallxsmall_time;
          dcmp_acc = &smallxsmall_dcmp;
          smallxsmall.increment();
        }

        // here we assume we're doing a distance comparison to every centroid
        // (this is not robust to clever centroid bucketing)

        if (this->filters_transpose.point_count(filter.a) > this->cutoff){

          dcmp_acc->add(static_cast<PostingListIndex<T, Point>*>(this->posting_lists[filter.a].get())->centroids.size()); // valid??? might be smarter to use dynamic_cast
        } 
        if (this->filters.point_count(filter.b) > this->cutoff){
          dcmp_acc->add(static_cast<PostingListIndex<T, Point>*>(this->posting_lists[filter.b].get())->centroids.size());
        }
        #endif

        auto a_size = this->filters_transpose.point_count(filter.a);
        auto b_size = this->filters_transpose.point_count(filter.b);
        // we xor below on matching the tiny case cutoff because if both are tiny we would rather just join them.
        // TODO: It's probable we actually would want to join in the tiny x small case as well
        if (a_size <= this->tiny_cutoff ^ b_size <= this->tiny_cutoff) {
          if (a_size <= b_size) {
            indices = parlay::filter(
               this->posting_lists[filter.a]->sorted_near(q),
               [&](index_type i) {
                 return this->filters.bin_match(i, filter.b);
               }
            );
          } else {
            indices = parlay::filter(
               this->posting_lists[filter.b]->sorted_near(q),
               [&](index_type i) {
                 return this->filters.bin_match(i, filter.a);
               }
            );
          }
        } else {
          indices = this->posting_lists[filter.a]->sorted_near(q);
          indices = join(indices,
                         this->posting_lists[filter.b]->sorted_near(q));
        }
      } else {
        #ifdef COUNTERS

        if (this->filters_transpose.point_count(filter.a) > this->cutoff){
          time_acc = &large_time;
          dcmp_acc = &large_dcmp;
          large.increment();

          dcmp_acc->add(static_cast<PostingListIndex<T, Point>*>(this->posting_lists[filter.a].get())->centroids.size()); 
        } else {
          time_acc = &small_time;
          dcmp_acc = &small_dcmp;
          small.increment();
        }

        #endif

        indices = this->posting_lists[filter.a]->sorted_near(q);
      }

      #ifdef COUNTERS

      dcmp_acc->add(indices.size());

      #endif

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

      #ifdef COUNTERS

      time_acc->add(t.stop());

      #endif

      for (unsigned int j = 0; j < knn; j++) {
        ids.mutable_data(i)[j] = static_cast<unsigned int>(frontier[j].first);
        dists.mutable_data(i)[j] = frontier[j].second;
      }
    });

    return std::make_pair(std::move(ids), std::move(dists));
  }

  void set_target_points(size_t n) {
    this->target_points = n;
    for (size_t i = 0; i < this->posting_lists.size(); i++) {
      this->posting_lists[i]->set_n_target_points(n);
    }
  }

  void set_tiny_cutoff(size_t n) { this->tiny_cutoff = n; }

  void set_max_iter(size_t n) { max_iter = n; }

  void reset() {
    #ifdef COUNTERS

    largexlarge.reset();
    largexsmall.reset();
    smallxsmall.reset();
    tinyxlarge.reset();
    large.reset();
    small.reset();

    largexlarge_dcmp.reset();
    largexsmall_dcmp.reset();
    smallxsmall_dcmp.reset();
    tinyxlarge_dcmp.reset();
    large_dcmp.reset();
    small_dcmp.reset();

    largexlarge_time.reset();
    largexsmall_time.reset();
    smallxsmall_time.reset();
    tinyxlarge_time.reset();
    large_time.reset();
    small_time.reset();

    #endif
  }

  void print_stats() {
    #ifdef COUNTERS

    std::cout << "Case       \tqueries\tavg. dc\tavg. time\ttotal dc\ttotal time" << std::endl;
    std::cout << "largexlarge\t" << largexlarge.total() << "\t" << largexlarge_dcmp.total() / std::max(largexlarge.total(), (size_t) 1) << "\t" << largexlarge_time.total() / std::max(largexlarge.total(), (size_t) 1) << "\t" << largexlarge_dcmp.total() << "\t" << largexlarge_time.total() << std::endl;
    std::cout << "largexsmall\t" << largexsmall.total() << "\t" << largexsmall_dcmp.total() / std::max(largexsmall.total(), (size_t) 1) << "\t" << largexsmall_time.total() / std::max(largexsmall.total(), (size_t) 1) << "\t" << largexsmall_dcmp.total() << "\t" << largexsmall_time.total() << std::endl;
    std::cout << "smallxsmall\t" << smallxsmall.total() << "\t" << smallxsmall_dcmp.total() / std::max(smallxsmall.total(), (size_t) 1) << "\t" << smallxsmall_time.total() / std::max(smallxsmall.total(), (size_t) 1) << "\t" << smallxsmall_dcmp.total() << "\t" << smallxsmall_time.total() << std::endl;
    std::cout << "tinyxlarge \t" << tinyxlarge.total() << "\t" << tinyxlarge_dcmp.total() / std::max(tinyxlarge.total(), (size_t) 1) << "\t" << tinyxlarge_time.total() / std::max(tinyxlarge.total(), (size_t) 1) << "\t" << tinyxlarge_dcmp.total() << "\t" << tinyxlarge_time.total() << std::endl;
    std::cout << "large      \t" << large.total() << "\t" << large_dcmp.total() / std::max(large.total(), (size_t) 1) << "\t" << large_time.total() / std::max(large.total(), (size_t) 1) << "\t" << large_dcmp.total() << "\t" << large_time.total() << std::endl;
    std::cout << "small      \t" << small.total() << "\t" << small_dcmp.total() / std::max(small.total(), (size_t) 1) << "\t" << small_time.total() / std::max(small.total(), (size_t) 1) << "\t" << small_dcmp.total() << "\t" << small_time.total() << std::endl;
    #endif
  }
};

#endif   // IVF_H