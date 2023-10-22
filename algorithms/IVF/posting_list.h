/* Posting list for the bottom level of an IVF index. */

#ifndef POSTING_LIST
#define POSTING_LIST

#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"

#include "../utils/filters.h"
#include "../utils/point_range.h"
#include "../utils/types.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <utility>

using index_type = int32_t;
/*
A basic, minimal implementation of a posting list.

The operations here are serial for simplcity, since posting lists should be
appropriately small that this is not a bottleneck.
 */
template <typename T, class Point>
struct NaivePostingList {
  PointRange<T, Point> points;            // reference to the full dataset
  parlay::sequence<index_type> indices;   // the indices of the points this
                                          // posting list actually cares about

  // NaivePostingList() {}

  NaivePostingList(PointRange<T, Point> points,
                   parlay::sequence<index_type> indices)
      : points(points), indices(indices) {}

  // see no reason to store its own centroid, but makes sense to be able to
  // generate it
  /*
  This generates (serially) the centroid of the points in the posting list.
   */
  Point centroid() {
    parlay::sequence<double> result(points.dimension(), 0);
    for (size_t i = 0; i < indices.size(); i++) {
      const T* values = points[indices[i]].get();
      for (size_t j = 0; j < points.dimension(); j++) {
        result[j] += values[j];
      }
    }
    for (size_t j = 0; j < points.dimension(); j++) {
      result[j] /= indices.size();
    }

    parlay::sequence<T> casted_result =
       parlay::map(result, [](double x) { return static_cast<T>(x); });
    return Point(static_cast<T*>(casted_result.data()), points.dimension(),
                 points.dimension(),
                 0);   // this may cause issues with alignment
                       // return parlay::map(result, [](double x) {return
                       // static_cast<T>(std::round(x));});
  }

  /*
  Takes a query point, k, and a sequence of pairs of indices and distances, and
  if there are points in the posting list closer than the farthest point in the
  sequence, adds them to the sequence.
   */
  void query(Point query, unsigned int k,
             parlay::sequence<std::pair<unsigned int, float>>& result) {
    float farthest = result[result.size() - 1].second;
    for (unsigned int i = 0; i < indices.size(); i++) {
      float dist = points[indices[i]].distance(query);
      if (dist < farthest) {
        result.push_back(std::make_pair(indices[i], dist));
        std::sort(result.begin(), result.end(),
                  [](std::pair<unsigned int, float> a,
                     std::pair<unsigned int, float> b) {
                    return a.second < b.second;
                  });
        result.pop_back();
        farthest = result[result.size() - 1].second;
      }
    }
  }

  /* Returns the mean sum squared error of the cluster */
  double msse() {
    Point centroid = this->centroid();
    double result = 0;
    for (size_t i = 0; i < indices.size(); i++) {
      result += points[indices[i]].distance(centroid);
    }
    return result / indices.size();
  }

  /* Returns the number of points in the cluster */
  size_t size() { return indices.end() - indices.begin(); }
};

/* Parent class for filtered posting lists

This should probably be an abstract class but I'm using it to hold a naive
implementation */
template <typename T, class Point>
class FilteredPostingList : public NaivePostingList<T, Point> {
 public:
  csr_filters
     filters;   // the filters for the dataset, which should probably be
                // transposed (filter-major) but this class shouldn't care so if
                // you want to do row major you can adjust accordingly
  // natively, this shares one filter object for the whole index, and adjusting
  // to one per posting list requires either defining a new constructor
  // (probably best) or making a subset filter object to pass in (adds more
  // complexity) having posting lists own their own filters is the only thing
  // that makes sense when every posting list has their own object, but if
  // they're all shared this would consume crazy amounts of memory.

  FilteredPostingList(PointRange<T, Point> points,
                      parlay::sequence<int32_t> indices,
                      const csr_filters& filters)
      : NaivePostingList<T, Point>(points, indices) {
    // assuming here that the filters are NOT transposed
    // this->filters = filters.subset_filters(indices);
    this->filters = filters.subset_rows_transpose(indices);
  }

  FilteredPostingList(PointRange<T, Point> points,
                      parlay::sequence<int32_t> indices)
      : NaivePostingList<T, Point>(points, indices) {
    this->filters = csr_filters();
  }

  virtual ~FilteredPostingList() {
    // want to delete the filters here if they're unique to this posting list
  }

  /* This function demonstrates the terrible naming of csr_filters */
  parlay::sequence<int32_t> nonempty_filters() {
    return this->filters.nonempty_rows();
  }

  virtual void filtered_query(
     const Point& query, const QueryFilter& f, unsigned int k,
     parlay::sequence<std::pair<unsigned int, float>>& result) {
    // same logic as query, but checking the filter matches
    parlay::sequence<int32_t> matches;
    if (f.b == -1) {
      matches = this->filters.nonempty_rows();
    } else {
      matches = join(
         this->filters.row_indices.get() + this->filters.row_offsets[f.a],
         this->filters.row_offsets[f.a + 1] - this->filters.row_offsets[f.a],
         this->filters.row_indices.get() + this->filters.row_offsets[f.b],
         this->filters.row_offsets[f.b + 1] - this->filters.row_offsets[f.b]);
    }
    float farthest = result[result.size() - 1].second;
    for (unsigned int i = 0; i < matches.size(); i++) {
      float dist = this->points[matches[i]].distance(query);
      if (dist < farthest) {
        result.push_back(std::make_pair(matches[i], dist));
        std::sort(result.begin(), result.end(),
                  [](std::pair<unsigned int, float> a,
                     std::pair<unsigned int, float> b) {
                    return a.second < b.second;
                  });
        result.pop_back();
        farthest = result[result.size() - 1].second;
      }
    }
  }
};

#endif   // POSTING_LIST