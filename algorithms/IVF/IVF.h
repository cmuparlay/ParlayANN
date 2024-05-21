/* An IVF index */
#ifndef IVF_H
#define IVF_H

#include "parlay/internal/get_time.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"
#include "parlay/delayed_sequence.h"
#include "parlay/slice.h"
#include "parlay/internal/group_by.h"

#include "../HCNNG/clusterEdge.h"
#include "../utils/beamSearch.h"
#include "../utils/graph.h"
#include "../utils/point_range.h"
#include "../utils/types.h"
#include "../vamana/index.h"
#include "bitvector.h"
#include "clustering.h"
#include "posting_list.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <filesystem>
#include <fstream>
#include <utility>
#include <variant>
#include <vector>

#include "pybind11/numpy.h"
#include "pybind11/stl.h"

#ifdef COUNTERS
#include "../utils/threadlocal.h"
#endif

// when we do tiny x large instead of small x large (default value)
#define TINY_CASE_CUTOFF 1000   

// cutoffs for the different sets of query/build params
#define L_CUTOFF 300'000
#define M_CUTOFF 50'000
// s is implicitly anything below M_CUTOFF

#define BITVECTOR_CUTOFF 10000

namespace py = pybind11;
using NeighborsAndDistances =
   std::pair<py::array_t<unsigned int>, py::array_t<float>>;

template <typename T>
using Seq_variant = std::variant<parlay::sequence<T>, parlay::slice<T*, T*>>;

index_type largest_filter_size(const csr_filters& filters,const QueryFilter& f) {
  if (f.is_and()) {
    index_type a_size = filters.point_count(f.a);
    index_type b_size = filters.point_count(f.b);
    return std::max(a_size, b_size);
  } else {
    return filters.point_count(f.a);
  }
}

index_type largest_filter_hash(const csr_filters& filters, const QueryFilter& f) {
  return largest_filter_size(filters, f) << 1 | f.is_and();
}

/* A hash function for pairs we can use for materialized joins 

Intentionally, hash(a, b) == hash(b, a).
*/
struct pair_hash {
    template <class T1, class T2>
    const std::size_t operator () (const std::pair<T1,T2> &pair) const {
        auto hash1 = std::hash<T1>{}(pair.first);
        auto hash2 = std::hash<T2>{}(pair.second);
        return hash1 ^ hash2;
    }
};

size_t cantor_pairing(int a, int b) {
    if (a >= b) {
        throw std::invalid_argument("a must be less than b");
    }
    return static_cast<size_t>(a + b) * (a + b + 1) / 2 + b;
}

/* equal function that ignores order */
struct pair_equal {
    template <class T1, class T2>
    const bool operator () (const std::pair<T1, T2>& x, const std::pair<T1, T2>& y) const {
        // Check if pairs are equal or reversed versions of each other
        return x == y || (x.first == y.second && x.second == y.first);
    }
};

pair_hash pair_hasher = pair_hash();

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
template <class Point>
struct MatchingPoints {
  // target points should probably be some multiple of the cutoff for
  // constructing a posting list
  virtual parlay::sequence<index_type> sorted_near(Point query,
                                                   int target_points) const = 0;
  virtual std::pair<parlay::sequence<std::pair<index_type, float>>, size_t> knn(
     Point query, int k) = 0;
  virtual bool bitmatch(index_type i) const = 0;
  virtual size_t footprint() const = 0;
};

/* An "index" which just stores an array of indices to matching points.

Meant to be used for the points associated with small filters

  TODO: make this not hold anything but the pointers, adjusting join
accordingly, use a slice with a template parameter for the sequence.
 */

template <typename T, class Point>
struct ArrayIndex : MatchingPoints<Point> {
  parlay::sequence<index_type> indices;
  PointRange<T, Point>& points;
  Bits bitvector;

  ArrayIndex(const index_type* start, const index_type* end,
             PointRange<T, Point>& points, int32_t ds_size = 0)
      : points(points) {
    this->indices = parlay::sequence<index_type>(start, end);
    if (ds_size) {
      this->bitvector = Bits(ds_size);
      for (size_t i = 0; i < this->indices.size(); i++) {
        this->bitvector.set_bit(this->indices[i]);
      }
    }
  }

  parlay::sequence<index_type> sorted_near(Point query,
                                           int target_points) const override {
    return this->indices;   // this copies (although if the calling functions are smart it might not have to)
  }

  std::pair<parlay::sequence<std::pair<index_type, float>>, size_t> knn(
     Point query, int k) override {
    parlay::sequence<std::pair<index_type, float>> frontier(
       k, std::make_pair(0, std::numeric_limits<float>::max()));
    for (size_t i = 0; i < indices.size(); i++) {
      float dist = query.distance(this->points[this->indices[i]]);
      if (dist < frontier[k - 1].second) {
        frontier.pop_back();
        frontier.push_back(std::make_pair(indices[i], dist));
        std::sort(
           frontier.begin(), frontier.end(),
           [&](std::pair<index_type, float> a, std::pair<index_type, float> b) {
             return a.second < b.second;
           });
      }
    }
    return std::make_pair(frontier, this->indices.size());
  }

  bool bitmatch(index_type i) const override {
    return this->bitvector.is_bit_set(i);
  }

  size_t footprint() const override {
    return sizeof(index_type) * this->indices.size() + this->bitvector.footprint();
  }
};

/* An extremely minimal IVF index which only returns full posting lists-worth of
   points

    For convenience, also does clustering on the points during initialization
   with a provided clusterer

    Might want to refactor to store the partitions in CSR format
*/
template <typename T, typename Point>
struct PostingListIndex : MatchingPoints<Point> {
  PointRange<T, Point>& points;
  parlay::sequence<parlay::sequence<index_type>> clusters;
  parlay::sequence<Point> centroids;
  std::unique_ptr<T[]>
     centroid_data;   // those point objects store const pointers to their data,
                      // so we need to keep it around
  size_t dim;
  size_t aligned_dim;
  size_t n;                                   // n points in the index
  index_type id;                              // id of this index
  std::pair<size_t, size_t> cluster_params;   // build params for the clustering

  Graph<index_type> index_graph;   // the vamana graph we do single search over
  QueryParams* QP;
  BuildParams BP;
  // index_type start_point;
  SubsetPointRange<T, Point> subset_points;

  size_t sq_ivf_n_lists;  // when  not building a graph, how many lists to check for ivf_knn

  Bits bitvector;

  PostingListIndex(PostingListIndex<T, Point>&& other) {
    std::cout << "Move constructor called" << std::endl;
  };

  template <typename Clusterer>
  PostingListIndex(PointRange<T, Point>& points, const index_type* start,
                   const index_type* end, Clusterer clusterer, BuildParams BP,
                   QueryParams* QP, index_type id,
                   std::string cache_path = "index_cache/",
                   index_type ds_size = 10'000'000,
                   size_t sq_ivf_n_lists = 96)
      : id(id),
        points(points),
        subset_points(points, parlay::sequence<index_type>(start, end)),
        bitvector(ds_size),
        sq_ivf_n_lists(sq_ivf_n_lists) {
    auto indices = parlay::sequence<index_type>(start, end);

    // this->points = points;
    this->dim = points.dimension();
    this->aligned_dim = points.aligned_dimension();
    this->QP = QP;
    this->BP = BP;

    this->n = indices.size();
    this->cluster_params = clusterer.get_build_params();

    if (cache_path != "" &&
        std::filesystem::exists(this->pl_filename(cache_path))) {
      // std::cout << "Loading posting list" << std::endl;
      this->load_posting_list(cache_path);
    } else {
      // std::cout << "Building posting list" << std::endl;

      this->clusters = clusterer.cluster(points, indices);

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
          T* data = points[this->clusters[i][p]].get();
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

      if (cache_path != "") {
        this->save_posting_list(cache_path);
      }
    }
    
    // build the vamana graph

    if (cache_path != "" &&
        std::filesystem::exists(this->graph_filename(cache_path))) {
      // std::cout << "Loading graph" << std::endl;

      std::string filename = this->graph_filename(cache_path);
      this->index_graph = Graph<index_type>(filename.data());
    } else {
      // std::cout << "Building graph" << std::endl;
      // this->start_point = indices[0];
      knn_index<Point, SubsetPointRange<T, Point>, index_type> I(BP);
      stats<index_type> BuildStats(this->points.size());

      // std::cout << "This filter has " << indices.size() << " points" <<
      // std::endl;

      this->index_graph = Graph<index_type>(BP.R, indices.size());
      I.build_index(this->index_graph, this->subset_points, BuildStats);

      if (cache_path != "") {
        this->save_graph(cache_path);
      }
    }
    // confirming the graph is real by printing the out neighbors of the first
    // point std::cout << "first point has " << this->index_graph[0].size() <<
    // "/" << this->index_graph.max_degree() <<" out neighbors:" << std::endl;
    // for (size_t i = 0; i < this->index_graph[0].size(); i++){
    //   std::cout << this->index_graph[0][i] << " ";
    // }
    // std::cout << std::endl;

    // auto [pairs, _] = this->knn(this->subset_points[0], 10);

    // for (size_t i = 0; i < pairs.size(); i++){
    //   std::cout << " (" << pairs[i].first << ", " << pairs[i].second << ") ";
    // }
    // std::cout << std::endl;

    for (size_t i = 0; i < indices.size(); i++) {
      this->bitvector.set_bit(indices[i]);
    }
  }

  parlay::sequence<index_type> sorted_near(Point query,
                                           int target_points) const override {
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
    } while (result.size() < target_points && i < this->clusters.size());

    std::sort(result.begin(), result.end());

    return result;
  }

  std::pair<parlay::sequence<std::pair<index_type, float>>, size_t> knn(
     Point query, int k) override {
    // will have to come back to do something clever with the distance
    // comparisons, perhaps track the comparisons for the single case inside the
    // posting lists std::cout << this->index_graph[0][0] << " " <<
    // this->index_graph[500][10] << std::endl;

    auto [pairElts, dist_cmps] =
       beam_search<Point, SubsetPointRange<T, Point>, index_type>(
          query, this->index_graph, this->subset_points, 0, *(this->QP));
    auto frontier = pairElts.first;

    return std::make_pair(
       parlay::map(frontier,
                   [&](std::pair<index_type, float> p) {
                     return std::make_pair(
                        this->subset_points.real_index(p.first), p.second);
                   }),
       dist_cmps);
  }

  /* computes the knn with the ivf index */
  std::pair<parlay::sequence<std::pair<index_type, float>>, size_t> ivf_knn(
     Point query, int k) {
      // we do linear traversal over the centroids to get the nearest ones
      parlay::sequence<std::pair<index_type, float>> pl_frontier(
         this->sq_ivf_n_lists, std::make_pair(0, std::numeric_limits<float>::max()));

      size_t dist_cmps = this->centroids.size();

      for (index_type i = 0; i < this->centroids.size(); i++) {
        float dist = query.distance(this->centroids[i]);
        if (dist < pl_frontier[this->sq_ivf_n_lists - 1].second) {
          pl_frontier.pop_back();
          pl_frontier.push_back(std::make_pair(i, dist));
          std::sort(pl_frontier.begin(), pl_frontier.end(),
                    [&](std::pair<index_type, float> a,
                        std::pair<index_type, float> b) {
                      return a.second < b.second;
                    });
        }
      }

      // now we do search on the points in the posting lists
      parlay::sequence<std::pair<index_type, float>> frontier(
         this->sq_ivf_n_lists, std::make_pair(0, std::numeric_limits<float>::max()));

      for (index_type i = 0; i < pl_frontier.size(); i++) {
        dist_cmps += this->clusters[pl_frontier[i].first].size();
        for (index_type j = 0; j < this->clusters[pl_frontier[i].first].size(); j++) {
          float dist = query.distance(this->points[this->clusters[pl_frontier[i].first][j]]);
          if (dist < frontier[this->sq_ivf_n_lists - 1].second) {
            frontier.pop_back();
            frontier.push_back(std::make_pair(this->clusters[pl_frontier[i].first][j], dist));
            std::sort(frontier.begin(), frontier.end(),
                      [&](std::pair<index_type, float> a,
                          std::pair<index_type, float> b) {
                        return a.second < b.second;
                      });
          }
        }
    }
    return std::make_pair(frontier, dist_cmps);
  }

  std::string graph_filename(std::string filename_prefix) {
    return filename_prefix + std::to_string(this->id) + "_graph_" +
           std::to_string(this->BP.R) + "_" + std::to_string(this->BP.L) + "_" +
           std::to_string(this->BP.alpha) + ".bin";
  }

  /* Saves the graph to a file named based on the build params

  Note that the prefix should include a trailing slash if it's a directory */
  void save_graph(std::string filename_prefix) {
    std::string filename = this->graph_filename(filename_prefix);
    this->index_graph.save(filename.data());
  }

  std::string pl_filename(std::string filename_prefix) {
    return filename_prefix + std::to_string(this->id) + "_postingList_" +
           std::to_string(this->cluster_params.first) + "_" +
           std::to_string(this->cluster_params.second) + ".bin";
  }

  /* Saves the posting list to a file named based on the cluster params */
  void save_posting_list(const std::string& filename_prefix) {
    std::string filename = pl_filename(filename_prefix);
    std::ofstream output(filename, std::ios::binary);

    if (!output.is_open()) {
      throw std::runtime_error("PostingListIndex: could not open file " +
                               filename);
    }

    // Write number of centroids/posting lists
    size_t num_centroids = this->clusters.size();
    output.write(reinterpret_cast<const char*>(&num_centroids), sizeof(size_t));

    // Write centroids
    size_t centroid_bytes = sizeof(T) * num_centroids * this->aligned_dim;
    output.write(reinterpret_cast<const char*>(this->centroid_data.get()),
                 centroid_bytes);

    // For CSR-like format
    parlay::sequence<index_type> indices =
       parlay::sequence<index_type>::uninitialized(this->n);
    parlay::sequence<index_type> indptr(1, 0);   // Start with an initial zero
    indptr.reserve(num_centroids + 1);

    for (const auto& list : this->clusters) {
      memcpy(indices.data() + indptr.back(), list.data(),
             sizeof(index_type) * list.size());
      indptr.push_back(indptr.back() + list.size());
    }

    // Write indptr array
    output.write(reinterpret_cast<const char*>(indptr.data()),
                 sizeof(index_type) * indptr.size());

    // Write indices array
    output.write(reinterpret_cast<const char*>(indices.data()),
                 sizeof(index_type) * indices.size());

    output.close();
  }

  /* Loads the posting list from a file named based on the cluster params */
  void load_posting_list(std::string filename_prefix) {
    std::string filename = pl_filename(filename_prefix);
    std::ifstream input(filename, std::ios::binary);

    if (!input.is_open()) {
      throw std::runtime_error("PostingListIndex: could not open file " +
                               filename);
    }

    // Read number of centroids/posting lists
    size_t num_centroids;
    input.read(reinterpret_cast<char*>(&num_centroids), sizeof(size_t));

    // Read centroids
    size_t centroid_bytes = sizeof(T) * num_centroids * this->aligned_dim;
    this->centroid_data =
       std::make_unique<T[]>(num_centroids * this->aligned_dim);
    input.read(reinterpret_cast<char*>(this->centroid_data.get()),
               centroid_bytes);

    for (size_t i = 0; i < num_centroids; i++) {
      this->centroids.push_back(
         Point(this->centroid_data.get() + i * this->aligned_dim, this->dim,
               this->aligned_dim, i));
    }
    // Read indptr array
    parlay::sequence<index_type> indptr(num_centroids + 1);
    input.read(reinterpret_cast<char*>(indptr.data()),
               sizeof(index_type) * indptr.size());

    // Read indices array
    parlay::sequence<index_type> indices(indptr.back());
    input.read(reinterpret_cast<char*>(indices.data()),
               sizeof(index_type) * indices.size());

    input.close();

    // Convert indptr and indices to CSR-like format
    this->clusters = parlay::sequence<parlay::sequence<index_type>>(
       num_centroids);   // should in theory be ininitialized but that causes a
                         // segfault
    for (size_t i = 0; i < num_centroids; i++) {
      this->clusters[i].reserve(indptr[i + 1] - indptr[i]);
      // this is heinous hopefully this is not needed
      for (size_t j = indptr[i]; j < indptr[i + 1]; j++) {
        this->clusters[i].push_back(indices[j]);
      }
    }
  }

  bool bitmatch(index_type i) const override {
    return this->bitvector.is_bit_set(i);
  }

  size_t footprint() const override {
    return sizeof(index_type) * this->n // posting lists
            + sizeof(T) * this->aligned_dim * this->clusters.size() // centroids
            + this->bitvector.footprint() // bitvector
            + sizeof(index_type) * this->BP.R * this->n; // graph
  }
};

/* The IVF^2 index the above structs are for */
template <typename T, typename Point>
struct IVF_Squared {
  PointRange<T, Point> points;   // hosting here for posting lists to use
  // TODO: uncomment the below;
  // This could be a hash-table per CSR.
  csr_filters filters;
  csr_filters
     filters_transpose;   // probably not actually needed after construction
  parlay::sequence<std::unique_ptr<MatchingPoints<Point>>>
     posting_lists;   // array of posting lists where indices correspond to
                      // filters

  size_t cutoff;
  size_t target_points = 10000;   // number of points for each filter to return
  size_t tiny_cutoff =
     TINY_CASE_CUTOFF;   // cutoff below which we use the tiny case
  size_t sq_target_points =
     500;   // number of points for each filter to return when not doing an and

  size_t large_cutoff = L_CUTOFF;    // cutoff for large case
  size_t medium_cutoff = M_CUTOFF;   // cutoff for medium case
  // small case is anything below large

  // TODO: update these when we receive the weight classes from the constructor
  QueryParams QP[3] = {QueryParams(10, 100, 1.35, M_CUTOFF, 16),
                       QueryParams(10, 100, 1.35, L_CUTOFF, 32),
                       QueryParams(10, 100, 1.35, 3'000'000, 64)};
  // TODO: increase L_build to 500 once we don't care about the build time
  // and/or are saving these graphs
  // TODO: modify build to do two passes once we have all the time in the world
  BuildParams BP[3] = {BuildParams(16, 500, 1.175), BuildParams(32, 500, 1.175),
                       BuildParams(64, 500, 1.175)};

  index_type bitvector_cutoff = BITVECTOR_CUTOFF;

  bool materialized_joins = true;
  // the below is PostingListIndex because we need to map back to original coords
  std::unordered_map<std::pair<index_type, index_type>, std::unique_ptr<PostingListIndex<T, Point>>, pair_hash, pair_equal> material_joins_map;
  size_t join_cutoff = 50000;

#ifdef COUNTERS
  // number of queries in each case
  threadlocal::accumulator<size_t> largexlarge{};   // most vexing parse???
  threadlocal::accumulator<size_t> largexsmall{};
  threadlocal::accumulator<size_t> smallxsmall{};
  threadlocal::accumulator<size_t> tinyxlarge{};  // also technically tinyxsmall
  threadlocal::accumulator<size_t> large{};   // these are for non-and queries
  threadlocal::accumulator<size_t> small{};

  // distance comparisons for each case
  // note that this is being done in a way that will not be robust to clever
  // centroid bucketing
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

#ifdef LUMBERJACK   // a play on logger

  threadlocal::logger<std::tuple<size_t, size_t, double>> logger{};

#endif

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
   * */
  void fit(PointRange<T, Point> points, csr_filters& filters,
           size_t cutoff = 10000, size_t cluster_size = 1000,
           std::string cache_path = "index_cache/", bool parallel_build = false) {
    this->filters = filters;   // rn we will not bother storing the filter sizes
                               // because it's one cache miss and a subtraction
                               // to compute one at query time
    filters.transpose_inplace();
    this->filters_transpose = filters;
    this->points = points;
    this->posting_lists =
       parlay::sequence<std::unique_ptr<MatchingPoints<Point>>>::
          uninitialized(filters.n_points);
    // auto clusterer = HCNNGClusterer<Point, PointRange<T, Point>,
    // index_type>(cluster_size);
    this->cutoff = cutoff;

    if (cluster_size <= 0) {
      throw std::runtime_error("IVF^2: cluster size must be positive");
    }

    auto above_cutoff = parlay::delayed_seq<size_t>(
       filters.n_points,
       [&](size_t i) { return filters.point_count(i) > cutoff; });

    size_t num_above_cutoff = parlay::reduce(above_cutoff);
    std::cout << "Num above cutoff = " << num_above_cutoff
              << " filters.n_points = " << filters.n_points << std::endl;
    std::atomic<int> ctr = 0;

    parlay::parallel_for(0, filters.n_points, [&](size_t i) {
      if (filters.point_count(i) >
          cutoff) {   // The name of this method is so bad that it accidentally
                      // describes what it's doing
        // weight classes here being the s, m, l cutoffs
        int weight_class = 0;
        if (filters.point_count(i) > this->large_cutoff) {
          weight_class = 2;
        } else if (filters.point_count(i) > this->medium_cutoff) {
          weight_class = 1;
        }

        // std::cout << "Filter with " << filters.point_count(i) << " points has
        // weight class " << weight_class << std::endl;

        this->posting_lists[i] = std::make_unique<PostingListIndex<T, Point>>(
           this->points, filters.row_indices.get() + filters.row_offsets[i],
           filters.row_indices.get() + filters.row_offsets[i + 1],
           KMeansClusterer<T, Point, index_type>(filters.point_count(i) /
                                                 cluster_size),
           BP[weight_class], QP + weight_class, i, cache_path,
           filters.n_filters);
        ctr += 1;
        if (ctr % 100 == 0) {
          std::cout << "===============  IVF^2: " << ctr << " / " << num_above_cutoff
                    << " PostingListIndex objects created  ===============" << std::endl;
        }
      } else {
        this->posting_lists[i] = std::make_unique<ArrayIndex<T, Point>>(
           filters.row_indices.get() + filters.row_offsets[i],
           filters.row_indices.get() + filters.row_offsets[i + 1], this->points,
           (filters.point_count(i) >= this->bitvector_cutoff ? filters.n_filters
                                                             : 0));
      }
    }, (parallel_build ? 0 : 10'000'000)); 

    if (this->materialized_joins) {
      std::cout << "Materializing joins larger than " << join_cutoff << " elements" << std::endl;

      this->material_joins_map = std::unordered_map<std::pair<index_type, index_type>, std::unique_ptr<PostingListIndex<T, Point>>, pair_hash, pair_equal>();

      // we store a 'manifest' of the intersection sizes so we don't have to recompute it every load
      std::string manifest_filename = cache_path + "manifest_" + std::to_string(cutoff) + "_" + std::to_string(join_cutoff);

      if (std::filesystem::exists(manifest_filename)) {
        std::ifstream manifest_file(manifest_filename);
        size_t n_intersections;

        std::cout << "Manifest file found with " << n_intersections << " intersections, loading..." << std::endl;

        manifest_file.read(reinterpret_cast<char*>(&n_intersections), sizeof(n_intersections)); // read number of materialized intersections
        for (size_t i = 0; i < n_intersections; i++) {
          index_type a, b;
          size_t size;
          manifest_file.read(reinterpret_cast<char*>(&a), sizeof(a));
          manifest_file.read(reinterpret_cast<char*>(&b), sizeof(b));
          manifest_file.read(reinterpret_cast<char*>(&size), sizeof(size));
          
          size_t id = filters.n_points + cantor_pairing(a, b); 

          int weight_class = 0;
          if (size > this->large_cutoff) {
            weight_class = 2;
          } else if (size > this->medium_cutoff) {
            weight_class = 1;
          }

          auto intersection = filters.point_intersection(a, b);

          // these should really be parallelized constructions
          this->material_joins_map[std::make_pair(a, b)] = std::make_unique<PostingListIndex<T, Point>>(
                  this->points, intersection.begin(), intersection.end(),
                  KMeansClusterer<T, Point, index_type>(1), 
                  BP[weight_class], QP + weight_class, id, cache_path,
                  filters.n_filters);

          if (i % 100 == 0) {
            std::cout << "===============  IVF^2: " << i << " / " << n_intersections << " materialized joins loaded  ===============" << std::endl;
          }
        }
        manifest_file.close();
      } else {
        std::cout << "No intersection manifest found, building materialized joins..." << std::endl;
        parlay::sequence<std::tuple<index_type, index_type, size_t>> intersection_sizes;
        // we compute the size of the join for each pair of large filters
        // and if a join is large enough, we build an index for it
        for (size_t i = 0; i < filters.n_points; i++) {
          if (filters.point_count(i) > join_cutoff) {
            for (size_t j = i + 1; j < filters.n_points; j++) {
              if (filters.point_count(j) > join_cutoff) {
                auto intersection = filters.point_intersection(i, j);

                if (intersection.size() > join_cutoff) {
                  intersection_sizes.push_back(std::make_tuple(i, j, intersection.size()));

                  int weight_class = 0;
                  if (intersection.size() > this->large_cutoff) {
                    weight_class = 2;
                  } else if (intersection.size() > this->medium_cutoff) {
                    weight_class = 1;
                  }

                  size_t id = filters.n_points + cantor_pairing(i, j);

                  this->material_joins_map[std::make_pair(i, j)] = std::make_unique<PostingListIndex<T, Point>>(
                    this->points, intersection.begin(), intersection.end(),
                    KMeansClusterer<T, Point, index_type>(1),
                    BP[weight_class], QP + weight_class, id, cache_path,
                    filters.n_filters);
                }
              }
            }
          }
          if (i % 100 == 0) {
            std::cout << "===============  IVF^2: " << i << " / " << filters.n_points << " filters processed for joins  ===============" << std::endl;
          }
        }
        // write manifest file
        std::ofstream manifest_file(manifest_filename);
        size_t n_intersections = intersection_sizes.size();
        manifest_file.write(reinterpret_cast<const char*>(&(n_intersections)), sizeof(size_t));
        for (auto [a, b, size] : intersection_sizes) {
          manifest_file.write(reinterpret_cast<const char*>(&a), sizeof(a));
          manifest_file.write(reinterpret_cast<const char*>(&b), sizeof(b));
          manifest_file.write(reinterpret_cast<const char*>(&size), sizeof(size));
        }
        manifest_file.close();
      }

      std::cout << "IVF^2: materialized joins built, indexing " << this->material_joins_map.size() << " intersections" << std::endl;
    }
  }

  void fit_from_filename(std::string filename, std::string filter_filename,
                         size_t cutoff = 10000, size_t cluster_size = 1000,
                         std::string cache_path = "",
                         std::pair<size_t, size_t> weight_classes =
                            std::make_pair(M_CUTOFF, L_CUTOFF), bool parallel_build = false) {
    this->medium_cutoff = weight_classes.first;
    this->large_cutoff = weight_classes.second;

    PointRange<T, Point> points(filename.c_str());
    std::cout << "IVF^2: points loaded" << std::endl;

    csr_filters filters(filter_filename.c_str());
    std::cout << "IVF^2: filters loaded" << std::endl;

    this->fit(points, filters, cutoff, cluster_size, cache_path, parallel_build);
    std::cout << "IVF^2: fit completed" << std::endl;
  }


  std::pair<parlay::sequence<index_type> /*indices*/, size_t /*dist_cmps*/> and_query(Point q, QueryFilter f) {
    index_type a = f.a;
    index_type b = f.b;
    index_type a_size = this->filters_transpose.point_count(a);
    index_type b_size = this->filters_transpose.point_count(b);

    if (this->materialized_joins) {
      // could check sizes are both above cutoff if really inclined,
      // but no way that makes a measureable difference
      auto and_index = this->material_joins_map.find(std::make_pair(a, b));
      if (and_index != this->material_joins_map.end()) {
        auto [output, dcmps] = and_index->second->knn(q, 10);
        auto indices = parlay::sequence<index_type>(output.size());
        for (size_t i = 0; i < output.size(); i++) {
          indices[i] = output[i].first;
        }
        return std::make_pair(std::move(indices), dcmps);
      }
    }

    parlay::sequence<index_type> indices;
    size_t dist_cmps = 0;

    // a will always be the smaller filter
    if (b_size < a_size) {
      std::swap(a, b);
      std::swap(a_size, b_size);
    }

    

    if (a_size > this->tiny_cutoff || (a_size <= this->tiny_cutoff & b_size <= this->bitvector_cutoff) ) { // large x large and small x large and small x small
      indices = join(this->posting_lists[a]->sorted_near(q, this->target_points),
                     this->posting_lists[b]->sorted_near(q, this->target_points));

      #ifdef COUNTERS
      if (b_size > this->cutoff) {
        dist_cmps += static_cast<PostingListIndex<T, Point>*>(
                           this->posting_lists[b].get())
                           ->centroids.size();
      }
      if (a_size > this->cutoff) {
        dist_cmps += static_cast<PostingListIndex<T, Point>*>(
                           this->posting_lists[a].get())
                           ->centroids.size();
      }
      #endif
    } else { // tiny x small (above bitvector cutoff) and tiny x large
      indices.reserve(a_size);

      for (index_type i = 0; i < a_size; i++) {
        const index_type point =
           this->filters_transpose.row_indices[this->filters_transpose.row_offsets[a] + i];
        if (this->posting_lists[b]->bitmatch(point)) {
          indices.push_back(point);
        }
      }
    }

    return std::make_pair(std::move(indices), dist_cmps);
  }

  NeighborsAndDistances batch_filter_search(
     py::array_t<T, py::array::c_style | py::array::forcecast>& queries,
     const std::vector<QueryFilter>& filters, uint64_t num_queries,
     uint64_t knn) {
    py::array_t<unsigned int> ids({num_queries, knn});
    py::array_t<float> dists({num_queries, knn});

    parlay::parallel_for(0, num_queries, [&](size_t i) {
      // for (size_t i = 0; i < num_queries; i++) {
      Point q = Point(queries.data(i), this->points.dimension(),
                      this->points.aligned_dimension(), i);
      const QueryFilter& filter = filters[i];

      parlay::sequence<index_type> indices;
      size_t dist_cmps = 0;

#ifdef COUNTERS

      parlay::internal::timer t;

      threadlocal::accumulator<double>*
         time_acc;   // to add the time/distance to at the end
      threadlocal::accumulator<size_t>* dcmp_acc;

      t.start();

#endif

      // We may want two different cutoffs for query / build.

      // Notice that this code doesn't care about the cutoff.
      // No distance comparisons yet other than looking at the
      // centroids.
      if (filter.is_and()) {
#ifdef COUNTERS
        size_t bigger = std::max(this->filters_transpose.point_count(filter.a),
                                 this->filters_transpose.point_count(filter.b));
        size_t smaller =
           std::min(this->filters_transpose.point_count(filter.a),
                    this->filters_transpose.point_count(filter.b));
        if (smaller > this->cutoff) {
          time_acc = &largexlarge_time;
          dcmp_acc = &largexlarge_dcmp;
          largexlarge.increment();
        } else if (bigger > this->cutoff) {
          if (smaller > this->tiny_cutoff) {
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

#endif
        auto tmp = this->and_query(q, filter);
        indices = tmp.first;
        dist_cmps = tmp.second;

      } else {
#ifdef COUNTERS

        if (this->filters_transpose.point_count(filter.a) > this->cutoff) {
          time_acc = &large_time;
          dcmp_acc = &large_dcmp;
          large.increment();

        } else {
          time_acc = &small_time;
          dcmp_acc = &small_dcmp;
          small.increment();
        }

#endif

        // indices = this->posting_lists[filter.a]->sorted_near(q,
        // this->sq_target_points);
        auto [frontier, cmps] = this->posting_lists[filter.a]->knn(q, 10);

#ifdef COUNTERS

        double elapsed = t.stop();

        dcmp_acc->add(cmps);
        time_acc->add(elapsed);

#ifdef LUMBERJACK

        logger.update(std::make_tuple(i, cmps, elapsed));

#endif

#endif

        for (unsigned int j = 0; j < knn; j++) {
          ids.mutable_data(i)[j] = static_cast<unsigned int>(frontier[j].first);
          dists.mutable_data(i)[j] = frontier[j].second;
        }
        return;
      }

#ifdef COUNTERS

      dist_cmps += indices.size();

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

      double elapsed = t.stop();
      time_acc->add(elapsed);
      dcmp_acc->add(dist_cmps);


#ifdef LUMBERJACK

      logger.update(std::make_tuple(i, dist_cmps, elapsed));

#endif

#endif

      for (unsigned int j = 0; j < knn; j++) {
        ids.mutable_data(i)[j] = static_cast<unsigned int>(frontier[j].first);
        dists.mutable_data(i)[j] = frontier[j].second;
      }
    });
    // }

    return std::make_pair(std::move(ids), std::move(dists));
  }


  NeighborsAndDistances sorted_batch_filter_search(
     py::array_t<T, py::array::c_style | py::array::forcecast>& queries,
     const std::vector<QueryFilter>& filters, uint64_t num_queries,
     uint64_t knn) {
    py::array_t<unsigned int> ids({num_queries, knn});
    py::array_t<float> dists({num_queries, knn});

    // we want to create a sequence of indices corresponding to an ordering which groups the queries by their largest filter
    // we can then iterate through this sequence and do the queries in that order

    // parlay::sequence<size_t> query_ordering = parlay::tabulate(
      //  num_queries, [&](size_t i) { return i; });   // initialize to iota

  // large to small, with ands coming first within each size
  // this should probably be parlay::rank
    parlay::sequence<size_t> query_ordering = parlay::sort(parlay::iota(num_queries), [&] (index_type x, index_type y) {
      index_type lfs_x = largest_filter_size(this->filters_transpose, filters[x]);
      index_type lfs_y = largest_filter_size(this->filters_transpose, filters[y]);
      if (lfs_x == lfs_y) {
        return filters[x].is_and() && !filters[y].is_and(); // this might be backwards
      } else {
        return lfs_x > lfs_y;
      }
    } );

    parlay::parallel_for(0, num_queries, [&] (index_type j) {
      index_type i = query_ordering[j]; // where we're applying this sort

      Point q = Point(queries.data(i), this->points.dimension(),
                      this->points.aligned_dimension(), i);
      const QueryFilter& filter = filters[i];

      parlay::sequence<index_type> indices;
      size_t dist_cmps = 0;

      #ifdef COUNTERS

      parlay::internal::timer t;

      threadlocal::accumulator<double>*
         time_acc;   // to add the time/distance to at the end
      threadlocal::accumulator<size_t>* dcmp_acc;

      t.start();

#endif

      // We may want two different cutoffs for query / build.

      // Notice that this code doesn't care about the cutoff.
      // No distance comparisons yet other than looking at the
      // centroids.
      if (filter.is_and()) {
#ifdef COUNTERS
        size_t bigger = std::max(this->filters_transpose.point_count(filter.a),
                                 this->filters_transpose.point_count(filter.b));
        size_t smaller =
           std::min(this->filters_transpose.point_count(filter.a),
                    this->filters_transpose.point_count(filter.b));
        if (smaller > this->cutoff) {
          time_acc = &largexlarge_time;
          dcmp_acc = &largexlarge_dcmp;
          largexlarge.increment();
        } else if (bigger > this->cutoff) {
          if (smaller > this->tiny_cutoff) {
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

#endif
        auto tmp = this->and_query(q, filter);
        indices = tmp.first;
        dist_cmps = tmp.second;

      } else {
#ifdef COUNTERS

        if (this->filters_transpose.point_count(filter.a) > this->cutoff) {
          time_acc = &large_time;
          dcmp_acc = &large_dcmp;
          large.increment();

        } else {
          time_acc = &small_time;
          dcmp_acc = &small_dcmp;
          small.increment();
        }

#endif

        // indices = this->posting_lists[filter.a]->sorted_near(q,
        // this->sq_target_points);
        auto [frontier, cmps] = this->posting_lists[filter.a]->knn(q, 10);

#ifdef COUNTERS

        double elapsed = t.stop();

        dcmp_acc->add(cmps);
        time_acc->add(elapsed);

#ifdef LUMBERJACK

        logger.update(std::make_tuple(i, cmps, elapsed));

#endif

#endif

        for (unsigned int j = 0; j < knn; j++) {
          ids.mutable_data(i)[j] = static_cast<unsigned int>(frontier[j].first);
          dists.mutable_data(i)[j] = frontier[j].second;
        }
        return;
      }

#ifdef COUNTERS

      dist_cmps += indices.size();

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

      double elapsed = t.stop();
      time_acc->add(elapsed);
      dcmp_acc->add(dist_cmps);


#ifdef LUMBERJACK

      logger.update(std::make_tuple(i, dist_cmps, elapsed));

#endif

#endif

      for (unsigned int j = 0; j < knn; j++) {
        ids.mutable_data(i)[j] = static_cast<unsigned int>(frontier[j].first);
        dists.mutable_data(i)[j] = frontier[j].second;
      }
      return;
    });

    return std::make_pair(std::move(ids), std::move(dists));
  }
  
  void set_target_points(size_t n) {
    this->target_points = n;
  }

  void set_sq_target_points(size_t n) {
    this->sq_target_points = n;
  }

  void set_tiny_cutoff(size_t n) {
    this->tiny_cutoff = n;
  }

  void set_max_iter(size_t n) {
    max_iter = n;
  }

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

  void print_stats() const {
#ifdef COUNTERS

    std::cout << "Case       \tqueries\tavg. dc\tavg. time\ttotal dc\ttotal "
                 "time\tQPS on 8c"
              << std::endl;
    std::cout << "largexlarge\t" << largexlarge.total() << "\t"
              << largexlarge_dcmp.total() /
                    std::max(largexlarge.total(), (size_t)1)
              << "\t"
              << largexlarge_time.total() /
                    std::max(largexlarge.total(), (size_t)1)
              << "\t" << largexlarge_dcmp.total() << "  \t"
              << largexlarge_time.total() << "    \t"
              << largexlarge.total() /
                    std::max(largexlarge_time.total() / 8, (double)1.)
              << std::endl;
    std::cout << "largexsmall\t" << largexsmall.total() << "\t"
              << largexsmall_dcmp.total() /
                    std::max(largexsmall.total(), (size_t)1)
              << "\t"
              << largexsmall_time.total() /
                    std::max(largexsmall.total(), (size_t)1)
              << "\t" << largexsmall_dcmp.total() << "  \t"
              << largexsmall_time.total() << "    \t"
              << largexsmall.total() /
                    std::max(largexsmall_time.total() / 8, (double)1.)
              << std::endl;
    std::cout << "smallxsmall\t" << smallxsmall.total() << "\t"
              << smallxsmall_dcmp.total() /
                    std::max(smallxsmall.total(), (size_t)1)
              << "\t"
              << smallxsmall_time.total() /
                    std::max(smallxsmall.total(), (size_t)1)
              << "\t" << smallxsmall_dcmp.total() << "  \t"
              << smallxsmall_time.total() << "    \t"
              << smallxsmall.total() /
                    std::max(smallxsmall_time.total() / 8, (double)1.)
              << std::endl;
    std::cout
       << "tinyxlarge \t" << tinyxlarge.total() << "\t"
       << tinyxlarge_dcmp.total() / std::max(tinyxlarge.total(), (size_t)1)
       << "\t"
       << tinyxlarge_time.total() / std::max(tinyxlarge.total(), (size_t)1)
       << "\t" << tinyxlarge_dcmp.total() << "    \t" << tinyxlarge_time.total()
       << "    \t"
       << tinyxlarge.total() / std::max(tinyxlarge_time.total() / 8, (double)1.)
       << std::endl;
    std::cout << "large      \t" << large.total() << "\t"
              << large_dcmp.total() / std::max(large.total(), (size_t)1) << "\t"
              << large_time.total() / std::max(large.total(), (size_t)1) << "\t"
              << large_dcmp.total() << "  \t" << large_time.total() << "    \t"
              << large.total() / std::max(large_time.total() / 8, (double)1.)
              << std::endl;
    std::cout << "small      \t" << small.total() << "\t"
              << small_dcmp.total() / std::max(small.total(), (size_t)1) << "\t"
              << small_time.total() / std::max(small.total(), (size_t)1) << "\t"
              << small_dcmp.total() << "  \t" << small_time.total() << "    \t"
              << small.total() / std::max(small_time.total() / 8, (double)1.)
              << std::endl;
#endif
  }

  void set_bitvector_cutoff(size_t n) {
    this->bitvector_cutoff = n;
  }

  void set_query_params(QueryParams qp, size_t weight_class) {
    this->QP[weight_class] = qp;
  }

  void set_build_params(BuildParams bp, size_t weight_class) {
    this->BP[weight_class] = bp;
  }

  void set_materialized_joins(bool b) {
    this->materialized_joins = b;
  }

  void set_materialized_join_cutoff(size_t n) {
    this->join_cutoff = n;
  }

  std::vector<py::tuple> get_log() const {
#ifdef LUMBERJACK

    auto log = logger.get();
    auto out = std::vector<py::tuple>(log.size());

    for (size_t i = 0; i < log.size(); i++) {
      out[i] = py::cast(log[i]);
    }

    return out;

#else

    return std::vector<py::tuple>();

#endif
  }

  int64_t get_dcmps() const {
#ifdef COUNTERS

    return largexlarge_dcmp.total() + largexsmall_dcmp.total() +
           smallxsmall_dcmp.total() + tinyxlarge_dcmp.total() +
           large_dcmp.total() + small_dcmp.total();

#else
  
      return 0;
#endif
  }

  size_t footprint() const {
    size_t total = 0;
    for (const auto& pl : this->posting_lists) {
      total += pl->footprint();
    }
    return total;
  }
};

#endif   // IVF_H