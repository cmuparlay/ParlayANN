//Interface for kmeans structs
//kmeans structs are required to have a "cluster_middle" function, which will run k-means on a set of data
//for explanations of common variable names, please see the README

#ifndef KMEANS_H
#define KMEANS_H

#include "distance.h"
#include "initialization.h"
#include "kmeans_bench.h"
#include "parse_files.h"

#include "parlay/internal/get_time.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"
#include "parlay/sequence.h"
#include "parlay/slice.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <set>
#include <type_traits>
#include <utility>

#include "../utils/euclidian_point.h"
#include "../utils/point_range.h"


// T = point data type e.g. float, uint8_t, int8_t
// CT = Center (data) type e.g. float, double
// index_type = type used for the indexing/assignments of points e.g., size_t
// Point = ParlayANN point object used (e.g. Euclidian_Point<uint8_t>)
// CenterPoint = ParlayANN center point object used (e.g. Euclidian_Point<float>)
template <typename T, typename Point, typename index_type, typename CT,
          typename CenterPoint>
struct KmeansInterface {

  //CITE this function adapted from from Ben's IVF branch code
  parlay::sequence<parlay::sequence<index_type>> get_clusters(index_type* asg,
                                                              size_t n,
                                                              size_t k) {

    auto pairs =
       parlay::tabulate(n, [&](size_t i) { return std::make_pair(asg[i], i); });
    return parlay::group_by_index(pairs, k);
  }

  // given a PointRange and k, cluster will use k-means clustering to partition
  // the points into k groups
  //returns a pair. The first element of the pair is the assignments, in the form of a parlay sequence of sequences. Here, seq[i] gives a sequence containing the ids of all of the points assigned to center i
  //The second element of the pair is a sequence of sequences, containing the coordinates of all of the centers
  //The point of cluster is to provide the user easy access to k-means, if they don't want to think about the internals. 
  //Thus the user cannot choose the max # of iterations, epsilon, initialization method, etc. If the user wants this control they should use cluster_middle
  virtual std::pair<parlay::sequence<parlay::sequence<index_type>>,
                    parlay::sequence<parlay::sequence<CT>>>
  cluster(PointRange<T, Point> points, size_t k) {

    T* v = points.get_values();               // v is array of point coordinates
    size_t d = points.dimension();            // d is # of dimensions
    size_t ad = points.aligned_dimension();   // ad is aligned dimension is # of
                                              // dimensions with padding
    size_t n = points.size();                 // n is # of points
    size_t max_iter = 5;                      // can change max_iter, epsilon
    double epsilon = 0;

    CT* c = new CT[k * ad];                // c stores the centers we find
    index_type* asg = new index_type[n];   // asg (assignment) stores the point
                                           // assignments to centers
    Lazy<T, CT, index_type> init;          // create an initalizer object
    Distance* D =
       new EuclideanDistanceFast();   // create a distance calculation object
    init(v, n, d, ad, k, c, asg);     // initialize c and asg
    kmeans_bench log =
       kmeans_bench(n, d, k, max_iter, epsilon, "Lazy",
                    "Naive");   // logging object that keeps track of time,
                                // center movements, msse, other useful  info

    cluster_middle(
       v, n, d, ad, k, c, asg, *D, log, max_iter,
       epsilon);   // call the parlaykmeans-style clustering algorithm

    std::cout << "Finished cluster" << std::endl;

    auto seq_seq_pt_asgs = get_clusters(asg, n, k); //get the assignments, in sequence form

    parlay::sequence<parlay::sequence<CT>> centers =
       parlay::tabulate(k, [&](size_t i) {
         return parlay::tabulate(d, [&](size_t j) { return c[i * ad + j]; });
       });

    delete[] c;
    delete[] asg;

    return std::make_pair(seq_seq_pt_asgs, centers);
  }

  // cluster_middle is the actual clustering function
  virtual void cluster_middle(T* v, size_t n, size_t d, size_t ad, size_t k,
                              CT* c, size_t* asg, Distance& D,
                              kmeans_bench& logger, size_t max_iter,
                              double epsilon,
                              bool suppress_logging = false) = 0;

  //returns the name of the k-means method
  virtual std::string name() = 0;

  // helpful function for center calculation
  // given a list of assignments (asg) of length n, format the list as a sequence of sequences and put this information into 'grouped'
  // requires integer/size_t keys (for assignments)
  void fast_int_group_by(
     parlay::sequence<std::pair<index_type, parlay::sequence<index_type>>>&
        grouped,
     size_t n, index_type* asg) {

    auto init_pairs = parlay::delayed_tabulate(
       n, [&](index_type i) { return std::make_pair(asg[i], i); });
    parlay::sequence<std::pair<index_type, index_type>> int_sorted =
       parlay::integer_sort(
          init_pairs,
          [&](std::pair<index_type, index_type> p) { return p.first; });

    // store where each center starts 
    auto start_pos =
       parlay::pack_index(parlay::delayed_tabulate(n, [&](size_t i) {
         return i == 0 || int_sorted[i].first != int_sorted[i - 1].first;
       }));
    start_pos.push_back(n);

    grouped = parlay::tabulate(start_pos.size() - 1, [&](size_t i) {
      return std::make_pair(
         int_sorted[start_pos[i]].first,
         parlay::map(
            int_sorted.subseq(start_pos[i], start_pos[i + 1]),
            [&](std::pair<index_type, index_type> ind) { return ind.second; }));
    });
  }

  // given assignments, compute the centers (new center is centroid of points
  // assigned to the center) and store in centers
  void compute_centers(T* v, size_t n, size_t d, size_t ad, size_t k, CT* c,
                       CT* centers, index_type* asg) {

    // copy center coords into centers
    parlay::parallel_for(0, k * ad, [&](size_t i) { centers[i] = 0; });

    // group points by center
    parlay::sequence<std::pair<index_type, parlay::sequence<index_type>>>
       pts_grouped_by_center;
    fast_int_group_by(pts_grouped_by_center, n, asg);

    // add points
    // caution: we can't parallel_for by k, must parallel_for by
    // pts_grouped_by_center.size() because a center can lose all points
    parlay::parallel_for(
       0, pts_grouped_by_center.size(),
       [&](size_t i) {
         size_t picked_center_d = pts_grouped_by_center[i].first * ad;
         for (size_t j = 0; j < pts_grouped_by_center[i].second.size(); j++) {
           size_t point_coord = pts_grouped_by_center[i].second[j] * ad;
           for (size_t coord = 0; coord < d; coord++) {
             centers[picked_center_d + coord] +=
                static_cast<CT>(v[point_coord + coord]);
           }
         }
       },
       1);

    parlay::parallel_for(0, pts_grouped_by_center.size(), [&](size_t i) {
      parlay::parallel_for(0, d, [&](size_t coord) {
        // note that this if condition is necessarily true, because if the list was empty that center wouldn't be in pts_grouped_by_center at all
        if (pts_grouped_by_center[i].second.size() > 0) {
          centers[pts_grouped_by_center[i].first * ad + coord] /=
             pts_grouped_by_center[i].second.size();
        }
      });
    });

    // we need to make sure that we don't wipe centers that lost all their
    // points
    parlay::sequence<bool> empty_center(k, true);

    for (size_t i = 0; i < pts_grouped_by_center.size(); i++) {
      empty_center[pts_grouped_by_center[i].first] = false;
    }
    parlay::parallel_for(0, k, [&](size_t i) {
      if (empty_center[i]) {
        for (size_t j = 0; j < d; j++) {
          centers[i * ad + j] = c[i * ad + j];
        }
      }
    });
  }
  //default destructor
  virtual ~KmeansInterface() {}
};

#endif   // KMEANS_H