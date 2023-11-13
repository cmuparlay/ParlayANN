#ifndef KMEANS_H
#define KMEANS_H

#include "naive.h"
#include "parse_files.h"
#include "distance.h"
#include "kmeans_bench.h"
#include "initialization.h"

#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"
#include "parlay/slice.h"
#include "parlay/random.h"
#include "parlay/internal/get_time.h"

#include <iostream>
#include <algorithm>
#include <chrono>
#include <random>
#include <set>
#include <atomic>
#include <utility>
#include <type_traits>
#include <cmath>

#include "../utils/point_range.h"
#include "../utils/euclidian_point.h"

//Kmeans struct interface -- needs a cluster function
template<typename T, typename Point, typename index_type, typename float_type, typename CenterPoint>
struct KmeansInterface {


  virtual std::pair<parlay::sequence<parlay::sequence<index_type>>,PointRange<float_type,CenterPoint>> cluster(PointRange<T,Point> points, size_t k)=0;

};

//helpful function for center calculation
//requires integer keys
template<typename index_type>
void fast_int_group_by(parlay::sequence<std::pair<index_type, parlay::sequence<index_type>>>& grouped, size_t n, index_type* asg) {
   
   auto init_pairs = parlay::delayed_tabulate(n,[&] (size_t i) {
    return std::make_pair(asg[i],i);
  });
   parlay::sequence<std::pair<size_t,size_t>> int_sorted = parlay::integer_sort(init_pairs, [&] (std::pair<size_t,size_t> p) {return p.first;});

    //store where each center starts in int_sorted
   auto start_pos = parlay::pack_index(parlay::delayed_tabulate(n,[&] (size_t i) {
    return i==0 || int_sorted[i].first != int_sorted[i-1].first;
  }));
  start_pos.push_back(n);
  grouped=parlay::tabulate(start_pos.size()-1, [&] (size_t i) {
    return std::make_pair(int_sorted[start_pos[i]].first, parlay::map(int_sorted.subseq(start_pos[i],start_pos[i+1]),[&] (std::pair<size_t,size_t> ind) { return ind.second;}) );

  });



}


#endif //KMEANS_H