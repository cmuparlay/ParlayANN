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



#endif //KMEANS_H