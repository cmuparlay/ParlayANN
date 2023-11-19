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
//CT = Center (data) type
template<typename T, typename Point, typename index_type, typename CT, typename CenterPoint>
struct KmeansInterface {

  parlay::sequence<parlay::sequence<index_type>> get_clusters(index_type* asg, size_t n, size_t k) {
   
    auto pairs = parlay::tabulate(n, [&] (size_t i) {
      return std::make_pair(asg[i], i);
    });
    return parlay::group_by_index(pairs, k);
  }

  //given a PointRange and k, cluster will use k-means clustering to partition the points into k groups
  virtual std::pair<parlay::sequence<parlay::sequence<index_type>>,PointRange<CT,CenterPoint>> cluster(PointRange<T,Point> points, size_t k) {
   
    T* v = points.get_values(); //v is array of point coordinates
    size_t d = points.dimension(); //d is # of dimensions
    size_t ad = points.aligned_dimension(); //ad is aligned dimension is # of dimensions with padding
    size_t n = points.size(); //n is # of points
    size_t max_iter = 5; //can change
    double epsilon = 0;

    CT* c = new CT[k*ad]; //c stores the centers we find
    index_type* asg = new index_type[n]; //asg (assignment) stores the point assignments to centers
    Lazy<T,CT,index_type> init; //create an initalizer object
    Distance* D = new EuclideanDistanceFast(); //create a distance calculation object
    init(v,n,d,ad,k,c,asg); //initialize c and asg
    kmeans_bench log = kmeans_bench(n,d,k,max_iter,
    epsilon,"Lazy","Naive"); //logging object that keeps track of time, center movements, msse, other useful  info

  
    cluster_middle(v,n,d,ad,k,c,asg,*D,log,max_iter,epsilon); //call the parlaykmeans-style clustering algorithm
      
    std::cout << "Finished cluster" << std::endl;

    CT* check;
    //TODO FIXME Seg fault when try to return the centers themselves, bye is a dummy value for now
    PointRange<CT,CenterPoint> bye(check,0,0,0);
    //PointRange<CT,CenterPoint> final_centers(c,n,d,ad);

    std::cout << "Assigned final centers" << std::endl;

    auto seq_seq_pt_asgs = get_clusters(asg,n,k);

    std::cout << "Here now" << std::endl;

    delete[] c;
    delete[] asg;

    std::cout << "Made it here" << std::endl;

    return std::make_pair(seq_seq_pt_asgs,bye);


  }

  //cluster_middle is the actual clustering function
  virtual void cluster_middle(T* v, size_t n, size_t d, size_t ad, size_t k, 
CT* c, size_t* asg, Distance& D, kmeans_bench& logger, size_t max_iter, double epsilon,bool suppress_logging=false) = 0;


//helpful function for center calculation
//requires integer keys
void fast_int_group_by(parlay::sequence<std::pair<index_type, parlay::sequence<index_type>>>& grouped, size_t n, index_type* asg) {
   
   auto init_pairs = parlay::delayed_tabulate(n,[&] (index_type i) {
    return std::make_pair(asg[i],i);
  });
   parlay::sequence<std::pair<index_type,index_type>> int_sorted = parlay::integer_sort(init_pairs, [&] (std::pair<index_type,index_type> p) {return p.first;});

    //store where each center starts in int_sorted
   auto start_pos = parlay::pack_index(parlay::delayed_tabulate(n,[&] (size_t i) {
    return i==0 || int_sorted[i].first != int_sorted[i-1].first;
  }));
  start_pos.push_back(n);
  
  grouped=parlay::tabulate(start_pos.size()-1, [&] (size_t i) {
    return std::make_pair(int_sorted[start_pos[i]].first, parlay::map(int_sorted.subseq(start_pos[i],start_pos[i+1]),[&] (std::pair<index_type,index_type> ind) { return ind.second;}) );

  });



}





//given assignments, compute the centers (new center is centroid of points assigned to the center) and store in centers
void compute_centers(T* v, size_t n, size_t d, size_t ad, size_t k, CT* c, CT* centers, index_type* asg) {

    //copy center coords into centers
    parlay::parallel_for(0,k*ad,[&] (size_t i) {
        centers[i] = 0;
    });

    //group points by center
    // auto rangn = parlay::delayed_tabulate(n,[&] (index_type i) {return i;});
    // parlay::sequence<std::pair<index_type,parlay::sequence<index_type>>>  pts_grouped_by_center = parlay::group_by_key(parlay::map(rangn,[&] (index_type i) {
    // return std::make_pair(asg[i],i);
    // }));
    //TODO change back to using int sort
    parlay::sequence<std::pair<index_type,parlay::sequence<index_type>>> pts_grouped_by_center; 
    fast_int_group_by(pts_grouped_by_center, n,asg);


    //add points
    //caution: we can't parallel_for by k, must parallel_for by pts_grouped_by_center.size() because a center can lose all points
    parlay::parallel_for(0,pts_grouped_by_center.size(),[&] (size_t i) {
        size_t picked_center_d = pts_grouped_by_center[i].first*ad;
        for (size_t j = 0; j < pts_grouped_by_center[i].second.size(); j++) {
          size_t point_coord = pts_grouped_by_center[i].second[j]*ad;
          for (size_t coord = 0; coord < d; coord++) {
            centers[picked_center_d + coord] += static_cast<CT>(v[point_coord + coord]);
          }
        }
    },1);

    parlay::parallel_for(0,pts_grouped_by_center.size(),[&] (size_t i) {

      parlay::parallel_for(0,d,[&] (size_t coord) {
        //note that this if condition is necessarily true, because if the list was empty that center wouldn't be in pts_grouped_by_center at all
        if (pts_grouped_by_center[i].second.size() > 0) {
          centers[pts_grouped_by_center[i].first*ad+coord] /= pts_grouped_by_center[i].second.size();
        }
       
      });
    
    });

    //we need to make sure that we don't wipe centers that lost all their points
    parlay::sequence<bool> empty_center(k,true);

    for (size_t i = 0; i < pts_grouped_by_center.size(); i++) {
      empty_center[pts_grouped_by_center[i].first]=false;
    }
    parlay::parallel_for(0,k,[&] (size_t i) {
      if (empty_center[i]) {
        for (size_t j = 0; j < d; j++) {
          centers[i*ad+j] = c[i*ad+j];

        }
      }
    });

}

};


#endif //KMEANS_H