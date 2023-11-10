//improved implementation of naive 

#ifndef Naive2_Kmeans
#define Naive2_Kmeans

#include "parlay/random.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"
#include "parlay/slice.h"
#include "parlay/delayed.h"
#include "parlay/io.h"
#include "parlay/internal/get_time.h"
#include "distance.h"
#include "kmeans_bench.h"
#include "kmeans.h"
#include "initialization.h"
#include "../utils/point_range.h"
#include "../utils/euclidian_point.h"


//TODO unclear which of these needed, remove that which are not
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>



// struct Kmeans : public KMeansInterface {
//   std::pair<parlay::sequence<parlay::sequence<index_type>>,PointRange<float,Point>> cluster(PointRange<T,Point> points, size_t k) {
//     T* vals = points.get_values();
//     size_t d = static_cast<size_t>(points.dimension());
//     size_t ad = static_cast<size_t>(points.algined_dimension());
//   }


// }

template <typename T, typename Point, typename index_type, typename float_type, typename CenterPoint>
struct NaiveKmeans : KmeansInterface<T,Point,index_type,float_type, CenterPoint> {


  // std::pair<parlay::sequence<parlay::sequence<size_t>>,PointRange<float,Euclidian_Point<float> >> cluster(PointRange<T,Euclidian_Point<T>> points, size_t k) {
  //   parlay::sequence<parlay::sequence<size_t>> hi;
  //   T* check;
  //   PointRange<float,Euclidian_Point<float>> bye = PointRange<float,Euclidian_Point<float>>(check,0,0,0);
  //   return std::make_pair(hi,bye);
  // }

  //get_cluster borrowed from Ben's IVF branch :)
  parlay::sequence<parlay::sequence<index_type>> get_clusters(index_type* asg, size_t n, size_t k) {
   
    auto pairs = parlay::tabulate(n, [&] (size_t i) {
      return std::make_pair(asg[i], i);
    });
    return parlay::group_by_index(pairs, k);
  }

   std::pair<parlay::sequence<parlay::sequence<index_type>>,PointRange<float_type,CenterPoint>> cluster(PointRange<T,Point> points, size_t k) {
    // dummy vals
    // parlay::sequence<parlay::sequence<index_type>> hi;
    // float_type* check;
    // PointRange<float_type,CenterPoint> bye(check,0,0,0);
    // return std::make_pair(hi,bye);

    T* v = points.get_values();
    size_t d = points.dimension();
    size_t ad = points.aligned_dimension();
    size_t n = points.size();
    size_t max_iter = 5; //can change
    double epsilon = 0;

    float_type* c = new float_type[k*ad];
    index_type* asg = new index_type[n];
    Lazy<T,index_type> init;
    Distance* D = new EuclideanDistanceFast();
    init(v,n,d,ad,k,c,asg);
    kmeans_bench log = kmeans_bench(n,d,k,max_iter,
    epsilon,"Lazy","Naive");

    

    cluster_middle(v,n,d,ad,k,c,asg,*D,log,max_iter,epsilon);
      
    std::cout << "Finished cluster" << std::endl;


    float_type* check;
    //TODO FIXME Seg fault when try to return the centers themselves, bye is a dummy value for now
    PointRange<float_type,CenterPoint> bye(check,0,0,0);
    //PointRange<float_type,CenterPoint> final_centers(c,n,d,ad);

    std::cout << "Assigned final centers" << std::endl;

    auto seq_seq_pt_asgs = get_clusters(asg,n,k);

    std::cout << "Aquí estamos" << std::endl;

    delete[] c;
    delete[] asg;

    std::cout << "Made it here" << std::endl;

    return std::make_pair(seq_seq_pt_asgs,bye);


  }

//**d before ad
void cluster_middle(T* v, size_t n, size_t d, size_t ad, size_t k, 
float* c, size_t* asg, Distance& D, kmeans_bench& logger, size_t max_iter, double epsilon,bool suppress_logging=false) {

  if (!suppress_logging) std::cout << "running pq" << std::endl;
  
  if (d > 2048) {
    std::cout << "d greater than 2048, too big, printing d: " << 
    d << std::endl;
    abort();
  }

  parlay::internal::timer t = parlay::internal::timer();
  t.start(); //start timer
  
  if(n == 0) return; //default case 

  size_t iterations = 0;
  float max_diff = 0;
  auto rangk = parlay::iota(k);
  auto rangn = parlay::iota(n);
  float* center_calc_float = new float[k*ad]; //do calculations for compute center inside here

  while (iterations < max_iter) {
    iterations++;

    // Assign each point to the closest center
 
    //TODO note that we can't use a closest point function here as what's the type for argument rangk?
    parlay::parallel_for(0, n, [&](size_t p) {
      float buf[2048];
      T* it = v+p*ad;
      for (size_t i = 0; i < d; i++) buf[i]=*(it++);
      
      auto distances = parlay::delayed::map(rangk, [&](size_t r) {
          return D.distance(buf, c+r*ad,d);
      });

      asg[p] = min_element(distances) - distances.begin();
    });

    float assignment_time = t.next_time();

    // Compute new centers
     //copy center coords into center_calc_float
    parlay::parallel_for(0,k*ad,[&] (size_t i) {
        center_calc_float[i] = 0;
    });
    //group points by center
    parlay::sequence<std::pair<size_t,parlay::sequence<size_t>>> pts_grouped_by_center = parlay::group_by_key(parlay::map(rangn,[&] (size_t i) {
    return std::pair(asg[i],i);
    }));
    //add points
    parlay::parallel_for(0,k,[&] (size_t i) {
        size_t picked_center_d = pts_grouped_by_center[i].first*ad;
        for (size_t j = 0; j < pts_grouped_by_center[i].second.size(); j++) {
          size_t point_coord = pts_grouped_by_center[i].second[j]*ad;
          for (size_t coord = 0; coord < d; coord++) {
            center_calc_float[picked_center_d + coord] += static_cast<float>(v[point_coord + coord]);
          }
        }
    },1);

    parlay::parallel_for(0,k,[&] (size_t i) {

      parlay::parallel_for(0,d,[&] (size_t coord) {
        if (pts_grouped_by_center[i].second.size() > 0) {
          center_calc_float[pts_grouped_by_center[i].first*ad+coord] /= pts_grouped_by_center[i].second.size();
        }
        else { //if no points belong to this center
          center_calc_float[pts_grouped_by_center[i].first*ad+coord] = c[pts_grouped_by_center[i].first*ad+coord];
        }
      });
    
    });
   
    parlay::sequence<float> deltas = parlay::tabulate(k, [&] (size_t i) {
      return D.distance(center_calc_float+i*ad, c + i*ad,d);
    });

    max_diff = *parlay::max_element(deltas);
   
    //copy back over centers
    parlay::parallel_for(0,k*d,[&](size_t i) {
      c[i] = center_calc_float[i];
    });

    float update_time = t.next_time();

    float msse = parlay::reduce(parlay::map(rangn,[&] (size_t i) { 
      float buf[2048];
      T* it = v+i*ad;
      for (size_t i = 0; i < d; i++) buf[i]=*(it++);
      return D.distance(buf,c+asg[i]*ad,d);
    }))/n; //calculate msse
    
    float setup_time = t.next_time(); //setup_time counts msse calculation time
    if (!suppress_logging) logger.add_iteration(assignment_time,update_time,
    msse, 0, 0, deltas,setup_time);

    if (max_diff <= epsilon) break;
    
  }

  delete[] center_calc_float;
}

};

#endif //Naive2