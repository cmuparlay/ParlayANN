//improved implementation of naive 

#ifndef Naive
#define Naive

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

template <typename T, typename Point, typename index_type, typename CT, typename CenterPoint>
struct NaiveKmeans : KmeansInterface<T,Point,index_type,CT, CenterPoint> {

//do the actual clustering
void cluster_middle(T* v, size_t n, size_t d, size_t ad, size_t k, 
CT* c, size_t* asg, Distance& D, kmeans_bench& logger, size_t max_iter, double epsilon,bool suppress_logging=false) {

  if (!suppress_logging) std::cout << "running kmeans, ad: " << ad << std::endl;

  //because we copy points into buffers of size 2048, we forbid larger values of d
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
  CT* new_centers = new CT[k*ad]; //do calculations for compute center inside here

  while (iterations < max_iter) {
    iterations++;

    // Assign each point to the closest center

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
    
    this->compute_centers(v,n,d,ad,k,c,new_centers,asg); // Compute new centers

    parlay::sequence<float> deltas = parlay::tabulate(k, [&] (size_t i) {
      return std::sqrt(D.distance(new_centers+i*ad, c + i*ad,d));
    }); //store how much each center moved

    max_diff = *parlay::max_element(deltas); //get the max movement of any center
   
    parlay::parallel_for(0,k*ad,[&](size_t i) { //copy back over centers
      c[i] = new_centers[i];
    });

    float update_time = t.next_time();

    float msse = parlay::reduce(parlay::map(rangn,[&] (size_t i) { 
      float buf[2048];
      T* it = v+i*ad;
      for (size_t i = 0; i < d; i++) buf[i]=*(it++);
      return D.distance(buf,c+asg[i]*ad,d);
    }))/n; //calculate msse
    
    float setup_time = t.next_time(); //setup_time counts msse calculation time
    if (!suppress_logging) logger.add_iteration(iterations,assignment_time,update_time,
    msse, 0, 0, deltas,setup_time);

    if (max_diff <= epsilon) break; //if our centers barely moved, we have essentially converged, so end k-means early
  }
  delete[] new_centers;
}

std::string name() {
  return "naive";
}
};
#endif //Naive