//use LSH bucketing for a good initialization
//CITE: the lsh code was adapted from here: https://github.com/rakri/grann/blob/main/src/lsh.cpp
#ifndef LSHCODE
#define LSHCODE

#include "distance.h"

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
#include <bitset>


template <typename T, typename CT>
struct LSH {

  static const size_t BITSET_MAX = 32; 

  std::string name() {
    return "lsh";
  }

  //comparator for pairs of size_t
  //compare the first elt then the second elt
  struct less_pair {
    bool operator() (const std::pair<size_t,size_t>& x, const std::pair<size_t,size_t>& y) const {
      if (x.first < y.first) {
        return true;
      }
      else if (x.first == y.first) {
        return x.second < y.second;
      }
      else {
        return false;
      }
      
    }

     bool comp(const std::pair<size_t,size_t>& x, const std::pair<size_t,size_t>& y) const {
      if (x.first < y.first) {
        return true;
      }
      else if (x.first == y.first) {
        return x.second < y.second;
      }
      else {
        return false;
      }
      
    }
  };

  //the internal version stores the sorted hashed points in pts_with_hash, for a center computation to later use
  void internal_set_init_asg(T* v, size_t n, size_t d, size_t ad, size_t k, size_t* asg, Distance& D, parlay::sequence<std::pair<size_t,size_t>>& pts_with_hash) {
    //store the coordinates of the hyperplanes we compare against
    parlay::sequence<parlay::sequence<float>> hps(BITSET_MAX,parlay::sequence<float>(d));
    generate_hps(hps,d);

     
    //<hash value, point id> pair
    //get the hash of each point
    pts_with_hash = parlay::tabulate(n, [&] (size_t i) {
    return std::make_pair(get_hash(v+i*ad,hps,n,d),i);
    });
     
    //sort the points by their hash
    parlay::sort_inplace(pts_with_hash,less_pair());

    //noverk = n over k = n divided by k = n/k
    size_t noverk = n/k;

    //bucket the points by their sorted value
    //we allow for uneven bucketing by putting all excess into the last bucket
    parlay::parallel_for(0,n,[&] (size_t i) {
      size_t chosen_center = std::min(i / noverk,k-1);
      asg[pts_with_hash[i].second] = chosen_center;

    });

  }
  
  //just give the assignments, not a full initialization
  void set_init_asg(T* v, size_t n,size_t d,size_t ad,size_t k, size_t* asg, Distance& D) {

    parlay::sequence<std::pair<size_t,size_t>> pts_with_hash;

    internal_set_init_asg(v,n,d,ad,k,asg,D,pts_with_hash);

  }

  //lsh as k-means initialization method
  void operator()(T* v, size_t n,size_t d,size_t ad,size_t k, CT* c, size_t* asg, Distance& D) {

    parlay::sequence<std::pair<size_t,size_t>> pts_with_hash;

    internal_set_init_asg(v,n,d,ad,k,asg,D,pts_with_hash);

    size_t noverk = n/k;

    //compute initial centers, taking advantage of the fact that pts_with_hash is sorted already by center ownership
    parlay::parallel_for(0,k,[&] (size_t i) { //for each center

    for (size_t j = 0; j < noverk; j++) { //for each point belonging to that center
      for (size_t coord = 0; coord < d; coord++) { //for each coordinate
        c[i*ad+coord] += v[pts_with_hash[noverk*i+j].second*ad+coord];

      }


    }
    //the last bucket has extra points
    if (i == k-1) {
      for (size_t j = (n/k)*k; j < n; j++) {
        for (size_t coord = 0; coord < d; coord++) {
          c[(k-1)*ad+coord]+=v[pts_with_hash[j].second*ad+coord];
        }
      }

    }

    });
    
    //divide by # of points in group
    parlay::parallel_for(0,k,[&] (size_t i) {
      if (i == k-1) {
        for (size_t coord = 0; coord < d; coord++) {
          c[i*ad+coord]/=(noverk+n%k);
        }

      }
      else {
        for (size_t coord = 0; coord < d; coord++) {
          c[i*ad+coord]/=noverk;
        }
      }

    });

  }

  void generate_hps(parlay::sequence<parlay::sequence<float>>& hps, size_t d) {
    std::random_device r;
    std::default_random_engine rng{r()};
    std::normal_distribution<float> gaussian_dist;
    for (size_t i = 0; i < BITSET_MAX; i++) {
      hps[i] = parlay::sequence<float>(d);
      for (size_t j = 0; j < d; j++) {
        hps[i][j] = gaussian_dist(rng);
      }
    }

  }

  //given a point and the hyperplanes, hash a point
  size_t get_hash(T* pt, const parlay::sequence<parlay::sequence<float>>& hps, size_t n, size_t d) {

    //we store the hashed value inside bin_list
    std::bitset<BITSET_MAX> bin_list; 

    auto rangd = parlay::iota(d);

    parlay::parallel_for(0,BITSET_MAX,[&] (size_t i) {
      float dot_p = parlay::reduce(parlay::map(rangd, [&] (size_t j) {return hps[i][j] * static_cast<float>(pt[j]);}));
      if (dot_p > 0) {
        bin_list[i] = 1;
      }
      else {
        bin_list[i] = 0;
      }
    });

    //to_ulong a bitset method to get a long out of a bitset
    return bin_list.to_ulong();
  }

};

#endif //LSHCODE