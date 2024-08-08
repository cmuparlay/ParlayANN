// This code is part of the Problem Based Benchmark Suite (PBBS)
// Copyright (c) 2011 Guy Blelloch and the PBBS team
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights (to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include <algorithm>
#include <set>

#include "beamSearch.h"
#include "csvfile.h"
#include "parse_results.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "types.h"
#include "stats.h"

template<typename distanceType, typename indexType>
void convergence_stats(parlay::sequence<parlay::sequence<std::pair<indexType, distanceType>>> visited_order){
  //get the max length of any visit order
  //we expect this to be just a little over L_search
  size_t max_len = 0;
  for(auto seq : visited_order){
    if(seq.size() > max_len) max_len = seq.size();
  }

  std::cout << "Max length: " << max_len << std::endl;

  parlay::sequence<double> avg_dists(max_len);

  //iterate over each coord of avg_dists
  parlay::parallel_for(0, max_len, [&] (size_t i){
    double total = 0;
    size_t nonzero = 0;
    for(size_t j=0; j<visited_order.size(); j++){
      if(visited_order[j].size() > i){
        total += static_cast<double>(visited_order[j][i].second);
        nonzero++;
      }
    }
    double avg = total/static_cast<double>(nonzero);

    avg_dists[i] = static_cast<double>(avg);
  });

  size_t index1 = 0;
  size_t index2 = visited_order.size()/2;
  size_t index3 = visited_order.size()/3;

  auto seq1 = parlay::tabulate(visited_order[index1].size(), [&] (size_t i){return visited_order[index1][i].second;});
  auto seq2 = parlay::tabulate(visited_order[index2].size(), [&] (size_t i){return visited_order[index2][i].second;});
  auto seq3 = parlay::tabulate(visited_order[index3].size(), [&] (size_t i){return visited_order[index3][i].second;});

  std::cout << std::setprecision(2) << std::endl;
  std::cout << "Point 1: ";
  for(float f: seq1){
    std::cout << f << ", ";
  }
  std::cout << std::endl;
  std::cout << "Point 2: ";
   for(float f: seq2){
    std::cout << f << ", ";
  }
  std::cout << std::endl;
  std::cout << "Point 3: ";
   for(float f: seq2){
    std::cout << f << ", ";
  }


  std::cout << std::setprecision(6) << std::endl;

  std::cout << parlay::to_chars(avg_dists) << std::endl;
  std::cout << std::endl;
}


template<typename Point, typename PointRange, typename indexType>
void checkRangeRecall(
        Graph<indexType> &G,
        PointRange &Base_Points,
        PointRange &Query_Points,
        RangeGroundTruth<indexType> GT,
        RangeParams RP,
        long start_point, 
        parlay::sequence<indexType> &active_indices, 
        bool converge_stats = false) {


  // parlay::sequence<parlay::sequence<indexType>> all_rr;

  parlay::internal::timer t;
  float query_time;
  stats<indexType> QueryStats(active_indices.size());
 
  auto [all_rr, visit_order]= RangeSearchOverSubset<Point, PointRange, indexType>(Query_Points, G, Base_Points, QueryStats, start_point, RP, active_indices);
  query_time = t.next_time();
  

  float pointwise_recall = 0.0;
  float reported_results = 0.0;
  float total_results = 0.0;
  float num_nonzero = 0.0;

    //since distances are exact, just have to cross-check number of results
    size_t n = active_indices.size();
    for (indexType i = 0; i < n; i++) {
      float num_reported_results = all_rr[i].size();
      float num_actual_results = GT[active_indices[i]].size();
      reported_results += num_reported_results;
      total_results += num_actual_results;
      if(num_actual_results != 0) {pointwise_recall += num_reported_results/num_actual_results; num_nonzero++;}
    }
    
    pointwise_recall /= num_nonzero;
    float cumulative_recall = reported_results/total_results;
  
  float QPS = active_indices.size() / query_time;
  auto stats_ = {QueryStats.dist_stats(), QueryStats.visited_stats()};
  
  std::cout << "For ";
  RP.print();
  std::cout << ", Pointwise Recall = " << pointwise_recall << ", Cumulative Recall = " << cumulative_recall << ", QPS = " << QPS << std::endl;
  
  // if(converge_stats) convergence_stats(visit_order);
  
}



template<typename Point, typename PointRange, typename indexType>
void range_search_wrapper(Graph<indexType> &G, PointRange &Base_Points,
   PointRange &Query_Points, 
  RangeGroundTruth<indexType> GT, double rad,
  indexType start_point=0){

  std::vector<long> beams;

  beams = {5, 10, 20, 30, 40, 50, 60, 70 , 80, 90 ,100, 125, 150, 175, 200, 350, 500, 1000}; 
  std::vector<double> slack = {1.0};

  //three categories: 0, 1-20, 20+

  parlay::sequence<indexType> zero_res = GT.results_between(0,0);
  parlay::sequence<indexType> nn_res = GT.results_between(1, 2);
  parlay::sequence<indexType> rng_res = GT.results_between(3, std::numeric_limits<indexType>::max());
  parlay::sequence<indexType> all = parlay::tabulate(Query_Points.size(), [&] (indexType i){return i;});

  std::cout << "For all points: " << std::endl;

  
  long bigann_early_stopping_rad = 38000;
  long ssnpp_early_stopping_rad = 200000;
  double msturing_early_stopping_rad = .9;
  double gist_early_stopping_rad = 1.3; 
  double deep_early_stopping_rad = .4;

  long early_stopping_rad = gist_early_stopping_rad;
  long steps_to_stopping = 15;

  // std::cout << "Sweeping once with regular beam search" << std::endl;
  // for(long b: beams){
  //   RangeParams RP(rad, b);
  //   checkRangeRecall<Point, PointRange, indexType>(G, Base_Points, Query_Points, GT, RP, start_point, all, true);

  // }
  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << "Regular range search" << std::endl;
  for(long b: beams){
    for(double sf: slack){
      RangeParams RP(rad, b, sf, true);
      checkRangeRecall<Point, PointRange, indexType>(G, Base_Points, Query_Points, GT, RP, start_point, all);
    }
  }

  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << "Regular range search and early stopping" << std::endl;
  for(long b: beams){
    for(double sf: slack){
      RangeParams RP(rad, b, sf, true, steps_to_stopping, early_stopping_rad);
      checkRangeRecall<Point, PointRange, indexType>(G, Base_Points, Query_Points, GT, RP, start_point, all);
    }
  }

  // std::cout << std::endl;

  // std::cout << "For all " << zero_res.size() << " points with zero results: " << std::endl;

  // std::cout << "Sweeping once with regular beam search" << std::endl;
  // for(long b: beams){
  //   RangeParams RP(rad, b);
  //   checkRangeRecall<Point, PointRange, indexType>(G, Base_Points, Query_Points, GT, RP, start_point, zero_res, true);

  // }
 
  // std::cout << std::endl;
  // std::cout << std::endl;
  // std::cout << "Trying again with early stopping" << std::endl;
  // for(long b: beams){
  //   for(double sf: slack){
  //     RangeParams RP(rad, b, sf, true, steps_to_stopping, early_stopping_rad);
  //     checkRangeRecall<Point, PointRange, indexType>(G, Base_Points, Query_Points, GT, RP, start_point, zero_res);
  //   }
  // }

  // std::cout << std::endl;

  // std::cout << "For all " << nn_res.size() <<  " points with 1 to 2 results" << std::endl;

  // std::cout << "Sweeping once with regular beam search" << std::endl;
  // for(long b: beams){
  //   RangeParams RP(rad, b);
  //   checkRangeRecall<Point, PointRange, indexType>(G, Base_Points, Query_Points, GT, RP, start_point, nn_res, true);

  // }

  // std::cout << std::endl;
  // std::cout << std::endl;
  // std::cout << "Trying again with two-round search" << std::endl;
  // for(long b: beams){
  //   for(double sf: slack){
  //     RangeParams RP(rad, b, sf, true);
  //     checkRangeRecall<Point, PointRange, indexType>(G, Base_Points, Query_Points, GT, RP, start_point, nn_res);
  //   }
  // }

  // std::cout << std::endl;
  // std::cout << std::endl;
  // std::cout << "Trying again with two-round search and early stopping" << std::endl;
  // for(long b: beams){
  //   for(double sf: slack){
  //     RangeParams RP(rad, b, sf, true, steps_to_stopping, early_stopping_rad);
  //     checkRangeRecall<Point, PointRange, indexType>(G, Base_Points, Query_Points, GT, RP, start_point, nn_res);
  //   }
  // }

  // std::cout << std::endl;

  // std::cout << "For all " << rng_res.size() <<  " points with greater than 3 results" << std::endl;

  // std::cout << "Sweeping once with regular beam search" << std::endl;
  // for(long b: beams){
  //   RangeParams RP(rad, b);
  //   checkRangeRecall<Point, PointRange, indexType>(G, Base_Points, Query_Points, GT, RP, start_point, rng_res, true);

  // }
  // std::cout << std::endl;
  // std::cout << std::endl;
  // std::cout << "Trying again with two-round search" << std::endl;
  // for(long b: beams){
  //   for(double sf: slack){
  //     RangeParams RP(rad, b, sf, true);
  //     checkRangeRecall<Point, PointRange, indexType>(G, Base_Points, Query_Points, GT, RP, start_point, rng_res);
  //   }
  // }
  // std::cout << std::endl;
  // std::cout << std::endl;
  // std::cout << "Trying again with two-round search and early stopping" << std::endl;
  // for(long b: beams){
  //   for(double sf: slack){
  //     RangeParams RP(rad, b, sf, true, steps_to_stopping, early_stopping_rad);
  //     checkRangeRecall<Point, PointRange, indexType>(G, Base_Points, Query_Points, GT, RP, start_point, rng_res);
  //   }
  // }

  // std::cout << std::endl;







  
}
