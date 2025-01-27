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
void convergence_stats(parlay::sequence<parlay::sequence<std::pair<indexType, distanceType>>> visited_order, std::string restype){
  if(restype=="") return;
  //get the max length of any visit order
  //we expect this to be just a little over L_search
  size_t max_len = 0;
  for(auto seq : visited_order){
    if(seq.size() > max_len) max_len = seq.size();
  }

  std::cout << "Max length: " << max_len << std::endl;

  parlay::sequence<parlay::sequence<double>> all_dists = parlay::tabulate(max_len, [&] (size_t i){
    parlay::sequence<double> dists;
    for(size_t j=0; j<visited_order.size(); j++){
      if(visited_order[j].size() > i){
        dists.push_back(static_cast<double>(visited_order[j][i].second));
      }
    }
    parlay::sort_inplace(dists);
    return dists;
  });

  parlay::sequence<parlay::sequence<double>> filtered_dists = parlay::tabulate(max_len, [&] (size_t i){
    parlay::sequence<double> dists;
    for(size_t j=0; j<visited_order.size(); j++){
      if(visited_order[j].size() > i && visited_order[j][i].first == 0){
        dists.push_back(static_cast<double>(visited_order[j][i].second));
      }
    }
    parlay::sort_inplace(dists);
    return dists;
  });



  std::cout << std::setprecision(6) << std::endl;

  for(size_t i=0; i<max_len; i++){
    std::cout << restype << ", Step " << i << ": " << parlay::to_chars(all_dists[i]) << std::endl;
  }

    for(size_t i=0; i<max_len; i++){
    std::cout << restype << ", Step " << i << ", Filtered: " << parlay::to_chars(filtered_dists[i]) << std::endl;
  }


  
}

template<typename Point, typename PointRange, typename indexType>
void calculateWeightedRecall(
    PointRange& Base_Points, 
    PointRange& Query_Points, 
    RangeGroundTruth<indexType> &GT, 
    parlay::sequence<indexType>& active_indices, 
    parlay::sequence<parlay::sequence<indexType>> range_results, 
    double rad,
    float QPS){

  float pointwise_weighted_recall = 0.0;
  float reported_weight = 0.0;
  float total_weight = 0.0;
  float num_nonzero = 0.0;
  
  for(size_t i=0; i<active_indices.size(); i++){
    if(GT[active_indices[i]].size()==0) continue;
    else{
      num_nonzero += 1;
      double point_weight = 0.0;
      double point_reported_weight = 0.0;
      for(auto j : GT[active_indices[i]]){
        double dist = Query_Points[active_indices[i]].distance(Base_Points[j]);
        double single_weight = (rad - dist)/rad;
        double weight = single_weight*single_weight;
        // std::cout << weight << std::endl;
        point_weight += weight;
        if(parlay::find(range_results[i], j) != range_results[i].end()){
          point_reported_weight += weight;
        }
      }
      if(point_weight > 0) pointwise_weighted_recall += point_reported_weight/point_weight;
      else num_nonzero--;
      total_weight += point_weight;
      reported_weight += point_reported_weight;
    }
    
  }
  double cumulative_weighted_recall = reported_weight/total_weight;
  double pointwise_recall = pointwise_weighted_recall/num_nonzero;

  std::cout << "Pointwise Weighted Recall = " << pointwise_recall << ", Cumulative Weighted Recall = " << cumulative_weighted_recall << ", QPS = " << QPS << std::endl;
}

template<typename Point, typename PointRange, typename indexType>
std::tuple<double, double, double> checkRangeRecall(
        Graph<indexType> &G,
        PointRange &Base_Points,
        PointRange &Query_Points,
        RangeGroundTruth<indexType> GT,
        RangeParams RP,
        long start_point, 
        parlay::sequence<indexType> &active_indices,
        std::string restype = "") {


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
  std::cout << "Convergence stats: " << std::endl;
  convergence_stats(visit_order, restype);
  return std::make_tuple(pointwise_recall, cumulative_recall, QPS);
  // calculateWeightedRecall<Point, PointRange, indexType>(Base_Points, Query_Points, GT, active_indices, all_rr, RP.rad, QPS);
  
  
}

size_t calc_steps_to_stopping(size_t b){
  return std::max<size_t>(10, b/4);
}

template<typename Point, typename PointRange, typename indexType>
void range_search_wrapper(Graph<indexType> &G, PointRange &Base_Points,
   PointRange &Query_Points, 
  RangeGroundTruth<indexType> GT, double rad, double esr,
  indexType start_point=0){

  std::vector<long> beams;

  // beams = {2,4,5,6,8,10,20,30,40,50,60,70,80,90,100,200,350,500,1000}; 
  // beams = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,22,25,28,30,32,35,38,40,42,45,47,50,52,55,58,60,68,70,72,75,78,80,85,90,95,100,110,120,125,150,160,175,200,250,300,315,330,340,350,500,625,750,875,1000}; 
  beams = {100};
  double sf = 1.0;

  //three categories: 0, 1-20, 20+

  parlay::sequence<indexType> zero_res = GT.results_between(0,0);
  parlay::sequence<indexType> nn_res = GT.results_between(1, 2);
  parlay::sequence<indexType> rng_res = GT.results_between(3, std::numeric_limits<indexType>::max());
  parlay::sequence<indexType> all = parlay::tabulate(Query_Points.size(), [&] (indexType i){return i;});

  long steps_to_stopping;


  std::cout << "Radius: " << rad << ", early stopping radius: " << esr << std::endl;
  parlay::sequence<double> pointwise_recall;
  parlay::sequence<double> cumulative_recall;
  parlay::sequence<double> qps;
  for(long b: beams){
    RangeParams RP(rad, b);
    std::tuple<double, double, double> stats = checkRangeRecall<Point, PointRange, indexType>(G, Base_Points, Query_Points, GT, RP, start_point, all);
    pointwise_recall.push_back(std::get<0>(stats));
    cumulative_recall.push_back(std::get<1>(stats));
    qps.push_back(std::get<2>(stats));
  }
  std::cout << "All, Beam Search, Pointwise Recall: " << parlay::to_chars(pointwise_recall) << std::endl;
  std::cout << "All, Beam Search, Cumulative Recall: " << parlay::to_chars(cumulative_recall) << std::endl;
  std::cout << "All, Beam Search, QPS: " << parlay::to_chars(qps) << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;
  pointwise_recall.clear();
  cumulative_recall.clear();
  qps.clear();
 
  for(long b: beams){
    RangeParams RP(rad, b, sf, true);
    std::tuple<double, double, double> stats = checkRangeRecall<Point, PointRange, indexType>(G, Base_Points, Query_Points, GT, RP, start_point, all);
    pointwise_recall.push_back(std::get<0>(stats));
    cumulative_recall.push_back(std::get<1>(stats));
    qps.push_back(std::get<2>(stats));
  }
  std::cout << "All, Greedy Search, Pointwise Recall: " << parlay::to_chars(pointwise_recall) << std::endl;
  std::cout << "All, Greedy Search, Cumulative Recall: " << parlay::to_chars(cumulative_recall) << std::endl;
  std::cout << "All, Greedy Search, QPS: " << parlay::to_chars(qps) << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;
  pointwise_recall.clear();
  cumulative_recall.clear();
  qps.clear();




  for(long b: beams){
    steps_to_stopping = std::max<size_t>(b/4, 10);
    RangeParams RP(rad, b, sf, true, steps_to_stopping, esr);
    std::tuple<double, double, double> stats = checkRangeRecall<Point, PointRange, indexType>(G, Base_Points, Query_Points, GT, RP, start_point, all);
    pointwise_recall.push_back(std::get<0>(stats));
    cumulative_recall.push_back(std::get<1>(stats));
    qps.push_back(std::get<2>(stats));
  }
  std::cout << "All, Early Stopping, Pointwise Recall: " << parlay::to_chars(pointwise_recall) << std::endl;
  std::cout << "All, Early Stopping, Cumulative Recall: " << parlay::to_chars(cumulative_recall) << std::endl;
  std::cout << "All, Early Stopping, QPS: " << parlay::to_chars(qps) << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;
  pointwise_recall.clear();
  cumulative_recall.clear();
  qps.clear();

  for(long b: beams){
    RangeParams RP(rad, b);
    std::tuple<double, double, double> stats = checkRangeRecall<Point, PointRange, indexType>(G, Base_Points, Query_Points, GT, RP, start_point, zero_res, "Zeros");
    qps.push_back(std::get<2>(stats));
  }

  std::cout << "Zeros, Beam Search, QPS: " << parlay::to_chars(qps) << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;
  qps.clear();

  for(long b: beams){
    RangeParams RP(rad, b, sf, true);
    std::tuple<double, double, double> stats = checkRangeRecall<Point, PointRange, indexType>(G, Base_Points, Query_Points, GT, RP, start_point, zero_res);
    qps.push_back(std::get<2>(stats));
  }

  std::cout << "Zeros, Greedy Search, QPS: " << parlay::to_chars(qps) << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;
  qps.clear();
 

  for(long b: beams){
    steps_to_stopping = calc_steps_to_stopping(b);
    RangeParams RP(rad, b, sf, true, steps_to_stopping, esr);
    std::tuple<double, double, double> stats = checkRangeRecall<Point, PointRange, indexType>(G, Base_Points, Query_Points, GT, RP, start_point, zero_res);
    qps.push_back(std::get<2>(stats));
  }

  std::cout << "Zeros, Early Stopping, QPS: " << parlay::to_chars(qps) << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;
  qps.clear();


  for(long b: beams){
    RangeParams RP(rad, b);
    std::tuple<double, double, double> stats = checkRangeRecall<Point, PointRange, indexType>(G, Base_Points, Query_Points, GT, RP, start_point, nn_res, "OneTwos");
    pointwise_recall.push_back(std::get<0>(stats));
    cumulative_recall.push_back(std::get<1>(stats));
    qps.push_back(std::get<2>(stats));
  }

  std::cout << "Onetwos, Beam Search, Pointwise Recall: " << parlay::to_chars(pointwise_recall) << std::endl;
  std::cout << "Onetwos, Beam Search, Cumulative Recall: " << parlay::to_chars(cumulative_recall) << std::endl;
  std::cout << "Onetwos, Beam Search, QPS: " << parlay::to_chars(qps) << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;
  pointwise_recall.clear();
  cumulative_recall.clear();
  qps.clear();

  std::cout << std::endl;
  std::cout << std::endl;
  for(long b: beams){
    RangeParams RP(rad, b, sf, true);
    std::tuple<double, double, double> stats = checkRangeRecall<Point, PointRange, indexType>(G, Base_Points, Query_Points, GT, RP, start_point, nn_res);
    pointwise_recall.push_back(std::get<0>(stats));
    cumulative_recall.push_back(std::get<1>(stats));
    qps.push_back(std::get<2>(stats));
  }

  std::cout << "Onetwos, Greedy Search, Pointwise Recall: " << parlay::to_chars(pointwise_recall) << std::endl;
  std::cout << "Onetwos, Greedy Search, Cumulative Recall: " << parlay::to_chars(cumulative_recall) << std::endl;
  std::cout << "Onetwos, Greedy Search, QPS: " << parlay::to_chars(qps) << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;
  pointwise_recall.clear();
  cumulative_recall.clear();
  qps.clear();

  std::cout << std::endl;
  std::cout << std::endl;
  for(long b: beams){
    steps_to_stopping = calc_steps_to_stopping(b);
    RangeParams RP(rad, b, sf, true, steps_to_stopping, esr);
    std::tuple<double, double, double> stats = checkRangeRecall<Point, PointRange, indexType>(G, Base_Points, Query_Points, GT, RP, start_point, nn_res);
    pointwise_recall.push_back(std::get<0>(stats));
    cumulative_recall.push_back(std::get<1>(stats));
    qps.push_back(std::get<2>(stats));
  }

  std::cout << "Onetwos, Early Stopping, Pointwise Recall: " << parlay::to_chars(pointwise_recall) << std::endl;
  std::cout << "Onetwos, Early Stopping, Cumulative Recall: " << parlay::to_chars(cumulative_recall) << std::endl;
  std::cout << "Onetwos, Early Stopping, QPS: " << parlay::to_chars(qps) << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;
  pointwise_recall.clear();
  cumulative_recall.clear();
  qps.clear();


  for(long b: beams){
    RangeParams RP(rad, b);
    std::tuple<double, double, double> stats = checkRangeRecall<Point, PointRange, indexType>(G, Base_Points, Query_Points, GT, RP, start_point, rng_res, "Threeplus");
    pointwise_recall.push_back(std::get<0>(stats));
    cumulative_recall.push_back(std::get<1>(stats));
    qps.push_back(std::get<2>(stats));
  }

  std::cout << "Threeplus, Beam Search, Pointwise Recall: " << parlay::to_chars(pointwise_recall) << std::endl;
  std::cout << "Threeplus, Beam Search, Cumulative Recall: " << parlay::to_chars(cumulative_recall) << std::endl;
  std::cout << "Threeplus, Beam Search, QPS: " << parlay::to_chars(qps) << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;
  pointwise_recall.clear();
  cumulative_recall.clear();
  qps.clear();

  for(long b: beams){
    RangeParams RP(rad, b, sf, true);
    std::tuple<double, double, double> stats = checkRangeRecall<Point, PointRange, indexType>(G, Base_Points, Query_Points, GT, RP, start_point, rng_res);
    pointwise_recall.push_back(std::get<0>(stats));
    cumulative_recall.push_back(std::get<1>(stats));
    qps.push_back(std::get<2>(stats));
  }

  std::cout << "Threeplus, Greedy Search, Pointwise Recall: " << parlay::to_chars(pointwise_recall) << std::endl;
  std::cout << "Threeplus, Greedy Search, Cumulative Recall: " << parlay::to_chars(cumulative_recall) << std::endl;
  std::cout << "Threeplus, Greedy Search, QPS: " << parlay::to_chars(qps) << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;
  pointwise_recall.clear();
  cumulative_recall.clear();
  qps.clear();


  for(long b: beams){
    steps_to_stopping = calc_steps_to_stopping(b);
    RangeParams RP(rad, b, sf, true, steps_to_stopping, esr);
    std::tuple<double, double, double> stats = checkRangeRecall<Point, PointRange, indexType>(G, Base_Points, Query_Points, GT, RP, start_point, rng_res);
    pointwise_recall.push_back(std::get<0>(stats));
    cumulative_recall.push_back(std::get<1>(stats));
    qps.push_back(std::get<2>(stats));
  }

  std::cout << "Threeplus, Early Stopping, Pointwise Recall: " << parlay::to_chars(pointwise_recall) << std::endl;
  std::cout << "Threeplus, Early Stopping, Cumulative Recall: " << parlay::to_chars(cumulative_recall) << std::endl;
  std::cout << "Threeplus, Early Stopping, QPS: " << parlay::to_chars(qps) << std::endl;
  std::cout << std::endl;
  std::cout << std::endl;
  pointwise_recall.clear();
  cumulative_recall.clear();
  qps.clear();

  
}