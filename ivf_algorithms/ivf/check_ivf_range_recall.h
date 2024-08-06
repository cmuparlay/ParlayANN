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

#include <algorithm>
#include <set>


#include "utils/csvfile.h"
#include "utils/parse_results.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "utils/types.h"
#include "utils/stats.h"

template<typename Point, typename PointRange, typename indexType, typename PL>
ivf_range_result checkRangeRecall(
        PL &PostingList,
        PointRange &Base_Points,
        PointRange &Query_Points,
        RangeGroundTruth<indexType> GT,
        long n_probes, 
        double rad) {


    parlay::sequence<parlay::sequence<indexType>> all_rr(Query_Points.size());

    parlay::internal::timer t;
    float query_time;
    stats<indexType> QueryStats(Query_Points.size());
    parlay::parallel_for(0, Query_Points.size(), [&] (size_t i){
        auto [frontier, dist_cmps] = PostingList.ivf_range(Query_Points[i], rad, n_probes);
        QueryStats.increment_dist(i, dist_cmps);
        all_rr[i] = parlay::tabulate(frontier.size(), [&] (size_t j) {return frontier[j].first;});
    });
    query_time = t.next_time();

    
    float pointwise_recall = 0.0;
    float reported_results = 0.0;
    float total_results = 0.0;
    float num_nonzero = 0.0;

    //since distances are exact, just have to cross-check number of results
    size_t n = Query_Points.size();
    int numCorrect = 0;
    for (indexType i = 0; i < n; i++) {
      float num_reported_results = all_rr[i].size();
      float num_actual_results = GT[i].size();
      reported_results += num_reported_results;
      total_results += num_actual_results;
      if(num_actual_results != 0) {pointwise_recall += num_reported_results/num_actual_results; num_nonzero++;}
    }
    
    pointwise_recall /= num_nonzero;
    float cumulative_recall = reported_results/total_results;
  
  float QPS = Query_Points.size() / query_time;
  auto stats_ = QueryStats.dist_stats();

  parlay::sequence<size_t> cast_stats = {static_cast<size_t>(stats_[0]), static_cast<size_t>(stats_[1])};
  ivf_range_result N(pointwise_recall, cumulative_recall, cast_stats, QPS, rad, n_probes, Query_Points.size());

  N.print();
  return N;
}

template<typename Point, typename PointRange, typename indexType, typename PL>
ivf_range_result checkRangeRecallOverSubset(
        PL &PostingList,
        PointRange &Base_Points,
        PointRange &Query_Points,
        RangeGroundTruth<indexType> GT,
        long n_probes, 
        double rad, parlay::sequence<indexType> &active_indices) {


    parlay::sequence<parlay::sequence<indexType>> all_rr(active_indices.size());

    parlay::internal::timer t;
    float query_time;
    stats<indexType> QueryStats(Query_Points.size());
    parlay::parallel_for(0, active_indices.size(), [&] (size_t i){
        auto [frontier, dist_cmps] = PostingList.ivf_range(Query_Points[active_indices[i]], rad, n_probes);
        QueryStats.increment_dist(i, dist_cmps);
        all_rr[i] = parlay::tabulate(frontier.size(), [&] (size_t j) {return frontier[j].first;});
    });
    query_time = t.next_time();

    
    float pointwise_recall = 0.0;
    float reported_results = 0.0;
    float total_results = 0.0;
    float num_nonzero = 0.0;

    //since distances are exact, just have to cross-check number of results
    size_t n = active_indices.size();
    int numCorrect = 0;
    for (indexType i = 0; i < n; i++) {
      float num_reported_results = all_rr[i].size();
      float num_actual_results = GT[active_indices[i]].size();
      reported_results += num_reported_results;
      total_results += num_actual_results;
      if(num_actual_results != 0) {pointwise_recall += num_reported_results/num_actual_results; num_nonzero++;}
    }
    
    pointwise_recall /= num_nonzero;
    float cumulative_recall = reported_results/total_results;
  
  float QPS = Query_Points.size() / query_time;
  auto stats_ = QueryStats.dist_stats();

  parlay::sequence<size_t> cast_stats = {static_cast<size_t>(stats_[0]), static_cast<size_t>(stats_[1])};
  ivf_range_result N(pointwise_recall, cumulative_recall, cast_stats, QPS, rad, n_probes, active_indices.size());

  N.print();
  return N;
}

template<typename Point, typename PointRange, typename indexType, typename PL>
ivf_range_result checkRangeRecallOverSubsetDistCmp(
        PL &PostingList,
        PointRange &Base_Points,
        PointRange &Query_Points,
        RangeGroundTruth<indexType> GT,
        long n_probes, 
        double rad, parlay::sequence<indexType> &active_indices) {


    parlay::sequence<parlay::sequence<indexType>> all_rr(active_indices.size());

    parlay::internal::timer t;
    float query_time;
    stats<indexType> QueryStats(Query_Points.size());
    size_t centroid_dist_cmp = 0;
    size_t search_dist_cmp = 0;
    parlay::parallel_for(0, active_indices.size(), [&] (size_t i){
        auto [frontier, dist_cmps_pair] = PostingList.ivf_range_dist_cmp(Query_Points[active_indices[i]], rad, n_probes);
        auto [dist_cmps_1,dist_cmps_2] = dist_cmps_pair;
        QueryStats.increment_dist(i, dist_cmps_1+dist_cmps_2);
        centroid_dist_cmp += dist_cmps_1;
        search_dist_cmp += dist_cmps_2;
        all_rr[i] = parlay::tabulate(frontier.size(), [&] (size_t j) {return frontier[j].first;});
    });
    query_time = t.next_time();
    std::cout << "centroid dist cmp: " << centroid_dist_cmp << ", search_dist_cmp: " << search_dist_cmp << std::endl;

    
    float pointwise_recall = 0.0;
    float reported_results = 0.0;
    float total_results = 0.0;
    float num_nonzero = 0.0;

    //since distances are exact, just have to cross-check number of results
    size_t n = active_indices.size();
    int numCorrect = 0;
    for (indexType i = 0; i < n; i++) {
      float num_reported_results = all_rr[i].size();
      float num_actual_results = GT[active_indices[i]].size();
      reported_results += num_reported_results;
      total_results += num_actual_results;
      if(num_actual_results != 0) {pointwise_recall += num_reported_results/num_actual_results; num_nonzero++;}
    }
    
    pointwise_recall /= num_nonzero;
    float cumulative_recall = reported_results/total_results;
  
  float QPS = Query_Points.size() / query_time;
  auto stats_ = QueryStats.dist_stats();

  parlay::sequence<size_t> cast_stats = {static_cast<size_t>(stats_[0]), static_cast<size_t>(stats_[1])};
  ivf_range_result N(pointwise_recall, cumulative_recall, cast_stats, QPS, rad, n_probes, active_indices.size());

  N.print();
  return N;
}




template<typename Point, typename PointRange, typename indexType, typename PL>
void search_and_parse_range(PL &PostingList, PointRange &Base_Points,
   PointRange &Query_Points, 
  RangeGroundTruth<indexType> GT, char* res_file, double rad, IVF_ I){

  parlay::sequence<ivf_range_result> results;
  std::vector<long> n_probes;

  std::vector<long> beams;

  beams = {1,2,3,4,5,8, 10, 20, 30, 40, 50, 100, 200, 500, 1000, 2000, 3000}; 
  std::vector<double> slack = {1.0};

  indexType start_point=0;

  //three categories: 0, 1-20, 20+

  parlay::sequence<indexType> zero_res = GT.results_between(0,0);
  parlay::sequence<indexType> nn_res = GT.results_between(1, 20);
  parlay::sequence<indexType> rng_res = GT.results_between(21, std::numeric_limits<indexType>::max());
  parlay::sequence<indexType> all = parlay::tabulate(Query_Points.size(), [&] (indexType i){return i;});

  std::cout << "For all points: " << std::endl;

  std::cout << "Sweeping once with regular beam search" << std::endl;
  for(long b: beams){
    checkRangeRecallOverSubsetDistCmp<Point, PointRange, indexType,PL>(PostingList, Base_Points, Query_Points, GT, b, rad, all);

  }
  std::cout << std::endl; 
  std::cout << std::endl;
  std::cout << "Trying again with two-round search" << std::endl;
  for(long b: beams){
      checkRangeRecallOverSubsetDistCmp<Point, PointRange, indexType,PL>(PostingList, Base_Points, Query_Points, GT, b, rad, all);
  }

  std::cout << std::endl;

  std::cout << "For all " << zero_res.size() << " points with zero results: " << std::endl;

  std::cout << "Sweeping once with regular beam search" << std::endl;
  for(long b: beams){
    checkRangeRecallOverSubset<Point, PointRange, indexType,PL>(PostingList, Base_Points, Query_Points, GT, b, rad, zero_res);

  }
 

  std::cout << std::endl;

  std::cout << "For all " << nn_res.size() <<  " points with 1 to 20 results" << std::endl;

  std::cout << "Sweeping once with regular beam search" << std::endl;
  for(long b: beams){
    checkRangeRecallOverSubset<Point, PointRange, indexType,PL>(PostingList, Base_Points, Query_Points, GT, b, rad, nn_res);

  }

  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << "Trying again with two-round search" << std::endl;
  for(long b: beams){
      checkRangeRecallOverSubset<Point, PointRange, indexType,PL>(PostingList, Base_Points, Query_Points, GT, b, rad, nn_res);
    
  }

  std::cout << std::endl;

  std::cout << "For all " << rng_res.size() <<  " points with greater than 20 results" << std::endl;

  std::cout << "Sweeping once with regular beam search" << std::endl;
  for(long b: beams){
    checkRangeRecallOverSubset<Point, PointRange, indexType,PL>(PostingList, Base_Points, Query_Points, GT, b, rad, rng_res);

  }
  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << "Trying again with two-round search" << std::endl;
  for(long b: beams){
      checkRangeRecallOverSubset<Point, PointRange, indexType,PL>(PostingList, Base_Points, Query_Points, GT, b, rad, rng_res);
  }

  std::cout << std::endl;


    n_probes = {1,2,3,4,5,10,20,50,100,200,500,1000};


    for (long np : n_probes) {
        results.push_back(checkRangeRecall<Point, PointRange, indexType>(PostingList, Base_Points, Query_Points, GT, np, rad));
    } 
}
