#include <algorithm>
#include <set>

#include "beamSearch.h"
#include "doublingSearch.h"
#include "rangeSearch.h"
#include "csvfile.h"
#include "parse_results.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "types.h"
#include "stats.h"

namespace parlayANN {

template<typename Point, typename PointRange, typename indexType>
void checkRangeRecall(
        Graph<indexType> &G,
        PointRange &Base_Points,
        PointRange &Query_Points,
        RangeGroundTruth<indexType> GT, RangeParams RP,
        long start_point,parlay::sequence<indexType> &active_indices) {

  if(RP.is_double_beam){
    
    parlay::internal::timer t;
    float query_time;
    stats<indexType> QueryStats(Query_Points.size());
    parlay::sequence<indexType> start_points = {static_cast<indexType>(start_point)};
    
  
    
    auto [all_rr,timings] = DoubleBeamRangeSearch<Point, PointRange, indexType>(Query_Points, G, Base_Points, QueryStats, start_points, RP, active_indices);
    query_time = t.next_time();
    

    float pointwise_recall = 0.0;
    float reported_results = 0.0;
    float total_results = 0.0;
    float num_nonzero = 0.0;

      //since distances are exact, just have to cross-check number of results
      size_t n = Query_Points.size();
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
    auto stats_ = {QueryStats.dist_stats(), QueryStats.visited_stats()};
    std::cout << "For ";
    RP.print();
    std::cout << ", Pointwise Recall = " << pointwise_recall << ", Cumulative Recall = " << cumulative_recall << ", QPS = " << QPS << std::endl;
    
  }else{
  parlay::sequence<parlay::sequence<indexType>> all_rr;

  parlay::internal::timer t;
  float query_time;
  stats<indexType> QueryStats(Query_Points.size());
  parlay::sequence<indexType> start_points = {static_cast<indexType>(start_point)};
  
 
  all_rr = RangeSearch<Point, PointRange, indexType>(Query_Points, G, Base_Points, QueryStats, start_point, RP);
  
  //auto [all_rr,timings] = DoubleBeamRangeSearch<Point, PointRange, indexType>(Query_Points, G, Base_Points, QueryStats, start_points, RP, active_indices);
  query_time = t.next_time();
  

  float pointwise_recall = 0.0;
  float reported_results = 0.0;
  float total_results = 0.0;
  float num_nonzero = 0.0;

    //since distances are exact, just have to cross-check number of results
    size_t n = Query_Points.size();
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
  auto stats_ = {QueryStats.dist_stats(), QueryStats.visited_stats()};
  std::cout << "For ";
  RP.print();
  std::cout << ", Pointwise Recall = " << pointwise_recall << ", Cumulative Recall = " << cumulative_recall << ", QPS = " << QPS << std::endl;
  
  }
  
 
  
}


template<typename Point, typename PointRange, typename indexType>
void range_search_wrapper(Graph<indexType> &G, PointRange &Base_Points,
   PointRange &Query_Points, 
  RangeGroundTruth<indexType> GT, double rad,
  indexType start_point=0, bool is_early_stopping = false, bool is_double_beam=false, double esr= 0.0){

  std::vector<long> beams;

  beams = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 100, 1000, 2000, 3000}; 

  long es = 0;

  parlay::sequence<indexType> all = parlay::tabulate(Query_Points.size(), [&] (indexType i){return i;});

  for(long b: beams){
    if(is_early_stopping){
      es = std::max((long)10, b/4);
    }
    RangeParams RP(rad, b, is_early_stopping, is_double_beam, es, esr);

    QueryParams QP(b, b, 0.0, G.size(), G.max_degree(), es,
                   esr, is_early_stopping, false, rad);
    
    checkRangeRecall<Point, PointRange, indexType>(G, Base_Points, Query_Points, GT, RP, start_point, all);
  }
  
}

} // end namespace
