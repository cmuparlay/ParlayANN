#ifndef ALGORITHMS_CHECK_RANGE_RECALL
#define ALGORITHMS_CHECK_RANGE_RECALL

#include <algorithm>
#include <set>

#include "beamSearch.h"
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
        RangeGroundTruth<indexType> GT,
        RangeParams RP,
        long start_point) {


  parlay::sequence<parlay::sequence<indexType>> all_rr;

  parlay::internal::timer t;
  float query_time;
  stats<indexType> QueryStats(Query_Points.size());
 
  all_rr = RangeSearch<Point, PointRange, indexType>(Query_Points, G, Base_Points, QueryStats, start_point, RP);
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


template<typename Point, typename PointRange, typename indexType>
void range_search_wrapper(Graph<indexType> &G, PointRange &Base_Points,
   PointRange &Query_Points, 
  RangeGroundTruth<indexType> GT, double rad,
  indexType start_point=0){

  std::vector<long> beams;

  beams = {10, 20, 30, 40, 50, 100, 1000, 2000, 3000}; 

  for(long b: beams){
    RangeParams RP(rad, b);
    checkRangeRecall<Point, PointRange, indexType>(G, Base_Points, Query_Points, GT, RP, start_point);
  }
  
}

} // end namespace
#endif // ALGORITHMS_CHECK_RANGE_RECALL
