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

template<typename Point, typename PointRange, typename QPointRange, typename indexType>
void checkRangeRecall(
        Graph<indexType> &G,
        PointRange &Base_Points, PointRange &Query_Points,
        QPointRange &Q_Base_Points, QPointRange &Q_Query_Points,
        RangeGroundTruth<indexType> GT, QueryParams QP,
        long start_point,parlay::sequence<indexType> &active_indices) {

  if(QP.range_query_type == Doubling) {
    
    parlay::internal::timer t;
    float query_time;
    stats<indexType> QueryStats(Query_Points.size());
    parlay::sequence<indexType> start_points = {static_cast<indexType>(start_point)};
    
    auto [all_rr,timings] = DoubleBeamRangeSearch(G,
                                                  Query_Points, Base_Points,
                                                  Q_Query_Points, Q_Base_Points,
                                                  QueryStats, start_points, QP, active_indices);
    query_time = t.next_time();
    auto [beam_search_time, other_time] = timings;
    
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
    QP.print();
    std::cout << ", Point Recall=" << pointwise_recall
              << ", Cum Recall=" << cumulative_recall
              << ", Comparisons=" << QueryStats.dist_stats()[0]
              << ", Visited=" << QueryStats.visited_stats()[0]
              << ", QPS=" << QPS
              << ", ctime=" << (1e9 / (QPS * QueryStats.dist_stats()[0]))
              << ", timings= [" << beam_search_time<< ","<< other_time <<"]"
              << std::endl;
    
  } else if (QP.range_query_type == Greedy || QP.range_query_type == Beam) {

  float query_time;
  stats<indexType> QueryStats(Query_Points.size());
  parlay::sequence<indexType> start_points = {static_cast<indexType>(start_point)};
  parlay::internal::timer t;  

  auto [all_rr, timings] = RangeSearch<Point,PointRange,QPointRange,indexType>(G,
                                                                    Query_Points, Base_Points,
                                                                    Q_Query_Points, Q_Base_Points,
                                                                    QueryStats, start_point, QP);
  auto [beam_search_time, other_time] = timings;
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
  QP.print();
    std::cout << ", Point Recall=" << pointwise_recall
              << ", Cum Recall=" << cumulative_recall
              << ", Comparisons=" << QueryStats.dist_stats()[0]
              << ", Visited=" << QueryStats.visited_stats()[0]
              << ", QPS=" << QPS
              << ", ctime=" << (1e9 / (QPS * QueryStats.dist_stats()[0]))
              << ", timings= [" << beam_search_time<< ","<< other_time <<"]"
              << std::endl;
  }
  else {
    std::cout << "Error: No beam search type provided, -seach_mode should be one of [doubling, greedy, beam]" << std::endl;
  }
}


template<typename Point, typename PointRange, typename QPointRange, typename indexType>
void range_search_wrapper(Graph<indexType> &G,
                          PointRange &Base_Points, PointRange &Query_Points,
                          QPointRange &Q_Base_Points, QPointRange &Q_Query_Points, 
                          RangeGroundTruth<indexType> GT, indexType start_point=0,
                          bool is_early_stopping = false, double esr = 0.0,
                          rangeQueryType rtype = None, double rad = 0.0) {

  std::vector<long> beams;

  beams = {1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 41, 42, 43, 44, 45, 50, 60, 70, 80, 90, 100, 150, 190, 200, 300, 400, 500, 600, 650, 700, 800, 1000, 2000, 3000}; 

  long es = 0;

  parlay::sequence<indexType> all = parlay::tabulate(Query_Points.size(), [&] (indexType i){return i;});
  parlay::sequence<double> cumulative_recall;
  parlay::sequence<std::pair<double,double>> timings;
  parlay::sequence<long> beam_size;



  for(long b: beams){
    if (is_early_stopping) 
      es = std::max((long)10, b/4);

    QueryParams QP(b, b, 0.0, G.size(), G.max_degree(),
                   is_early_stopping, esr, es, rtype, rad);

    
    checkRangeRecall<Point>(G,
                            Base_Points, Query_Points,
                            Q_Base_Points, Q_Query_Points,
                            GT, QP, start_point, all);

  }
  

  
}

} // end namespace
