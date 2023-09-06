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

#include "beamSearch.h"
#include "csvfile.h"
#include "parse_results.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "types.h"
#include "stats.h"

template<typename Point, typename PointRange, typename indexType>
nn_result checkRecall(
        Graph<indexType> &G,
        PointRange &Base_Points,
        PointRange &Query_Points,
        groundTruth<indexType> GT,
        bool random,
        long start_point, 
        long k,
        QueryParams &QP) {
  if (GT.size() > 0 && k > GT.dimension()) {
    std::cout << k << "@" << k << " too large for ground truth data of size "
              << GT.dimension() << std::endl;
    abort();
  }

  parlay::sequence<parlay::sequence<indexType>> all_ngh;

  parlay::internal::timer t;
  float query_time;
  stats<indexType> QueryStats(Query_Points.size());
  if(random){
    all_ngh = beamSearchRandom<Point, PointRange, indexType>(Query_Points, G, Base_Points, QueryStats, QP);
    t.next_time();
    QueryStats.clear();
    all_ngh = beamSearchRandom<Point, PointRange, indexType>(Query_Points, G, Base_Points, QueryStats, QP);
    query_time = t.next_time();
  }else{
    all_ngh = searchAll<Point, PointRange, indexType>(Query_Points, G, Base_Points, QueryStats, start_point, QP);
    t.next_time();
    QueryStats.clear();
    all_ngh = searchAll<Point, PointRange, indexType>(Query_Points, G, Base_Points, QueryStats, start_point, QP);
    query_time = t.next_time();
  }

  float recall = 0.0;
  //TODO deprecate this after further testing
  bool dists_present = true;
  if (GT.size() > 0 && !dists_present) {
    size_t n = Query_Points.size();
    int numCorrect = 0;
    for (indexType i = 0; i < n; i++) {
      std::set<indexType> reported_nbhs;
      for (indexType l = 0; l < k; l++) reported_nbhs.insert((all_ngh[i])[l]);
      for (indexType l = 0; l < k; l++) {
        if (reported_nbhs.find((GT.coordinates(i,l))) !=
            reported_nbhs.end()) {
          numCorrect += 1;
        }
      }
    }
    recall = static_cast<float>(numCorrect) / static_cast<float>(k * n);
  } else if (GT.size() > 0 && dists_present) {
    size_t n = Query_Points.size();
    
    int numCorrect = 0;
    for (indexType i = 0; i < n; i++) {
      parlay::sequence<int> results_with_ties;
      for (indexType l = 0; l < k; l++)
        results_with_ties.push_back(GT.coordinates(i,l));
      float last_dist = GT.distances(i, k-1);
      for (indexType l = k; l < GT.dimension(); l++) {
        if (GT.distances(i,l) == last_dist) {
          results_with_ties.push_back(GT.coordinates(i,l));
        }
      }
      std::set<int> reported_nbhs;
      for (indexType l = 0; l < k; l++) reported_nbhs.insert((all_ngh[i])[l]);
      for (indexType l = 0; l < results_with_ties.size(); l++) {
        if (reported_nbhs.find(results_with_ties[l]) != reported_nbhs.end()) {
          numCorrect += 1;
        }
      }
    }
    recall = static_cast<float>(numCorrect) / static_cast<float>(k * n);
  }
  float QPS = Query_Points.size() / query_time;
  auto stats_ = {QueryStats.dist_stats(), QueryStats.visited_stats()};
  parlay::sequence<indexType> stats = parlay::flatten(stats_);
  nn_result N(recall, stats, QPS, k, QP.beamSize, QP.cut, Query_Points.size(), QP.limit, QP.degree_limit, k);
  return N;
}

void write_to_csv(std::string csv_filename, parlay::sequence<float> buckets,
                  parlay::sequence<nn_result> results, Graph_ G) {
  csvfile csv(csv_filename);
  csv << "GRAPH"
      << "Parameters"
      << "Size"
      << "Build time"
      << "Avg degree"
      << "Max degree" << endrow;
  csv << G.name << G.params << G.size << G.time << G.avg_deg << G.max_deg
      << endrow;
  csv << endrow;
  csv << "Num queries"
      << "Target recall"
      << "Actual recall"
      << "QPS"
      << "Average Cmps"
      << "Tail Cmps"
      << "Average Visited"
      << "Tail Visited"
      << "k"
      << "Q"
      << "cut" << endrow;
  for (int i = 0; i < results.size(); i++) {
    nn_result N = results[i];
    csv << N.num_queries << buckets[i] << N.recall << N.QPS << N.avg_cmps
        << N.tail_cmps << N.avg_visited << N.tail_visited << N.k << N.beamQ
        << N.cut << endrow;
  }
  csv << endrow;
  csv << endrow;
}

parlay::sequence<long> calculate_limits(size_t upper_bound) {
  parlay::sequence<long> L(9);
  for (float i = 1; i < 10; i++) {
    L[i - 1] = (long)(i * ((float) upper_bound) * .1);
  }
  auto limits = parlay::remove_duplicates(L);
  return limits;
}

template<typename Point, typename PointRange, typename indexType>
void search_and_parse(Graph_ G_, Graph<indexType> &G, PointRange &Base_Points,
   PointRange &Query_Points, 
  groundTruth<indexType> GT, char* res_file, long k,
  bool random=true, indexType start_point=0){

  parlay::sequence<nn_result> results;
  std::vector<long> beams;
  std::vector<long> allr;
  std::vector<double> cuts;

  QueryParams QP;
  QP.limit = (long) G.size();
  QP.degree_limit = (long) G.max_degree();
  beams = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 26, 28, 30, 32, 
          34, 36, 38, 40, 45, 50, 55, 60, 65, 70, 80, 90, 100, 120, 140, 160, 
          180, 200, 225, 250, 275, 300, 375, 500, 750, 1000}; 
  if(k==0) allr = {10};
  else allr = {k};
  cuts = {1.35};

    for (long r : allr) {
      results.clear();
      QP.k = r;
      for (float cut : cuts){
        QP.cut = cut;
        for (float Q : beams){
          QP.beamSize = Q;
          if (Q > r){
            results.push_back(checkRecall<Point, PointRange, indexType>(G, Base_Points, Query_Points, GT, random, start_point, r, QP));
          }
        }
      }
      // check "limited accuracy"
      parlay::sequence<long> limits = calculate_limits(results[0].avg_visited);
      parlay::sequence<long> degree_limits = calculate_limits(G.max_degree());
      degree_limits.push_back(G.max_degree());
      QP = QueryParams(r, r, 1.35, (long) G.size(), (long) G.max_degree());
      for(long l : limits){
        QP.limit = l;
        QP.beamSize = std::max<long>(l, r);
        for(long dl : degree_limits){
          QP.degree_limit = dl;
	        results.push_back(checkRecall<Point, PointRange, indexType>(G, Base_Points, Query_Points, GT, random, start_point, r, QP));
        }
      }
      // check "best accuracy"
      QP = QueryParams((long) 100, (long) 1000, (double) 10.0, (long) G.size(), (long) G.max_degree());
      results.push_back(checkRecall<Point, PointRange, indexType>(G, Base_Points, Query_Points, GT, random, start_point, r, QP));

    parlay::sequence<float> buckets =  {.1, .2, .3,  .4,  .5,  .6, .7, .75,  .8, .85,                                                                                            
                                        .9, .93, .95, .97, .98, .99, .995, .999, .9995, 
                                        .9999, .99995, .99999};
    auto [res, ret_buckets] = parse_result(results, buckets);
    std::cout << std::endl;
    if (res_file != NULL)
      write_to_csv(std::string(res_file), ret_buckets, res, G_);
  }
}
