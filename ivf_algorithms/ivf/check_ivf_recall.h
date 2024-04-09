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
ivf_result checkRecall(
        PL &PostingList,
        PointRange &Base_Points,
        PointRange &Query_Points,
        groundTruth<indexType> GT,
        long n_probes, 
        long k) {
    if (GT.size() > 0 && k > GT.dimension()) {
        std::cout << k << "@" << k << " too large for ground truth data of size "
                << GT.dimension() << std::endl;
        abort();
    }

    parlay::sequence<parlay::sequence<indexType>> all_ngh(Query_Points.size());

    parlay::internal::timer t;
    float query_time;
    stats<indexType> QueryStats(Query_Points.size());
    parlay::parallel_for(0, Query_Points.size(), [&] (size_t i){
        auto [frontier, dist_cmps] = PostingList.ivf_knn(Query_Points[i], k, n_probes);
        QueryStats.increment_dist(i, dist_cmps);
        all_ngh[i] = parlay::tabulate(frontier.size(), [&] (size_t j) {return frontier[j].first;});
    });
    query_time = t.next_time();



    float recall = 0.0;

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
  
  float QPS = Query_Points.size() / query_time;
  auto stats_ = QueryStats.dist_stats();

  parlay::sequence<size_t> cast_stats = {static_cast<size_t>(stats_[0]), static_cast<size_t>(stats_[1])};
  ivf_result N(recall, cast_stats, QPS, k, n_probes, Query_Points.size());
  return N;
}

void write_to_csv(std::string csv_filename, parlay::sequence<float> buckets,
                  parlay::sequence<ivf_result> results, IVF_ I) {
  csvfile csv(csv_filename);
  csv << "Algorithm type" << "Parameters" << "Index size" << "Build Time" << endrow;
  csv << I.name << I.params << I.size << I.time << endrow;
  csv << "Num queries"
      << "Target recall"
      << "Actual recall"
      << "QPS"
      << "Average Cmps"
      << "Tail Cmps"
      << "k"
      << "n_probes" << endrow;
  for (int i = 0; i < results.size(); i++) {
    ivf_result N = results[i];
    csv << N.num_queries << buckets[i] << N.recall << N.QPS << N.avg_cmps
        << N.tail_cmps << N.k << N.n_probes << endrow;
  }
  csv << endrow;
  csv << endrow;
}



template<typename Point, typename PointRange, typename indexType, typename PL>
void search_and_parse(PL &PostingList, PointRange &Base_Points,
   PointRange &Query_Points, 
  groundTruth<indexType> GT, char* res_file, long k, IVF_ I){

  parlay::sequence<ivf_result> results;
  std::vector<long> n_probes;



//   n_probes = {1,2,5,10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 26, 28, 30, 32, 
//           34, 36, 38, 40, 45, 50, 55, 60, 65, 70, 80, 90, 100, 120, 140, 160, 
//           180, 200, 225, 250, 275, 300, 375, 500, 750, 1000}; 

    n_probes = {1,5,10,20,50,100,200,500,1000};


    for (long np : n_probes) {
        results.push_back(checkRecall<Point, PointRange, indexType>(PostingList, Base_Points, Query_Points, GT, np, k));
    }


    parlay::sequence<float> buckets =  {.1, .2, .3,  .4,  .5,  .6, .7, .75,  .8, .85,                                                                                            
                                        .9, .93, .95, .97, .98, .99, .995, .999, .9995, 
                                        .9999, .99995, .99999};
    auto [res, ret_buckets] = parse_result(results, buckets);
    std::cout << std::endl;
    if (res_file != NULL)
      write_to_csv(std::string(res_file), ret_buckets, res, I);
  
}
