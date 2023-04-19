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
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "indexTools.h"
#include <set>
#include "types.h"
#include "beamSearch.h"
#include "csvfile.h"

template<typename T>
range_result checkRecall(
        parlay::sequence<Tvec_point<T>*> &v,
        parlay::sequence<Tvec_point<T>*> &q,
        parlay::sequence<ivec_point> groundTruth,
        int k,
        int beamQ,
        float cut,
        double rad,
        double slack,
        bool random=true,
        int start_point=0) {
  //run twice
  parlay::internal::timer t;
  unsigned d = (v[0]->coordinates).size();
  float query_time;
  if(random){
    rangeSearchRandom(q, v, beamQ, d, rad, k, cut, slack);
    t.next_time();
    rangeSearchRandom(q, v, beamQ, d, rad, k, cut, slack);
    query_time = t.next_time();
  } else{
    rangeSearchAll(q, v, beamQ, d, v[start_point], rad, k, cut, slack);
    t.next_time();
    rangeSearchAll(q, v, beamQ, d, v[start_point], rad, k, cut, slack);
    query_time = t.next_time();
  }
  
  //for range search, disambiguate zero and nonzero queries
  float nonzero_correct = 0.0;
  float zero_correct = 0.0;
  int num_nonzero=0;
  int num_zero=0;

  float num_entries=0;
  float num_reported=0;

  size_t n = q.size();
  int numCorrect = 0;
  for(int i=0; i<n; i++){
    if(groundTruth[i].coordinates.size() == 0){
      num_zero += 1;
      if(q[i]->ngh.size()==0) {zero_correct += 1.0;}
    }else{
      //since the graph-based nearest neighbor algorithms are exact, no need to check IDs
      num_nonzero += 1;
      int num_real_results = groundTruth[i].coordinates.size();
      int num_correctly_reported = q[i]->ngh.size();
      num_entries += (float) num_real_results;
      num_reported += (float) num_correctly_reported;
      nonzero_correct += static_cast<float>(num_correctly_reported)/static_cast<float>(num_real_results);
    }
  }
  
  float nonzero_recall = nonzero_correct/static_cast<float>(num_nonzero);
  float zero_recall = zero_correct/static_cast<float>(num_zero);
  float total_recall = (nonzero_correct + zero_correct)/static_cast<float>(num_nonzero + num_zero);

  float alt_recall = num_reported/num_entries;

  float QPS = q.size()/query_time;

  auto res = query_stats(q);

  range_result R(q.size(), num_nonzero, nonzero_recall, alt_recall, res, QPS, k, beamQ, cut, slack);
  return R;
}

void write_to_csv(std::string csv_filename, parlay::sequence<float> buckets, 
  parlay::sequence<range_result> results, Graph G){
  csvfile csv(csv_filename);
  csv << "GRAPH" << "Parameters" << "Size" << "Build time" << "Avg degree" << "Max degree" << endrow;
  csv << G.name << G.params << G.size << G.time << G.avg_deg << G.max_deg << endrow;
  csv << endrow;
  csv << "Num queries" << "Num nonzero queries" << "Target recall" << "Actual recall" << 
  "Alternative recall" << "QPS" << "Average cmps" << "Tail cmps" << "Average visited" <<
  "Tail visited" <<  "k" << "Q" << "cut" << "slack" << endrow;
  for(int i=0; i<results.size(); i++){
    range_result R = results[i];
    csv << R.num_queries << R.num_nonzero_queries << buckets[i] << R.recall <<
    R.alt_recall << R.QPS << R.avg_cmps << R.tail_cmps << R.avg_visited << R.tail_visited <<
    R.k << R.beamQ << R.cut << R.slack << endrow;
  }
  csv << endrow;
  csv << endrow; 
}

template<typename T>
void search_and_parse(Graph G, parlay::sequence<Tvec_point<T>*> &v, parlay::sequence<Tvec_point<T>*> &q, 
    parlay::sequence<ivec_point> groundTruth, double rad, char* res_file, bool random=true, int start_point=0){
    unsigned d = v[0]->coordinates.size();

    parlay::sequence<range_result> R;
    std::vector<int> beams = {15, 20, 30, 50, 75, 100, 125, 250, 500};
    std::vector<int> allk = {10, 15, 20, 30, 50, 100};
    std::vector<float> slacks = {1.5};
    for(float slack : slacks){
        for(float Q : beams){
            for(float K : allk){
                if(Q>K) R.push_back(checkRecall(v, q, groundTruth, K, Q, 1.14, rad, slack, random, start_point));
            }
        }
    }

    // check "high accuracy accuracy"
    std::vector<int> highbeams = {1000, 2000};
    for(float Q : highbeams){
      R.push_back(checkRecall(v, q, groundTruth, 100, Q, 10.0, rad, 5.0, random, start_point));
    }
    

    parlay::sequence<float> buckets = {.1, .15, .2, .25, .3, .35, .4, .45, .5, .55, .6, .65, .7, .75, .8, .83, .85, .87, .9, .83, .95, .97, .99, .995, .999};
    auto [results, res_buckets] = parse_result(R, buckets);
    if(res_file != NULL) write_to_csv(std::string(res_file), res_buckets, results, G);
}
