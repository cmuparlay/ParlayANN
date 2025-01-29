#ifndef ALGORITHMS_CHECK_NN_RECALL_H_
#define ALGORITHMS_CHECK_NN_RECALL_H_

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

template<typename PointRange, typename QPointRange, typename QQPointRange, typename indexType>
nn_result checkRecall(const Graph<indexType> &G,
                      const PointRange &Base_Points,
                      const PointRange &Query_Points,
                      const QPointRange &Q_Base_Points,
                      const QPointRange &Q_Query_Points,
                      const QQPointRange &QQ_Base_Points,
                      const QQPointRange &QQ_Query_Points,
                      const groundTruth<indexType> &GT,
                      const bool random,
                      const long start_point,
                      const long k,
                      const QueryParams &QP,
                      const bool verbose) {
  using Point = typename PointRange::Point;

  if (GT.size() > 0 && k > GT.dimension()) {
    std::cout << k << "@" << k << " too large for ground truth data of size "
              << GT.dimension() << std::endl;
    abort();
  }

  parlay::sequence<parlay::sequence<indexType>> all_ngh;

  parlay::internal::timer t;
  float query_time;
  stats<indexType> QueryStats(Query_Points.size());
  QueryStats.clear();
  // to help clear the cache between runs
  auto volatile xx = parlay::random_permutation<long>(5000000);
  t.next_time();
  if (random) {
    all_ngh = qsearchAll<PointRange, QPointRange, QQPointRange, indexType>(Query_Points, Q_Query_Points, QQ_Query_Points,
                                                                           G,
                                                                           Base_Points, Q_Base_Points, QQ_Base_Points,
                                                                           QueryStats, start_point, QP, /*random=*/true);
  } else {
    all_ngh = qsearchAll<PointRange, QPointRange, QQPointRange, indexType>(Query_Points, Q_Query_Points, QQ_Query_Points,
                                                                           G,
                                                                           Base_Points, Q_Base_Points, QQ_Base_Points,
                                                                           QueryStats, start_point, QP);
  }
  query_time = t.next_time();
  
  float recall = 0.0;
  //TODO deprecate this after further testing
  bool dists_present = true;
  if (GT.size() > 0 && !dists_present) {
    size_t n = Query_Points.size();
    int numCorrect = 0;
    for (indexType i = 0; i < n; i++) {
      std::set<indexType> reported_nbhs;
      if (all_ngh[i].size() != k) {
        std::cout << "bad number of neighbors reported: " << all_ngh[i].size() << std::endl;
        abort();
      }
      for (indexType l = 0; l < k; l++) reported_nbhs.insert((all_ngh[i])[l]);
      if (reported_nbhs.size() != k) {
        std::cout << "duplicate entries in reported neighbors" << std::endl;
        abort();
      }
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
      Point qp = Query_Points[i];
      float last_dist = qp.distance(Base_Points[GT.coordinates(i, k-1)]);
      //float last_dist = GT.distances(i, k-1);
      for (indexType l = k; l < GT.dimension(); l++) {
        //if (GT.distances(i,l) == last_dist) {
        if (qp.distance(Base_Points[GT.coordinates(i, l)]) == last_dist) {
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
  if (verbose)
    std::cout << "search: Q=" << QP.beamSize << ", k=" << QP.k
              << ", limit=" << QP.limit
      //<< ", dlimit=" << QP.degree_limit
              << ", recall=" << recall
              << ", visited=" << QueryStats.visited_stats()[0]
              << ", comparisons=" << QueryStats.dist_stats()[0]
              << ", QPS=" << QPS
              << ", ctime=" << 1/(QPS*QueryStats.dist_stats()[0]) * 1e9 << std::endl;

  auto stats_ = {QueryStats.dist_stats(), QueryStats.visited_stats()};
  parlay::sequence<indexType> stats = parlay::flatten(stats_);
  nn_result N(recall, stats, QPS, k, QP.beamSize, QP.cut, Query_Points.size(), QP.limit, QP.degree_limit, k);
  return N;
}

void write_to_csv(std::string csv_filename,
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
    csv << N.num_queries << N.recall << N.QPS << N.avg_cmps
        << N.tail_cmps << N.avg_visited << N.tail_visited << N.k << N.beamQ
        << N.cut << endrow;
  }
  csv << endrow;
  csv << endrow;
}

parlay::sequence<long> calculate_limits(size_t upper_bound) {
  parlay::sequence<long> L(6);
  for (float i = 0; i < 6; i++) {
    L[i] = (long)((4 + i) * ((float) upper_bound) * .1);
    //std::cout << L[i - 1] << std::endl;
  }
  //auto limits = parlay::remove_duplicates(L);
  return L; //limits;
}

template<typename PointRange, typename indexType>
void search_and_parse(Graph_ G_,
                      Graph<indexType> &G,
                      PointRange &Base_Points,
                      PointRange &Query_Points,
                      groundTruth<indexType> GT, char* res_file, long k,
                      bool verbose = false,
                      long fixed_beam_width = 0) {
  search_and_parse(G_, G, Base_Points, Query_Points, Base_Points, Query_Points, Base_Points, Query_Points, GT, res_file, k, false, 0u, verbose, fixed_beam_width);
}

template<typename PointRange, typename QPointRange, typename QQPointRange, typename indexType>
void search_and_parse(Graph_ G_,
                      Graph<indexType> &G,
                      PointRange &Base_Points,
                      PointRange &Query_Points,
                      QPointRange &Q_Base_Points,
                      QPointRange &Q_Query_Points,
                      QQPointRange &QQ_Base_Points,
                      QQPointRange &QQ_Query_Points,
                      groundTruth<indexType> GT, char* res_file, long k,
                      bool random = true,
                      indexType start_point = 0,
                      bool verbose = false,
                      long fixed_beam_width = 0,
                      int rerank_factor = 100) {
  parlay::sequence<nn_result> results;
  std::vector<long> beams;
  std::vector<long> allr;
  std::vector<double> cuts;

  auto check = [&] (const long k, const QueryParams QP) {
    return checkRecall(G,
                       Base_Points, Query_Points,
                       Q_Base_Points, Q_Query_Points,
                       QQ_Base_Points, QQ_Query_Points,
                       GT,
                       random,
                       start_point, k, QP, verbose);};

  QueryParams QP;
  QP.limit = (long) G.size();
  QP.rerank_factor = rerank_factor;
  QP.degree_limit = (long) G.max_degree();
  beams = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 200, 250, 275, 300, 375, 500, 750, 1000};
  if(k==0) allr = {10};
  else allr = {k};
  cuts = {1.35};
  
  std::cout<<"Fixed beam width: "<<fixed_beam_width<<std::endl;
  for (long r : allr) {
    results.clear();
    QP.k = r;
    for (float cut : cuts){
      QP.cut = cut;
      for (float Q : beams){
        QP.beamSize = Q;
        if (Q >= r){
          results.push_back(check(r, QP));
        }
      }
    }
    std::cout << std::endl;
    if (res_file != NULL)
      write_to_csv(std::string(res_file), results, G_);
  }
}

// template<typename Point, typename PointRange, typename indexType>
// void search_and_parse(Graph_ G_,
//                       Graph<indexType> &G,
//                       PointRange &Base_Points,
//                       PointRange &Query_Points,
//                       groundTruth<indexType> GT, char* res_file, long k,
//                       bool random=true, indexType start_point=0,
//                       bool verbose=false) {
//   search_and_parse<Point>(G_, G, Base_Points, Query_Points, Base_Points, Query_Points, GT,
//                           res_file, k, random, start_point, verbose);
// }

} // end namespace

#endif // ALGORITHMS_CHECK_NN_RECALL_H_
