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

#include <math.h>

#include <algorithm>
#include <random>
#include <set>

#include "../utils/NSGDist.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"
#include "../utils/graph.h"
#include "../utils/types.h"
#include "../utils/stats.h"
#include "../utils/beamSearch.h"


template<typename Point, typename PointRange, typename indexType>
struct knn_index {
  using distanceType = typename Point::distanceType;
  using pid = std::pair<indexType, distanceType>;
  using PR = PointRange;
  using GraphI = Graph<indexType>;
  

  BuildParams BP;
  std::set<indexType> delete_set; 
  indexType start_point;


  knn_index(BuildParams &BP) : BP(BP) {}

  indexType get_start() { return start_point; }

  //robustPrune routine as found in DiskANN paper, with the exception
  //that the new candidate set is added to the field new_nbhs instead
  //of directly replacing the out_nbh of p
  parlay::sequence<indexType> robustPrune(indexType p, parlay::sequence<pid>& cand,
                    GraphI &G, PR &Points,  bool add = true) {
    // add out neighbors of p to the candidate set.
    size_t out_size = G[p].size();
    std::vector<pid> candidates;
    for (auto x : cand) candidates.push_back(x);

    if(add){
      for (size_t i=0; i<out_size; i++) {
        // candidates.push_back(std::make_pair(v[p]->out_nbh[i], Points[v[p]->out_nbh[i]].distance(Points[p])));
        candidates.push_back(std::make_pair(G[p][i], Points[G[p][i]].distance(Points[p])));
      }
    }

    // Sort the candidate set in reverse order according to distance from p.
    auto less = [&](pid a, pid b) { return a.second < b.second; };
    std::sort(candidates.begin(), candidates.end(), less);

    std::vector<indexType> new_nbhs;
    new_nbhs.reserve(BP.R);

    size_t candidate_idx = 0;

    while (new_nbhs.size() < BP.R && candidate_idx < candidates.size()) {
      // Don't need to do modifications.
      int p_star = candidates[candidate_idx].first;
      candidate_idx++;
      if (p_star == p || p_star == -1) {
        continue;
      }

      new_nbhs.push_back(p_star);

      for (size_t i = candidate_idx; i < candidates.size(); i++) {
        int p_prime = candidates[i].first;
        if (p_prime != -1) {
          distanceType dist_starprime = Points[p_star].distance(Points[p_prime]);
          distanceType dist_pprime = candidates[i].second;
          if (BP.alpha * dist_starprime <= dist_pprime) {
            candidates[i].first = -1;
          }
        }
      }
    }

    auto new_neighbors_seq = parlay::to_sequence(new_nbhs);
    return new_neighbors_seq;
  }

  //wrapper to allow calling robustPrune on a sequence of candidates 
  //that do not come with precomputed distances
  parlay::sequence<indexType> robustPrune(indexType p, parlay::sequence<indexType> candidates,
                    GraphI &G, PR &Points, bool add = true){

    parlay::sequence<pid> cc;
    cc.reserve(candidates.size()); // + size_of(p->out_nbh));
    for (size_t i=0; i<candidates.size(); ++i) {
      cc.push_back(std::make_pair(candidates[i], Points[candidates[i]].distance(Points[p])));
    }
    return robustPrune(p, cc, G, Points, add);
  }

  void build_index(GraphI &G, PR &Points, stats<indexType> &BuildStats, parlay::sequence<indexType> inserts=parlay::sequence<indexType>()) {
    // std::cout << "Building graph..." << std::endl;
    if (inserts.size() == 0) {
      inserts = parlay::tabulate(G.size(), [&](indexType i) { return i; });
    }
    start_point = inserts[0];

    batch_insert(inserts, G, Points, BuildStats, true, 2, .02);
    parlay::parallel_for (0, G.size(), [&] (long i) {
      auto less = [&] (indexType j, indexType k) {
		    return Points[i].distance(Points[j]) < Points[i].distance(Points[k]);};
      G[i].sort(less);});
  }

  void lazy_delete(parlay::sequence<indexType> deletes, GraphI &G) {
    for (indexType p : deletes) {
      if (p > (int)G.size()) {
        std::cout << "ERROR: invalid point " << p << " given to lazy_delete"
                  << std::endl;
        abort();
      }
      if (p != start_point)
        delete_set.insert(p);
      else
        std::cout << "Deleting start_point not permitted; continuing" << std::endl;
    }
  }

  void lazy_delete(indexType p, GraphI &G) {
    if (p > (int)G.size()) {
      std::cout << "ERROR: invalid point " << p << " given to lazy_delete"
                << std::endl;
      abort();
    }
    if (p == start_point) {
      std::cout << "Deleting start_point not permitted; continuing" << std::endl;
      return;
    }
    delete_set.insert(p);
  }

  // void consolidate_deletes(parlay::sequence<Tvec_point<T>*> &v){
  //   //clear deleted neighbors out of delete set for preprocessing

  //   parlay::parallel_for(0, v.size(), [&] (size_t i){
  //     if (delete_set.find(i) != delete_set.end()){
  //       parlay::sequence<int> new_edges; 
  //       for(int j=0; j<size_of(v[i]->out_nbh); j++){
  //         if(delete_set.find(v[i]->out_nbh[j]) == delete_set.end())
  //           new_edges.push_back(v[i]->out_nbh[j]);
  //        }
  //        if(new_edges.size() < size_of(v[i]->out_nbh))
  //          add_out_nbh(new_edges, v[i]); 
  //      } });

  //   parlay::parallel_for(0, v.size(), [&] (size_t i){
  //     if (delete_set.find(i) == delete_set.end() && size_of(v[i]->out_nbh) != 0) {
  //       std::set<int> new_edges;
  //       bool modify = false;
  //       for(int j=0; j<size_of(v[i]->out_nbh); j++){
  //         if(delete_set.find(v[i]->out_nbh[j]) == delete_set.end()){
  //           new_edges.insert(v[i]->out_nbh[j]);
  //         } else{
  //           modify = true;
  //           int index = v[i]->out_nbh[j];
  //           for(int k=0; k<size_of(v[index]->out_nbh); k++)
  //             new_edges.insert(v[index]->out_nbh[k]);
  //         }
  //       }
  //       //TODO only prune if overflow happens
  //       //TODO modify in separate step with new memory initialized in one block
  //       if(modify){ 
  //         parlay::sequence<int> candidates;
  //         for(int j : new_edges) candidates.push_back(j);
  //         parlay::sequence<int> new_neighbors(BP.R, -1);
  //         v[i]->new_nbh = parlay::make_slice(new_neighbors.begin(), new_neighbors.end());
  //         robustPrune(v[i], std::move(candidates), v, r2_alpha, false);
  //         synchronize(v[i]);
  //       }       
  //     }  });
  //   parlay::parallel_for(0, v.size(), [&] (size_t i){
  //     if (delete_set.find(i) != delete_set.end()){
  //       clear(v[i]);
  //     } });
 
  //   delete_set.clear();
  // }

  void batch_insert(parlay::sequence<indexType> &inserts,
                     GraphI &G, PR &Points, stats<indexType> &BuildStats,
                    bool random_order = false, double base = 2,
                    double max_fraction = .02, bool print=true) {
    for(int p : inserts){
      if(p < 0 || p > (int) Points.size()){
        std::cout << "ERROR: invalid or already inserted point "
                  << p << " given to batch_insert" << std::endl;
        abort();
      }
    }
    size_t n = G.size();
    size_t m = inserts.size();
    size_t inc = 0;
    size_t count = 0;
    float frac = 0.0;
    float progress_inc = .1;
    size_t max_batch_size = std::min(
        static_cast<size_t>(max_fraction * static_cast<float>(n)), 1000000ul);
    parlay::sequence<int> rperm;
    if (random_order)
      rperm = parlay::random_permutation<int>(static_cast<int>(m));
    else
      rperm = parlay::tabulate(m, [&](int i) { return i; });
    auto shuffled_inserts =
        parlay::tabulate(m, [&](size_t i) { return inserts[rperm[i]]; });
    parlay::internal::timer t_beam("beam search time");
    parlay::internal::timer t_bidirect("bidirect time");
    parlay::internal::timer t_prune("prune time");
    t_beam.stop();
    t_bidirect.stop();
    t_prune.stop();
    while (count < m) {
      size_t floor;
      size_t ceiling;
      if (pow(base, inc) <= max_batch_size) {
        floor = static_cast<size_t>(pow(base, inc)) - 1;
        ceiling = std::min(static_cast<size_t>(pow(base, inc + 1)), m) - 1;
        count = std::min(static_cast<size_t>(pow(base, inc + 1)), m) - 1;
      } else {
        floor = count;
        ceiling = std::min(count + static_cast<size_t>(max_batch_size), m);
        count += static_cast<size_t>(max_batch_size);
      }
      if (print) {
        auto ind = frac * n;
        if (floor <= ind && ceiling > ind) {
          frac += progress_inc;
          // std::cout << "Index build " << 100 * frac << "% complete"
          //           << std::endl;
        }
      }
      parlay::sequence<parlay::sequence<indexType>> new_out_(ceiling-floor);
      // search for each node starting from the start_point, then call
      // robustPrune with the visited list as its candidate set
      t_beam.start();
      parlay::parallel_for(floor, ceiling, [&](size_t i) {
        size_t index = shuffled_inserts[i];
        QueryParams QP((long) 0, BP.L, (double) 0.0, (long) Points.size(), (long) G.max_degree());
        parlay::sequence<pid> visited = 
          (beam_search<Point, PointRange, indexType>(Points[index], G, Points, start_point, QP)).first.second;
        BuildStats.increment_visited(index, visited.size());
        new_out_[i-floor] = robustPrune(index, visited, G, Points); });
      t_beam.stop();
      // make each edge bidirectional by first adding each new edge
      //(i,j) to a sequence, then semisorting the sequence by key values
      t_bidirect.start();
      auto to_flatten = parlay::tabulate(ceiling - floor, [&](size_t i) {
        indexType index = shuffled_inserts[i + floor];
        auto edges =
            parlay::tabulate(new_out_[i].size(), [&](size_t j) {
              return std::make_pair(new_out_[i][j], index);
            });
        return edges;
      });

      parlay::parallel_for(floor, ceiling, [&](size_t i) {
        G[shuffled_inserts[i]].update_neighbors(new_out_[i-floor]);
      });
      auto grouped_by = parlay::group_by_key(parlay::flatten(to_flatten));
      t_bidirect.stop();
      t_prune.start();
      // finally, add the bidirectional edges; if they do not make
      // the vertex exceed the degree bound, just add them to out_nbhs;
      // otherwise, use robustPrune on the vertex with user-specified alpha
      parlay::parallel_for(0, grouped_by.size(), [&](size_t j) {
        auto &[index, candidates] = grouped_by[j];
        size_t newsize = candidates.size() + G[index].size();
        if (newsize <= BP.R) {
          G[index].append_neighbors(candidates);
        } else {
          auto new_out_2_ = robustPrune(index, std::move(candidates), G, Points);  
          G[index].update_neighbors(new_out_2_);    
        }
      });
      t_prune.stop();
      inc += 1;
    }
    t_beam.total();
    t_bidirect.total();
    t_prune.total();
  }

  void batch_insert(indexType p, Graph<indexType> &G) {
    parlay::sequence<indexType> inserts;
    inserts.push_back(p);
    batch_insert(inserts, G, true);
  }

  // void check_index(parlay::sequence<Tvec_point<T>*> &v){
  //   parlay::parallel_for(0, v.size(), [&] (size_t i){
  //     if(v[i]->id > 1000000 && v[i]->id != start_point->id){
  //       if(size_of(v[i]->out_nbh) != 0) {
  //         std::cout << "ERROR : deleted point " << i << " still in graph" << std::endl; 
  //         abort();
  //       }
  //     } else {
  //       for(int j=0; j<size_of(v[i]->out_nbh); j++){
  //         int nbh = v[i]->out_nbh[j];
  //         if(nbh > 1000000 && nbh != start_point->id){
  //           std::cout << "ERROR : point " << i << " contains deleted neighbor "
  //                     << nbh << std::endl; 
  //           abort();
  //         }
  //       }
  //     }
  //  });
  // }

};
