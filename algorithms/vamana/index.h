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
#include <mutex>

#include "../utils/NSGDist.h"
#include "../utils/point_range.h"
#include "../utils/graph.h"
#include "../utils/types.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"
#include "../utils/beamSearch.h"


template<typename Point, typename PointRange, typename indexType, typename GraphType>
struct knn_index {
  using distanceType = typename Point::distanceType;
  using pid = std::pair<indexType, distanceType>;
  using PR = PointRange;
  using GraphI = typename GraphType::Graph;
  using LockGuard = std::lock_guard<std::mutex>;
  

  BuildParams BP;
  std::set<indexType> delete_set; 
  std::set<indexType> old_delete_set;
  std::mutex delete_lock;  // lock for delete_set which can only be updated sequentially
  bool epoch_running = false;
  indexType start_point;
  bool start_set = false;


  knn_index(BuildParams &BP) : BP(BP) {}

  indexType get_start() { return start_point; }

  //robustPrune routine as found in DiskANN paper, with the exception
  //that the new candidate set is added to the field new_nbhs instead
  //of directly replacing the out_nbh of p
  parlay::sequence<indexType> robustPrune(indexType p, parlay::sequence<pid>& cand,
                    GraphI &G, PR &Points, double alpha, bool add = true) {
    // add out neighbors of p to the candidate set.
    std::vector<pid> candidates;
    for (auto x : cand) candidates.push_back(x);

    if(add){
      for (indexType i : G[p].neighbors()) {
        candidates.push_back(std::make_pair(i, Points[i].distance(Points[p])));
      }
    }

    // Sort the candidate set according to distance from p
    auto less = [&](pid a, pid b) { return a.second < b.second; };
    std::sort(candidates.begin(), candidates.end(), less);

    // remove any duplicates
    auto new_end =std::unique(candidates.begin(), candidates.end(),
			      [&] (auto x, auto y) {return x.first == y.first;});
    candidates = std::vector(candidates.begin(), new_end);
    
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
          if (alpha * dist_starprime <= dist_pprime) {
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
                    GraphI &G, PR &Points, double alpha, bool add = true){

    parlay::sequence<pid> cc;
    cc.reserve(candidates.size()); // + size_of(p->out_nbh));
    for (size_t i=0; i<candidates.size(); ++i) {
      cc.push_back(std::make_pair(candidates[i], Points[candidates[i]].distance(Points[p])));
    }
    return robustPrune(p, cc, G, Points, alpha, add);
  }

  // add ngh to candidates without adding any repeats
  template<typename rangeType1, typename rangeType2>
  void add_neighbors_without_repeats(const rangeType1 &ngh, rangeType2& candidates) {
    std::unordered_set<indexType> a;
    for (auto c : candidates) a.insert(c);
    for (int i=0; i < ngh.size(); i++) 
      if (a.count(ngh[i]) == 0) candidates.push_back(ngh[i]);
  }

  void set_start(GraphI &G){
    start_point = 0;
    parlay::sequence<indexType> nbh = {};
    parlay::sequence<std::pair<indexType, parlay::sequence<indexType>>> update = {std::make_pair(start_point, nbh)};
    G.batch_update(update);
    start_set = true;
  }

  void build_index(GraphType &Graph, PR &Points, stats<indexType> &BuildStats){
    std::cout << "Building graph..." << std::endl;
    
    parlay::sequence<indexType> inserts = parlay::tabulate(Points.size(), [&] (size_t i){
					    return static_cast<indexType>(i);});
    GraphI G = Graph.Get_Graph();
    if(!start_set) set_start(G);
    if(BP.two_pass) batch_insert(inserts, G, Points, BuildStats, 1.0, true, true, 2, .02);
    batch_insert(inserts, G, Points, BuildStats, BP.alpha, true, true, 2, .02);
    // parlay::parallel_for (0, G.size(), [&] (long i) {
    //   auto less = [&] (indexType j, indexType k) {
		//     return Points[i].distance(Points[j]) < Points[i].distance(Points[k]);};
    //   G[i].sort(less);});
    Graph.Update_Graph(std::move(G));
  }

  void insert(GraphType &Graph, PR &Points, stats<indexType> &BuildStats, parlay::sequence<indexType> inserts){
    std::cout << "Inserting points " << std::endl;
    GraphI G = Graph.Get_Graph();
    if(!start_set) set_start(G);
    batch_insert(inserts, G, Points, BuildStats, BP.alpha, false, true, 2, .02);
    Graph.Update_Graph(std::move(G));
  }

  void lazy_delete(parlay::sequence<indexType> deletes) {
    {
      LockGuard guard(delete_lock);
      for (indexType p : deletes) {
        if (p != start_point)
          delete_set.insert(p);
        else
          std::cout << "Deleting start_point not permitted; continuing" << std::endl;
      }
    }
  }

  void lazy_delete(indexType p) {
    parlay::sequence<indexType> deletes = {p};
    lazy_delete(deletes);
  }

   void start_delete_epoch() {
    // freeze the delete set and start a new one before consolidation
    if (!epoch_running) {
      {
        LockGuard guard(delete_lock);
        delete_set.swap(old_delete_set);
      }
      epoch_running = true;
    } else {
      std::cout << "ERROR: cannot start new epoch while previous epoch is running"<< std::endl;
      abort();
    }
  }

  void end_delete_epoch(GraphType &Graph) {
    if (epoch_running) {
      parlay::sequence<indexType> delete_vec;
      for (auto d : old_delete_set) delete_vec.push_back(d);
      GraphI G = Graph.Get_Graph();
      G.batch_delete(delete_vec);
      Graph.Update_Graph(std::move(G));
      old_delete_set.clear();
      epoch_running = false;
    } else {
      std::cout << "ERROR: cannot end epoch while epoch is not running" << std::endl;
      abort();
    }
  }

  void consolidate(GraphType &Graph, PR &Points){
    GraphI G = Graph.Get_Graph();
    consolidate_deletes(G, Points);
    check_deletes_correct(G, Points);
    Graph.Update_Graph(std::move(G));
  }

  void consolidate_deletes_simple(GraphI &G, PR &Points){
    parlay::sequence<std::pair<indexType, parlay::sequence<indexType>>> edge_updates(Points.size());
    parlay::sequence<bool> updated(Points.size(), false);
    parlay::parallel_for(0, Points.size(), [&] (size_t i){
      if(old_delete_set.find(i) == old_delete_set.end()){
        auto nbh = G[i].neighbors();
        parlay::sequence<indexType> new_nbh;
        bool modify = false;
        for(indexType j : nbh){
          if(old_delete_set.find(j) == old_delete_set.end()){
            new_nbh.push_back(j);
          }else modify = true;
        }
        if(modify){
          updated[i] = true;
          edge_updates[i] = std::make_pair((indexType) i, new_nbh);
        }
      }
    });
    auto to_delete = parlay::pack(edge_updates, updated);
    G.batch_update(to_delete);
  }

  void check_deletes_correct(GraphI &G, PR &Points){
    parlay::sequence<std::pair<indexType, parlay::sequence<indexType>>> edge_updates(Points.size());
    parlay::sequence<bool> updated(Points.size(), false);
    parlay::parallel_for(0, Points.size(), [&] (size_t i){
      if(old_delete_set.find(i) == old_delete_set.end()){
        auto nbh = G[i].neighbors();
        for(indexType j : nbh){
          if(old_delete_set.find(j) != old_delete_set.end()){
            std::cout << "ERROR: point " << i << " has deleted neighbor " << j << std::endl;
            abort();
          }
        }
      }
    });
  }

  void consolidate_deletes(GraphI &G, PR &Points){
    //clear deleted neighbors out of delete set for preprocessing
    parlay::sequence<std::pair<indexType, parlay::sequence<indexType>>> edge_updates(Points.size());
    parlay::sequence<bool> updated(Points.size(), false);
    parlay::parallel_for(0, Points.size(), [&] (size_t i){
      if(old_delete_set.find(i) != old_delete_set.end()){
        auto nbh = G[i].neighbors();
        parlay::sequence<indexType> new_nbh;
        bool modify = false;
        for(indexType j : nbh){
          if(old_delete_set.find(j) == old_delete_set.end()) new_nbh.push_back(j);
          else modify = true;
        }
        if(modify){
          updated[i] = true;
          edge_updates[i] = std::make_pair((indexType) i, new_nbh);
        }
      }
    });
    auto to_delete = parlay::pack(edge_updates, updated);
    G.batch_update(to_delete);

    parlay::sequence<std::pair<indexType, parlay::sequence<indexType>>> edge_updates_round2(Points.size());
    parlay::sequence<bool> updated_round2(Points.size(), false);
    parlay::parallel_for(0, Points.size(), [&] (size_t i){
      if(old_delete_set.find(i) == old_delete_set.end()){
        auto nbh = G[i].neighbors();
        parlay::sequence<indexType> new_nbh;
        bool modify = false;
        for(indexType j : nbh){
          if(old_delete_set.find(j) == old_delete_set.end()) new_nbh.push_back(j);
          else{
            modify = true;
            auto j_nbh = G[j].neighbors();
            for(auto l : j_nbh) {
              if(old_delete_set.find(l) != old_delete_set.end()){
                std::cout << "ERROR: deleted point " << j << " still contains deleted neighbor " << l << std::endl;
                abort();
              }
              new_nbh.push_back(l);
            }
          }
        }
        if(modify){
          updated_round2[i] = true;
          new_nbh = parlay::remove_duplicates(new_nbh);
          if(new_nbh.size() > BP.R){
            auto pruned_nbh = robustPrune(i, new_nbh, G, Points, BP.alpha);
            edge_updates_round2[i] = std::make_pair((indexType) i, pruned_nbh);
          } else edge_updates_round2[i] = std::make_pair((indexType) i, new_nbh);
        }
      }
    });
    auto updates_round2 = parlay::pack(edge_updates_round2, updated_round2);
    G.batch_update(updates_round2);
  }

  void batch_insert(parlay::sequence<indexType> &inserts,
                     GraphI &G, PR &Points, stats<indexType> &BuildStats, double alpha,
                     bool print=true, bool random_order = false, double base = 2,
                    double max_fraction = .02) {
    size_t n = G.size();
    size_t m = inserts.size();
    size_t inc = 0;
    size_t count = 0;
    float frac = 0.0;
    float progress_inc = .1;
    size_t max_batch_size = std::min(
        static_cast<size_t>(max_fraction * static_cast<float>(m)), 1000000ul);
    //fix bug where max batch size could be set to zero 
    if(max_batch_size == 0) max_batch_size = n;
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
      parlay::sequence<std::pair<indexType, parlay::sequence<indexType>>> new_out_(ceiling-floor);
      // search for each node starting from the start_point, then call
      // robustPrune with the visited list as its candidate set
      t_beam.start();
      std::cout << "Running next batch, floor = " << floor << " ceil = " << ceiling << std::endl;
      parlay::parallel_for(floor, ceiling, [&](size_t i) {
        size_t index = shuffled_inserts[i];
        QueryParams QP((long) 0, BP.L, (double) 0.0, (long) Points.size(), (long) G.max_degree());
        parlay::sequence<pid> visited = 
          (beam_search<Point, PointRange, indexType>(Points[index], G, Points, start_point, QP)).first.second;
        BuildStats.increment_visited(index, visited.size());
        new_out_[i-floor] = std::make_pair(index, robustPrune(index, visited, G, Points, alpha)); 
      });
      std::cout << "Calling batch update!" << std::endl;
      G.batch_update(new_out_);
      t_beam.stop();
      // make each edge bidirectional by first adding each new edge
      //(i,j) to a sequence, then semisorting the sequence by key values
      t_bidirect.start();
      auto to_flatten = parlay::tabulate(ceiling - floor, [&](size_t i) {
        indexType index = shuffled_inserts[i + floor];
        auto edges =
            parlay::tabulate(new_out_[i].second.size(), [&](size_t j) {
              return std::make_pair(new_out_[i].second[j], index);
            });
        return edges;
      });
      auto grouped_by = parlay::group_by_key(parlay::flatten(to_flatten));
      t_bidirect.stop();
      t_prune.start();
      // finally, add the bidirectional edges; if they do not make
      // the vertex exceed the degree bound, just add them to out_nbhs;
      // otherwise, use robustPrune on the vertex with user-specified alpha

      auto bidirectional_edges = parlay::tabulate(grouped_by.size(), [&] (size_t j){
        auto &[index, candidates] = grouped_by[j];
	      size_t newsize = candidates.size() + G[index].size();
        if (newsize <= BP.R) {
	        add_neighbors_without_repeats(G[index].neighbors(), candidates);
	        return std::make_pair(index, std::move(candidates));
        } else {
          auto new_out_2_ = robustPrune(index, std::move(candidates), G, Points, alpha);
	        return std::make_pair(index, std::move(new_out_2_));   
        }
      });
      G.batch_update(bidirectional_edges);
      t_prune.stop();
      if (print) {
        auto ind = frac * m;
        if (floor <= ind && ceiling > ind) {
          frac += progress_inc;
          std::cout << "Pass " << 100 * frac << "% complete"
                    << std::endl;
        }
      }
      inc += 1;
    }
    t_beam.total();
    t_bidirect.total();
    t_prune.total();
  }

};
