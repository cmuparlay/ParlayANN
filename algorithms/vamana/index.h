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
#include "parlay/random.h"
#include "../utils/indexTools.h"
#include "../utils/NSGDist.h"
#include <random>
#include <set>
#include <math.h>

extern bool report_stats;

template<typename T>
struct knn_index {
  int maxDeg;
  int beamSize;
  double r2_alpha; //alpha parameter for round 2 of robustPrune
  unsigned int d;
  Distance* D;
  std::set<int> delete_set; 
  using tvec_point = Tvec_point<T>;
  using fvec_point = Tvec_point<float>;
  tvec_point* medoid;
  using pid = std::pair<int, float>;
  using slice_tvec = decltype(make_slice(parlay::sequence<tvec_point*>()));
  using index_pair = std::pair<int, int>;
  using slice_idx = decltype(make_slice(parlay::sequence<index_pair>()));

  knn_index(int md, int bs, double a, unsigned int dim, Distance* DD)
    : maxDeg(md), beamSize(bs), r2_alpha(a), d(dim), D(DD) {}

  parlay::sequence<float> centroid_helper(slice_tvec a){
    if(a.size() == 1){
      parlay::sequence<float> centroid_coords = parlay::sequence<float>(d);
      for(int i=0; i<d; i++) centroid_coords[i] = static_cast<float>((a[0]->coordinates)[i]);
      return centroid_coords;
    }
    else{
      size_t n = a.size();
      parlay::sequence<float> c1;
      parlay::sequence<float> c2;
      parlay::par_do_if(n>1000,
                        [&] () {c1 = centroid_helper(a.cut(0, n/2));},
                        [&] () {c2 = centroid_helper(a.cut(n/2, n));}
                        );
      parlay::sequence<float> centroid_coords = parlay::sequence<float>(d);
      for(int i=0; i<d; i++){
        float result = (c1[i] + c2[i])/2;
        centroid_coords[i] = result;
      }
      return centroid_coords;
    }
  }

  tvec_point* medoid_helper(tvec_point* centroid, slice_tvec a){
    if(a.size() == 1){
      return a[0];
    }
    else{
      size_t n = a.size();
      tvec_point* a1;
      tvec_point* a2;
      parlay::par_do_if(n>1000,
          [&] () {a1 = medoid_helper(centroid, a.cut(0, n/2));},
          [&] () {a2 = medoid_helper(centroid, a.cut(n/2, n));} );
      float d1 = D->distance(centroid->coordinates.begin(), a1->coordinates.begin(), d);
      float d2 = D->distance(centroid->coordinates.begin(), a2->coordinates.begin(), d);
      if(d1<d2) return a1;
      else return a2;
    }
  }

  //computes the centroid and then assigns the approx medoid as the point in v
  //closest to the centroid
  void find_approx_medoid(parlay::sequence<Tvec_point<T>*> &v){
    size_t n = v.size();
    parlay::sequence<float> centroid = centroid_helper(parlay::make_slice(v));
    auto c = parlay::tabulate(centroid.size(), [&] (size_t i){
               return static_cast<T>(centroid[i]);});
    tvec_point centroidp = tvec_point();
    centroidp.coordinates = parlay::make_slice(c);
    medoid = medoid_helper(&centroidp, parlay::make_slice(v));
    std::cout << "Medoid ID: " << medoid->id << std::endl;
  }

  int get_medoid(){return medoid->id;}

  //robustPrune routine as found in DiskANN paper, with the exception
  //that the new candidate set is added to the field new_nbhs instead
  //of directly replacing the out_nbh of p
  void robustPrune(tvec_point* p, parlay::sequence<pid>& cand,
                   parlay::sequence<tvec_point*> &v,
                   double alpha, bool add = true) {
    // add out neighbors of p to the candidate set.
    auto get_coords = [&] (auto q) {
        auto coord_len = (v[1]->coordinates.begin() - v[0]->coordinates.begin());
        return v[0]->coordinates.begin() + q * coord_len;};
    
    int out_size = size_of(p->out_nbh);
    std::vector<pid> candidates;
    for (auto x : cand) candidates.push_back(x);

    if(add){
      for (size_t i=0; i<out_size; i++) {
        candidates.push_back(std::make_pair(p->out_nbh[i],
                                            D->distance(get_coords(p->out_nbh[i]),
                                                        p->coordinates.begin(), d)));
      }
    }
    
    // Sort the candidate set in reverse order according to distance from p.
    auto less = [&] (pid a, pid b) {return a.second < b.second;};
    std::sort(candidates.begin(), candidates.end(), less);

    std::vector<int> new_nbhs;
    new_nbhs.reserve(maxDeg);

    size_t candidate_idx = 0;

    while (new_nbhs.size() < maxDeg && candidate_idx < candidates.size()) {
      // Don't need to do modifications.
      int p_star = candidates[candidate_idx].first;
      candidate_idx++;
      if (p_star == p->id || p_star == -1) {
        continue;
      }

      new_nbhs.push_back(p_star);

      for (size_t i = candidate_idx; i < candidates.size(); i++) {
        int p_prime = candidates[i].first;
        if (p_prime != -1) {
          float dist_starprime = D->distance(get_coords(p_star), get_coords(p_prime), d);
          float dist_pprime = candidates[i].second;
          if (alpha * dist_starprime <= dist_pprime) {
            candidates[i].first = -1;
          }
        } 
      }
    }
    add_new_nbh(parlay::to_sequence(new_nbhs), p);
  }

  //wrapper to allow calling robustPrune on a sequence of candidates 
  //that do not come with precomputed distances
  void robustPrune(Tvec_point<T>* p, parlay::sequence<int> candidates,
                   parlay::sequence<Tvec_point<T>*> &v,
                   double alpha, bool add = true){
    auto get_coords = [&] (auto q) {
        auto coord_len = (v[1]->coordinates.begin() - v[0]->coordinates.begin());
        return v[0]->coordinates.begin() + q * coord_len;};

    parlay::sequence<pid> cc;
    cc.reserve(candidates.size()); // + size_of(p->out_nbh));
    for (size_t i=0; i<candidates.size(); ++i) {
      cc.push_back(std::make_pair(candidates[i], D->distance(get_coords(candidates[i]),
                                                             p->coordinates.begin(), d)));
    }
    return robustPrune(p, cc, v, alpha, add);
  }

  void build_index(parlay::sequence<Tvec_point<T>*> &v, parlay::sequence<int> inserts){
    clear(v);
    std::cout << "Building graph..." << std::endl;
    find_approx_medoid(v);
    batch_insert(inserts, v, true, r2_alpha, 2, .02);
  }

  void lazy_delete(parlay::sequence<int> deletes, parlay::sequence<Tvec_point<T>*> &v){
    for(int p : deletes){
      if(p < 0 || p > (int) v.size() ){
        std::cout << "ERROR: invalid point " << p << " given to lazy_delete" << std::endl; 
        abort();
      }
      if(p != medoid->id) delete_set.insert(p);
      else std::cout << "Deleting medoid not permitted; continuing" << std::endl; 
    } 
  }

  void lazy_delete(int p, parlay::sequence<Tvec_point<T>*> &v){
    if(p < 0 || p > (int) v.size()){
      std::cout << "ERROR: invalid point " << p << " given to lazy_delete" << std::endl; 
      abort();
    }
    if(p == medoid->id){
      std::cout << "Deleting medoid not permitted; continuing" << std::endl; 
      return;
    } 
    delete_set.insert(p);
  }

  void consolidate_deletes(parlay::sequence<Tvec_point<T>*> &v){
    //clear deleted neighbors out of delete set for preprocessing

    parlay::parallel_for(0, v.size(), [&] (size_t i){
      if (delete_set.find(i) != delete_set.end()){
        parlay::sequence<int> new_edges; 
        for(int j=0; j<size_of(v[i]->out_nbh); j++){
          if(delete_set.find(v[i]->out_nbh[j]) == delete_set.end())
            new_edges.push_back(v[i]->out_nbh[j]);
         }
         if(new_edges.size() < size_of(v[i]->out_nbh))
           add_out_nbh(new_edges, v[i]); 
       } });

    parlay::parallel_for(0, v.size(), [&] (size_t i){
      if (delete_set.find(i) == delete_set.end() && size_of(v[i]->out_nbh) != 0) {
        std::set<int> new_edges;
        bool modify = false;
        for(int j=0; j<size_of(v[i]->out_nbh); j++){
          if(delete_set.find(v[i]->out_nbh[j]) == delete_set.end()){
            new_edges.insert(v[i]->out_nbh[j]);
          } else{
            modify = true;
            int index = v[i]->out_nbh[j];
            for(int k=0; k<size_of(v[index]->out_nbh); k++)
              new_edges.insert(v[index]->out_nbh[k]);
          }
        }
        //TODO only prune if overflow happens
        //TODO modify in separate step with new memory initialized in one block
        if(modify){ 
          parlay::sequence<int> candidates;
          for(int j : new_edges) candidates.push_back(j);
          parlay::sequence<int> new_neighbors(maxDeg, -1);
          v[i]->new_nbh = parlay::make_slice(new_neighbors.begin(), new_neighbors.end());
          robustPrune(v[i], std::move(candidates), v, r2_alpha, false);
          synchronize(v[i]);
        }       
      }  });
    parlay::parallel_for(0, v.size(), [&] (size_t i){
      if (delete_set.find(i) != delete_set.end()){
        clear(v[i]);
      } });
 
    delete_set.clear();
  }

  void batch_insert(parlay::sequence<int> &inserts,
                    parlay::sequence<Tvec_point<T>*> &v,
                    bool random_order = false, double alpha = 1.2, double base = 2,
                    double max_fraction = .02, bool print=true) {
    for(int p : inserts){
      if(p < 0 || p > (int) v.size() || (v[p]->out_nbh[0] != -1 && v[p]->id != medoid->id)){
        std::cout << "ERROR: invalid or already inserted point "
                  << p << " given to batch_insert" << std::endl;
        abort();
      }
    }
    size_t n = v.size();
    size_t m = inserts.size();
    size_t inc = 0;
    size_t count = 0;
    float frac=0.0;
    float progress_inc=.1;
    size_t max_batch_size = std::min(static_cast<size_t>(max_fraction*static_cast<float>(n)),
                                     1000000ul);
    parlay::sequence<int> rperm;
    if(random_order) rperm = parlay::random_permutation<int>(static_cast<int>(m));
    else rperm = parlay::tabulate(m, [&] (int i) {return i;});
    auto shuffled_inserts = parlay::tabulate(m, [&] (size_t i) {return inserts[rperm[i]];});
    parlay::internal::timer t_beam("beam search time");
    parlay::internal::timer t_bidirect("bidirect time");
    parlay::internal::timer t_prune("prune time");
    t_beam.stop();
    t_bidirect.stop();
    t_prune.stop();
    while(count < m){
      size_t floor;
      size_t ceiling;
      if(pow(base,inc) <= max_batch_size){
        floor = static_cast<size_t>(pow(base, inc))-1;
        ceiling = std::min(static_cast<size_t>(pow(base, inc+1)), m)-1;
        count = std::min(static_cast<size_t>(pow(base, inc+1)), m)-1;
      } else{
        floor = count;
        ceiling = std::min(count + static_cast<size_t>(max_batch_size), m);
        count += static_cast<size_t>(max_batch_size);
      }
      if(print){
        auto ind = frac*n;
        if(floor <= ind && ceiling > ind){
          frac += progress_inc;
          std::cout << "Index build " << 100*frac << "% complete" << std::endl;
        }
      }
      parlay::sequence<int> new_out = parlay::sequence<int>(maxDeg*(ceiling-floor), -1);
      //search for each node starting from the medoid, then call
      //robustPrune with the visited list as its candidate set
      t_beam.start();
      parlay::parallel_for(floor, ceiling, [&] (size_t i){
        size_t index = shuffled_inserts[i];
        v[index]->new_nbh = parlay::make_slice(new_out.begin()+maxDeg*(i-floor),
                                               new_out.begin()+maxDeg*(i+1-floor));
        parlay::sequence<pid> visited = (beam_search(v[index], v, medoid, beamSize, d, D)).first.second;
        if (report_stats) v[index]->visited = visited.size();
        robustPrune(v[index], visited, v, alpha); });
      t_beam.stop();
      //make each edge bidirectional by first adding each new edge
      //(i,j) to a sequence, then semisorting the sequence by key values
      t_bidirect.start();
      auto to_flatten = parlay::tabulate(ceiling-floor, [&] (size_t i){
                          int index = shuffled_inserts[i+floor];
                          auto edges = parlay::tabulate(size_of(v[index]->new_nbh), [&] (size_t j){
                            return std::make_pair((v[index]->new_nbh)[j], index); });
                          return edges; });

      parlay::parallel_for(floor, ceiling, [&] (size_t i) {
         synchronize(v[shuffled_inserts[i]]);} );
      auto grouped_by = parlay::group_by_key(parlay::flatten(to_flatten));
      t_bidirect.stop();

      t_prune.start();
      //finally, add the bidirectional edges; if they do not make
      //the vertex exceed the degree bound, just add them to out_nbhs;
      //otherwise, use robustPrune on the vertex with user-specified alpha
      parlay::parallel_for(0, grouped_by.size(), [&] (size_t j){
        auto& [index, candidates] = grouped_by[j];
        int newsize = candidates.size() + size_of(v[index]->out_nbh);
        if (newsize <= maxDeg) {
          add_nbhs(candidates, v[index]);
        } else{
          parlay::sequence<int> new_out_2(maxDeg, -1);
          v[index]->new_nbh=parlay::make_slice(new_out_2.begin(), new_out_2.begin()+maxDeg);
          robustPrune(v[index], std::move(candidates), v, r2_alpha);  
          synchronize(v[index]);
        } });
      t_prune.stop();
      inc += 1;
    }
    t_beam.total();
    t_bidirect.total();
    t_prune.total();
  }

  void batch_insert(int p, parlay::sequence<Tvec_point<T>*> &v){
    parlay::sequence<int> inserts;
    inserts.push_back(p);
    batch_insert(inserts, v, true);
  }

  void check_index(parlay::sequence<Tvec_point<T>*> &v){
    parlay::parallel_for(0, v.size(), [&] (size_t i){
      if(v[i]->id > 1000000 && v[i]->id != medoid->id){
        if(size_of(v[i]->out_nbh) != 0) {
          std::cout << "ERROR : deleted point " << i << " still in graph" << std::endl; 
          abort();
        }
      } else {
        for(int j=0; j<size_of(v[i]->out_nbh); j++){
          int nbh = v[i]->out_nbh[j];
          if(nbh > 1000000 && nbh != medoid->id){
            std::cout << "ERROR : point " << i << " contains deleted neighbor "
                      << nbh << std::endl; 
            abort();
          }
        }
      }
   });
  }

  void searchNeighbors(parlay::sequence<Tvec_point<T>*> &q,
                       parlay::sequence<Tvec_point<T>*> &v, int beamSizeQ, int k, float cut){
    searchAll(q, v, beamSizeQ, k, d, medoid, D, cut);
  }

  void rangeSearch(parlay::sequence<Tvec_point<T>*> &q, parlay::sequence<Tvec_point<T>*> &v, 
                   int beamSizeQ, double r, int k, float cut=1.14, double slack = 3.0){
    rangeSearchAll(q, v, beamSizeQ, d, medoid, r, D, k, cut, slack);
  }
};
