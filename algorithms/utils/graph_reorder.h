#include <atomic>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>

#include <parlay/delayed.h>
#include <parlay/parallel.h>
#include <parlay/primitives.h>
#include <parlay/random.h>
#include <parlay/sequence.h>
#include <parlay/utilities.h>
#include <parlay/internal/get_time.h>

using vertex = int;
using w_type = float;
using edge_id = int;

using edge = std::pair<vertex,vertex>;
using w_edge = std::pair<edge,w_type>;
struct tagged_w_type {w_type w; edge_id i;};
bool greater(tagged_w_type a, tagged_w_type b) {
  return (a.w > b.w) ? true : ((a.w == b.w) ? a.i > b.i : false);}
struct vertex_info {
  std::atomic<tagged_w_type> tw;
  vertex size = 1;
};

// Uses recursive graph contraction to renumber a graph
//   E : sequence of weighted edges (only needed in one direction)
//   V : remaining vertices
//   Sizes : keeps size of contracted vertices on the way
//           down the recursion, and offsets on the way up
//   W : used for temporary space to write priorities
//   P : used for temporary space to write parent of contracted vertex
//   m : original number of edges (not currently used)
// Idea: each round of contraction identifies edges (u,v) that maximize
//     w(u,v)/(|u||v|) on both u and v.  These edges are contracted.
//     |u| is the number of vertices in the component and
//     w(u,v) is the number of edges between components u and v.
void recursive_reorder(parlay::sequence<w_edge>& E,
                       parlay::sequence<vertex>& V,
                       parlay::sequence<vertex_info>& W,
                       parlay::sequence<vertex>& P,
                       int i,
                       long m) {
  // std::cout << E.size() << ", " << V.size() << std::endl;
  
  // Base case: need to scan if more than one component
  if (i > 300 || E.size() == 0) { 
    auto vsizes = parlay::tabulate(V.size(), [&] (long i) {return W[V[i]].size;});
    auto [offsets, sum] = parlay::scan(vsizes);
    parlay::parallel_for(0, V.size(), [&] (long i) {W[V[i]].size = offsets[i];});
    return;
  }

  // Write with max into W the priority (w(u,v)/(|u||v|)) to each endpoint
  // Priorities are tagged with id to break ties
  // Must firsrt clear W at all active vertices
  float empty = std::numeric_limits<float>::lowest();
  parlay::for_each(V, [&] (vertex& v) {
      W[v].tw.store(tagged_w_type{empty, 0});});
  parlay::parallel_for(0, E.size(), [&] (edge_id i) {
      auto [u, v] = E[i].first;
      auto w = tagged_w_type{E[i].second / (W[u].size * W[v].size), i};
      parlay::write_min(&(W[v].tw), w, greater);
      parlay::write_min(&(W[u].tw), w, greater);});

  // Check for each active vertex u which edge (u,v) won on it.  If
  // the edge also won on v, then it is matched, we contract v into u,
  // and return the edge along with the old weight of u.
  auto matches = parlay::map_maybe(V, [&] (vertex& u) {
      long i = W[u].tw.load().i;
      if (W[u].tw.load().w != empty && E[i].first.first == u) {
        vertex v = E[i].first.second;
        if (W[v].tw.load().i == i) {
          vertex usize = W[u].size;
          W[u].size += W[v].size;
          P[v] = u;
          return std::optional(std::tuple(u, v, usize));
        }
      }
      return std::optional<std::tuple<vertex,vertex,vertex>>();});
  
  // Update edge endpoints and remove self edges
  E = parlay::map_maybe(E, [&] (w_edge e) {
        auto [u,v] = e.first;
        vertex pu = P[u];
        vertex pv = P[v];
        if (pu > pv) std::swap(pu,pv); // keep oriented low to high
        if (pu == pv) return std::optional<w_edge>();
        return std::optional(w_edge(edge(pu, pv), e.second));});
  
  // Combine redundant edges
  // For efficiency, only do every three steps
  if (i % 4 == 3)
    E = parlay::reduce_by_key(E);

  // These are the remaining vertices after contraction
  V = parlay::filter(V, [&] (vertex v) {return P[v] == v;});
  
  // recurse
  recursive_reorder(E, V, W, P, i+1, m);

  // update Sizes to give right offsets
  parlay::for_each(matches, [&] (auto match) {
        auto [u,v,usize] = match;
        W[v].size = W[u].size + usize;});
}

// E is a sequence of edges, only needed in one direction
// n is the number of vertices
parlay::sequence<vertex> graph_reorder(parlay::sequence<edge>& E, long n) {
  E = parlay::random_shuffle(E); // randomly permute the edges

  // Initialize the five arguments
  auto WE = parlay::map(E, [&] (edge e) { return w_edge(e,1); });
  auto V = parlay::tabulate(n, [] (vertex i) {return i;});
  parlay::sequence<vertex_info> W(n);
  auto P = V;

  // Call main routine
  recursive_reorder(WE, V, W, P, 0, E.size());
  return parlay::map(W, [] (auto& w) {return w.size;});
}
