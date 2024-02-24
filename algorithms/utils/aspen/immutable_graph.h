#pragma once


namespace aspen {

template <class weight, typename vertex_id>
struct symmetric_graph {
  struct edge_entry {
    using key_t = vertex_id;  // a vertex_id
    using val_t = weight;     // placeholder
    static inline bool comp(key_t a, key_t b) { return a < b; }
    using entry_t = std::tuple<key_t, val_t>;
  };


  using edge_tree = cpam::pam_map<edge_entry>;


  using edge_node = typename edge_tree::node;

  struct vertex_entry {
    using key_t = vertex_id;
    using val_t = edge_tree;
    using aug_t = std::pair<vertex_id, size_t>;
    static inline bool comp(key_t a, key_t b) { return a < b; }
    static aug_t get_empty() { return std::make_pair(0, 0); }
    static aug_t from_entry(const key_t& k, const val_t& v) {
      return std::make_pair(k, v.size());
    }
    static aug_t combine(aug_t a, aug_t b) {
      auto& [a_v, a_e] = a;
      auto& [b_v, b_e] = b;
      return {std::max(a_v, b_v), a_e + b_e};
    }
    using entry_t = std::tuple<key_t, val_t>;
  };

  using vertex_tree = cpam::aug_map<vertex_entry>;

  using vertex_node = typename vertex_tree::node;
  using vertex_gc = typename vertex_tree::GC;


  using ngh_and_weight = std::tuple<vertex_id, weight>;

  using edge = std::pair<vertex_id, ngh_and_weight>;

  struct neighbors {
    vertex_id id;
    edge_node* edges;

    neighbors(vertex_id id, edge_node* edges) : id(id), edges(edges) {}

    template <class F, class G>
    void copy(size_t offset, F& f, G& g) {
      auto map_f = [&](const auto& et, size_t i) {
        auto[ngh, wgh] = et;
        auto val = f(id, ngh, wgh);
        g(ngh, offset + i, val);
      };
      edge_tree tree;
      tree.root = edges;
      tree.foreach_index(tree, map_f);
      tree.root = nullptr;
    }

    template <class F>
    void map_index(F& f) {
      auto map_f = [&](const auto& et, size_t i) {
        auto[ngh, wgh] = et;
        f(id, ngh, wgh, i);
      };
      edge_tree tree;
      tree.root = edges;
      tree.foreach_index(tree, map_f);
      tree.root = nullptr;
    }

    template <class F>
    void map(F& f) {
      auto map_f = [&](const auto& et, size_t i) {
        auto[ngh, wgh] = et;
        f(id, ngh, wgh);
      };
      edge_tree tree;
      tree.root = edges;
      tree.foreach_index(tree, map_f);
      tree.root = nullptr;
    }

    template <class _T>
    struct Add {
      using T = _T;
      static T identity() { return 0; }
      static T add(T a, T b) { return a + b; }
    };

    // Count the number of neighbors satisfying the predicate p.
    template <class P>
    size_t count(P& p) {
      edge_tree tree;
      tree.root = edges;
      auto map_f = [&](const auto& et) -> size_t {
        auto[ngh, wgh] = et;
        return p(id, ngh, wgh);
      };
      auto addm = Add<size_t>();
      auto ct = edge_tree::map_reduce(tree, map_f, addm);
      tree.root = nullptr;
      return ct;
    }

    template <class F, class C>
    void map_cond(F& f, C& c) {
      auto map_f = [&](const auto& et, size_t i) {
        auto[ngh, wgh] = et;
        return f(id, ngh, wgh);
      };
      edge_tree tree;
      tree.root = edges;
      tree.foreach_cond_par(tree, map_f, c);
      tree.root = nullptr;
    }

    template <class F>
    void foreach_cond(F& f) {
      auto map_f = [&](const auto& et) -> bool {
        return f(id, std::get<0>(et), std::get<1>(et));
      };
      edge_tree tree;
      tree.root = edges;
      tree.foreach_cond(tree, map_f);
      tree.root = nullptr;
    }
  };

  struct vertex {
    vertex_id id;
    edge_node* edges;
    size_t out_degree() {
      return edge_tree::size(edges);
    }
    size_t in_degree() { return out_degree(); }
    size_t ref_cnt() {
      edge_tree tree;
      tree.root = edges;
      auto sz = tree.ref_cnt();
      tree.root = nullptr;
      return sz;
    }
    auto out_neighbors() const { return neighbors(id, edges); }
    auto in_neighbors() const { return neighbors(id, edges); }
    vertex(vertex_id id, edge_node* edges) : id(id), edges(edges) {}
    vertex() : id(std::numeric_limits<vertex_id>::max()), edges(nullptr) {}
  };

  using maybe_vertex = std::optional<vertex>;
  using weight_type = weight;
  using SymGraph = symmetric_graph<weight, vertex_id>;

  vertex_tree V;


  // Set from a provided root (no ref-ct bump)
  symmetric_graph(vertex_node* root) {
    set_root(root);
  }

  // Build from a sequence of edges.
  symmetric_graph(parlay::sequence<edge>& edges) { V = from_edges(edges); }

  symmetric_graph(vertex_tree&& _V) : V(std::move(_V)) {}

  symmetric_graph() { V.root = nullptr; }

  vertex_tree& get_vertices() { return V; }

  void clear_root() { V.root = nullptr; }

  vertex_node* get_root() { return V.root; }

  size_t ref_cnt() {
    return V.ref_cnt();
  }

  void set_root(vertex_node* root) {
    V.root = root;
  }

  // Note that it's important to use n and not V.size() here.
  size_t num_vertices() const { return V.aug_val().first + 1; }

  size_t num_edges() const { return V.aug_val().second; }

  vertex get_vertex(vertex_id v) const{
    auto opt = V.find(v);
    if (opt.has_value()) {
      const auto& in_opt = *opt;
      // auto ref_cnt = edge_tree::Tree::ref_cnt(in_opt);
      // assert(ref_cnt == 1);
      return vertex(v, in_opt.root);
    }
    return vertex(v, nullptr);
  }

  template <class F>
  void map_vertices(const F& f) {
    using entry_t = typename vertex_entry::entry_t;
    auto map_f = [&](const entry_t& vtx_entry, size_t i) {
      const vertex_id& v = std::get<0>(vtx_entry);
      auto vtx = vertex(v, std::get<1>(vtx_entry).root);
      f(vtx);
    };
    vertex_tree::foreach_index(V, map_f, 0, 1);
  }

  static vertex_tree from_edges(parlay::sequence<edge>& edges) {
    auto reduce = [&](parlay::slice<ngh_and_weight*, ngh_and_weight*> R) {
      // return edge_tree(R);
      auto tree = edge_tree(R.begin(), R.begin() + R.size());
      auto root = tree.root;
      tree.root = nullptr;
      assert(edge_tree::Tree::ref_cnt(root) == 1);
      return root;
    };
    vertex_tree vertices;
    return vertex_tree::multi_insert_reduce(vertices, edges, reduce);
  }

  // Reserve space for n vertices and m edges.
  static void reserve(size_t n, size_t m) {
    vertex_tree::reserve(n);
    edge_tree::reserve(m);
  }

  struct AddFourTup {
    using T = std::tuple<size_t, size_t, size_t, size_t>;
    static T identity() { return {0, 0, 0, 0}; }
    static T add(T a, T b) {
      return {std::get<0>(a) + std::get<0>(b), std::get<1>(a) + std::get<1>(b),
              std::get<2>(a) + std::get<2>(b), std::get<3>(a) + std::get<3>(b)};
    }
  };

  void get_tree_sizes(const std::string& graphname, const std::string& mode) {
    auto noop = [](const auto& q) { return 0; };
    size_t vertex_tree_bytes = V.size_in_bytes(noop);
    auto[outer_internal, outer_leafs, outer_leaf_sizes] = V.node_stats();
    std::cout << "Num vertex_tree outer_nodes = " << outer_internal
              << " Num vertex_tree inner_nodes = " << outer_leafs
              << " Total vertex tree leaf sizes = " << outer_leaf_sizes
              << std::endl;

    auto map_f =
        [&](const auto& et) -> std::tuple<size_t, size_t, size_t, size_t> {
      auto[key, root] = et;
      if (root != nullptr) {
        edge_tree tree;
        tree.root = root;
        auto sz = tree.size_in_bytes(noop);
        auto[internal, leafs, leaf_sizes] = tree.node_stats();
        tree.root = nullptr;
        return {sz, internal, leafs, leaf_sizes};
      }
      return {0, 0, 0, 0};
    };

    auto addm = AddFourTup();
    auto[edge_tree_bytes, inner_internal, inner_leaf, inner_sizes] =
        vertex_tree::map_reduce(V, map_f, addm);

    std::cout << "Num edge_trees outer_nodes = " << inner_internal
              << " Num edge_trees inner_nodes = " << inner_leaf
              << " Total edge_trees leaf sizes = " << inner_sizes << std::endl;

    std::cout << "Edge trees size in bytes = " << edge_tree_bytes << std::endl;
    std::cout << "Vertex tree size in bytes = " << vertex_tree_bytes
              << std::endl;

    size_t total_bytes = edge_tree_bytes + vertex_tree_bytes;
    size_t m = num_edges();

    std::cout << "csv: " << graphname << "," << num_vertices() << "," << num_edges() << "," << mode
              << "," << total_bytes << "," << vertex_tree_bytes << ","
              << edge_tree_bytes << std::endl;
  }

  void print_stats() {
#ifndef USE_PAM
    size_t sz = 0;
    size_t edges_bytes = 0;
    auto f = [&](const auto& et) {
      const auto& incident = std::get<1>(et);
      auto noop = [](const auto& q) { return 0; };
      size_t edges_size = incident.size();
      edges_bytes += incident.size_in_bytes(noop);
      //      if (edges_size < 2*cpam::utils::B) {
      //	assert(incident.root_is_compressed());
      //      }
    };
    vertex_tree::foreach_seq(V, f);
    std::cout << "num_edges = " << sz << std::endl;
    std::cout << "edges_size = " << edges_bytes << std::endl;
#else

#endif
  }

  /* ============= Update Operations ================ */

  
  void insert_vertex_inplace(vertex_id id, edge_tree* e) {
    auto et = typename vertex_entry::entry_t(id, e);
    V.insert(et);
  }

  // m : number of edges
  // edges: pairs of edges to insert. Currently working with undirected graphs;
  template <class VtxEntry>
  void insert_vertices_batch(size_t m, VtxEntry* E) {
    timer pt("Insert", false);
    timer t("Insert", false);
    auto E_slice = parlay::make_slice(E, E + m);
    auto key_less = [&] (const VtxEntry& l, const VtxEntry& r) {
      return std::get<0>(l) < std::get<0>(r);
    };
    parlay::sort_inplace(E_slice, key_less);

    auto combine_op = [&] (edge_tree cur, edge_tree inc) {
      // Let cur get decremented here.
      return inc;
    };
    V = vertex_tree::multi_insert_sorted(std::move(V), E_slice, combine_op);
  }

  template <class VtxEntry>
  SymGraph insert_vertices_batch_functional(size_t m, VtxEntry* E) {
    timer pt("Insert", false);
    timer t("Insert", false);
    auto E_slice = parlay::make_slice(E, E + m);
    auto key_less = [&] (const VtxEntry& l, const VtxEntry& r) {
      return std::get<0>(l) < std::get<0>(r);
    };
    parlay::sort_inplace(E_slice, key_less);

    auto combine_op = [&] (edge_tree cur, edge_tree inc) {
      // Let cur get decremented here.
      return inc;
    };
    std::cout << "Multiinsert size: " << E_slice.size() << std::endl;
    auto new_V = vertex_tree::multi_insert_sorted(V, E_slice, combine_op);
    return SymGraph(std::move(new_V));
  }


  SymGraph functional_copy(){
    auto new_V = V;
    return SymGraph(std::move(new_V));
  }

  // m : number of vertices to delete
  // D : array of the deleted vertex ids
  void delete_vertices_batch(size_t m, vertex_id* D) {
    timer pt("Insert", false);
    timer t("Insert", false);
    auto D_slice = parlay::make_slice(D, D + m);
    auto key_less = std::less<vertex_id>();
    parlay::sort_inplace(D_slice, key_less);
    V = vertex_tree::multi_delete_sorted(std::move(V), D_slice);
  }

  SymGraph delete_vertices_batch_functional(size_t m, vertex_id* D) {
    timer pt("Insert", false);
    timer t("Insert", false);
    auto D_slice = parlay::make_slice(D, D + m);
    auto key_less = std::less<vertex_id>();
    parlay::sort_inplace(D_slice, key_less);
    auto new_V = vertex_tree::multi_delete_sorted(V, D_slice);
    return SymGraph(std::move(new_V));
  }



};

}  // namespace aspen
