#pragma once

// TODO can we get rid of this "weight" struct? It seems like it's always empty
// TODO can we templatize over the edge and vertex ID types? it seems like at
// the moment they are hardcoded

namespace aspenflat {

template <typename indexType>
struct symmetric_graph {

  // questions: how to store/allocate without wasting space?
  // should we work with a max size and use parlay allocator or new?
  // or have a variety of allocators based on size? it would need to be pretty
  // granular
  // TODO could we just have a parlay::sequence?
  struct edge_array {

    template <typename RangeType>
    edge_array(RangeType R) {
      // +1 to include the ref_cnt
      edges = (indexType*)malloc(sizeof(indexType) * (R.size() + 1));
      size_ = R.size();
      for (size_t i=0; i<R.size(); ++i) {
        edges[i+1] = R[i];
      }
      edges[0] = 1;
      std::cout << "End of range constructor, ref_cnt = " << get_ref_cnt() << std::endl;
    }

    // copy constructor, increment reference count
    edge_array(const edge_array& E) {
      std::cout << "Copy constructor! ref_cnt of E = " << E.get_ref_cnt() << std::endl;
      edges = E.edges;
      size_ = E.size_;
      increment();
      std::cout << "End of copy constructor! ref_cnt of our edges = " << get_ref_cnt() << std::endl;
    }

    // copy assignment, clear target, increment reference count,
    edge_array& operator = (const edge_array& E) {
      std::cout << "Copy assignment!" << std::endl;
      clear();

      edges = E.edges;
      size_ = E.size_;
      increment();
      return *this;
    }

    // move constructor
    edge_array(edge_array&& E) {
      std::cout << "Move constructor!: edges = " << edges << std::endl;
      edges = E.edges;
      size_ = E.size_;
      E.edges = nullptr;
      E.size_ = 0;
    }

    // move assignment, clear target, leave reference count as is
    edge_array& operator = (edge_array&& E) {
      std::cout << "Move assignment!" << std::endl;
      if (this != &E) {
        clear();
        edges = E.edges;
        size_ = E.size_;
        E.edges = nullptr;
        E.size_ = 0;
      }
      return *this;
    }

    edge_array() {
      // std::cout << "Default constructor" << std::endl;
      edges = nullptr;
      size_ = 0;
    }

    parlay::slice<indexType*, indexType*> get_edges() {
      return parlay::make_slice(edges + 1, edges + 1 + size_);
    }

    size_t size() { return size_; }

    size_t get_ref_cnt() const {
      if (edges) {
        return edges[0];
      }
      return 0;
    }

    void increment() { utils::write_add(edges, 1); }

    void clear() {
      // Check that the following is OK with unsigned integers (e.g.,
      // indexType).

      if (edges) {
        std::cout << "Clear on : " << edges << " ref_cnt = " << get_ref_cnt() << std::endl;
        std::cout << "Before decrement, ref_cnt = " << get_ref_cnt() << std::endl;
        if (utils::fetch_and_add(edges, -1) == 1) {
          std::cout << "After decrement, ref_cnt = " << get_ref_cnt() << std::endl;
          // do the free since we were the last owner
          free(edges);
          edges = nullptr;
          size_ = 0;
        }
        edges = nullptr;
        size_ = 0;
      }
    }

    // Implicitly we encode ref_cnt as the first entry of edges. The
    // type of ref_cnt is indexType.
    indexType* edges = nullptr;
    size_t size_;

    ~edge_array() { clear(); }

  };   // end edge_array

  // aug_t is (max_id, total # edges)
  struct vertex_entry {
    using key_t = indexType;
    using val_t = edge_array;
    using entry_t = std::tuple<key_t, val_t>;

    static inline bool comp(key_t a, key_t b) { return a < b; }
  };   // end vertex_entry

  using vertex_tree = cpam::pam_map<vertex_entry>;
  using vtx_entry = typename vertex_tree::Entry;
  using vertex_node = typename vertex_tree::node;
  using vertex_gc = typename vertex_tree::GC;
  using SymGraph = symmetric_graph<indexType>;

  vertex_tree V;

  // Set from a provided root (no ref-ct bump)
  symmetric_graph(vertex_node* root) { set_root(root); }

  symmetric_graph(vertex_tree&& _V) : V(std::move(_V)) {}

  symmetric_graph() { V.root = nullptr; }

  vertex_tree& get_vertices() { return V; }

  void clear_root() { V.root = nullptr; }

  vertex_node* get_root() { return V.root; }

  size_t ref_cnt() { return V.ref_cnt(); }

  void set_root(vertex_node* root) { V.root = root; }

  // Note that it's important to use n and not V.size() here.
  // TODO replace with something that accurately counts vertices
  // use a foreach
  size_t num_vertices() const { return V.size(); }

  // size_t num_edges() const { return V.aug_val().second; }

  // TODO can we return a pointer here instead?
  edge_array get_vertex(indexType v) const {
    auto opt = V.find(v);
    std::cout << "Looking for v = " << v << " opt return value = " << opt.has_value() << std::endl;
    return *opt;
  }

  // Reserve space for n vertices and m edges.
  static void reserve(size_t n) { vertex_tree::reserve(n); }

  // TODO re-implement stats stuff

  // struct AddFourTup {
  //   using T = std::tuple<size_t, size_t, size_t, size_t>;
  //   static T identity() { return {0, 0, 0, 0}; }
  //   static T add(T a, T b) {
  //     return {std::get<0>(a) + std::get<0>(b), std::get<1>(a) +
  //     std::get<1>(b),
  //             std::get<2>(a) + std::get<2>(b), std::get<3>(a) +
  //             std::get<3>(b)};
  //   }
  // };

  // void get_tree_sizes(const std::string& graphname, const std::string& mode)
  // {
  //   auto noop = [](const auto& q) { return 0; };
  //   size_t vertex_tree_bytes = V.size_in_bytes(noop);
  //   auto[outer_internal, outer_leafs, outer_leaf_sizes] = V.node_stats();
  //   std::cout << "Num vertex_tree outer_nodes = " << outer_internal
  //             << " Num vertex_tree inner_nodes = " << outer_leafs
  //             << " Total vertex tree leaf sizes = " << outer_leaf_sizes
  //             << std::endl;

  //   auto map_f =
  //       [&](const auto& et) -> std::tuple<size_t, size_t, size_t, size_t> {
  //     auto[key, root] = et;
  //     if (root != nullptr) {
  //       edge_tree tree;
  //       tree.root = root;
  //       auto sz = tree.size_in_bytes(noop);
  //       auto[internal, leafs, leaf_sizes] = tree.node_stats();
  //       tree.root = nullptr;
  //       return {sz, internal, leafs, leaf_sizes};
  //     }
  //     return {0, 0, 0, 0};
  //   };

  //   auto addm = AddFourTup();
  //   auto[edge_tree_bytes, inner_internal, inner_leaf, inner_sizes] =
  //       vertex_tree::map_reduce(V, map_f, addm);

  //   std::cout << "Num edge_trees outer_nodes = " << inner_internal
  //             << " Num edge_trees inner_nodes = " << inner_leaf
  //             << " Total edge_trees leaf sizes = " << inner_sizes <<
  //             std::endl;

  //   std::cout << "Edge trees size in bytes = " << edge_tree_bytes <<
  //   std::endl; std::cout << "Vertex tree size in bytes = " <<
  //   vertex_tree_bytes
  //             << std::endl;

  //   size_t total_bytes = edge_tree_bytes + vertex_tree_bytes;
  //   size_t m = num_edges();

  //   std::cout << "csv: " << graphname << "," << num_vertices() << "," <<
  //   num_edges() << "," << mode
  //             << "," << total_bytes << "," << vertex_tree_bytes << ","
  //             << edge_tree_bytes << std::endl;
  // }

  // void print_stats() {
  //   size_t sz = 0;
  //   size_t edges_bytes = 0;
  //   auto f = [&](const auto& et) {
  //     const auto& incident = std::get<1>(et);
  //     auto noop = [](const auto& q) { return 0; };
  //     size_t edges_size = incident.size();
  //     edges_bytes += incident.size_in_bytes(noop);
  //     //      if (edges_size < 2*cpam::utils::B) {
  //     //	assert(incident.root_is_compressed());
  //     //      }
  //   };
  //   vertex_tree::foreach_seq(V, f);
  //   std::cout << "num_edges = " << sz << std::endl;
  //   std::cout << "edges_size = " << edges_bytes << std::endl;
  // }

  /* ============= Update Operations ================ */

  // m : number of edges
  // edges: pairs of edges to insert. Currently working with undirected graphs;
  template <class VtxEntry>
  void insert_vertices_batch(size_t m, VtxEntry* E) {
    timer pt("Insert", false);
    timer t("Insert", false);
    auto E_slice = parlay::make_slice(E, E + m);
    auto key_less = [&](const VtxEntry& l, const VtxEntry& r) {
      return std::get<0>(l) < std::get<0>(r);
    };
    parlay::sort_inplace(E_slice, key_less);

    auto combine_op = [&](edge_array cur, edge_array inc) {
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
    auto key_less = [&](const VtxEntry& l, const VtxEntry& r) {
      return std::get<0>(l) < std::get<0>(r);
    };
    parlay::sort_inplace(E_slice, key_less);

    auto combine_op = [&](edge_array cur, edge_array inc) {
      // Let cur get decremented here.
      return inc;
    };
    std::cout << "Multiinsert size: " << E_slice.size() << std::endl;
    auto new_V = vertex_tree::multi_insert_sorted(V, E_slice, combine_op);
    return SymGraph(std::move(new_V));
  }

  SymGraph functional_copy() {
    auto new_V = V;
    return SymGraph(std::move(new_V));
  }

  // m : number of vertices to delete
  // D : array of the deleted vertex ids
  void delete_vertices_batch(size_t m, indexType* D) {
    timer pt("Insert", false);
    timer t("Insert", false);
    auto D_slice = parlay::make_slice(D, D + m);
    auto key_less = std::less<indexType>();
    parlay::sort_inplace(D_slice, key_less);
    V = vertex_tree::multi_delete_sorted(std::move(V), D_slice);
  }

  SymGraph delete_vertices_batch_functional(size_t m, indexType* D) {
    timer pt("Insert", false);
    timer t("Insert", false);
    auto D_slice = parlay::make_slice(D, D + m);
    auto key_less = std::less<indexType>();
    parlay::sort_inplace(D_slice, key_less);
    auto new_V = vertex_tree::multi_delete_sorted(V, D_slice);
    return SymGraph(std::move(new_V));
  }
};

}   // namespace aspen
