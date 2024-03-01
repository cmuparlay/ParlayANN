#pragma once

// Defines the API that readers and the writer use to acquire versions of the
// graph and commit new versions, making them readable.
// #include "traversable_graph.h"
#include "sequentialHT.h"

#include <limits>

namespace aspen {

namespace refct_utils {
  static uint64_t make_refct(uint32_t current_ts, uint32_t ref_ct) {
    return (static_cast<uint64_t>(current_ts) << 32UL) | static_cast<uint64_t>(ref_ct);
  }
  static uint32_t get_rfct(uint64_t ts_and_ct) {
    static constexpr const uint64_t mask = std::numeric_limits<uint32_t>::max();
    return ts_and_ct & mask;
  }
}


using ts = uint32_t;
static constexpr const ts max_ts = std::numeric_limits<uint32_t>::max();
static constexpr const ts tombstone = max_ts - 1;

template <class snapshot_graph>
struct versioned_graph {

  ts current_timestamp;

  using Tree = typename snapshot_graph::vertex_tree::Tree;
  using Node = typename snapshot_graph::vertex_node;
  using Node_GC = typename snapshot_graph::vertex_gc;

  // Currently wasteful; ts is duplicated.
  using K = uint64_t;
  using V = tuple<uint64_t, Node*>;
  using T = tuple<K, V>;
  using table = sequentialHT<K, V>;
  table live_versions;

  static constexpr typename table::T empty = std::make_tuple(max_ts, std::make_tuple(0, nullptr));

  struct version {
    ts timestamp;
    T* table_entry;
    snapshot_graph graph;
    version(ts _timestamp, T* _table_entry, snapshot_graph&& _graph) :
      timestamp(_timestamp), table_entry(_table_entry) {
      graph.set_root(_graph.get_root());
      _graph.clear_root();
    }

    void* get_root() {
      return graph.get_root();
    }

    size_t get_graph_ref_cnt() {
      return graph.ref_cnt();
    }

    size_t get_ref_cnt() {
      uint64_t refct_and_ts = std::get<0>(std::get<1>(*table_entry));
      return refct_utils::get_rfct(refct_and_ts);
    }
  };

  versioned_graph() {
    current_timestamp=0;
    size_t initial_ht_size = 512;
    live_versions = table(initial_ht_size, empty, tombstone);

    auto initial_graph = nullptr;
    ts timestamp = current_timestamp++;
    live_versions.insert(std::make_tuple(timestamp, std::make_tuple(refct_utils::make_refct(timestamp, 1), std::move(initial_graph))));
  }

  versioned_graph(snapshot_graph&& G) : current_timestamp(0) {
    size_t initial_ht_size = 512;
    live_versions = table(initial_ht_size, empty, tombstone);

    ts timestamp = current_timestamp++;
    live_versions.insert(std::make_tuple(timestamp, std::make_tuple(refct_utils::make_refct(timestamp, 1), G.get_root())));
    std::cout << "inserted with ts = " << timestamp << std::endl;
    G.clear_root();
  }


  ts latest_timestamp() {
    return current_timestamp-1;
  }

  // Lock-free, but not wait-free
  version acquire_version() {
    while (true) {
      size_t latest_ts = latest_timestamp();
//      std::cout << "looking up with ts = " << latest_ts << std::endl;
      tuple<T&, bool> ref_and_valid = live_versions.find(latest_ts);
      T& table_ref = std::get<0>(ref_and_valid); bool valid = std::get<1>(ref_and_valid);
      if (valid) {
        while(true) {
          // can't be max_ts in a probe sequence
          if (std::get<0>(table_ref) == tombstone) break;
          uint64_t refct_and_ts = std::get<0>(std::get<1>(table_ref));
          uint64_t next_value = refct_and_ts + 1;
          size_t ref_ct_before = refct_utils::get_rfct(refct_and_ts);
          size_t ts = std::get<0>(table_ref);
          if (ref_ct_before > 0) {
            if (cpam::utils::atomic_compare_and_swap(&std::get<0>(std::get<1>(table_ref)), refct_and_ts, next_value)) {
              auto graph = snapshot_graph(std::get<1>(std::get<1>(table_ref)));
//              std::cout << "Success in CAS! graph = " << graph.get_root() << " ref_cnt = " << graph.ref_cnt() << std::endl;
              return version(ts, &table_ref, std::move(graph));
            }
          } else { // refct == 0
            break;
          }
        }
      }
    }
  }

  void release_version(version&& S) {
    ts timestamp = S.timestamp;
    T* table_entry = S.table_entry;
    auto root = S.graph.get_root();
    S.graph.clear_root(); // relinquish ownership

    uint64_t* ref_ct_loc = &(std::get<0>(std::get<1>(*table_entry)));
    if (refct_utils::get_rfct(cpam::utils::fetch_and_add(ref_ct_loc, -1)) == 2 &&
        timestamp != latest_timestamp()) {
      // read again and try to free.
      uint64_t cur_val = *ref_ct_loc;
      if (refct_utils::get_rfct(cur_val) == 1) {

        if (cpam::utils::atomic_compare_and_swap(ref_ct_loc, cur_val, cur_val-1)) {
          // no longer possible for new readers to acquire
          if (root) { // might be an empty graph
            Node_GC::decrement_recursive(root);
          }

          typename table::T first_empty = std::make_tuple(timestamp, std::make_tuple(0, nullptr));
          *table_entry = first_empty;
          typename table::T tomb_empty = std::make_tuple(tombstone, std::make_tuple(0, nullptr));
          *table_entry = tomb_empty;
        }
      }
    }
  }



  void add_version_from_graph(snapshot_graph G_next){
    live_versions.insert(std::make_tuple(current_timestamp,
                                    std::make_tuple(refct_utils::make_refct(current_timestamp, 1),
                                               G_next.get_root())));
    G_next.clear_root();

    std::cout << "New version released with timestamp " << current_timestamp << std::endl;
    // 2. Make the new version visible
    cpam::utils::fetch_and_add(&current_timestamp, 1);
  }

  // Returns the number of live versions in the versioning structure.
  size_t num_live_versions() {
    return live_versions.num_nonempty_slots();
  }

};

}  // namespace aspen
