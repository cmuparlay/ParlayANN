#pragma once

template <class K, class V>
class sequentialHT {
 public:
  typedef tuple<K,V> T;

  size_t m, n;
  size_t mask;
  T empty;
  K max_key;
  K tombstone;
  parlay::sequence<T> table;

  inline size_t toRange(size_t h) {return h & mask;}
  inline size_t firstIndex(K v) {return toRange(parlay::hash32_2(v >> 32UL));}
  inline size_t incrementIndex(size_t h) {return toRange(h+1);}

  // m must be a power of two
  sequentialHT(size_t _m, tuple<K, V> _empty, K _tombstone) :
    m((size_t)_m), mask(m-1), empty(_empty), tombstone(_tombstone)
  {
    max_key = std::get<0>(empty);
    table = parlay::sequence<T>(_m, empty);
  }

  sequentialHT() {}

  bool insert(tuple<K, V> v) {
    K key = std::get<0>(v);
    size_t h = firstIndex(key);
    while (1) {
      auto k = std::get<0>(table[h]);
      if (k == max_key || k == tombstone) {
        table[h] = v;
        n++;
        return true;
      } else if (k == key) {
        return false;
      }
      h = incrementIndex(h);
    }
  }

  template <class F>
  inline bool remove(K key, F deletion_fn) {
    size_t h = firstIndex(key);
    while (1) {
      auto k = std::get<0>(table[h]);
      if (k == max_key) {
        return false;
      } else if (k == key) {
        deletion_fn(table[h]);
        std::get<0>(table[h]) = tombstone;
      }
      h = incrementIndex(h);
    }
  }

  inline tuple<T&,bool> find(K key) {
    size_t h = firstIndex(key);
    while (1) {
      auto& table_ref = table[h];
      if (std::get<0>(table_ref) == max_key) {
        return std::forward_as_tuple(table_ref, false);
      } else if (std::get<0>(table_ref) == key) {
      	return std::forward_as_tuple(table_ref, true);
      }
      h = incrementIndex(h);
    }
  }

  template <class Eq>
  inline tuple<T&,bool> find(K key, Eq& eq) {
    size_t h = firstIndex(key);
    while (1) {
      auto& table_ref = table[h];
      if (eq(std::get<0>(table_ref), max_key)) {
        return std::forward_as_tuple(table_ref, false);
      } else if (eq(std::get<0>(table_ref), key)) {
      	return std::forward_as_tuple(table_ref, true);
      }
      h = incrementIndex(h);
    }
  }

};

