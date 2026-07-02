#ifndef ALGORITHMS_ANN_HASHSET_H_
#define ALGORITHMS_ANN_HASHSET_H_

#include <vector>
#include <cmath>
namespace parlayANN {
  // K must be an signed integer type
  // keys must only be non-negative (reserves -1 for empty).
  template <typename K>
  struct hashset {
    static constexpr K empty = (K) -1;
    int bits;  // log of size
    std::vector<K> entries;
    size_t mask = 0;
    long num_entries = 0;
    size_t hash(K const& k) const noexcept {
      auto x =  (uint64_t)(k) * UINT64_C(0xbf58476d1ce4e5b9); 
      return (x ^ (x >> 31)); }  // linear then non-linear transform
  
    bool operator () (const K k) {
      int loc = hash(k) & mask;
      if (entries.size() == 0) return {};
      while (entries[loc] != empty && entries[loc] != k)
        loc = (loc + 1) & mask;
      return (entries[loc] == k);
    }

    void insert_fits(const K k) {
      int loc = hash(k) & mask;
      while (entries[loc] != empty && entries[loc] != k)
        loc = (loc + 1) & mask;
      entries[loc] = k;
    }
  
    void insert(const K k) {
      if (num_entries >= entries.size()/2) {
        bits = bits + 1;
        std::vector<K> new_entries(1ul << bits, empty);
        mask = new_entries.size() - 1;
        swap(entries, new_entries);
        for (auto k : new_entries)
          if (k != empty) insert_fits(k);
      }
      insert_fits(k);
      num_entries++;
    }
  
    hashset(long n) :
      bits(std::ceil(std::log2(n))),
      entries(std::vector<K>((1ul << bits), -1)),
      mask(entries.size() - 1)
    {}
  };

} // end namespace parlayANN
#endif // ALGORITHMS_ANN_HASHSET_H_
