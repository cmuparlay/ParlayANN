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
      if (entries[loc] == k)
        return true;
      if (entries[loc] == empty) {
        insert_at(k, loc);
        return false;
      }
      loc = (loc + 1) & mask;
      while (entries[loc] != empty && entries[loc] != k)
        loc = (loc + 1) & mask;
      if (entries[loc] == k)
        return true;
      insert_at(k, loc);
      return false;
    }

    void insert_at(const K k, const int loc) {
      entries[loc] = k;
      num_entries++;
      if (num_entries >= entries.size()/2) {
        bits = bits + 1;
        std::vector<K> new_entries(1ul << bits, empty);
        mask = new_entries.size() - 1;
        swap(entries, new_entries);
        for (auto k : new_entries)
          if (k != empty) {
            int rloc = hash(k) & mask;
            while (entries[rloc] != empty)
              rloc = (rloc + 1) & mask;
            entries[rloc] = k;
          }
      }
    }
  
    hashset(long n) :
      bits(std::ceil(std::log2(n))),
      entries(std::vector<K>((1ul << bits), -1)),
      mask(entries.size() - 1)
    {}
  };

} // end namespace parlayANN
#endif // ALGORITHMS_ANN_HASHSET_H_
