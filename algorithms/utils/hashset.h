#ifndef ALGORITHMS_ANN_HASHSET_H_
#define ALGORITHMS_ANN_HASHSET_H_

#include <vector>
#include <cmath>
namespace parlayANN {

// a hashset that enters integer keys and can give a false negative
// grows as needed
//   hashset x(n); : creates an empty hashset x of initial capacity n
//   x(i) : returns true if i in set, otherwise adds i to set and returns false
  template <typename K>
  struct hashset {
    static constexpr K empty = (K) -1;
    int bits;
    std::vector<K> entries;
    size_t mask = 0;
    long num_entries = 0;
    size_t hash(K const& k) const noexcept {
      return k * UINT64_C(0xbf58476d1ce4e5b9); }

    bool operator () (K a) {
      int loc = hash(a) & mask;
      if (entries[loc] == a) return true;
      if (num_entries > entries.size()/2) {
        bits = bits + 1;
        std::vector<K> new_entries(1ul << bits, empty);
        mask = new_entries.size() - 1;
        swap(entries, new_entries);
        for (auto k : new_entries)
          if (k != empty) {
            int loc = hash(k) & mask;
            while (entries[loc] != empty && entries[loc] != k)
              loc = (loc + 1) & mask;
            entries[loc] = k;
            num_entries++;
          }
      }
      if (entries[loc] != empty) {
        loc = (loc + 1) & mask;
        while (entries[loc] != -1 && entries[loc] != a)
          loc = (loc + 1) & mask;
        if (entries[loc] == a) return true;
      }
      entries[loc] = a;
      num_entries++;
      return false;
    }
  
    hashset(long n) :
      bits(std::ceil(std::log2(n))),
      entries(std::vector<K>((1ul << bits), -1)),
      mask(entries.size() - 1)
    {}
  };

}
#endif // ALGORITHMS_ANN_HASHSET_H_
