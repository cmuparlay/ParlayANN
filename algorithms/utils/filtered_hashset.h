#ifndef ALGORITHMS_ANN_HASHSET_H_
#define ALGORITHMS_ANN_HASHSET_H_

#include <vector>
#include <cmath>

// a hashset that enters integer keys and can give a false negative
// grows as needed
//   filtered_hashset x(n); : creates an empty hashset x of initial capacity n
//   x(i) : returns true if i in set, otherwise adds i to set and returns false
template <typename intT>
struct filtered_hashset {
  int bits;
  std::vector<intT> filter;
  size_t mask;
  long num_entries = 0;
  size_t hash(intT const& k) const noexcept {
    return k * UINT64_C(0xbf58476d1ce4e5b9); }
    
  bool operator () (intT a) {
    int loc = hash(a) & mask;
    if (filter[loc] == a) return true;
    if (num_entries > filter.size()/2) {
      bits = bits + 1;
      std::vector<intT> new_filter(1ul << bits, -1);
      mask = new_filter.size() - 1;
      for (auto x : filter)
        new_filter[hash(x) & mask] = x;
      loc = hash(a) & mask;
      filter = new_filter;
    }
    filter[loc] = a;
    num_entries++;
    return false;
  };
  filtered_hashset(long n) :
    bits(std::ceil(std::log2(n))),
    filter(std::vector<intT>(1ul << bits, -1)),
    mask(filter.size() - 1)
  {}
};

template <typename intT>
struct hashset {
  int bits;
  std::vector<intT> filter;
  size_t mask;
  long num_entries = 0;
  size_t hash(intT const& k) const noexcept {
    return k * UINT64_C(0xbf58476d1ce4e5b9); }
    
  bool operator () (intT a) {
    int loc = hash(a) & mask;
    if (filter[loc] == a) return true;
    if (filter[loc] != -1) {
      loc = (loc + 1) & mask;
      while (filter[loc] != -1 && filter[loc] != a)
        loc = (loc + 1) & mask;
      if (filter[loc] == a) return true;
    }
    if (num_entries > filter.size()/2) {
      bits = bits + 1;
      std::vector<intT> new_filter(1ul << bits, -1);
      mask = new_filter.size() - 1;
      for (auto x : filter) {
        int loc = hash(x) & mask;
        while (new_filter[loc] != -1)
          loc = (loc + 1) & mask;
        new_filter[loc] = x;
      }
      loc = hash(a) & mask;
      filter = new_filter;
    }
    filter[loc] = a;
    num_entries++;
    return false;
  };
  hashset(long n) :
    bits(std::ceil(std::log2(n))),
    filter(std::vector<intT>((1ul << bits), -1)),
    mask(filter.size() - 1)
  {}
};

#endif // ALGORITHMS_ANN_HASHSET_H_
