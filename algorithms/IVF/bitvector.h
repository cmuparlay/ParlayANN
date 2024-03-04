#pragma once

#include "parlay/primitives.h"

// A bitvector over n indices. Initially all bits are set to 0. You can
// *sequentially* call set_bit(i) to set the i-th bit to 1; note that we are
// currently not using CAS so concurrent calls should be avoided.
struct Bits {
 public:

  Bits() { }

  // n: number of points
  Bits(size_t n) {
    // Calculate the number of 64-bit words needed
    size_t num_words = (n + 64 - 1) / 64;
    // Initialize to all 0
    bits = parlay::sequence<size_t>(num_words, 0);
  }

  // Set the index-th bit.
  void set_bit(size_t index) {
    size_t word_ind = get_word(index);
    uint64_t cur_word = bits[word_ind];
    size_t bit_pos = get_bit_pos(index);
    uint64_t new_word = cur_word | (1UL << bit_pos);
    bits[word_ind] = new_word;
  }

  // Return true iff the index-th bit is set.
  inline bool is_bit_set(size_t index) const {
    uint64_t cur_word = bits[get_word(index)];
    size_t bit_pos = get_bit_pos(index);
    return (cur_word >> bit_pos) & 1;
  }

  // Return the footprint of the bitvector in bytes.
  size_t footprint() const {
    return bits.size() * sizeof(size_t);
  }

 private:
  // The compiler should optimize / 64 right? We can try the
  // shift-version too and check if it helps.
  // godbolt says yes at -O3
  constexpr inline size_t get_word(size_t index) const { return index / 64; }
  // The bit position in the word for this index.
  constexpr inline size_t get_bit_pos(size_t index) const {
    return index - ((index / 64) * 64);
  }
  parlay::sequence<uint64_t> bits;
};
