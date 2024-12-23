#ifndef ALGORITHMS_ANN_UNION_
#define ALGORITHMS_ANN_UNION_

#include <set>
#include "parlay/parallel.h"
#include "parlay/primitives.h"

namespace parlayANN {

// takes in two sorted sequences and returns a sorted union
// of length at most K, with a bool denoting whether P has changed
template <typename F, typename T>
std::pair<parlay::sequence<T>, bool> seq_union_bounded(
    parlay::sequence<T>& P, parlay::sequence<T>& Q, int K, F&& less) {
  T* first1 = P.begin();
  T* last1 = P.end();
  T* first2 = Q.begin();
  T* last2 = Q.end();
  bool changed = false;
  parlay::sequence<T> result = parlay::sequence<T>();
  result.reserve(K);
  int count = 0;
  while (true && count < K) {
    if (first1 == last1) {
      while (first2 != last2 && count < K) {
        changed = true;
        result.push_back(*first2);
        count++;
        ++first2;
      }
      return std::make_pair(result, changed);
    } else if (first2 == last2) {
      while (first1 != last1 && count < K) {
        result.push_back(*first1);
        count++;
        ++first1;
      }
      return std::make_pair(result, changed);
    }
    if (less(*first1, *first2)) {
      result.push_back(*first1);
      count++;
      ++first1;
    } else if (less(*first2, *first1)) {
      result.push_back(*first2);
      changed = true;
      count++;
      ++first2;
    } else {
      if (first1->first == first2->first) {
        result.push_back(*first1);
        count++;
        ++first1;
        ++first2;
      } else {
        result.push_back(*first1);
        count++;
        if (count == K)
          break;
        else {
          result.push_back(*first2);
          changed = true;
          count++;
          ++first1;
          ++first2;
        }
      }
    }
  }
  return std::make_pair(result, changed);
}

// takes in two sorted sequences and returns a sorted union
template <typename F, typename T>
parlay::sequence<T> seq_union(parlay::sequence<T>& P, parlay::sequence<T>& Q, F&& less) {
  T* first1 = P.begin();
  T* last1 = P.end();
  T* first2 = Q.begin();
  T* last2 = Q.end();
  parlay::sequence<T> result = parlay::sequence<T>();
  result.reserve(P.size() + Q.size());
  while (true) {
    if (first1 == last1) {
      while (first2 != last2) {
        result.push_back(*first2);
        ++first2;
      }
      return result;
    } else if (first2 == last2) {
      while (first1 != last1) {
        result.push_back(*first1);
        ++first1;
      }
      return result;
    }
    if (less(*first1, *first2)) {
      result.push_back(*first1);
      ++first1;
    } else if (less(*first2, *first1)) {
      result.push_back(*first2);
      ++first2;
    } else {
      if (first1->first == first2->first) {
        result.push_back(*first1);
        ++first1;
        ++first2;
      } else {
        result.push_back(*first1);
        result.push_back(*first2);
        ++first1;
        ++first2;
      }
    }
  }
  return result;
}

} // end namespace

#endif // ALGORITHMS_ANN_UNION_
