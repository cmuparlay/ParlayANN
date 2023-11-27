#pragma once
#include "../parlay/utilities.h"

namespace dataGen {

#define HASH_MAX_INT ((unsigned) 1 << 31)

  //#define HASH_MAX_LONG ((unsigned long) 1 << 63)

  template <class T> T hash(size_t i);
  
  template <>
  inline int hash<int>(size_t i) {
    return parlay::hash64(i) & ((((size_t) 1) << 31) - 1);}

  template <>
  inline long  hash<long>(size_t i) {
    return parlay::hash64(i) & ((((size_t) 1) << 63) - 1);}

  template <>
  inline unsigned int hash<unsigned int>(size_t i) {
    return parlay::hash64(i);}

  template <>
  inline size_t hash<size_t>(size_t i) {
    return parlay::hash64(i);}

  template <>
  inline double hash<double>(size_t i) {
    return ((double) hash<int>(i)/((double) ((((size_t) 1) << 31) - 1)));}

  template <>
  inline float hash<float>(size_t i) {
    return ((double) hash<int>(i)/((double) ((((size_t) 1) << 31) - 1)));}
};
