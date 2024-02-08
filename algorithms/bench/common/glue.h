#pragma once

#include "../pbbslib/hash_table.h"
#include "../pbbslib/integer_sort.h"

using intT = int;
using uintT = unsigned int;

#define newA(__E,__n) (__E*) malloc((__n)*sizeof(__E))

namespace utils {

  static void myAssert(int cond, std::string s) {
    if (!cond) {
      std::cout << s << std::endl;
      abort();
    }
  }

  inline unsigned int hash(unsigned int a)
  {
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
  }

  inline int hashInt(unsigned int a) {  
    return hash(a) & (((unsigned) 1 << 31) - 1);
  }

  // template <class E>
  // struct identityF { E operator() (const E& x) {return x;}};

  // template <class E>
  // struct addF { E operator() (const E& a, const E& b) const {return a+b;}};

  // template <class E>
  // struct absF { E operator() (const E& a) const {return std::abs(a);}};

  // template <class E>
  // struct zeroF { E operator() (const E& a) const {return 0;}};

  // template <class E>
  // struct maxF { E operator() (const E& a, const E& b) const {return (a>b) ? a : b;}};

  // template <class E>
  // struct minF { E operator() (const E& a, const E& b) const {return (a<b) ? a : b;}};

  // template <class E1, class E2>
  // struct firstF {E1 operator() (std::pair<E1,E2> a) {return a.first;} };

  // template <class E1, class E2>
  // struct secondF {E2 operator() (std::pair<E1,E2> a) {return a.second;} };

}

// template <class T>
// struct _seq {
//   T* A;
//   long n;
//   _seq() {A = NULL; n=0;}
//   _seq(T* _A, long _n) : A(_A), n(_n) {}
//   void del() {free(A);}
// };

// namespace osequence {

//   template <class ET, class intT, class F>
//   ET scan(ET *In, ET* Out, intT n, F f, ET zero) {
//     if (In == Out)
//       return pbbs::scan_inplace(pbbs::range<ET*>(In,In+n),pbbs::make_monoid(f,zero));
//     else {
//       std::cout << "NYI in scan" << std::endl;
//       return zero;
//     }
//   }

//   template <class ET>
//   ET plusScan(ET *In, ET* Out, size_t n) {
//     return scan(In, Out, n, [&] (ET a, ET b) {return a + b;}, (ET) 0);
//   }
  
//   template <class ET, class PRED>
//   size_t filter(ET* In, ET* Out, size_t n, PRED p) {
//     pbbs::sequence<ET> r = pbbs::filter(pbbs::range<ET*>(In,In+n), p);
//     parallel_for(0, r.size(), [&] (size_t i) {Out[i] = r[i];});
//     return r.size();
//   }

// };

namespace dataGen {

  using namespace std;

#define HASH_MAX_INT ((unsigned) 1 << 31)

  //#define HASH_MAX_LONG ((unsigned long) 1 << 63)

  template <class T> T hash(intT i);
  
  template <>
  inline intT hash<intT>(intT i) {
    return utils::hash(i) & (HASH_MAX_INT-1);}

  template <>
  inline uintT hash<uintT>(intT i) {
    return utils::hash(i);}

  template <>
  inline double hash<double>(intT i) {
    return ((double) hash<intT>(i)/((double) HASH_MAX_INT));}

};

// template <class HASH, class ET>
// _seq<ET> removeDuplicates(_seq<ET> S, HASH hashF) {
//   return pbbs::remove_duplicates(pbbs::range<ET*>(S.A, S.A+S.n), hashF);
// }

