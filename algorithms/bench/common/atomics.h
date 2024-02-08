#pragma once

namespace pbbs {

  template <typename ET>
  inline bool atomic_compare_and_swap(ET* a, ET oldval, ET newval) {
    static_assert(sizeof(ET) <= 8, "Bad CAS length");
    if (sizeof(ET) == 1) {
      uint8_t r_oval, r_nval;
      std::memcpy(&r_oval, &oldval, sizeof(ET));
      std::memcpy(&r_nval, &newval, sizeof(ET));
      return __sync_bool_compare_and_swap(reinterpret_cast<uint8_t*>(a), r_oval, r_nval);
    } else if (sizeof(ET) == 4) {
      uint32_t r_oval, r_nval;
      std::memcpy(&r_oval, &oldval, sizeof(ET));
      std::memcpy(&r_nval, &newval, sizeof(ET));
      return __sync_bool_compare_and_swap(reinterpret_cast<uint32_t*>(a), r_oval, r_nval);
    } else { // if (sizeof(ET) == 8) {
      uint64_t r_oval, r_nval;
      std::memcpy(&r_oval, &oldval, sizeof(ET));
      std::memcpy(&r_nval, &newval, sizeof(ET));
      return __sync_bool_compare_and_swap(reinterpret_cast<uint64_t*>(a), r_oval, r_nval);
    } 
  }

  template <typename E, typename EV>
  inline E fetch_and_add(E *a, EV b) {
    volatile E newV, oldV;
    do {oldV = *a; newV = oldV + b;}
    while (!atomic_compare_and_swap(a, oldV, newV));
    return oldV;
  }

  template <typename E, typename EV>
  inline void write_add(E *a, EV b) {
    //volatile E newV, oldV;
    E newV, oldV;
    do {oldV = *a; newV = oldV + b;}
    while (!atomic_compare_and_swap(a, oldV, newV));
  }

  template <typename E, typename EV>
  inline void write_add(std::atomic<E> *a, EV b) {
    //volatile E newV, oldV;
    E newV, oldV;
    do {oldV = a->load(); newV = oldV + b;}
    while (!std::atomic_compare_exchange_strong(a, &oldV, newV));
  }

  template <typename ET, typename F>
  inline bool write_min(ET *a, ET b, F less) {
    ET c; bool r=0;
    do c = *a;
    while (less(b,c) && !(r=atomic_compare_and_swap(a,c,b)));
    return r;
  }

  template <typename ET, typename F>
  inline bool write_min(std::atomic<ET> *a, ET b, F less) {
    ET c; bool r=0;
    do c = a->load();
    while (less(b,c) && !(r=std::atomic_compare_exchange_strong(a, &c, b)));
    return r;
  }

  template <typename ET, typename F>
  inline bool write_max(ET *a, ET b, F less) {
    ET c; bool r=0;
    do c = *a;
    while (less(c,b) && !(r=atomic_compare_and_swap(a,c,b)));
    return r;
  }

  template <typename ET, typename F>
  inline bool write_max(std::atomic<ET> *a, ET b, F less) {
    ET c; bool r=0;
    do c = a->load();
    while (less(c,b) && !(r=std::atomic_compare_exchange_strong(a, &c, b)));
    return r;
  }
}
