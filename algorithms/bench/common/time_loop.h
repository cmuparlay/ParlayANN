#include "../parlay/internal/get_time.h"

template<class F, class G, class H>
void time_loop(int rounds, double delay, F initf, G runf, H endf) {
  parlay::internal::timer t;
  // run for delay seconds to "warm things up"
  // will skip if delay is zero
  while (t.total_time() < delay) {
    initf(); runf(); endf();
  } 
  for (int i=0; i < rounds; i++) {
    initf();
    t.start();
    runf();
    t.next("");
    endf();
  }
}
