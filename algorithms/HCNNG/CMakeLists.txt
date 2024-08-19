add_executable(neighbors-hcnng ../bench/neighborsTime.C)
  target_link_libraries(neighbors-hcnng PRIVATE parlay)
  target_precompile_headers(neighbors-hcnng PRIVATE neighbors.h)

