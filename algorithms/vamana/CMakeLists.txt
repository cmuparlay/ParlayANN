add_executable(neighbors-vamana ../bench/neighborsTime.C)
  target_link_libraries(neighbors-vamana PRIVATE parlay)
  target_precompile_headers(neighbors-vamana PRIVATE neighbors.h)

