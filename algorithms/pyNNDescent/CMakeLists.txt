add_executable(neighbors-pynndescent ../bench/neighborsTime.C)
  target_link_libraries(neighbors-pynndescent PRIVATE parlay)
  target_precompile_headers(neighbors-pynndescent PRIVATE neighbors.h)

