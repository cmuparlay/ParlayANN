function(add_parlay_app NAME)
  add_executable(${NAME} ${NAME}.cpp)
    target_link_libraries(${NAME} PRIVATE parlay)
endfunction()

add_parlay_app(vec_to_bin)
add_parlay_app(compute_groundtruth)
add_parlay_app(compute_range_groundtruth)
add_parlay_app(crop)
add_parlay_app(random_sample)

add_executable(evaluate_partitioning evaluate_partitioning.cpp)
  target_link_libraries(evaluate_partitioning PRIVATE parlay ParlayANN)
  target_link_libraries(evaluate_partitioning PUBLIC KaMinPar::KaMinPar)  # Shared-memory partitioning

