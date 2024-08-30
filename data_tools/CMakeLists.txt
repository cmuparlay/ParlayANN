function(add_parlayann_app NAME)
  add_executable(${NAME} ${NAME}.cpp)
    target_link_libraries(${NAME} PRIVATE parlay ParlayANN)
endfunction()

add_parlayann_app(vec_to_bin)
add_parlayann_app(compute_groundtruth)
add_parlayann_app(compute_range_groundtruth)
add_parlayann_app(crop)
add_parlayann_app(random_sample)
add_parlayann_app(build_knn_graph)
add_parlayann_app(graph_analysis)

add_executable(partition_graph partition_graph.cpp)
  target_link_libraries(partition_graph PRIVATE parlay ParlayANN)
  target_link_libraries(partition_graph PUBLIC KaMinPar::KaMinPar)  # Shared-memory partitioning

