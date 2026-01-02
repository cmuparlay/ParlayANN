function(add_neighbors NAME DEFS)
  add_executable(${NAME} ../bench/neighborsTime.C)
  target_link_libraries(${NAME} PRIVATE parlay)
  target_compile_definitions(${NAME} PRIVATE ${DEFS})
  target_precompile_headers(${NAME} PRIVATE neighbors.h)
  target_include_directories(${NAME} PRIVATE ${PARLAY_INCLUDE_DIR})
endfunction()

add_neighbors(neighbors "")


