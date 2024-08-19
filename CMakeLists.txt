cmake_minimum_required(VERSION 3.15)
project(ParlayANN VERSION 1
        DESCRIPTION "ParlayANN is a library of approximate nearest neighbor search algorithms, along with a set of useful tools for designing such algorithms. It is written in C++ and uses parallel primitives from ParlayLib. Currently it includes implementations of the ANNS algorithms DiskANN, HNSW, HCNNG, and pyNNDescent."
        LANGUAGES CXX)

include(CheckCXXCompilerFlag)
include (FetchContent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

FetchContent_Declare(parlaylib
  GIT_REPOSITORY  https://github.com/cmuparlay/parlaylib.git
  GIT_TAG         master
)
FetchContent_GetProperties(parlaylib)
if(NOT parlaylib_POPULATED)
  FetchContent_Populate(parlaylib)  
  add_subdirectory(${parlaylib_SOURCE_DIR} EXCLUDE_FROM_ALL)
endif()

# Set module path
list(APPEND CMAKE_MODULE_PATH "${ParlayANN_SOURCE_DIR}/cmake")

add_library(ParlayANN INTERFACE)
set(ParlayANN_INCLUDE_DIR "${ParlayANN_SOURCE_DIR}/algorithms")
target_include_directories(ParlayANN INTERFACE
  $<BUILD_INTERFACE:${ParlayANN_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

add_subdirectory(algorithms)
#add_subdirectory(data_tools)
#add_subdirectory(range_search)


