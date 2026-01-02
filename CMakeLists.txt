cmake_minimum_required(VERSION 3.15)
project(PARLAYANN VERSION 1
        DESCRIPTION "ParlayANN is a library of approximate nearest neighbor search algorithms, along with a set of useful tools for designing such algorithms. It is written in C++ and uses parallel primitives from ParlayLib. Currently it includes implementations of the ANNS algorithms DiskANN, HNSW, HCNNG, and pyNNDescent."
        LANGUAGES CXX)

include(CheckCXXCompilerFlag)
include(GNUInstallDirs)

# Set a default build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RELEASE" CACHE STRING "Build type (Release)" FORCE)
  message(STATUS "No build type specified. Defaulted to RELEASE.")
  message(STATUS "To specify a build type, add -DCMAKE_BUILD_TYPE=<DEBUG/RELEASE/RELWITHDEBINFO/MINSIZEREL>")
endif(NOT CMAKE_BUILD_TYPE)

set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -fno-omit-frame-pointer")
set (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_DEBUG} -O3 -mcx16 -march=native -g")

message(STATUS "PARLAYANN VERSION ${PARLAYANN_VERSION}")
message(STATUS "---------------------------- General configuration -----------------------------")
message(STATUS "CMake Generator:                ${CMAKE_GENERATOR}")
message(STATUS "Compiler:                       ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Build type:                     ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_CXX_FLAGS:                ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG:          ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "CMAKE_CXX_FLAGS_RELEASE:        ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "CMAKE_CXX_FLAGS_RELWITHDEBINFO: ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
message(STATUS "CMAKE_EXE_LINKER_FLAGS          ${CMAKE_EXE_LINKER_FLAGS}")
message(STATUS "CMAKE_INSTALL_PREFIX:           ${CMAKE_INSTALL_PREFIX}" )

# Set module path
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(parlay INTERFACE)
target_include_directories(parlay INTERFACE "${PROJECT_SOURCE_DIR}/parlaylib/include")

# Link against system threads
find_package(Threads REQUIRED)
target_link_libraries(parlay INTERFACE Threads::Threads)

add_subdirectory(algorithms)



