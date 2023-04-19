// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
#pragma once

#include <map>
#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>
#include "tsl/robin_set.h"
#include "tsl/robin_map.h"
#include <omp.h>
#include "timer.h"
#include "boost/dynamic_bitset.hpp"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <queue>
#include <random>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <vector>
#include <mutex>

#include "parameters.h"

// from aligned_file_reader.h
#define MAX_IO_DEPTH 128

// from utils.h

// taken from
// https://github.com/Microsoft/BLAS-on-flash/blob/master/include/utils.h
// round up X to the nearest multiple of Y
#define ROUND_UP(X, Y) \
  ((((uint64_t)(X) / (Y)) + ((uint64_t)(X) % (Y) != 0)) * (Y))

#define DIV_ROUND_UP(X, Y) (((uint64_t)(X) / (Y)) + ((uint64_t)(X) % (Y) != 0))

// round down X to the nearest multiple of Y
#define ROUND_DOWN(X, Y) (((uint64_t)(X) / (Y)) * (Y))

// alignment tests
#define IS_ALIGNED(X, Y) ((uint64_t)(X) % (uint64_t)(Y) == 0)
#define IS_512_ALIGNED(X) IS_ALIGNED(X, 512)
#define IS_4096_ALIGNED(X) IS_ALIGNED(X, 4096)

#define SECTOR_LEN 4096
#define MAX_N_SECTOR_READS 128
#define MAX_BLOCK_SIZE (_u64) 1000000
#define MAX_K_MEANS_REPS 10

typedef uint64_t _u64;
typedef int64_t  _s64;
typedef uint32_t _u32;
typedef int32_t  _s32;
typedef uint16_t _u16;
typedef int16_t  _s16;
typedef uint8_t  _u8;
typedef int8_t   _s8;

namespace grann {
  static const _u64 MAX_SIZE_OF_STREAMBUF = 2LL * 1024 * 1024 * 1024;

  enum Metric { L2 = 0, INNER_PRODUCT = 1, FAST_L2 = 2, PQ = 3 };
};  // namespace grann

// USEFUL MACROS FOR VAMANA
#define VAMANA_SLACK_FACTOR 1.3

#define ESTIMATE_VAMANA_RAM_USAGE(size, dim, datasize, degree) \
  (1.30 * (((double) size * dim) * datasize +                  \
           ((double) size * degree) * sizeof(unsigned) * VAMANA_SLACK_FACTOR))
