#pragma once

#include <limits.h>

namespace aspen {
#define LONG 1

// size of edge-offsets.
// If the number of edges is more than sizeof(MAX_UINT),
// you should set the LONG flag on the command line.
#if defined(LONG)
typedef long intT;
typedef unsigned long uintT;
#define INT_T_MAX LONG_MAX
#define UINT_T_MAX ULONG_MAX
#else
typedef int intT;
typedef unsigned int uintT;
#define INT_T_MAX INT_MAX
#define UINT_T_MAX UINT_MAX
#endif

// edge size macros.
// If the number of vertices is more than sizeof(MAX_UINT)
// you should set the EDGELONG flag on the command line.
#if defined(EDGELONG)
typedef long intE;
typedef unsigned long uintE;
#define INT_E_MAX LONG_MAX
#define UINT_E_MAX ULONG_MAX
#else
typedef int intE;
typedef unsigned int uintE;
#define INT_E_MAX INT_MAX
#define UINT_E_MAX UINT_MAX
#endif

}  // namespace aspen
