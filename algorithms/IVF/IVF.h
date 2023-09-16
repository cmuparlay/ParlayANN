/* An IVF index */

#include "parlay/sequence.h"
#include "parlay/primitives.h"
#include "parlay/parallel.h"

#include "point_range.h"
#include "posting_list.h"
#include "types.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <utility>

/* A reasonably extensible ivf index */
template<typename T, class Point>