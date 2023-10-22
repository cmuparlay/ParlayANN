#include "IVF.h"
#include "clustering.h"
#include "posting_list.h"

#include "../utils/euclidian_point.h"
#include "../utils/filters.h"
#include "../utils/mips_point.h"
#include "../utils/point_range.h"
#include "../utils/types.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <utility>

#define DATA_DIR "/ssd1/data/bigann/data/yfcc100M/"

const std::string points =
   std::string(DATA_DIR) + std::string("base.10M.u8bin.crop_nb_10000000");
const std::string filters =
   std::string(DATA_DIR) + std::string("base.metadata.10M.spmat");

int main() {
  auto fivf =
     FilteredIVFIndex<uint8_t, Mips_Point<uint8_t>,
                      FilteredPostingList<uint8_t, Mips_Point<uint8_t>>>();
  fivf.fit_from_filename(points, filters, 1000);

  return 0;
}