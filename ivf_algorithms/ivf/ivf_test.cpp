#include "ivf.h"
#include "clustering.h"
#include "posting_list.h"

#include "utils/euclidian_point.h"
// #include "utils/filters.h"
#include "utils/mips_point.h"
#include "utils/point_range.h"
#include "utils/types.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <utility>



int main(int argc, char* argv[]) {
  commandLine P(argc,argv,
  "[-base_path <b>] [-query_path <q>] "
      "[-data_type <d>] [-k <k> ] [-dist_func <d>] [-gt_path <outfile>]");

  char* gFile = P.getOptionValue("-gt_path");
  char* qFile = P.getOptionValue("-query_path");
  char* bFile = P.getOptionValue("-base_path");
  char* vectype = P.getOptionValue("-data_type");
  char* dfc = P.getOptionValue("-dist_func");
  int k = P.getOptionIntValue("-k", 100);

  std::string df = std::string(dfc);
  if(df != "Euclidian" && df != "mips"){
    std::cout << "Error: invalid distance type: specify Euclidian or mips" << std::endl;
    abort();
  }

  std::string tp = std::string(vectype);
  if((tp != "uint8") && (tp != "int8") && (tp != "float")){
    std::cout << "Error: data type not specified correctly, specify int8, uint8, or float" << std::endl;
    abort();
  }

  if(tp == "float"){
    std::cout << "Detected float coordinates" << std::endl;
    if(df == "Euclidian"){
      PointRange<float, Euclidian_Point<float>> B = PointRange<float, Euclidian_Point<float>>(bFile);
      PointRange<float, Euclidian_Point<float>> Q = PointRange<float, Euclidian_Point<float>>(qFile);
      KMeansClusterer<float, Euclidian_Point<float>, unsigned int> K(1000);
      PostingListIndex<float, Euclidian_Point<float>, unsigned int> PostingList = 
        PostingListIndex<float, Euclidian_Point<float>, unsigned int>(B, K, 0);
    } else if(df == "mips"){
      PointRange<float, Mips_Point<float>> B = PointRange<float, Mips_Point<float>>(bFile);
      PointRange<float, Mips_Point<float>> Q = PointRange<float, Mips_Point<float>>(qFile);

    }
  }else if(tp == "uint8"){
    std::cout << "Detected uint8 coordinates" << std::endl;
    if(df == "Euclidian"){
      PointRange<uint8_t, Euclidian_Point<uint8_t>> B = PointRange<uint8_t, Euclidian_Point<uint8_t>>(bFile);
      PointRange<uint8_t, Euclidian_Point<uint8_t>> Q = PointRange<uint8_t, Euclidian_Point<uint8_t>>(qFile);

    } else if(df == "mips"){
      PointRange<uint8_t, Mips_Point<uint8_t>> B = PointRange<uint8_t, Mips_Point<uint8_t>>(bFile);
      PointRange<uint8_t, Mips_Point<uint8_t>> Q = PointRange<uint8_t, Mips_Point<uint8_t>>(qFile);

    }
  }else if(tp == "int8"){
    std::cout << "Detected int8 coordinates" << std::endl;
    if(df == "Euclidian"){
      PointRange<int8_t, Euclidian_Point<int8_t>> B = PointRange<int8_t, Euclidian_Point<int8_t>>(bFile);
      PointRange<int8_t, Euclidian_Point<int8_t>> Q = PointRange<int8_t, Euclidian_Point<int8_t>>(qFile);

    } else if(df == "mips"){
      PointRange<int8_t, Mips_Point<int8_t>> B = PointRange<int8_t, Mips_Point<int8_t>>(bFile);
      PointRange<int8_t, Mips_Point<int8_t>> Q = PointRange<int8_t, Mips_Point<int8_t>>(qFile);

    }
  }

  return 0;
}