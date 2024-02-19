#include <iostream>
#include <algorithm>
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/io.h"
#include "utils/euclidian_point.h"
#include "utils/mips_point.h"
#include "utils/point_range.h"
#include "utils/types.h"



template<typename PointRange, typename T>
void write_nonzero_elts(RangeGroundTruth<unsigned int> &result, PointRange Query_Points, const std::string outFile){
    size_t n = result.size();
    parlay::sequence<int> sizes = parlay::tabulate(n, [&] (size_t i){
        if(result[i].size() > 0) return 1;
        return 0;
    });
    size_t num_nonzero = parlay::reduce(sizes);
    std::cout << "Number of nonzero elements: " << num_nonzero << std::endl;

    int d = Query_Points.dimension();
    parlay::sequence<int> preamble = {static_cast<int>(num_nonzero), static_cast<int>(d)};
    parlay::sequence<T> data(num_nonzero*d);
    parlay::sequence<parlay::sequence<unsigned int>> to_flatten = parlay::tabulate(n, [&] (size_t i){
      parlay::sequence<unsigned int> ret;
      if(result[i].size() > 0) ret.push_back(i);
      return ret;
    });
    parlay::sequence<unsigned int> indices = parlay::flatten(to_flatten);
    if(indices.size() != num_nonzero) abort();
    parlay::parallel_for(0, indices.size(), [&] (size_t i){
      for(int j=0; j<d; j++) data[d*i+j] = Query_Points[indices[i]][j];
    });

    std::ofstream writer;
    writer.open(outFile, std::ios::binary | std::ios::out);
    writer.write((char *) (preamble.begin()), 2*sizeof(int));
    writer.write((char *) (data.begin()), num_nonzero * sizeof(T) * d);
    writer.close();


}




int main(int argc, char* argv[]) {
  commandLine P(argc,argv,
  "[-base_path <b>] [-query_path <q>] "
      "[-data_type <d>] [-r <r> ] [-dist_func <d>] [-gt_path <outfile>]");

  char* gFile = P.getOptionValue("-vec_path");
  char* gtFile = P.getOptionValue("-gt_path");
  char* qFile = P.getOptionValue("-query_path");
  char* vectype = P.getOptionValue("-data_type");
  char* dfc = P.getOptionValue("-dist_func");

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

  RangeGroundTruth<unsigned int> RGT(gtFile);


  if(tp == "float"){
    std::cout << "Detected float coordinates" << std::endl;
    if(df == "Euclidian"){
      PointRange<float, Euclidian_Point<float>> Q = PointRange<float, Euclidian_Point<float>>(qFile);
      write_nonzero_elts<PointRange<float, Euclidian_Point<float>>, float>(RGT, Q, std::string(gFile));
    } else if(df == "mips"){
      PointRange<float, Mips_Point<float>> Q = PointRange<float, Mips_Point<float>>(qFile);
      write_nonzero_elts<PointRange<float, Mips_Point<float>>, float>(RGT, Q, std::string(gFile));
    }
  }else if(tp == "uint8"){
    std::cout << "Detected uint8 coordinates" << std::endl;
    if(df == "Euclidian"){
      PointRange<uint8_t, Euclidian_Point<uint8_t>> Q = PointRange<uint8_t, Euclidian_Point<uint8_t>>(qFile);
      write_nonzero_elts<PointRange<uint8_t, Euclidian_Point<uint8_t>>, uint8_t>(RGT, Q, std::string(gFile));
    } else if(df == "mips"){
      PointRange<uint8_t, Mips_Point<uint8_t>> Q = PointRange<uint8_t, Mips_Point<uint8_t>>(qFile);
      write_nonzero_elts<PointRange<uint8_t, Mips_Point<uint8_t>>, uint8_t>(RGT, Q, std::string(gFile));
    }
  }else if(tp == "int8"){
    std::cout << "Detected int8 coordinates" << std::endl;
    if(df == "Euclidian"){
      PointRange<int8_t, Euclidian_Point<int8_t>> Q = PointRange<int8_t, Euclidian_Point<int8_t>>(qFile);
      write_nonzero_elts<PointRange<int8_t, Euclidian_Point<int8_t>>, uint8_t>(RGT, Q, std::string(gFile));
    } else if(df == "mips"){
      PointRange<int8_t, Mips_Point<int8_t>> Q = PointRange<int8_t, Mips_Point<int8_t>>(qFile);
      write_nonzero_elts<PointRange<int8_t, Mips_Point<int8_t>>, uint8_t>(RGT, Q, std::string(gFile));
    }
  }
  

  return 0;
}

