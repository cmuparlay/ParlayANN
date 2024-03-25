#include <iostream>
#include <algorithm>
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/io.h"
// #include "utils/types.h"
#include "utils/euclidian_point.h"
#include "utils/mips_point.h"
#include "utils/point_range.h"



template<typename PointRange>
parlay::sequence<parlay::sequence<int>> compute_range_groundtruth(PointRange &B, 
  PointRange &Q, float r){
    unsigned d = B.dimension();
    size_t q = Q.size();
    size_t b = B.size();
    auto answers = parlay::tabulate(q, [&] (size_t i){  
        parlay::sequence<int> results;
        for(size_t j=0; j<b; j++){
            float dist = Q[i].distance(B[j]);
            if(dist <= r) results.push_back(j);
        }
        return results;
    });
    std::cout << "Done computing groundtruth" << std::endl;
    return answers;
}

template<typename PointRange, typename T>
void write_nonzero_elts(parlay::sequence<parlay::sequence<int>> &result, PointRange Query_Points, const std::string outFile){
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
    parlay::sequence<parlay::sequence<int>> to_flatten = parlay::tabulate(n, [&] (size_t i){
      parlay::sequence<int> ret;
      if(result[i].size() > 0) ret.push_back(i);
      return ret;
    });
    parlay::sequence<int> indices = parlay::flatten(to_flatten);
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

void write_rangeres(parlay::sequence<parlay::sequence<int>> &result, const std::string outFile){
    std::cout << "File contains range groundtruth for " << result.size() << " data points" << std::endl;

    
    size_t n = result.size();
    parlay::sequence<int> sizes = parlay::tabulate(n, [&] (size_t i){
        return static_cast<int>(result[i].size());
    });
    size_t num_matches = parlay::reduce(sizes);

    std::cout << "Number of nonzero matches: " << num_matches << std::endl;
    parlay::sequence<int> preamble = {static_cast<int>(n), static_cast<int>(num_matches)};

    auto flat_ids = parlay::flatten(result);

    auto pr = preamble.begin();
    auto size_data = sizes.begin();
    auto id_data = flat_ids.begin();
    std::ofstream writer;
    writer.open(outFile, std::ios::binary | std::ios::out);
    writer.write((char *) pr, 2*sizeof(int));
    writer.write((char *) size_data, n * sizeof(int));
    writer.write((char *) id_data, num_matches * sizeof(int));
    writer.close();
}


int main(int argc, char* argv[]) {
  commandLine P(argc,argv,
  "[-base_path <b>] [-query_path <q>] "
      "[-data_type <d>] [-r <r> ] [-dist_func <d>] [-gt_path <outfile>]");

  char* gFile = P.getOptionValue("-gt_path");
  char* qFile = P.getOptionValue("-query_path");
  char* bFile = P.getOptionValue("-base_path");
  char* vectype = P.getOptionValue("-data_type");
  char* dfc = P.getOptionValue("-dist_func");
  float r = P.getOptionDoubleValue("-r", 0);

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

  std::cout << "Computing the groundtruth for radius " << r << std::endl;

  parlay::sequence<parlay::sequence<int>> answers;

  if(tp == "float"){
    std::cout << "Detected float coordinates" << std::endl;
    if(df == "Euclidian"){
      PointRange<float, Euclidian_Point<float>> B = PointRange<float, Euclidian_Point<float>>(bFile);
      PointRange<float, Euclidian_Point<float>> Q = PointRange<float, Euclidian_Point<float>>(qFile);
      answers = compute_range_groundtruth<PointRange<float, Euclidian_Point<float>>>(B, Q, r);
    } else if(df == "mips"){
      PointRange<float, Mips_Point<float>> B = PointRange<float, Mips_Point<float>>(bFile);
      PointRange<float, Mips_Point<float>> Q = PointRange<float, Mips_Point<float>>(qFile);
      answers = compute_range_groundtruth<PointRange<float, Mips_Point<float>>>(B, Q, r);
    }
  }else if(tp == "uint8"){
    std::cout << "Detected uint8 coordinates" << std::endl;
    if(df == "Euclidian"){
      PointRange<uint8_t, Euclidian_Point<uint8_t>> B = PointRange<uint8_t, Euclidian_Point<uint8_t>>(bFile);
      PointRange<uint8_t, Euclidian_Point<uint8_t>> Q = PointRange<uint8_t, Euclidian_Point<uint8_t>>(qFile);
      answers = compute_range_groundtruth<PointRange<uint8_t, Euclidian_Point<uint8_t>>>(B, Q, r);
    } else if(df == "mips"){
      PointRange<uint8_t, Mips_Point<uint8_t>> B = PointRange<uint8_t, Mips_Point<uint8_t>>(bFile);
      PointRange<uint8_t, Mips_Point<uint8_t>> Q = PointRange<uint8_t, Mips_Point<uint8_t>>(qFile);
      answers = compute_range_groundtruth<PointRange<uint8_t, Mips_Point<uint8_t>>>(B, Q, r);
    }
  }else if(tp == "int8"){
    std::cout << "Detected int8 coordinates" << std::endl;
    if(df == "Euclidian"){
      PointRange<int8_t, Euclidian_Point<int8_t>> B = PointRange<int8_t, Euclidian_Point<int8_t>>(bFile);
      PointRange<int8_t, Euclidian_Point<int8_t>> Q = PointRange<int8_t, Euclidian_Point<int8_t>>(qFile);
      answers = compute_range_groundtruth<PointRange<int8_t, Euclidian_Point<int8_t>>>(B, Q, r);
    } else if(df == "mips"){
      PointRange<int8_t, Mips_Point<int8_t>> B = PointRange<int8_t, Mips_Point<int8_t>>(bFile);
      PointRange<int8_t, Mips_Point<int8_t>> Q = PointRange<int8_t, Mips_Point<int8_t>>(qFile);
      answers = compute_range_groundtruth<PointRange<int8_t, Mips_Point<int8_t>>>(B, Q, r);
    }
  }
  write_rangeres(answers, std::string(gFile));
  

  return 0;
}

