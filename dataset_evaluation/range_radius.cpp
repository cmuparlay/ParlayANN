#include <iostream>
#include <algorithm>
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/io.h"
#include "utils/euclidian_point.h"
#include "utils/mips_point.h"
#include "utils/point_range.h"

void range_res_stats(parlay::sequence<parlay::sequence<int>> &result, size_t dataset_size){

    size_t n = result.size();
    parlay::sequence<int> sizes = parlay::tabulate(n, [&] (size_t i){
        return static_cast<int>(result[i].size());
    });

    parlay::sequence<double> pct_contained =parlay::tabulate(n, [&] (size_t i){
        return static_cast<double>(result[i].size())/static_cast<double>(dataset_size);
    });

    double percent_covered = (parlay::reduce(pct_contained)/static_cast<double>(result.size()))*100;
    std::cout << "Percent covered: " << percent_covered << std::endl;

    size_t num_matches = parlay::reduce(sizes);

    std::cout << "Number of nonzero matches: " << num_matches << std::endl;
    parlay::sort_inplace(sizes);
    std::cout << "Largest num matches: " << sizes[sizes.size()-1] << std::endl;
    size_t zero = 0;
    size_t small = 0;
    size_t big = 0;
    for(auto i : sizes){
        if(i == 0) zero ++;
        else if(i >0 && i<=20) small++;
        else big++;
    }
    std::cout << "Number of points with zero results: " << zero << std::endl;
    std::cout << "Number of points with one to twenty results: " << small << std::endl;
    std::cout << "Number of points with more than twenty results: " << big << std::endl;


  
}

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




int main(int argc, char* argv[]) {
  commandLine P(argc,argv,
  "[-base_path <b>] [-query_path <q>] "
      "[-data_type <d>] [-r <r> ] [-dist_func <d>] [-gt_path <outfile>]");

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
      range_res_stats(answers, B.size());
    } else if(df == "mips"){
      PointRange<float, Mips_Point<float>> B = PointRange<float, Mips_Point<float>>(bFile);
      PointRange<float, Mips_Point<float>> Q = PointRange<float, Mips_Point<float>>(qFile);
      answers = compute_range_groundtruth<PointRange<float, Mips_Point<float>>>(B, Q, r);
      range_res_stats(answers, B.size());
    }
  }else if(tp == "uint8"){
    std::cout << "Detected uint8 coordinates" << std::endl;
    if(df == "Euclidian"){
      PointRange<uint8_t, Euclidian_Point<uint8_t>> B = PointRange<uint8_t, Euclidian_Point<uint8_t>>(bFile);
      PointRange<uint8_t, Euclidian_Point<uint8_t>> Q = PointRange<uint8_t, Euclidian_Point<uint8_t>>(qFile);
      answers = compute_range_groundtruth<PointRange<uint8_t, Euclidian_Point<uint8_t>>>(B, Q, r);
      range_res_stats(answers, B.size());
    } else if(df == "mips"){
      PointRange<uint8_t, Mips_Point<uint8_t>> B = PointRange<uint8_t, Mips_Point<uint8_t>>(bFile);
      PointRange<uint8_t, Mips_Point<uint8_t>> Q = PointRange<uint8_t, Mips_Point<uint8_t>>(qFile);
      answers = compute_range_groundtruth<PointRange<uint8_t, Mips_Point<uint8_t>>>(B, Q, r);
      range_res_stats(answers, B.size());
    }
  }else if(tp == "int8"){
    std::cout << "Detected int8 coordinates" << std::endl;
    if(df == "Euclidian"){
      PointRange<int8_t, Euclidian_Point<int8_t>> B = PointRange<int8_t, Euclidian_Point<int8_t>>(bFile);
      PointRange<int8_t, Euclidian_Point<int8_t>> Q = PointRange<int8_t, Euclidian_Point<int8_t>>(qFile);
      answers = compute_range_groundtruth<PointRange<int8_t, Euclidian_Point<int8_t>>>(B, Q, r);
      range_res_stats(answers, B.size());
    } else if(df == "mips"){
      PointRange<int8_t, Mips_Point<int8_t>> B = PointRange<int8_t, Mips_Point<int8_t>>(bFile);
      PointRange<int8_t, Mips_Point<int8_t>> Q = PointRange<int8_t, Mips_Point<int8_t>>(qFile);
      answers = compute_range_groundtruth<PointRange<int8_t, Mips_Point<int8_t>>>(B, Q, r);
      range_res_stats(answers, B.size());
    }
  }
  
  

  return 0;
}

