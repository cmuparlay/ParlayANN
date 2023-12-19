#include "parlay/internal/get_time.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"
#include "parlay/sequence.h"
#include "parlay/slice.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <set>
#include <type_traits>
#include <utility>   //for unique_ptr

#include "../bench/parse_command_line.h"
#include "distance.h"
#include "initialization.h"
#include "kmeans.h"
#include "kmeans_bench.h"
#include "naive.h"
#include "parse_files.h"
#include "yy.h"

// this version uses the "cluster" method from the KmeansInterface
template <typename T>
inline void bench_three(T* v, size_t n, size_t d, size_t k, bool debug_print=false) {
  NaiveKmeans<T, Euclidian_Point<T>, size_t, float, Euclidian_Point<float>>
     runner;
  auto output =
     runner.cluster(PointRange<T, Euclidian_Point<T>>(v, n, d, d), k);
  std::cout << "finished" << std::endl;
  if (debug_print) {
    std::cout << "Printing out partitions: " << std::endl;
    auto parts = output.first;
    for (size_t i = 0; i < parts.size(); i++) {
      std::cout << i << ": ";
      for (size_t j = 0; j < parts[i].size(); j++) {
        std::cout << parts[i][j] << " ";
      }
      std::cout << std::endl;
    }

    std::cout << "Printing out centers: " << std::endl;
    auto centers = output.second;
    for (size_t i = 0; i < centers.size(); i++) {
      for (size_t j = 0; j < centers[i].size(); j++) {
        std::cout << centers[i][j] << " ";
      }
      std::cout << std::endl;
    }
  }

}
 
template <typename T>
inline void bench_two_stable(T* v, size_t n, size_t d, size_t ad, size_t k, Distance& D,
                             size_t max_iter = 1000, double epsilon = 0,
                             bool output_log_to_csv = false,
                             std::string output_file_name1 = "data.csv",
                             std::string output_file_name2 = "data2.csv") {
  std::cout << "fill in bench two stable" << std::endl;



  float* c = new float[k * ad];   // centers
  size_t* asg = new size_t[n];

  // initialization
  Lazy<T, float, size_t> init;
  // note that here, d=ad
  init(v, n, d, ad, k, c, asg);

  Yinyang<T, Euclidian_Point<T>, size_t, float, Euclidian_Point<float>> yy_runner;

  kmeans_bench logger_yy = kmeans_bench(n, d, k, max_iter, epsilon, "Lazy", "YY");
  logger_yy.start_time();
  yy_runner.cluster_middle(v, n, d, ad, k, c, asg, D, logger_yy, max_iter,
                           epsilon);
  logger_yy.end_time();

  delete[] c;
  delete[] asg;
}

// bench two is the basic version I mess with
// currently, we bench naive against yy, yy with point and center grouping, and yy with point but no center grouping
template <typename T>
inline void bench_two(T* v, size_t n, size_t d, size_t ad, size_t k,
                      Distance& D, size_t max_iter = 1000, double epsilon = 0,
                      bool output_log_to_csv = false,
                      std::string output_file_name1 = "data.csv",
                      std::string output_file_name2 = "data2.csv") {

  std::cout << "Running bench two " << std::endl;

  float* c = new float[k * ad];   // centers
  size_t* asg = new size_t[n];

  // initialization
  Lazy<T, float, size_t> init;
  // note that here, d=ad
  init(v, n, d, ad, k, c, asg);
  

  // c2 and asg2 for the yy run
  // make sure to copy over AFTER initialization, but BEFORE kmeans run
  float* c2 = new float[k * ad];
  size_t* asg2 = new size_t[n];
  float* c3 = new float[k*ad];
  size_t* asg3 = new size_t[n];
  float* c4 = new float[k*ad];
  size_t* asg4 = new size_t[n];
  parlay::parallel_for(0, k * ad, [&](size_t i) { c2[i] = c[i]; c3[i]=c[i]; c4[i]=c[i]; });
  parlay::parallel_for(0, n, [&](size_t i) { asg2[i] = asg[i]; asg3[i]=asg[i]; asg4[i]=asg[i]; });
 

  //uncomment to run Naive
  // NaiveKmeans<T, Euclidian_Point<T>, size_t, float, Euclidian_Point<float>> nie2;
  // kmeans_bench logger_nie2 = kmeans_bench(n, d, k, max_iter, epsilon, "Lazy", "Naive");
  // logger_nie2.start_time();
  // // note that d=ad here
  // nie2.cluster_middle(v, n, d, ad, k, c, asg, D, logger_nie2, max_iter,
  //                     epsilon);
  // logger_nie2.end_time();

  Yinyang<T, Euclidian_Point<T>, size_t, float, Euclidian_Point<float>> yy_runner;

  //uncomment to run Yinyang with point and center grouping
  yy_runner.do_center_groups=true;
  yy_runner.do_point_groups=true;

  kmeans_bench logger_yy = kmeans_bench(n, d, k, max_iter, epsilon, "Lazy", "YY");
  logger_yy.start_time();
  yy_runner.cluster_middle(v, n, d, ad, k, c2, asg2, D, logger_yy, max_iter,
                           epsilon);
  logger_yy.end_time();

  yy_runner.do_center_groups=true;
  yy_runner.do_point_groups=false;

  kmeans_bench logger_cgyy = kmeans_bench(n, d, k, max_iter, epsilon, "Lazy", "cgYY");
  logger_cgyy.start_time();
  yy_runner.cluster_middle(v,n,d,ad,k,c3,asg3,D,logger_cgyy,max_iter,epsilon);
  logger_cgyy.end_time();

  //uncomment to run yy with point grouping and no center grouping
  // kmeans_bench logger_pgyy = kmeans_bench(n, d, k, max_iter, epsilon, "Lazy", "pgYY");
  // logger_pgyy.start_time();
  // yy_runner.do_center_groups=false;
  // yy_runner.do_point_groups=true;
  // yy_runner.cluster_middle(v,n,d,ad,k,c4,asg4,D,logger_pgyy,max_iter,epsilon);
  // logger_pgyy.end_time();

  delete[] c;
  delete[] c2;
  delete[] c3;
  delete[] c4;
  delete[] asg;
  delete[] asg2;
  delete[] asg3;
  delete[] asg4;
}

// if new_d is the default value, go with the d given by the dataset. Otherwise,
// use the custom value of d.
size_t pick_num(long orig_d, long new_d) {
  if (new_d == -1) {
    return orig_d;
  }
  return new_d;
}

int main(int argc, char* argv[]) {
  commandLine P(argc, argv,
                "[-k <n_clusters>] [-m <iterations>] [-o <output>] [-i "
                "<input>] [-f <ft>] [-t <tp>] [-D <dist>]");

  long newn = P.getOptionLongValue("-n", -1); //n is # of points. Default value is the # of points in the input file. If we specify n as an argument, however, we use the value of the argument instead.
  size_t k = P.getOptionLongValue("-k", 10);   // k is number of clusters
  long newd = P.getOptionLongValue("-d", -1); //we can set a custom value of d that is no more than the dimension of the points in the dataset.
  size_t max_iterations = P.getOptionLongValue(
     "-m",
     1000);   // max_iterations is the max # of Lloyd iters kmeans will run
 
  std::string input =
     std::string(P.getOptionValue("-i", ""));   // the data input file
  std::string ft =
     std::string(P.getOptionValue("-f", "bin"));   // file type, bin or vecs
  std::string tp = std::string(P.getOptionValue("-t", "uint8"));   // data type
  std::string dist =
     std::string(P.getOptionValue("-D", "Euclidian"));   // distance choice
  std::string bench_version =
     std::string(P.getOptionValue("-bench_version", "no")); //controls which bench function is run
  bool output_log_to_csv = false; //variable for whether we output the logging info to a csv file
  std::string output_to_csv_str =
     std::string(P.getOptionValue("-csv_log", "false"));
  if (output_to_csv_str == "true") {
    output_log_to_csv = true;
  }
  std::string output_log_file_name =
     std::string(P.getOptionValue("-csv_log_file_name", "data.csv")); //csv logging output file name
  std::string output_log_file_name2 =
     std::string(P.getOptionValue("-csv_log_file_name2", "data2.csv")); //2nd csv logging output file name (if benching two different algorithms)
 
  float epsilon = static_cast<float>(P.getOptionDoubleValue("-epsilon", 0.0)); //threshold for stopping k-means run early

  std::cout << "using " << parlay::num_workers << " workers." << std::endl; //print out # of parlay workers being used

  if (input == "") {   // if no input file given, quit
    std::cout << "Error: input file not specified" << std::endl;
    abort();
  }

  if ((ft != "bin") &&
      (ft != "vec")) {   // if the file type chosen is not one of the two
                         // approved file types, quit
    std::cout << "Error: file type not specified correctly, specify bin or vec"
              << std::endl;
    abort();
  }

  if ((tp != "uint8") && (tp != "int8") &&
      (tp != "float")) {   // if the data type isn't one of the three approved
                           // data types, quit
    std::cout << "Error: vector type not specified correctly, specify int8, "
                 "uint8, or float"
              << std::endl;
    abort();
  }

  if ((ft == "vec") &&
      (tp ==
       "int8")) {   // you can't store int8s in a vec file apparently I guess
    std::cout << "Error: incompatible file and vector types" << std::endl;
    abort();
  }

  // TODO: add support for vec files
  if (ft == "vec") {
    std::cout << "Error: vec file type not supported yet" << std::endl;
    abort();
  }

  Distance* D;

  // create a distance object, it can either by Euclidian or MIPS
  if (dist == "mips") {
    std::cout << "Using MIPS distance" << std::endl;
    D = new Mips_Distance();
  } else if (dist == "short") {
    std::cout << "Using short Euclidean" << std::endl;
    D = new EuclideanDistanceSmall();
  } else if (dist == "fast") {
    std::cout << "Using fast Euclidean" << std::endl;
    D = new EuclideanDistanceFast();
  } else {
    std::cout << "Error: distance type not specified correctly, specify "
                 "Euclidean or mips"
              << std::endl;
    abort();
  }

  if (ft == "bin") {
    if (tp == "float") {
      auto [v, n, d] = parse_fbin(input.c_str());
      if (bench_version == "two") {   
        bench_two<float>(v, pick_num(n, newn), pick_num(d, newd), d, k, *D,
                         max_iterations, epsilon, output_log_to_csv,
                         output_log_file_name, output_log_file_name2);
      } else if (bench_version == "stable") {
        bench_two_stable<float>(v, pick_num(n, newn), pick_num(d, newd), d, k, *D,
                                max_iterations, epsilon, output_log_to_csv,
                                output_log_file_name, output_log_file_name2);
      } else if (bench_version == "three") {
        bench_three<float>(v, pick_num(n, newn), pick_num(d, newd), k);
      }

      else {
        std::cout << "Must specify bench path, aborting" << std::endl;
        abort();
      }

    } else if (tp == "uint8") {
      auto [v, n, d] = parse_uint8bin(input.c_str());
      if (bench_version == "two") {
        bench_two<uint8_t>(v, pick_num(n, newn), pick_num(d, newd), d, k, *D,
                           max_iterations, epsilon, output_log_to_csv,
                           output_log_file_name, output_log_file_name2);
      } else if (bench_version == "stable") {
        bench_two_stable<uint8_t>(v, pick_num(n, newn), pick_num(d, newd), d, k,
                                  *D, max_iterations, epsilon,
                                  output_log_to_csv, output_log_file_name,
                                  output_log_file_name2);

      } else if (bench_version == "three") {
        bench_three<uint8_t>(v, pick_num(n, newn), pick_num(d, newd), k);

      } else {
        std::cout << "Must specify bench path, aborting" << std::endl;
        abort();
      }
    } else if (tp == "int8") {
      auto [v, n, d] = parse_int8bin(input.c_str());
      if (bench_version == "two") {
        bench_two<int8_t>(v, pick_num(n, newn), pick_num(d, newd), d, k, *D,
                          max_iterations, epsilon, output_log_to_csv,
                          output_log_file_name, output_log_file_name2);

      } else if (bench_version == "stable") {
        bench_two_stable<int8_t>(v, pick_num(n, newn), pick_num(d, newd), d, k, *D,
                                 max_iterations, epsilon, output_log_to_csv,
                                 output_log_file_name, output_log_file_name2);

      } else if (bench_version == "three") {
        bench_three<int8_t>(v, pick_num(n, newn), pick_num(d, newd), k);

      } else {
        std::cout << "Must specify bench path, aborting" << std::endl;
        abort();
      }
    } else {
      //  this should actually be unreachable
      std::cout << "Error: bin type can only be float, uint8, or int8. "
                   "Supplied type is "
                << tp << "." << std::endl;
      abort();
    }
  }

  delete D;

  std::cout << "program finished" << std::endl;

  return 0;
}