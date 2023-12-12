// file for benching a kmeans method with many varied parameters
// kmeans.cpp is for running a single instance of kmeans, or perhaps two methods
// on the same initialization **on the same data** whereas bench.cpp is meant
// for running cross-data

// TODO purge the include list to include only what's actually needed
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
#include <utility>

#include "../bench/parse_command_line.h"
#include "distance.h"
#include "initialization.h"
#include "kmeans.h"
#include "kmeans_bench.h"
#include "naive.h"
#include "parse_files.h"
#include "yy.h"

// get runtime from given kmeans run
template <typename T, typename Runner>
kmeans_bench get_run_time(T* v, size_t n, size_t d, size_t ad, size_t k,
                          Distance& D, Runner& r, size_t max_iter = 1000,
                          double epsilon = 0) {

  float* c = new float[k * ad];   // centers
  size_t* asg = new size_t[n];

  // initialization
  Lazy<T, float, size_t> init;
  // note that here, d=ad
  init(v, n, d, ad, k, c, asg);

  kmeans_bench logger =
     kmeans_bench(n, d, k, max_iter, epsilon, "Lazy", r.name());
  logger.start_time();
  // note that d=ad here
  r.cluster_middle(v, n, d, ad, k, c, asg, D, logger, max_iter, epsilon, false);
  logger.end_time();

  std::cout << "printing dist calcs: " <<std::endl;
  for (int i = 0; i < logger.iterations.size(); i++) {
    std::cout << i << ": " << logger.iterations[i].distance_calculations << std::endl;
  }
  return logger;
}


// //alternate function, for benching against the IVF k-means implementation
// template <typename T, typename OuterRunner>
// kmeans_bench get_run_time_IVF(T* v, size_t n, size_t d, size_t ad, size_t k, Distance& D, Runner& r,
// size_t max_iter=1000, double epsilon=0) { 
  
//     float* c = new float[k*ad]; // centers
//     size_t* asg = new size_t[n];

  
//     //initialization
//     Lazy<T,float,size_t> init;
//     //note that here, d=ad
//     init(v,n,d,ad,k,c,asg);

   

//     kmeans_bench logger = kmeans_bench(n,d,k,max_iter,
//     epsilon,"Lazy",r.name());
//     logger.start_time();
//     //note that d=ad here
//     r.cluster_middle(v,n,d,ad,k,c,asg,D,logger,max_iter,epsilon,true);
//     logger.end_time();
//     return logger;
    
// }

// bench many
// n samples <- values of n to try
// k samples <- values of k to try
// d samples < values of d to try
// output_file <- where to put results
// iter_samples <- values of max_iter to tryu
// var_samples <- how many times we do the run (if > 1, then average)
// limiter <- cap the value of n*d*k*max_iter*var_samples to prevent a single
// run from taking too long
// TODO: this version of the code will only take in a single data file -- and we
// would just run this code over multiple data files in separate runs, for
// simplicity
template <typename T>
inline void bench_many(T* v, size_t ad, std::string n_samples,
                       std::string k_samples, std::string d_samples,
                       std::string runner_name, std::string output_file,
                       std::string iter_samples, std::string var_samples,
                       size_t limiter, Distance& D) {
  std::cout << "Run bench many " << std::endl;
  std::cout << "N sample: " << n_samples << std::endl;
  std::vector<size_t> n_vec = extract_vector<size_t>(n_samples.c_str());
  std::cout << "made it here" << std::endl;
  std::vector<size_t> d_vec = extract_vector<size_t>(d_samples.c_str());
  std::cout << "made it here2" << std::endl;


  std::vector<size_t> k_vec = extract_vector<size_t>(k_samples.c_str());
  std::cout << "made it here3" << std::endl;
  std::cout << "Iter file: " << iter_samples << std::endl;

  std::vector<size_t> iter_vec = extract_vector<size_t>(iter_samples.c_str());
  std::cout << "made it here4" << std::endl;

  std::vector<size_t> var_vec = extract_vector<size_t>(var_samples.c_str());
  std::cout << "made it here5" << std::endl;

  KmeansInterface<T, Euclidian_Point<T>, size_t, float, Euclidian_Point<float>>*
     runner;
  if (runner_name == "naive") {
    runner = new NaiveKmeans<T, Euclidian_Point<T>, size_t, float,
                             Euclidian_Point<float>>();
  } else if (runner_name == "yy") {
    runner = new Yinyang<T, Euclidian_Point<T>, size_t, float,
                         Euclidian_Point<float>>();
  } else {
    std::cout << "please provide a kmeans method to run, aborting" << std::endl;
    abort();
  }

  // std::vector<std::pair,std::string,std::string>> data_samples =
  // extract_string_pair_vector(data_samples_name.c_str());
  //  std::vector<DataWrapper> ex_data;
  //  for (int i = 0; i < data_samples.size(); i++) {
  //      ex_data.push_back(DataWrapper(data_samples[i].first,data_samples[i].second));
  //  }

  std::ofstream file(output_file);
  file << "n"
       << ", "
       << "d"
       << ", "
       << "k"
       << ", "
       << "n_iter"
       << ", "
       << "reps(vars)"
       << ", "
       << "time"
       << ", "
       << "dist_calcs"
       << "\n";

  std::vector<size_t> capacities = {n_vec.size(), k_vec.size(), d_vec.size(),
                                    iter_vec.size(), var_vec.size()};
  std::vector<size_t> cur_parms = {
     0, 0, 0, 0, 0};   // starting at -1 for ease of while loop

  do {
    for (size_t i = 0; i < cur_parms.size(); i++) {
      std::cout << cur_parms[i] << " ";
    }
    // TODO add var_vec to get runtime function
    // note that we need ad to access the right points
    kmeans_bench logger =
       get_run_time(v, n_vec[cur_parms[0]], d_vec[cur_parms[2]], ad,
                    k_vec[cur_parms[1]], D, *runner, iter_vec[cur_parms[3]], 0);
    double result_time = logger.total_time;
    size_t dist_calcs = parlay::reduce(
       parlay::map(logger.iterations, [&](iteration_bench& iter_data) {
         return iter_data.distance_calculations;
       }));

    file << n_vec[cur_parms[0]] << ", " << d_vec[cur_parms[2]] << ", "
         << k_vec[cur_parms[1]] << ", " << iter_vec[cur_parms[3]] << ", " << 1
         << ", " << result_time << ", " << dist_calcs << "\n";


    } while (iterate_multidim(capacities,cur_parms));

    file.close();

    delete runner;


}

int main(int argc, char* argv[]) {
  commandLine P(argc, argv,
                "[-ns <n_samples_file>] [-ks <k_file>] [-i <input data file>] "
                "[-is <iter_file>] [-vs <var_file>] [-D <dist>] [-o "
                "<output_file>] [-lim <limiter>] [-tp <type>]");

  std::string output = std::string(P.getOptionValue(
     "-o", "kmeans_bench_results.txt"));   // maybe the kmeans results get
                                           // written into this csv

  std::string dist =
     std::string(P.getOptionValue("-D", "fast"));   // distance choice

  std::string n_samples_name = std::string(P.getOptionValue("-ns", "n.txt"));
  std::string k_samples_name = std::string(P.getOptionValue("-ks", "k.txt"));
  std::string d_samples_name = std::string(P.getOptionValue("-ds", "d.txt"));
  std::string iter_samples_name =
     std::string(P.getOptionValue("-is", "iter.txt"));
  std::string var_samples_name =
     std::string(P.getOptionValue("-vs", "var.txt"));
  std::string data_samples_name =
     std::string(P.getOptionValue("-ds", "data.txt"));
  long limiter = P.getOptionLongValue("-lim", 1'000'000'000);
  std::string input =
     std::string(P.getOptionValue("-i", ""));   // the data input file
  std::string tp = std::string(P.getOptionValue("-t", "uint8"));   // data type
  std::string runner_name = std::string(P.getOptionValue("-rn", "none"));

  Distance* D;   // create a distance object, it can either by Euclidian or MIPS
  // if (dist == "Euclidean") {
  //     std::cout << "Using Euclidean distance" << std::endl;
  //     D = new EuclideanDistance();
  // } else
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

  if (tp == "float") {
    auto [tv, tn, td] = parse_fbin(input.c_str());
    bench_many(tv, td, n_samples_name, k_samples_name, d_samples_name,
               runner_name, output, iter_samples_name, var_samples_name,
               limiter, *D);

  } else if (tp == "uint8") {
    auto [tv, tn, td] = parse_uint8bin(input.c_str());
    bench_many<uint8_t>(tv, td, n_samples_name, k_samples_name, d_samples_name,
                        runner_name, output, iter_samples_name,
                        var_samples_name, limiter, *D);

  } else if (tp == "int8") {
    auto [tv, tn, td] = parse_int8bin(input.c_str());
    bench_many<int8_t>(tv, td, n_samples_name, k_samples_name, d_samples_name,
                       runner_name, output, iter_samples_name, var_samples_name,
                       limiter, *D);

  } else {
    std::cout << "invalid type, aborting " << std::endl;
    abort();
  }

  delete D;
}