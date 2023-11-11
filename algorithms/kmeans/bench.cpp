//file for benching a kmeans method with many varied parameters
//kmeans.cpp is for running a single instance of kmeans, or perhaps two methods on the same initialization **on the same data**
//whereas bench.cpp is meant for running cross-data 

//TODO purge the include list to include only what's actually needed
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"
#include "parlay/slice.h"
#include "parlay/random.h"
#include "parlay/internal/get_time.h"

#include <iostream>
#include <algorithm>
#include <chrono>
#include <random>
#include <set>
#include <atomic>
#include <utility>
#include <type_traits>
#include <cmath>

#include "../bench/parse_command_line.h"
#include "parse_files.h"
#include "distance.h"
#include "kmeans_bench.h"
#include "initialization.h"
#include "naive.h"
#include "kmeans.h"


//bench many
//n samples <- values of n to try
//k samples <- values of k to try
//d samples < values of d to try
//output_file <- where to put results
//iter_samples <- values of max_iter to tryu
//var_samples <- how many times we do the run (if > 1, then average)
//limiter <- cap the value of n*d*k*max_iter*var_samples to prevent a single run from taking too long
inline void bench_many(std::string data_samples_name, std::string n_samples, std::string k_samples, std::string d_samples, std::string output_file, std::string iter_samples, std::string var_samples, size_t limiter, Distance& D) {
    std::cout << "Run bench many " << std::endl;
    std::vector<size_t> n_vec = extract_vector<size_t>(n_samples.c_str());
    std::vector<size_t> d_vec = extract_vector<size_t>(d_samples.c_str());
    std::vector<size_t> k_vec = extract_vector<size_t>(k_samples.c_str());
    std::vector<size_t> iter_vec = extract_vector<size_t>(iter_samples.c_str());
    std::vector<size_t> var_vec = extract_vector<size_t>(var_samples.c_str());

    std::vector<std::string> data_samples = extract_string_vector(data_samples_name.c_str());

}




int main(int argc, char* argv[]){
    commandLine P(argc, argv, " [-ds <data_file_names>] [-ns <n_samples_file>] [-ks <k_file>] [-ds <d_file>] [-is <iter_file>] [-vs <var_file>] [-D <dist>] [-o <output_file>] [-lim <limiter>]");

    std::string output = std::string(P.getOptionValue("-o", "kmeans_bench_results.txt")); // maybe the kmeans results get written into this csv
    
    std::string dist = std::string(P.getOptionValue("-D", "fast")); // distance choice

    std::string n_samples_name = std::string(P.getOptionValue("-ns","n.txt"));
    std::string k_samples_name = std::string(P.getOptionValue("-ks","k.txt"));
    std::string d_samples_name = std::string(P.getOptionValue("-ds","d.txt"));
    std::string iter_samples_name = std::string(P.getOptionValue("-iter_vals","iter.txt"));
    std::string var_samples_name = std::string(P.getOptionValue("-vs","var.txt"));
    std::string data_samples_name = std::string(P.getOptionValue("-ds","data.txt"));
    long limiter = P.getOptionLongValue("-lim",1'000'000'000);


    Distance* D; // create a distance object, it can either by Euclidian or MIPS
    // if (dist == "Euclidean") { 
    //     std::cout << "Using Euclidean distance" << std::endl;
    //     D = new EuclideanDistance();
    // } else 
    if (dist == "mips") {
        std::cout << "Using MIPS distance" << std::endl;
        D = new Mips_Distance();
    } else if (dist=="short") {
        std::cout << "Using short Euclidean" << std::endl;
        D = new EuclideanDistanceSmall();
    } 
    else if (dist=="fast") {
        std::cout << "Using fast Euclidean" << std::endl;
        D = new EuclideanDistanceFast();
    }
    else {
        std::cout << "Error: distance type not specified correctly, specify Euclidean or mips" << std::endl;
        abort();
    }

    bench_many(data_samples_name,n_samples_name,k_samples_name,d_samples_name,output,iter_samples_name,var_samples_name,limiter,*D);




}