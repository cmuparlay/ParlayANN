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
#include "yy.h"

//other information default
//this version uses the "cluster" method from the KmeansInterface
template <typename T>
inline void bench_three(T*v, size_t n, size_t d, size_t k) {
    NaiveKmeans<T,Euclidian_Point<T>,size_t,float,Euclidian_Point<float>> runner;
    auto output = runner.cluster(PointRange<T,Euclidian_Point<T>>(v,n,d,d),k);
    std::cout << "finished" << std::endl;
    std::cout << "Printing out partitions: " << std::endl;
    auto parts = output.first;
    for (size_t i = 0; i < parts.size(); i++) {
        std::cout << i << ": ";
        for (size_t j = 0; j < parts[i].size(); j++) {
            std::cout << parts[i][j] << " ";
        }
        std::cout << std::endl;
    }

}

//bench stable has a promise not to be changed/updated
template <typename T>
inline void bench_two_stable(T* v, size_t n, size_t d, size_t k, Distance& D, 
size_t max_iter=1000, double epsilon=0, bool output_log_to_csv=false, std::string output_file_name1="data.csv", std::string output_file_name2="data2.csv") { 
    std::cout << "fill in bench two stable" << std::endl;

}

//bench two is the basic version I mess with
template <typename T>
inline void bench_two(T* v, size_t n, size_t d, size_t ad, size_t k, Distance& D, 
size_t max_iter=1000, double epsilon=0, bool output_log_to_csv=false, std::string output_file_name1="data.csv", std::string output_file_name2="data2.csv") { 
    std::cout << "fill in bench 2" << std::endl;

     std::cout << "Running bench stable " << std::endl;

    float* c = new float[k*ad]; // centers
    size_t* asg = new size_t[n];

    
  
    //initialization
    Lazy<T,float,size_t> init;
    //note that here, d=ad
    init(v,n,d,ad,k,c,asg);

    //c2 and asg2 for the yy run
    //make sure to copy over AFTER initialization, but BEFORE kmeans run
    float* c2 = new float[k*ad];
    size_t* asg2 = new size_t[n];
    parlay::parallel_for(0,k*ad,[&] (size_t i) {
        c2[i]=c[i];
    });
    parlay::parallel_for(0,n,[&] (size_t i) {
        asg2[i]=asg[i];
    });


   
    NaiveKmeans<T,Euclidian_Point<T>,size_t,float,Euclidian_Point<float>> nie2;
    kmeans_bench logger_nie2 = kmeans_bench(n,d,k,max_iter,
    epsilon,"Lazy","Naive");
    logger_nie2.start_time();
    //note that d=ad here
    nie2.cluster_middle(v,n,d,ad,k,c,asg,D,logger_nie2,max_iter,epsilon);
    logger_nie2.end_time();

    Yinyang<T,Euclidian_Point<T>,size_t,float,Euclidian_Point<float>> yy_runner;
    kmeans_bench logger_yy = kmeans_bench(n,d,k,max_iter,epsilon,"Lazy","YY");
    logger_yy.start_time();
    yy_runner.cluster_middle(v,n,d,ad,k,c2,asg2,D,logger_yy,max_iter,epsilon);
    logger_yy.end_time();


    
}

//bench many
//n samples <- values of n to try
//k samples <- values of k to try
//d samples < values of d to try
//output_file <- where to put results
//iter_samples <- values of max_iter to tryu
//var_samples <- how many times we do the run (if > 1, then average)
//limiter <- cap the value of n*d*k*max_iter*var_samples to prevent a single run from taking too long
// template<typename T>
// inline void bench_many(T* v, size_t n, size_t d, std::string n_samples, std::string k_samples, std::string d_samples, std::string output_file, std::string iter_samples, std::string var_samples, size_t limiter) {
//     std::cout << "Run bench many " << std::endl;
//     std::cout << "Bench many moved to bench.cpp file" << std::endl;
//     std::vector<size_t> n_vec = extract_vector<size_t>(n_samples);
//     std::vector<size_t> d_vec = extract_vector<size_t>(d_samples);
//     std::vector<size_t> k_vec = extract_vector<size_t>(k_samples);
//     std::vector<size_t> iter_vec = extract_vector<size_t>(iter_samples);
//     std::vector<size_t> var_vec = extract_vector<size_t>(var_samples);

// }

//if new_d is the default value, go with the d given by the dataset. Otherwise, use the custom value of d.
size_t pick_num(long orig_d,long new_d) {
    if (new_d==-1) {
        return orig_d;
    }
    return new_d;
}


int main(int argc, char* argv[]){
    commandLine P(argc, argv, "[-k <n_clusters>] [-m <iterations>] [-o <output>] [-i <input>] [-f <ft>] [-t <tp>] [-D <dist>]");

    long newn = P.getOptionLongValue("-n",-1);
    size_t k = P.getOptionLongValue("-k", 10); // k is number of clusters
    long newd = P.getOptionLongValue("-d",-1);
    size_t max_iterations = P.getOptionLongValue("-m", 1000); // max_iterations is the max # of Lloyd iters kmeans will run
    std::string output = std::string(P.getOptionValue("-o", "kmeans_results.csv")); // maybe the kmeans results get written into this csv
    std::string input = std::string(P.getOptionValue("-i", "")); // the data input file
    std::string ft = std::string(P.getOptionValue("-f", "bin")); // file type, bin or vecs
    std::string tp = std::string(P.getOptionValue("-t", "uint8")); // data type
    std::string dist = std::string(P.getOptionValue("-D", "Euclidian")); // distance choice
    std::string use_bench_two = std::string(P.getOptionValue("-two","no"));
    bool output_log_to_csv = false;
    std::string output_to_csv_str = std::string(P.getOptionValue("-csv_log","false"));
    if (output_to_csv_str == "true") {
        output_log_to_csv=true;
    }
    std::string output_log_file_name = std::string(P.getOptionValue("-csv_log_file_name","data.csv"));
    std::string output_log_file_name2 = std::string(P.getOptionValue("-csv_log_file_name2","data2.csv"));
    std::string n_samples_name = std::string(P.getOptionValue("-n_vals","n.txt"));
    std::string k_samples_name = std::string(P.getOptionValue("-k_vals","k.txt"));
    std::string d_samples_name = std::string(P.getOptionValue("-d_vals","d.txt"));
    std::string iter_samples_name = std::string(P.getOptionValue("-iter_vals","iter.txt"));
    std::string var_samples_name = std::string(P.getOptionValue("-var_vals","var.txt"));
    long limiter = P.getOptionLongValue("-lim",1'000'000'000);
    limiter +=1;//to avoid unused warning TODO FIXME

    float epsilon = static_cast<float>(P.getOptionDoubleValue("-epsilon",0.0));


    if(input == ""){ // if no input file given, quit
        std::cout << "Error: input file not specified" << std::endl;
        abort();
    }

    if((ft != "bin") && (ft != "vec")){ // if the file type chosen is not one of the two approved file types, quit 
    std::cout << "Error: file type not specified correctly, specify bin or vec" << std::endl;
    abort();
    }

    if((tp != "uint8") && (tp != "int8") && (tp != "float")){ // if the data type isn't one of the three approved data types, quit
        std::cout << "Error: vector type not specified correctly, specify int8, uint8, or float" << std::endl;
        abort();
    }

    if((ft == "vec") && (tp == "int8")){ // you can't store int8s in a vec file apparently I guess
        std::cout << "Error: incompatible file and vector types" << std::endl;
        abort();
    }

    // TODO: add support for vec files
    if (ft == "vec") {
        std::cout << "Error: vec file type not supported yet" << std::endl;
        abort();
    }

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

    if (ft == "bin"){
        if (tp == "float") {
            auto [v, n, d] = parse_fbin(input.c_str());
            if (use_bench_two=="yes") { //can't use switch for strings sadly
                    bench_two<float>(v,pick_num(n,newn),pick_num(d,newd),d,k,*D,max_iterations,epsilon,output_log_to_csv,output_log_file_name,output_log_file_name2);
            }
            else if (use_bench_two=="stable") {
                    bench_two_stable<float>(v,n,d,k,*D,max_iterations,epsilon,output_log_to_csv,output_log_file_name,output_log_file_name2);
            }
            else if (use_bench_two=="three") {
                    bench_three<float>(v,n,d,k);
            }
                // case "many":
                //     bench_many<float>(v,n,d,n_samples_name,k_samples_name,d_samples_name,output_log_to_csv,iter_samples_name,var_samples_name,limiter);
            else {
                    std::cout << "Must specify bench path, aborting" << std::endl;
                    abort();

            }

        } else if (tp == "uint8") {
            auto [v, n, d] = parse_uint8bin(input.c_str());
            if (use_bench_two=="yes") {
                bench_two<uint8_t>(v,pick_num(n,newn),pick_num(d,newd),d,k,*D,max_iterations,epsilon,output_log_to_csv,output_log_file_name,output_log_file_name2);
            }
            else if (use_bench_two=="stable") {
                bench_two_stable<uint8_t>(v,n,d,k,*D,max_iterations,epsilon,output_log_to_csv,output_log_file_name,output_log_file_name2);

            }
            else if (use_bench_two=="three") {
                bench_three<uint8_t>(v,n,d,k);

            }
            else {
                std::cout << "Must specify bench path, aborting" << std::endl;
                abort();


            }
        } else if (tp == "int8") {
            auto [v, n, d] = parse_int8bin(input.c_str());
            if (use_bench_two == "yes") {
                bench_two<int8_t>(v,pick_num(n,newn),pick_num(d,newd),d,k,*D,max_iterations,epsilon,output_log_to_csv,output_log_file_name,output_log_file_name2);

            }
            else if (use_bench_two=="stable") {
                bench_two_stable<int8_t>(v,n,d,k,*D,max_iterations,epsilon,output_log_to_csv,output_log_file_name,output_log_file_name2);

            }
            else if (use_bench_two=="three") {
                bench_three<int8_t>(v,n,d,k);

            }
            else {
                std::cout << "Must specify bench path, aborting" << std::endl;
                abort();
            }
        } else {
            //  this should actually be unreachable
            std::cout << "Error: bin type can only be float, uint8, or int8. Supplied type is " << tp << "." << std::endl;
            abort();
        }
    }

    return 0;

}