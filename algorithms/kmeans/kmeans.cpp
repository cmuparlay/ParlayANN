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

//other information default
//this version uses the "cluster" method from the KmeansInterface
template <typename T>
inline void bench_three(T*v, size_t n, size_t d, size_t k) {
    NaiveKmeans<T,Euclidian_Point<T>,size_t,float,Euclidian_Point<float>> runner;
    auto output = runner.cluster(PointRange<T,Euclidian_Point<T>>(v,n,d,d),k);
    std::cout << "finished" << std::endl;
    std::cout << "Printing out partitions: " << std::endl;
    auto parts = output.first;
    for (int i = 0; i < parts.size(); i++) {
        std::cout << i << ": ";
        for (int j = 0; j < parts[i].size(); j++) {
            std::cout << parts[i][j] << " ";
        }
        std::cout << std::endl;
    }

}


template <typename T>
inline void bench_two_stable(T* v, size_t n, size_t d, size_t k, Distance& D, 
size_t max_iter=1000, double epsilon=0, bool output_log_to_csv=false, std::string output_file_name1="data.csv", std::string output_file_name2="data2.csv") { 
    std::cout << "fill in bench two stable" << std::endl;

}

template <typename T>
inline void bench_two(T* v, size_t n, size_t d, size_t k, Distance& D, 
size_t max_iter=1000, double epsilon=0, bool output_log_to_csv=false, std::string output_file_name1="data.csv", std::string output_file_name2="data2.csv") { 
    std::cout << "fill in bench 2" << std::endl;

     std::cout << "Running bench stable " << std::endl;

    float* c = new float[k*d]; // centers
    size_t* asg = new size_t[n];

  
    //initialization
    Lazy<T,size_t> init;
    //note that here, d=ad
    init(v,n,d,d,k,c,asg);

   
    NaiveKmeans<T,Euclidian_Point<T>,size_t,float,Euclidian_Point<float>> nie2;
    kmeans_bench logger_nie2 = kmeans_bench(n,d,k,max_iter,
    epsilon,"Lazy","Naive");
    logger_nie2.start_time();
    //note that d=ad here
    nie2.cluster_middle(v,n,d,d,k,c,asg,D,logger_nie2,max_iter,epsilon);
    logger_nie2.end_time();
    

}


int main(int argc, char* argv[]){
    commandLine P(argc, argv, "[-k <n_clusters>] [-m <iterations>] [-o <output>] [-i <input>] [-f <ft>] [-t <tp>] [-D <dist>]");

    size_t k = P.getOptionLongValue("-k", 10); // k is number of clusters
    size_t max_iterations = P.getOptionLongValue("-m", 1000); // max_iterations is the max # of Lloyd iters kmeans will run
    std::string output = std::string(P.getOptionValue("-o", "kmeans_results.csv")); // maybe the kmeans results get written into this csv
    std::string input = std::string(P.getOptionValue("-i", "")); // the input file
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
            if (use_bench_two == "yes") {
                bench_two<float>(v,n,d,k,*D,max_iterations,epsilon,output_log_to_csv,output_log_file_name,output_log_file_name2);

            }
            else if (use_bench_two=="stable") {
                bench_two_stable<float>(v,n,d,k,*D,max_iterations,epsilon,output_log_to_csv,output_log_file_name,output_log_file_name2);

            }
            else if (use_bench_two=="three") {
                bench_three<float>(v,n,d,k);

            }
            else {
                std::cout << "Must specify bench path, aborting" << std::endl;
                abort();


            }
        } else if (tp == "uint8") {
            auto [v, n, d] = parse_uint8bin(input.c_str());
            if (use_bench_two=="yes") {
                bench_two<uint8_t>(v,n,d,k,*D,max_iterations,epsilon,output_log_to_csv,output_log_file_name,output_log_file_name2);
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
                bench_two<int8_t>(v,n,d,k,*D,max_iterations,epsilon,output_log_to_csv,output_log_file_name,output_log_file_name2);

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