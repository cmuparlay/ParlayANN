

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


#define INITIALIZER MacQueen
#define INITIALIZER_NAME "MacQueen"
#define RUNNER NaiveKmeans 
#define RUNNER_NAME "Naive"

template<typename T, typename Initializer, typename Runner>         
void Kmeans(T* v, size_t n, size_t d, size_t k, float* c, size_t* asg, 
Distance& D, kmeans_bench logger, size_t max_iter = 1000, double epsilon=0.01) {

    Initializer init;
    init(v,n,d,k,c,asg,D);

    std::cout << "Initialization complete" << std::endl;

    Runner run;
    run.cluster(v,n,d,k,c,asg,D,logger,max_iter,epsilon);

    std::cout << "Clustering complete" << std::endl;

}

template <typename T, typename initializer, typename runner>
inline void bench(T* v, size_t n, size_t d, size_t k, Distance& D, size_t max_iter = 1000, double epsilon=0) {
    float* centers = new float[k*d];
    size_t* asg = new size_t[n];
    kmeans_bench logger = kmeans_bench(n, d, k, max_iter, epsilon, INITIALIZER_NAME, RUNNER_NAME);
    logger.start_time();
    Kmeans<T, initializer, runner>(v, n, d, k, centers, asg, D, logger, max_iter, epsilon);
    logger.end_time();

    return;
}

template <typename T>
inline void bench_two_stable(T* v, size_t n, size_t d, size_t k, Distance& D, 
size_t max_iter=1000, double epsilon=0, bool output_log_to_csv=false, std::string output_file_name1="data.csv", std::string output_file_name2="data2.csv") { 
    std::cout << "fill in bench two stable" << std::endl;

}

template <typename T, typename Initializer, typename Runner1, typename Runner2>
inline void bench_two(T* v, size_t n, size_t d, size_t k, Distance& D, 
size_t max_iter=1000, double epsilon=0, bool output_log_to_csv=false, std::string output_file_name1="data.csv", std::string output_file_name2="data2.csv") { 
    std::cout << "fill in bench 2" << std::endl;

     std::cout << "Running bench stable " << std::endl;

    float* c = new float[k*d]; // centers
    size_t* asg = new size_t[n];

   float* c2 = new float[k*d];
   size_t* asg2 = new size_t[n];
  
    //initialization
    LazyStart<T> init;
    init(v,n,d,k,c,asg,D);

    for (size_t i = 0; i < k*d; i++) {
        c2[i] = c[i];
    }
    for (size_t i = 0; i < n; i++) {
        asg2[i] = asg[i];
    }

    NaiveKmeans<T> nie2;
    kmeans_bench logger_nie2 = kmeans_bench(n,d,k,max_iter,
    epsilon,"Lazy","Naive");
    logger_nie2.start_time();
    nie2.cluster(v,n,d,k,c2,asg2,D,logger_nie2,max_iter,epsilon);
    logger_nie2.end_time();
    

}

// //bench two kmeans methods on the same data
// template <typename T>
// inline void bench_two_stable(T* v, size_t n, size_t d, size_t k, Distance& D, 
// size_t max_iter=1000, double epsilon=0, bool output_log_to_csv=false, std::string output_file_name1="data.csv", std::string output_file_name2="data2.csv") {

//     std::cout << "Running bench stable " << std::endl;

//     float* c = new float[k*d]; // centers
//     size_t* asg = new size_t[n];

//    float* c2 = new float[k*d];
//    size_t* asg2 = new size_t[n];
  
//     //initialization
//     LazyStart<T> init;
//     init(v,n,d,k,c,asg,D);

//     for (size_t i = 0; i < k*d; i++) {
//         c2[i] = c[i];
//     }
//     for (size_t i = 0; i < n; i++) {
//         asg2[i] = asg[i];
//     }

//     NaiveKmeans2<T> nie2;
//     kmeans_bench logger_nie2 = kmeans_bench(n,d,k,max_iter,
//     epsilon,"Lazy","Naive2");
//     logger_nie2.start_time();
//     nie2.cluster(v,n,d,k,c2,asg2,D,logger_nie2,max_iter,epsilon);
//     logger_nie2.end_time();
    
//     YinyangImproved<T> yy;
//     kmeans_bench logger_yy = kmeans_bench(n,d,k,max_iter,epsilon,
//     "Lazy","YY Imp");
//     logger_yy.start_time();

//     yy.cluster(v,n,d,k,c2,asg2,D,logger_yy, max_iter,epsilon);

//     logger_yy.end_time();

//     std::cout << "finished bench stable" << std::endl;

//     delete[] c;
//     delete[] asg;
//     delete[] asg2;

// }

// //bench two kmeans methods on the same data
// template <typename T, typename Initializer, typename Runner1, typename Runner2>
// inline void bench_two(T* v, size_t n, size_t d, size_t k, Distance& D, 
// size_t max_iter=1000, double epsilon=0, bool output_log_to_csv=false, std::string output_file_name1="data.csv", std::string output_file_name2="data2.csv") {


//     // std::cout << "shortening d for debugging" << std::endl;
//     // //d = 2;

//     std::cout << "Running bench two " << std::endl;
//     std::cout << "n d " << n << " " << d << std::endl;

//     std::cout << "printing 1st 3 points, first 10 dim of each" << std::endl;
//     std::cout << "int cast needed for uint8s" << std::endl;
//     for (size_t i = 0; i < 3; i++) {
//         for (size_t j = 0; j < std::min(d,(size_t) 10); j++) {
//             std::cout << static_cast<int>(v[i*d +j]) << " ";
//         }
//         std::cout << std::endl;
//     }
//     std::cout << std::endl;

//     float* c = new float[k*d]; // centers
//     size_t* asg = new size_t[n];

//    float* c2 = new float[k*d];
//    size_t* asg2 = new size_t[n];
// //    float* c3 = new float[k*d];
// //    size_t* asg3 = new size_t[n];

//     std::cout << "made it hey 1" << std::endl;
//     // KmeansPlusPlus<T> init;
//     // init(v,n,d,k,c,asg,D);

//     //using LSH not LazyStart to be more realistic

//     // LSH<T> lsh_init;
//     // lsh_init(v,n,d,k,c,asg,D);
//     //Lazy better??
//     LazyStart<T> init;
//     init(v,n,d,k,c,asg,D);

// //      std::cout << "printing different initializations, first 50: " << std::endl;
// //    for (size_t i = 0; i < 50; i++) {
// //     std::cout << asg[i] << " " << asg2[i] << std::endl;
// //    }


//     for (size_t i = 0; i < k*d; i++) {
//         c2[i] = c[i];
//         //c3[i]=c[i];
//     }
//     for (size_t i = 0; i < n; i++) {
//         asg2[i] = asg[i];
//         //asg3[i]=asg[i];
//     }

//     std::cout << "Trying to use the Kmeans<templatted> function call" << std::endl;
//     // kmeans_bench logger_yy = kmeans_bench(n,d,k,max_iter,epsilon,
//     // "LSH","YY");
//     //Kmeans<T,LSH<T>,YinyangSimp<T>>(v,n,d,k,c,asg,D,logger_yy,max_iter,0);
    
    
//     //Kmeans<T,LSH<T>,NaiveKmeans<T>>(v,n,d,k,c2,asg2,D,logger_yy,max_iter,0);


//     // QuantizedKmeans<T> quant;
//     // kmeans_bench logger_quant = kmeans_bench(n,d,k,max_iter,epsilon,"Lazy","Quant");
//     // logger_quant.start_time();
//     // quant.cluster(v,n,d,k,c,asg,D,logger_quant,max_iter,epsilon);
//     // logger_quant.end_time();

//     //  if (output_log_to_csv) {
//     //     logger_quant.output_to_csv(output_file_name1);
//     // }


//     // std::cout << "cutting out after quant" << std::endl;
//     // abort();

//     // NiskKmeans<T> sk;
//     // kmeans_bench logger_sk = kmeans_bench(n,d,k,max_iter, epsilon,"Lazy","Skln");
//     // logger_sk.start_time();
//     // sk.cluster(v,n,d,k,c,asg,D,logger_sk,max_iter,epsilon);
//     // logger_sk.end_time();


//     // LSHQuantizedKmeans<T> lshq;
//     // kmeans_bench logger_lshq = kmeans_bench(n,d,k,max_iter, epsilon,"Lazy","LSHQuantized");
//     // logger_lshq.start_time();
//     // lshq.cluster(v,n,d,k,c,asg,D,logger_lshq,max_iter,epsilon);
//     // logger_lshq.end_time();

//     //Don't need to run as already know how long it takes
//     // NaiveKmeans<T> nie;
//     // kmeans_bench logger_nie = kmeans_bench(n,d,k,max_iter,
//     // epsilon,"LSH","NaiveKmeans");
//     // logger_nie.start_time();
//     // nie.cluster(v,n,d,k,c,asg,D,logger_nie,max_iter,epsilon);
//     // logger_nie.end_time();
//     // if (output_log_to_csv) {
//     //     logger_nie.output_to_csv(output_file_name1);
//     // }

//     // NaiveKmeans<T> nie2;
//     // kmeans_bench logger_nie2 = kmeans_bench(n,d,k,max_iter,
//     // epsilon,"Lazy","NaiveKmeans");
//     // logger_nie2.start_time();
//     // nie2.cluster(v,n,d,k,c2,asg2,D,logger_nie2,max_iter,epsilon);
//     // logger_nie2.end_time();
//     //logger_nie2.output_to_csv(output_file_name1);

//       NaiveKmeans2<T> nie2;
//     kmeans_bench logger_nie2 = kmeans_bench(n,d,k,max_iter,
//     epsilon,"Lazy","Naive2");
//     logger_nie2.start_time();
//     nie2.cluster(v,n,d,k,c2,asg2,D,logger_nie2,max_iter,epsilon);
//     logger_nie2.end_time();
  

//     // std::cout << "cutting out after my naive" << std::endl;
//     // abort();
//     // std::cout << "starting naive" << std::endl;

//     // YinyangImproved<T> yy;
//     // kmeans_bench logger_yy = kmeans_bench(n,d,k,max_iter,epsilon,
//     // "LSH","YY Imp");
//     // logger_yy.start_time();

//     // yy.cluster(v,n,d,k,c2,asg2,D,logger_yy, max_iter,epsilon);

//     // logger_yy.end_time();

//     // PQKmeans<T> pq;
//     // kmeans_bench logger_pq = kmeans_bench(n,d,k,max_iter,epsilon,"LSH","PQ Kmeans actually Naive rn");
//     // logger_pq.start_time();
//     // pq.cluster(v,n,d,k,c2,asg2,D,logger_pq,max_iter,epsilon);
//     // logger_pq.end_time();
//     // if (output_log_to_csv) { logger_yy.output_to_csv(output_file_name2); }

//     // YinyangSimp<T> yy_simp;
//     // kmeans_bench logger_yy_simp = kmeans_bench(n,d,k,max_iter,epsilon,
//     // "LSH","YY Simp");
//     // logger_yy_simp.start_time();

//     // yy_simp.cluster(v,n,d,k,c,asg,D,logger_yy_simp, max_iter,epsilon);

//     // logger_yy_simp.end_time();
//     // if (output_log_to_csv) { logger_yy_simp.output_to_csv(output_file_name2); }

//     // std::cout << "Cutting out after yy" << std::endl;
//     // abort();


//     // Naive<T> ben_naive;
//     // kmeans_bench logger = 
//     // kmeans_bench(n, d, k, max_iter, epsilon, "Lazy", "Naive");
//     // logger.start_time();
//     // ben_naive.cluster(v,n,d,k,c3,asg3,D,logger,max_iter,epsilon);
//     // logger.end_time();

//     // std::cout << "Printing out first 10 final centers, the first 10 dim: "  << std::endl;
//     // for (size_t i = 0; i < std::min((size_t) 10,k); i++) {
//     //     for (size_t j = 0; j < std::min((size_t) 10,d); j++) {
//     //         std::cout << c[i*d + j] <<  "|" << c3[i*d+j] << " ";
//     //     }
//     //     std::cout << std::endl;
//     // }

//     // std::cout << "Printing out 5 final assignments: " << std::endl;
//     // for (size_t i = 0; i < std::min(n,(size_t) 50); i++) {
//     //     std::cout << asg[i] << " " << std::endl;// << asg2[i] << " " << std::endl;
        
//     // }
//     // std::cout << std::endl << std::endl;
//     std::cout << "finished" << std::endl;

//     delete[] c;
//     delete[] asg;

// }

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
                bench_two<float,LazyStart<float>,NaiveKmeans<float>,NaiveKmeans<float>>(v,n,d,k,*D,max_iterations,epsilon,output_log_to_csv,output_log_file_name,output_log_file_name2);

            }
            else if (use_bench_two=="stable") {
                bench_two_stable<float>(v,n,d,k,*D,max_iterations,epsilon,output_log_to_csv,output_log_file_name,output_log_file_name2);

            }
            else {
                bench<float, INITIALIZER<float>, RUNNER<float>>(v, n, d, k, *D, max_iterations, epsilon);


            }
        } else if (tp == "uint8") {
            auto [v, n, d] = parse_uint8bin(input.c_str());
            if (use_bench_two=="yes") {
                bench_two<uint8_t,LazyStart<uint8_t>,NaiveKmeans<uint8_t>,NaiveKmeans<uint8_t>>(v,n,d,k,*D,max_iterations,epsilon,output_log_to_csv,output_log_file_name,output_log_file_name2);
            }
            else if (use_bench_two=="stable") {
                bench_two_stable<uint8_t>(v,n,d,k,*D,max_iterations,epsilon,output_log_to_csv,output_log_file_name,output_log_file_name2);

            }
            else {
                bench<uint8_t, INITIALIZER<uint8_t>, RUNNER<uint8_t>>(v, n, d, k, *D, max_iterations, epsilon);


            }
        } else if (tp == "int8") {
            auto [v, n, d] = parse_int8bin(input.c_str());
            if (use_bench_two == "yes") {
                bench_two<int8_t,LazyStart<int8_t>,NaiveKmeans<int8_t>,NaiveKmeans<int8_t>>(v,n,d,k,*D,max_iterations,epsilon,output_log_to_csv,output_log_file_name,output_log_file_name2);

            }
            else if (use_bench_two=="stable") {
                bench_two_stable<int8_t>(v,n,d,k,*D,max_iterations,epsilon,output_log_to_csv,output_log_file_name,output_log_file_name2);

            }
            else {
                bench<int8_t, INITIALIZER<int8_t>, RUNNER<int8_t>>(v, n, d, k, *D, max_iterations, epsilon);


            }
        } else {
            //  this should actually be unreachable
            std::cout << "Error: bin type can only be float, uint8, or int8. Supplied type is " << tp << "." << std::endl;
            abort();
        }
    }

    return 0;

}