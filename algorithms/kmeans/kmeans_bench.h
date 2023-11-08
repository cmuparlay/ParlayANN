#ifndef KMEANS_BENCH_H
#define KMEANS_BENCH_H

#include "parlay/sequence.h"
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/random.h"
#include "parlay/internal/get_time.h"

#include <iostream>

/* 
    stores benchmarking properties for a single iteration of kmeans
*/
struct iteration_bench {
    double assign_time;
    double update_time;

    
    double msse;
    size_t distance_calculations;
    size_t center_reassignments;
    parlay::sequence<float> center_movements;

    //setup time tracks time used before the first iteration
    double setup_time;

    iteration_bench(double assign_time, double update_time, double msse, size_t distance_calculations, size_t center_reassignments, parlay::sequence<float> center_movements,
    double setup_time=0.0) : assign_time(assign_time), update_time(update_time), msse(msse), distance_calculations(distance_calculations), center_reassignments(center_reassignments), setup_time(setup_time) {
        this->center_movements = center_movements;
    };

    void print() {
        std::cout << "assignment time:       \t" << assign_time << std::endl;
        std::cout << "update time:           \t" << update_time << std::endl;
        std::cout << "setup time:            \t" << setup_time << std::endl;
        std::cout << "msse:                  \t" << msse << std::endl;
        std::cout << "distance calculations: \t" << distance_calculations << std::endl;
        std::cout << "center reassignments:  \t" << center_reassignments << std::endl;
        std::cout << "center movements:" << std::endl;
        std::cout << "\tmean: " << parlay::reduce(center_movements) / center_movements.size() << std::endl;
        std::cout << "\tmin:  " << parlay::reduce(center_movements, parlay::minm<float>()) << std::endl;
        std::cout << "\tmax:  " << parlay::reduce(center_movements, parlay::maxm<float>()) << std::endl;
        // could also throw quartiles in here
    }
};

/* 
    stores benchmarking properties for a kmeans run
*/
struct kmeans_bench {
    size_t n;
    size_t d;
    size_t k;
    size_t max_iter;
    double epsilon;
    parlay::sequence<iteration_bench> iterations;
    parlay::internal::timer t;
    parlay::internal::timer iteration_timer;
    double total_time = 0.0;
    std::string initializer;
    std::string runner;

    size_t n_iterations = 0;


    kmeans_bench(size_t n, size_t d, size_t k, size_t max_iter, double epsilon, std::string initializer, std::string runner) : n(n), d(d), k(k), max_iter(max_iter), epsilon(epsilon), initializer(initializer), runner(runner){
        iterations = parlay::sequence<iteration_bench>();
    }

    void start_time() {
        std::cout << initializer << " initialization with " << runner << " iterations." << std::endl;
        std::cout << "n:         \t" << n << std::endl;
        std::cout << "d:         \t" << d << std::endl;
        std::cout << "k:         \t" << k << std::endl;
        std::cout << "max_iter:  \t" << max_iter << std::endl;
        std::cout << "epsilon:   \t" << epsilon << std::endl;

        t.start();
        iteration_timer.start();
    }

    void end_time() {
        total_time = t.stop();

        std::cout << "iterations:\t" << n_iterations << std::endl;
        // std::cout << "msse:      \t" << iterations[iterations.size() - 1].msse << std::endl;
        std::cout << "total time:\t" << total_time << std::endl;
        std::cout << "avg iteration time:\t" << total_time / n_iterations << std::endl;

        //print out final assignment, update time
        double total_update_time = 0;
        double total_assign_time = 0;
        double total_setup_time = 0;
        for (size_t i = 0; i < iterations.size(); i++) {
            total_update_time += iterations[i].update_time;
            total_assign_time += iterations[i].assign_time;
            total_setup_time += iterations[i].setup_time;
        }
        std::cout << "Total assign time:\t" << total_assign_time << std::endl;
        std::cout << "Total update time:\t" << total_update_time << std::endl;
        std::cout << "Total setup time:\t" << total_setup_time << std::endl;
        // std::cout << "total assignment time:\t" << parlay::reduce(parlay::tabulate(iterations, iterations.size(), )) << std::endl;
        // std::cout << "total update time:\t" << parlay::reduce(iterations, parlay::addm<double>(), [](iteration_bench b) {return b.update_time;}) << std::endl;
    }

    /* 
    args:
        assign_time: time to assign points to centers
        update_time: time to update centers
        msse: mean sum squared error
        distance_calculations: number of distance calculations
        center_reassignments: number of points assigned to a new center
        center_movements: sequence of distances moved by each center in an iteration
     */
    void add_iteration(double assign_time, double update_time, double msse, size_t distance_calculations, size_t center_reassignments, parlay::sequence<float> center_movements,double setup_time=0) {
        iterations.push_back(iteration_bench(assign_time, update_time, msse, distance_calculations, center_reassignments, center_movements,setup_time));
        n_iterations++;

        iterations[iterations.size()-1].print();

        // std::cout << "iteration " << n_iterations << " complete. (" << iteration_timer.next_time() << "s) \tmsse: " << msse << std::endl;
        // std::cout << "distance calc: " << distance_calculations << ". center reasg: " << center_reassignments 
        // << ". center move: " << parlay::reduce(center_movements) << std::endl;
    }

    void print() {
        std::cout << "n:         \t" << n << std::endl;
        std::cout << "d:         \t" << d << std::endl;
        std::cout << "k:         \t" << k << std::endl;
        std::cout << "max_iter:  \t" << max_iter << std::endl;
        std::cout << "epsilon:   \t" << epsilon << std::endl;
        std::cout << "iterations:\t" << n_iterations << std::endl;
        std::cout << "msse:      \t" << iterations[iterations.size() - 1].msse << std::endl;
        std::cout << "total time:\t" << total_time << std::endl;
        std::cout << "avg iteration time:\t" << total_time / iterations.size() << std::endl;
        // std::cout << "total assignment time:\t" << parlay::reduce(parlay::tabulate(iterations, iterations.size(), )) << std::endl;
        // std::cout << "total update time:\t" << parlay::reduce(iterations, parlay::addm<double>(), [](iteration_bench b) {return b.update_time;}) << std::endl;
    }


    //take logging info and put it into a CSV
    //first line: outputs
    //n, d, k, max_iter, epsilon, n-iter, msse, total_time
    //then on each line:
    //iter#, assign time, update time, setup time, 
    //msse, distance_cals, center reassgs, mean center movement, 
    //min center movement, max center movement
    void output_to_csv(std::string output_folder) {


         auto c_t = std::time(nullptr);
        auto tm = *std::localtime(&c_t);
        std::ostringstream oss;
        oss << std::put_time(&tm, "%m_%d_%Y"); //don't write hour and day, annoying
        auto date_str = oss.str();

        std::string fname = output_folder+"bench_"+runner+"_"+std::to_string(n)+"_"+std::to_string(d)+"_"+std::to_string(k)+"_"+date_str + ".csv";
        std::cout << "outputting to CSV: " << fname << std::endl;

        std::ofstream file(fname);
        // file << "n, d, k, max_iter, epsilon, n_iter, msse, total_time" << std::endl;
        
        file  << "n" 
        << ", " << 
        "d"
        << ", " << 
        "k"
        << ", " << 
        "max_iter"
        << ", " << 
        "epsilon"
        << ", " << 
        "n_iter"
        << ", " << 
        "msse"
        << ", " << 
        "total_time" <<  ", "
        "init name" 
        << ", " << 
        "runner name" << std::endl;
       




        file << n << ", " << d << ", " << k << ", "<<
       max_iter << ", " << epsilon << ", " << n_iterations <<
       "," << iterations[iterations.size()-1].msse << ", " << total_time 
        << ", " << 
        initializer 
        << ", " << 
        runner << std::endl;


        file << "iter#" << "," << "asg time" << ", " << "update time" 
        << ", " << 
        "setup_time" <<
        ", " << 
        "msse"
        << ", " << 
        "dist calcs" 
        << ", " << 
        "center reasg"
        << ", " << 
        "mean center move"
        << ", " << 
        "min center move"
        << ", " << 
        "max center move" << std::endl;
        
        // setup time, msse, dist calcs, center reasg, mean center move, min center move, max center move" << std::endl;

       for (size_t i = 0; i < n_iterations; i++) {
        file<< i << ", " << iterations[i].assign_time << ", " << iterations[i].update_time
         << ", " << iterations[i].setup_time << ", " << iterations[i].msse << ", " << 
         iterations[i].distance_calculations << ", " << iterations[i].center_reassignments
         << ", " << parlay::reduce(iterations[i].center_movements) / iterations[i].center_movements.size() << ", " << 
         parlay::reduce(iterations[i].center_movements, parlay::minm<float>()) 
          << ", " <<
          parlay::reduce(iterations[i].center_movements, parlay::maxm<float>()) 
          << std::endl;
       }



        
    }

};

//struct to hold initialization info
//mainly to help contain output to csv
struct initialization_bench {
    size_t n;
    size_t d;
    size_t k;
    
    std::string name;
    parlay::internal::timer t;
    // parlay::internal::timer t_assign;
    // parlay::internal::timer t_update;
    // parlay::internal::timer t_setup;

    double total_time;
    double msse;
    // double center_update_time;
    // double assign_time;
    // double setup_time;
    parlay::sequence<std::pair<std::string, double>> labeled_times;

    initialization_bench(size_t n, size_t d, size_t k, std::string initializer) : n(n), d(d), k(k), name(initializer) {
    }

    void start_time() {
        std::cout << " initialization with " << name << std::endl;
        std::cout << "n:         \t" << n << std::endl;
        std::cout << "d:         \t" << d << std::endl;
        std::cout << "k:         \t" << k << std::endl;
     
        t.start();
    }

    void end_time() {
        total_time = t.stop();

        std::cout << "total time:\t" << total_time << std::endl;
       
    }

    void set_msse(double msse) {
        this->msse = msse;
    }
   

    void add_labeled_time(std::string label, double labels_time) {
        labeled_times.push_back(std::make_pair(label,labels_time));

    }

    void output_to_csv(std::string output_folder) {
        //https://stackoverflow.com/questions/16357999/current-date-and-time-as-string

         auto c_t = std::time(nullptr);
        auto tm = *std::localtime(&c_t);
        std::ostringstream oss;
        oss << std::put_time(&tm, "%m_%d_%Y"); //don't write hour and day, annoying
        auto date_str = oss.str();

        std::string fname = output_folder+"bench_"+name+"_"+std::to_string(n)+"_"+std::to_string(d)+"_"+std::to_string(k)+"_"+date_str + ".csv";
        std::cout << "outputting to CSV: " << fname << std::endl;
        std::ofstream file(fname);
        file << "n"
        << ", " << 
        "d"
        << ", " << 
        "k"
        << ", " << 
        "init_name"
        << ", " << 
        "time" <<
        ", " << 
         "msse" << std::endl;

        file << n
        << ", " << 
        d
        << ", " << 
        k
        << ", " << 
        name 
        << ", " << 
        total_time << 
        ", " << msse << std::endl;

        for (size_t i = 0; i < labeled_times.size(); i++) {
            file << labeled_times[i].first << ", " << labeled_times[i].second << std::endl;
        }


    }


};

#endif