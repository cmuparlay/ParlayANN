#ifndef THREADLOCAL_H
#define THREADLOCAL_H

#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"
#include "parlay/slice.h"

#include <type_traits>
#include <string>

// for accumulator, no performance difference between 64 and higher values, but 32 is significantly slower (<- on benchmark)
// in actual code for some reason 4 is fastest, with smaller values being immediately slower and larger values gradually slower,
// although this causes incorrect results when T is a double
// #define BUFFER_ACC 8
// // for padding buffer, larger values seem to be markedly slower
// #define PADDING 64
namespace threadlocal {

template <typename T, size_t pad_bytes = 128, size_t threads = 192>
struct accumulator {
    static const size_t width = pad_bytes / sizeof(T);
    T counts[width * threads];

    accumulator() {
        std::fill(counts, counts + width * threads, (T)0);
    }

    void increment(){
        counts[parlay::worker_id()*width]++;
    }

    void decrement(){
        // probably not really valid on unsigned types but as long as the total is positive overflow should in theory make this work as intented anyways
        counts[parlay::worker_id()*width]--;
    }

    void add(T val){
        counts[parlay::worker_id()*width] += val;
    }

    void subtract(T val){
        counts[parlay::worker_id()*width] -= val;
    }

    T total(){
        return parlay::reduce(counts, parlay::addm<T>());
    }

    void reset() {
        std::fill(counts, counts + width * threads, (T)0);
    }
};


// template <typename T, >
// struct accumulator {

//     size_t width = BUFFER_ACC / sizeof(T);
//     parlay::sequence<T> counts;

//     accumulator() {
//         // if (sizeof(T) < 8) {
//         //     width /= 2;
//         // }

//         counts = parlay::sequence<T>(parlay::num_workers() * width, (T)0);
//     }

//     void increment(){
//         counts[parlay::worker_id()*width]++;
//     }

//     void decrement(){
//         // probably not really valid on unsigned types but as long as the total is positive overflow should in theory make this work as intented anyways
//         counts[parlay::worker_id()*width]--;
//     }

//     void add(T val){
//         counts[parlay::worker_id()*width] += val;
//     }

//     void subtract(T val){
//         counts[parlay::worker_id()*width] -= val;
//     }

//     T total(){
//         return parlay::reduce(counts);
//     }

//     void reset(){
//         std::fill(counts.begin(), counts.end(), (T)0);
//     }
// };

// template <typename T>
// struct minimizer {

//     size_t width = BUFFER_ACC / sizeof(T);
//     parlay::sequence<T> counts;

//     minimizer(T init) {
//         counts = parlay::sequence<T>(parlay::num_workers() * width, (T)init);
//     }

//     void update(T val){
//         counts[parlay::worker_id()*width] = std::min(counts[parlay::worker_id()*width], val);
//     }

//     T total(){
//         return parlay::reduce(counts, parlay::minm<T>());
//     }

//     void reset(){
//         counts = parlay::sequence<T>(parlay::num_workers() * width, (T)0);
//     }
// };

// template <typename T>
// struct maximizer {

//         size_t width = BUFFER_ACC / sizeof(T);
//         parlay::sequence<T> counts;

//         maximizer(T init) {
//             counts = parlay::sequence<T>(parlay::num_workers() * width, (T)init);
//         }
    
//         void update(T val){
//             counts[parlay::worker_id()*width] = std::max(counts[parlay::worker_id()*width], val);
//         }
    
//         T total(){
//             return parlay::reduce(counts, parlay::maxm<T>());
//         }
    
//         void reset(){
//             counts = parlay::sequence<T>(parlay::num_workers() * width, (T)0);
//         }
//     };

// // TODO: write a general thread local reducer that takes a binary function and a neutral element

// // TODO: write a thread local average-er that computes minimally lossy averages with small types
// // (floats are going to be too small for large n)

// /* 
//     Provides a thread local scratch buffer for copying data into.

//     as-is will probably break if T is int8_t or uint8_t
//  */
// template <typename T>
// struct buffer {
//     const size_t padding = PADDING / sizeof(T);
//     size_t length;
//     T* data;

//     /* 
//     args:
//         length: the length of the buffer (the number of elements it can hold)
//      */
//     buffer(size_t length) : length(length) {
//         data = new T[(length + padding) * parlay::num_workers()];
//     }

//     ~buffer() {
//         delete[] data;
//     }

//     void write(T* src) {
//         size_t id = parlay::worker_id();
//         size_t offset = id * (length + padding);
//         std::memcpy(data + offset, src, length * sizeof(T));
//     }

    
//     void write(int8_t* src) {
//         size_t id = parlay::worker_id();
//         size_t offset = id * (length + padding);
//         for (size_t i = 0; i < length; i++) {
//             data[offset + i] = static_cast<T>(src[i]);
//         }
//     }
    

//     void write(uint8_t* src) {
//         size_t id = parlay::worker_id();
//         size_t offset = id * (length + padding);
//         for (size_t i = 0; i < length; i++) {
//             data[offset + i] = static_cast<T>(src[i]);
//         }
//     }
    

//     T* begin() {
//         return data + parlay::worker_id() * (length + padding);
//     }

//     size_t size() {
//         return length;
//     }
// };
// TODO: write tests to benchmark the above

}
#endif