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

    T total() const {
        return parlay::reduce(counts, parlay::addm<T>());
    }

    void reset() {
        std::fill(counts, counts + width * threads, (T)0);
    }
};


template <typename T, size_t pad_bytes = 128, size_t threads = 192>
struct minimizer {
    static const size_t width = pad_bytes / sizeof(T);
    T counts[width * threads];

    minimizer() {
        std::fill(counts, counts + width * threads, std::numeric_limits<T>::max());
    }

    void update(T val){
        counts[parlay::worker_id()*width] = std::min(counts[parlay::worker_id()*width], val);
    }

    T total() const {
        return parlay::reduce(counts, parlay::minm<T>());
    }

    void reset() {
        std::fill(counts, counts + width * threads, (T)0);
    }
};

template <typename T, size_t pad_bytes = 128, size_t threads = 192>
struct maximizer {
    static const size_t width = pad_bytes / sizeof(T);
    T counts[width * threads];

    maximizer() {
        std::fill(counts, counts + width * threads, std::numeric_limits<T>::min());
    }

    void update(T val){
        counts[parlay::worker_id()*width] = std::max(counts[parlay::worker_id()*width], val);
    }

    T total() const {
        return parlay::reduce(counts, parlay::maxm<T>());
    }

    void reset() {
        std::fill(counts, counts + width * threads, (T)0);
    }
};

template <typename T, size_t pad_bytes = 128, size_t threads = 192>
struct extremeizer {
    static const size_t width = pad_bytes / sizeof(std::pair<T, T>);
    T counts[width * threads];

    extremeizer() {
        std::fill(counts, counts + width * threads, std::make_pair(std::numeric_limits<T>::max(), std::numeric_limits<T>::min()));
    }

    
    void update(T val){
        auto& pair = counts[parlay::worker_id()*width];
        if (val < pair.first) {
            counts[parlay::worker_id()*width].first = val;
        }
        if (val > counts[parlay::worker_id()*width].second) {
            counts[parlay::worker_id()*width].second = val;
        }
    }

    std::pair<T, T> total() const {
        T min = std::numeric_limits<T>::max();
        T max = std::numeric_limits<T>::min();
        for (size_t t = 0; t < threads; t++) {
            auto& pair = counts[t*width];
            if (pair.first < min) {
                min = pair.first;
            }
            if (pair.second > max) {
                max = pair.second;
            }
        }

        return std::make_pair(min, max);
    }

    void reset() {
        std::fill(counts, counts + width * threads, std::make_pair(std::numeric_limits<T>::max(), std::numeric_limits<T>::min()));
    }
};

/* not sure this is even benefiting much from being spaced out  

T is going to end up being a length 3 or 4 tuple*/
template <typename T, size_t pad_bytes = 128, size_t threads = 192>
struct logger {
    static const size_t width = pad_bytes / sizeof(T);
    parlay::sequence<parlay::sequence<T>> counts;

    logger() {
        counts = parlay::sequence<parlay::sequence<T>>::uninitialized(width * threads);

        for (size_t i = 0; i < threads; i++) {
            counts[i] = parlay::sequence<T>();
        }
    }

    void reserve(size_t n) {
        for (size_t i = 0; i < threads; i++) {
            counts[i].reserve(n / threads + 1);
        }
    }

    void update(T val){
        counts[parlay::worker_id()*width].push_back(val);
    }

    parlay::sequence<T> get() const {
        int total_size = 0;
        for (size_t i = 0; i < threads; i++) {
            total_size += counts[i * width].size();
        }

        parlay::sequence<T> result = parlay::sequence<T>::uninitialized(total_size);
        size_t offset = 0;
        for (size_t i = 0; i < threads; i++) {
            size_t size = counts[i * width].size();
            std::memcpy(result.begin() + offset, counts[i * width].begin(), size * sizeof(T));
            offset += size;
        }

        return result;
    }

    void reset() {
        std::fill(counts, counts + width * threads, (T)0);
    }
};
}


#endif