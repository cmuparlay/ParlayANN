#ifndef PARSING
#define PARSING

#include <iostream>
#include <algorithm>
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"
#include "parlay/slice.h"

#include "../utils/mmap.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <utility>
#include <unistd.h>

/* 
***********************************
*  Parsing functions for binary files
***********************************
*/

auto parse_uint8bin(const char* filename){
    auto [fileptr, length] = mmapStringFromFile(filename);

    int num_vectors = *((int*) fileptr);
    int d = *((int*) (fileptr+4));

    std::cout << "Detected " << num_vectors << " points with dimension " << d << std::endl;

    return std::make_tuple((uint8_t*)fileptr + 8, num_vectors, d);
}

auto parse_int8bin(const char* filename){
    auto [fileptr, length] = mmapStringFromFile(filename);

    int num_vectors = *((int*) fileptr);
    int d = *((int*) (fileptr+4));
 
    std::cout << "Detected " << num_vectors << " points with dimension " << d << std::endl;
 
    return std::make_tuple((int8_t*)fileptr + 8, num_vectors, d);
}

auto parse_fbin(const char* filename){
    auto [fileptr, length] = mmapStringFromFile(filename);

    int num_vectors = *((int*) fileptr);
    int d = *((int*) (fileptr+4));

    std::cout << "Detected " << num_vectors << " points with dimension " << d << std::endl;

    return std::make_tuple((float*) fileptr + 8, num_vectors, d);
}

#endif //PARSING