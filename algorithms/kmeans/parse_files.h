#ifndef PARSING
#define PARSING

#include <iostream>
#include <algorithm>
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"
#include "parlay/slice.h"
//#include "../parse_command_line.h"
//#include "types.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <utility>
#include <unistd.h>

// returns a pointer and a length
std::pair<char*, size_t> mmapStringFromFile(const char* filename) {
  struct stat sb;
  int fd = open(filename, O_RDONLY);
  if (fd == -1) {
    perror("open");
    exit(-1);
  }
  if (fstat(fd, &sb) == -1) {
    perror("fstat");
    exit(-1);
  }
  if (!S_ISREG(sb.st_mode)) {
    perror("not a file\n");
    exit(-1);
  }
  char* p =
      static_cast<char*>(mmap(0, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0));
  if (p == MAP_FAILED) {
    perror("mmap");
    exit(-1);
  }
  if (close(fd) == -1) {
    perror("close");
    exit(-1);
  }
  size_t n = sb.st_size;
  return std::make_pair(p, n);
}

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

    // parlay::sequence<point<uint8_t>> points(num_vectors);
    
    // parlay::parallel_for(0, num_vectors, [&] (size_t i) {
    //     points[i].id = i; 

    //     uint8_t* start = (uint8_t*)(fileptr + 8 + i*d); //8 bytes at the start for size + dimension
    //     uint8_t* end = start + d;
    //     points[i].coordinates = parlay::make_slice(start, end);
    // });

    // is passing this pointer without copying safe? I think yes but not sure
    return std::make_tuple((uint8_t*)fileptr + 8, num_vectors, d);
}

auto parse_int8bin(const char* filename){
    auto [fileptr, length] = mmapStringFromFile(filename);

    int num_vectors = *((int*) fileptr);
    int d = *((int*) (fileptr+4));
 
    std::cout << "Detected " << num_vectors << " points with dimension " << d << std::endl;
    // parlay::sequence<point<int8_t>> points(num_vectors);

    // parlay::parallel_for(0, num_vectors, [&] (size_t i) {
    //     points[i].id = i; 

    //     int8_t* start = (int8_t*)(fileptr + 8 + i*d); //8 bytes at the start for size + dimension
    //     int8_t* end = start + d;
    //     points[i].coordinates = parlay::make_slice(start, end);
    // });

    return std::make_tuple((int8_t*)fileptr + 8, num_vectors, d);
}

auto parse_fbin(const char* filename){
    auto [fileptr, length] = mmapStringFromFile(filename);

    int num_vectors = *((int*) fileptr);
    int d = *((int*) (fileptr+4));

    std::cout << "Detected " << num_vectors << " points with dimension " << d << std::endl;

    // parlay::sequence<point<float>> points(num_vectors);

    // parlay::parallel_for(0, num_vectors, [&] (size_t i) {
    //     points[i].id = i; 

    //     float* start = (float*)(fileptr + 8 + 4*i*d); //8 bytes at the start for size + dimension
    //     float* end = start + d;
    //     points[i].coordinates = parlay::make_slice(start, end);
    // });

    return std::make_tuple((float*) fileptr + 8, num_vectors, d);
}

// the below filetypes are not trivially mmap-able to a flat array

/* auto parse_fvecs(const char* filename) {
  auto [fileptr, length] = mmapStringFromFile(filename);

  // Each vector is 4 + 4*d bytes.
  // * first 4 bytes encode the dimension (as an integer)
  // * next d values are floats representing vector components
  // See http://corpus-texmex.irisa.fr/ for more details.

  int d = *((int*)fileptr);

  size_t vector_size = 4 + 4*d;
  size_t num_vectors = length / vector_size;
  // std::cout << "Num vectors = " << num_vectors << std::endl;

  parlay::sequence<point<float>> points(num_vectors);

  parlay::parallel_for(0, num_vectors, [&] (size_t i) {
    size_t offset_in_bytes = vector_size * i + 4;  // skip dimension
    float* start = (float*)(fileptr + offset_in_bytes);
    float* end = start + d;
    points[i].id = i; 
    points[i].coordinates = parlay::make_slice(start, end);  
  });

  return points;
}

auto parse_bvecs(const char* filename) {
  auto [fileptr, length] = mmapStringFromFile(filename);
  // Each vector is 4 + d bytes.
  // * first 4 bytes encode the dimension (as an integer)
  // * next d values are unsigned chars representing vector components
  // See http://corpus-texmex.irisa.fr/ for more details.

  int d = *((int*)fileptr);
  size_t vector_size = 4 + d;
  size_t num_vectors = length / vector_size;

  parlay::sequence<point<uint8_t>> points(num_vectors);

  parlay::parallel_for(0, num_vectors, [&] (size_t i) {
    size_t offset_in_bytes = vector_size * i + 4;  // skip dimension
    uint8_t* start = (uint8_t*)(fileptr + offset_in_bytes);
    uint8_t* end = start + d;
    points[i].id = i; 
    points[i].coordinates = parlay::make_slice(start, end);  
  });

  return points;
} */

#endif //PARSING