#ifndef PARSING
#define PARSING

#include <algorithm>
#include <iostream>
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/sequence.h"
#include "parlay/slice.h"

#include "../utils/mmap.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <utility>

#include <fstream>
#include <string>
#include <vector>
/*
***********************************
*  Parsing functions for binary files
***********************************
*/

auto parse_uint8bin(const char* filename) {
  auto [fileptr, length] = mmapStringFromFile(filename);

  int num_vectors = *((int*)fileptr);
  int d = *((int*)(fileptr + 4));

  std::cout << "Detected " << num_vectors << " points with dimension " << d
            << std::endl;

  return std::make_tuple((uint8_t*)fileptr + 8, num_vectors, d);
}

auto parse_int8bin(const char* filename) {
  auto [fileptr, length] = mmapStringFromFile(filename);

  int num_vectors = *((int*)fileptr);
  int d = *((int*)(fileptr + 4));

  std::cout << "Detected " << num_vectors << " points with dimension " << d
            << std::endl;

  return std::make_tuple((int8_t*)fileptr + 8, num_vectors, d);
}

auto parse_fbin(const char* filename) {
  auto [fileptr, length] = mmapStringFromFile(filename);

  int num_vectors = *((int*)fileptr);
  int d = *((int*)(fileptr + 4));

  std::cout << "Detected " << num_vectors << " points with dimension " << d
            << std::endl;

  return std::make_tuple((float*)fileptr + 8, num_vectors, d);
}

// given a file of the form
// NUMVALS \n x1 \n x2 \n x3 \n ... \n xn, extract the numbers and put them in a
// vector and return it note that T must be a numeric type (because we use stoi
// on the line before casting to T)
template <typename T>
std::vector<T> extract_vector(const char* filename) {
  std::ifstream myfile(filename);

  std::string line;
  std::getline(myfile, line);
  int num_vals = std::stoi(line);
  std::vector<T> vals;
  T temp;
  for (int i = 0; i < num_vals; i++) {
    std::getline(myfile, line);
    temp = static_cast<T>(std::stoi(line));
    vals.push_back(temp);
  }

  return vals;
}

std::vector<std::string> extract_string_vector(const char* filename) {
  std::ifstream myfile(filename);

  std::string line;
  std::getline(myfile, line);
  int num_vals = std::stoi(line);
  std::vector<std::string> vals;
  for (int i = 0; i < num_vals; i++) {
    std::getline(myfile, line);
    vals.push_back(line);
  }

  return vals;
}

std::vector<std::pair<std::string, std::string>> extract_string_pair_vector(
   const char* filename) {
  std::ifstream myfile(filename);

  std::string line;
  std::string line2;
  std::getline(myfile, line);
  int num_vals = std::stoi(line);
  std::vector<std::pair<std::string, std::string>> vals;
  for (int i = 0; i < num_vals; i++) {
    std::getline(myfile, line, ' ');
    std::getline(myfile, line2);
    vals.push_back(std::make_pair(line, line2));
  }

  return vals;
}

// iterate by one the curr array
// Ex
// if capacities = 3 2 2 and
// curr = 1 0 1, then iterate would return
// 1 1 0 (because the rightmost one increases, hits capacity and so we carry the
// value over to the 2nd place)
bool iterate_multidim(std::vector<size_t>& capacities,
                      std::vector<size_t>& curr) {
  // first, make sure we can iterate
  bool isFull = true;
  for (size_t i = 0; i < capacities.size(); i++) {
    if (capacities[i] - 1 != curr[i]) {
      isFull = false;
    }
  }
  if (isFull) {
    return false;   // false means can no longer iterate
  }
  // if we can iterate, iterate
  int position = curr.size() - 1;
  curr[position] += 1;
  while (curr[position] == capacities[position]) {

    curr[position] = 0;
    curr[position - 1] += 1;
    position -= 1;
  }
  return true;
}

// //hold a point set
// struct DataWrapper {
//     float* fd;
//     int8_t* ind;
//     uint8_t* uid;
//     size_t n;
//     size_t d;
//     size_t type=-1; //0 means float, 1 means int8, 2 means uint8
//     size_t type_string = "";

//     DataWrapper(std::string filename, std::string tp) {
//         if (tp == "float") {
//             auto [tv, tn, td] = parse_fbin(input.c_str());
//             fd=tv;
//             n=tn;
//             d=td;
//             type=0;
//             type_string="float";
//         }
//         else if (tp == "uint8") {
//             auto [tv, tn, td] = parse_uint8_t(input.c_str());
//             fd=tv;
//             n=tn;
//             d=td;
//             type=2;
//             type_string="uint8_t";
//         }
//         else if (tp == "int8") {
//             auto [tv, tn, td] = parse_int8bin(input.c_str());
//             fd=tv;
//             n=tn;
//             d=td;
//             type=1;
//             type_string="int8_t";
//         }
//         else {
//             std::cout << "invalid type, aborting " << std::endl;
//             abort();
//         }

//     }

// };

#endif   // PARSING