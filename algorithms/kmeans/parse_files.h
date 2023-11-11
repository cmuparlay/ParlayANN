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

#include<vector>
#include<fstream>
#include<string>
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

//given a file of the form
//NUMVALS \n x1 \n x2 \n x3 \n ... \n xn, extract the numbers and put them in a vector and return it
//note that T must be a numeric type (because we use stoi on the line before casting to T)
template<typename T>
std::vector<T> extract_vector(const char* filename) {
    std::ifstream myfile(filename);
    
    std::string line;
    std::getline(myfile,line);
    int num_vals = std::stoi(line);
    std::vector<T> vals;
    T temp;
    for (int i = 0; i < num_vals; i++) {
        std::getline(myfile,line);
        temp = static_cast<T>(std::stoi(line));
        vals.push_back(temp);
    }

    return vals;

}

std::vector<std::string> extract_string_vector(const char* filename) {
    std::ifstream myfile(filename);
    
    std::string line;
    std::getline(myfile,line);
    int num_vals = std::stoi(line);
    std::vector<std::string> vals;
    for (int i = 0; i < num_vals; i++) {
        std::getline(myfile,line);
        vals.push_back(line);
    }

    return vals;

}

std::vector<std::pair<std::string,std::string>> extract_string_pair_vector(const char* filename) {
    std::ifstream myfile(filename);
    
    std::string line;
    std::string line2;
    std::getline(myfile,line);
    int num_vals = std::stoi(line);
    std::vector<std::pair<std::string,std::string>> vals;
    for (int i = 0; i < num_vals; i++) {
        std::getline(myfile,line, ' ');
        std::getline(myfile,line2);
        vals.push_back(std::make_pair(line,line2));
    }

    return vals;

}

#endif //PARSING