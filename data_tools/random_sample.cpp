#include <iostream>
#include <algorithm>
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/io.h"
#include "parlay/random.h"
#include "utils/mmap.h"

#include <random>





template<typename T>
void random_sample(char* iFile, int n, char* oFile){
    auto [fileptr, length] = mmapStringFromFile(iFile);

    int fsize = *((int*) fileptr);
    int dim = *((int*) (fileptr+4));
    std::cout << "Writing " << n << " points with dimension " << dim << std::endl;
    parlay::sequence<int> preamble = {n, dim};

    parlay::random_generator gen;
    std::uniform_int_distribution<long> dis(0, fsize - 1);
    auto indices = parlay::tabulate(n, [&](size_t i) {
        auto r = gen[i];
        return dis(r);
    });

    T* start = (T*)(fileptr + 8);

    auto to_flatten = parlay::tabulate(n, [&] (size_t i){
        parlay::sequence<T> data;
        for(int j=0; j<dim; j++){
            data.push_back(*(start + dim*indices[i] + j));
        }
        return data;
    });

    auto data = parlay::flatten(to_flatten);

    std::ofstream writer;
    writer.open(oFile, std::ios::binary | std::ios::out);
    writer.write((char *)(preamble.begin()), 2*sizeof(int));
    writer.write((char *)(data.begin()), dim*n*sizeof(T));
    writer.close();
}

int main(int argc, char* argv[]) {
  if (argc != 5) {
    std::cout << "usage: random_sample <base> <num_points_to_crop> <tp> <oF>" << std::endl;
    return 1;
  }
  

  int n = atoi(argv[2]);

  std::string tp = std::string(argv[3]);

  if(tp == "float") random_sample<float>(argv[1], n, argv[4]);
  else if(tp == "uint8") random_sample<uint8_t>(argv[1], n, argv[4]);
  else if(tp == "int8") random_sample<int8_t>(argv[1], n, argv[4]);
  else{
    std::cout << "Invalid type, specify float, uint8, or int8" << std::endl;
  }

  return 0;
}