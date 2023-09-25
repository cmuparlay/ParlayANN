#include <iostream>
#include <algorithm>
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/io.h"
#include "../algorithms/utils/mmap.h"

template<typename T>
void crop_file(char* iFile, int n, char* oFile){
  auto [fileptr, length] = mmapStringFromFile(iFile);

  int dim = *((int*) (fileptr+4));
  std::cout << "Writing " << n << " points with dimension " << dim << std::endl;
  parlay::sequence<int> preamble = {n, dim};

  T* data = (T*)(fileptr+8);
  std::ofstream writer;
  writer.open(oFile, std::ios::binary | std::ios::out);

  writer.write((char *)(preamble.begin()), 2*sizeof(int));
  writer.write((char *) data, dim*n*sizeof(T));
  writer.close();
}

int main(int argc, char* argv[]) {
  if (argc != 5) {
    std::cout << "usage: crop <base> <num_points_to_crop> <tp> <oF>" << std::endl;
    return 1;
  }
  

  int n = atoi(argv[2]);

  std::string tp = std::string(argv[3]);

  if(tp == "float") crop_file<float>(argv[1], n, argv[4]);
  else if(tp == "uint8") crop_file<uint8_t>(argv[1], n, argv[4]);
  else if(tp == "int8") crop_file<int8_t>(argv[1], n, argv[4]);
  else{
    std::cout << "Invalid type, specify float, uint8, or int8" << std::endl;
  }

  return 0;
}