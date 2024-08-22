#include <iostream>
#include <algorithm>
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/io.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

// using namespace benchIO;
// *************************************************************
// Parsing code (should move to common?)
// *************************************************************

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

template<typename T>
void crop_file(char* iFile, int n, char* oFile){
  auto [fileptr, length] = mmapStringFromFile(iFile);

  int dim = *((int*) (fileptr+4));
  std::cout << "Writing " << n << " points with dimension " << dim << std::endl;
  parlay::sequence<int> preamble = {n, dim};

  T* data = (T*)(fileptr+8);
  std::ofstream writer;
  writer.open(oFile, std::ios::binary | std::ios::out);

  size_t bytes_to_write = n;
  bytes_to_write *= dim;
  bytes_to_write *= sizeof(T);

  writer.write((char *)(preamble.begin()), 2*sizeof(int));
  writer.write((char *) data, bytes_to_write);
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