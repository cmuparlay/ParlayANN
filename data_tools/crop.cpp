#include <iostream>
#include <algorithm>
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/io.h"
#include "../algorithms/bench/common/parse_command_line.h"
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
void crop_file(char* iFile, int min, int max, char* oFile){
  auto [fileptr, length] = mmapStringFromFile(iFile);
  int n = max - min;
  int dim = *((int*) (fileptr+4));
  std::cout << "Writing " << n << " points with dimension " << dim << std::endl;
  parlay::sequence<int> preamble = {n, dim};

  T* data = (T*)(fileptr+8 + sizeof(T)*min);
  std::ofstream writer;
  writer.open(oFile, std::ios::binary | std::ios::out);

  writer.write((char *)(preamble.begin()), 2*sizeof(int));
  writer.write((char *) data, dim*static_cast<size_t>(n)*sizeof(T));
  writer.close();
}

int main(int argc, char* argv[]) {
  commandLine P(argc,argv,
  "[-file_path <b>] "
      "[-data_type <d>] [-min <min>] [-max <max>] [-write_path <outfile>]");

  char* iFile = P.getOptionValue("-file_path");
  char* oFile = P.getOptionValue("-write_path");
  int min = P.getOptionIntValue("-min", 0);
  char* vectype = P.getOptionValue("-data_type");
  int max = P.getOptionIntValue("-max", 0);
  

  std::string tp = std::string(vectype);
  if(min >= max){
    std::cout << "ERROR: min " << min << " less than or equal to max " << max << std::endl;
    abort();
  }

  if(tp == "float") crop_file<float>(iFile, min, max, oFile);
  else if(tp == "uint8") crop_file<uint8_t>(iFile, min, max, oFile);
  else if(tp == "int8") crop_file<int8_t>(iFile, min, max, oFile);
  else{
    std::cout << "Invalid type, specify float, uint8, or int8" << std::endl;
  }

  return 0;
}