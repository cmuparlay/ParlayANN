#include <iostream>
#include <algorithm>
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/io.h"

// convert from .bvec file to .u8bin file

auto convert(const char* infile, const char* outfile) {
  auto str = parlay::chars_from_file(infile);
  int dims = *((int *) str.data());
  int n = str.size()/(dims+4);
  std::cout << "n = " << n << " d = " << dims << std::endl;
  auto vects = parlay::tabulate(n, [&] (int i) {
		     return parlay::to_sequence(str.cut(4 + i * (4 + dims), (i+1) * (4 + dims)));});
  parlay::sequence<char> head(8);
  *((int *) head.data()) = n;
  *(((int *) head.data()) + 1) = dims;
  auto strout = parlay::append(head, parlay::flatten(vects));
  parlay::chars_to_file(strout, outfile);
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    std::cout << "usage: bvec_to_u8bin <infile> <outfile>" << std::endl;
    return 1;
  }
  convert(argv[1], argv[2]);
  return 0;
}
