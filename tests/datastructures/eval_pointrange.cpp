#include <iostream>
#include <algorithm>
#include "parlay/parallel.h"
#include "parlay/primitives.h"
#include "parlay/io.h"
#include "parlay/random.h"
#include "utils/dynamic_point_range.h"
#include "utils/euclidian_point.h"




int main(int argc, char* argv[]) {
  commandLine P(argc,argv,
      "[-base_path <outfile>]");

  char* iFile = P.getOptionValue("-base_path");


  using PR = DynamicPointRange<uint8_t, Euclidian_Point<uint8_t>, unsigned int>;
  using PointRange = PR::PointRange;
  // PointRange Points = PointRange(iFile);
  // PR test = PR(iFile);

  size_t n = 10000000;
  size_t init_inserts = 1000000;
  
  std::ifstream reader(iFile);
  assert(reader.is_open());

  //read num points and max degree
  unsigned int num_points;
  unsigned int d;   
  reader.read((char*)(&num_points), sizeof(unsigned int));
  reader.read((char*)(&d), sizeof(unsigned int));
  std::cout << "Detected " << num_points << " points with dimension " << d << std::endl;
  PR test = PR(6000000, d);

  unsigned int index=0;
  unsigned int BLOCK_SIZE = 1000000;

  using T = uint8_t;

  // auto modifier = [&] () {
  PointRange& Points = test.Get_PointRange();
  while(index < num_points){
    size_t floor = index;
    size_t ceiling = index+BLOCK_SIZE <= n ? index+BLOCK_SIZE : n;
    T* data_start = new T[(ceiling-floor)*d];
    reader.read((char*)(data_start), sizeof(T)*(ceiling-floor)*d);
    T* data_end = data_start + (ceiling-floor)*d;
    parlay::slice<T*, T*> data = parlay::make_slice(data_start, data_end);
    parlay::sequence<unsigned int> ids = parlay::tabulate(ceiling-floor, [&] (size_t i) {return static_cast<unsigned int>(floor+i);});
    Points.insert(data, ids);
    Points.check();
    parlay::sequence<unsigned int> to_delete = parlay::tabulate((ceiling-floor)/2, [&] (size_t i){return static_cast<unsigned int>(floor+i);});
    Points.delete_points(to_delete);
    Points.check();
    index = ceiling;
  }
  test.Release_PointRange();
  // };

  // auto reader = [&] (){
  //   auto wp = test.Get_PointRange_Read_Only();
  //   //choose random ids to read if they are live
  //   test.Release_PointRange(wp.worker_id);
  // }; 

  // parlay::par_do([&] {deleter();}, [&] {holder();});
  


  return 0;
}
