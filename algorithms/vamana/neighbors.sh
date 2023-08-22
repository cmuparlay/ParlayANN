#!/bin/bash
cd ~/ParlayANN/algorithms/vamana
make 

P=/ssd1/data/vecs/sift
./neighbors -R 64 -L 128 -file_type bin -data_type float -dist_func Euclidian -base_path $P/sift.fbin

