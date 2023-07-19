#!/bin/bash
cd ~/ParlayANN/algorithms/vamana
make 

P=/ssd1/data
G=/ssd1/results
BP=$P/bigann
BG=$G/bigann
for i in 1 2 8 12 24 48 96 144 192; do
    PARLAY_NUM_THREADS=$i nohup ./neighbors -R 64 -L 128 -file_type bin -data_type uint8 -dist_func Euclidian -base_path $BP/base.1B.u8bin.crop_nb_1000000 >> test_parallel.out
done
