#!/bin/bash
cd ~/ParlayANN/algorithms/vamana
make 

P=/ssd1/data/bigann
# ./neighbors -R 64 -L 128 -data_type uint8 -dist_func Euclidian -base_path $P/base.1B.u8bin.crop_nb_1000000

./neighbors -R 64 -L 128 -a 1.2 -data_type uint8 -dist_func Euclidian -query_path $P/query.public.10K.u8bin -gt_path $P/bigann-1M -res_path test.csv -base_path $P/base.1B.u8bin.crop_nb_1000000

