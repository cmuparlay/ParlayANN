#!/bin/bash
cd ~/ParlayANN/algorithms/vamana
make 

P=/ssd1/data/bigann
# ./neighbors -R 64 -L 128 -data_type uint8 -dist_func Euclidian -base_path $P/base.1B.u8bin.crop_nb_1000000

# PARLAY_NUM_THREADS=1 
./neighbors -R 64 -L 128 -alpha 1.2 -data_type uint8 -dist_func Euclidian -query_path $P/query.public.10K.u8bin -gt_path $P/bigann-1M -res_path test.csv -base_path $P/base.1B.u8bin.crop_nb_1000000
# ./neighbors -R 64 -L 128 -alpha 1.2 -data_type uint8 -dist_func Euclidian -graph_path $P/graph-10M -query_path $P/query.public.10K.u8bin -gt_path $P/bigann-10M -res_path test.csv -base_path $P/base.1B.u8bin.crop_nb_10000000

# Q=/ssd1/data/text2image1B
# ./neighbors -R 64 -L 128 -alpha 1.0 -data_type float -dist_func mips -query_path $Q/query.public.100K.fbin -gt_path $Q/text2image-1M -res_path test.csv -base_path $Q/base.1B.fbin.crop_nb_1000000

# ./neighbors -R 64 -L 128 -a 1.2 -data_type uint8 -dist_func Euclidian -query_path /ssd1/data/bigann/query.public.10K.u8bin -gt_path /ssd1/data/bigann/bigann-1M -res_path test.csv -base_path /ssd1/data/bigann/base.1B.u8bin.crop_nb_1000000
