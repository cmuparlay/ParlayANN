#!/bin/bash
cd ~/ParlayANN/algorithms/vamana
make 

P=/ssd1/data/bigann
# ./neighbors -R 64 -L 128 -data_type uint8 -dist_func Euclidian -base_path $P/base.1B.u8bin.crop_nb_1000000

# PARLAY_NUM_THREADS=1 
# ./neighbors -R 64 -L 128 -alpha 1.2 -two_pass 0 -data_type uint8 -dist_func Euclidian -query_path $P/query.public.10K.u8bin -gt_path $P/bigann-1M -res_path test.csv -base_path $P/base.1B.u8bin.crop_nb_1000000
# ./neighbors -R 64 -L 128 -alpha 1.2 -data_type uint8 -dist_func Euclidian -graph_path $P/graph-10M -query_path $P/query.public.10K.u8bin -gt_path $P/bigann-10M -res_path test.csv -base_path $P/base.1B.u8bin.crop_nb_10000000

Q=/ssd1/data/text2image1B

# ./neighbors -R 64 -L 128 -alpha 1.0 -data_type float -dist_func mips -sample_path $Q/query_rs_1000.fbin -query_path $Q/query.public.10K.fbin -gt_path $Q/text2image-10K-1M -res_path test.csv -base_path $Q/base.1B.fbin.crop_nb_1000000
# ./neighbors -R 64 -L 128 -alpha 1.0 -data_type float -dist_func Euclidian -query_path $Q/query.public.10K.fbin -gt_path $Q/text2image-10K-10M -res_path test.csv -base_path $Q/base.1B.fbin.crop_nb_10000000
# ./neighbors -R 64 -L 128 -alpha 1.0 -two_pass 1 -data_type float -dist_func Euclidian -sample_path $Q/query_rs_10000.fbin -query_path $Q/query.public.10K.fbin -gt_path $Q/text2image-10K-1M -res_path test.csv -base_path $Q/base.1B.fbin.crop_nb_1000000
# ./neighbors -R 64 -L 128 -alpha 1.0 -data_type float -dist_func Euclidian -sample_path $Q/query_rs_10000.fbin -query_path $Q/query.public.10K.fbin -gt_path $Q/text2image-10K-10M -res_path test.csv -base_path $Q/base.1B.fbin.crop_nb_10000000
# ./neighbors -R 64 -L 128 -alpha 1.0 -data_type float -dist_func Euclidian -sample_path $Q/query_rs_100000.fbin -query_path $Q/query.public.10K.fbin -gt_path $Q/text2image-10K-10M -res_path test.csv -base_path $Q/base.1B.fbin.crop_nb_10000000

./neighbors -R 64 -L 128 -alpha 1.0 -data_type float -two_pass 1 -dist_func mips -query_path $Q/query.public.10K.fbin -gt_path $Q/text2image-10K-1M -res_path test.csv -base_path $Q/base.1B.fbin.crop_nb_1000000
# ./neighbors -R 64 -L 128 -alpha 1.0 -data_type float -two_pass 0 -dist_func mips -query_path $Q/base_rs_10000.fbin -gt_path $Q/text2image-base-10K-1M -res_path test.csv -base_path $Q/base.1B.fbin.crop_nb_1000000
./neighbors -R 64 -L 128 -alpha 1.0 -data_type float -two_pass 1 -dist_func mips -query_path $Q/query_rs_10000_2.fbin -gt_path $Q/query-query-10K.fbin -res_path test.csv -base_path $Q/query_crop_1000000.fbin
