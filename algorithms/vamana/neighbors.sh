#!/bin/bash
cd ~/ParlayANN/algorithms/vamana
make 

P=/ssd1/data/bigann
# ./neighbors -R 64 -L 128 -alpha 1.2 -data_type uint8 -dist_func Euclidian -base_path $P/base.1B.u8bin.crop_nb_1000000

# PARLAY_NUM_THREADS=1 
# ./neighbors -R 64 -L 128 -alpha 1.15 -two_pass 0 -data_type uint8 -dist_func Euclidian -query_path $P/query.public.10K.u8bin -gt_path $P/bigann-1M -res_path bigann.csv -base_path $P/base.1B.u8bin.crop_nb_1000000
# ./neighbors -R 64 -L 128 -alpha 1.15 -two_pass 0 -data_type uint8 -dist_func Euclidian -query_path $P/query.public.10K.u8bin -gt_path $P/bigann-10M -res_path bigann.csv -base_path $P/base.1B.u8bin.crop_nb_10000000
# ./neighbors -R 64 -L 128 -alpha 1.15 -two_pass 0 -data_type uint8 -dist_func Euclidian -query_path $P/query.public.10K.u8bin -gt_path $P/bigann-100M -res_path bigann.csv -base_path $P/base.1B.u8bin.crop_nb_100000000

Q=/ssd1/data/text2image1B
# ./neighbors -R 64 -L 128 -alpha 1.0 -data_type float -two_pass 0 -dist_func mips -query_path $Q/query.public.10K.fbin -gt_path $Q/text2image-10K-1M -res_path test.csv -base_path $Q/base.1B.fbin.crop_nb_1000000

V=/ssd1/data/MSSPACEV1B
# ./neighbors -R 64 -L 128 -alpha 1.2 -two_pass 0 -data_type int8 -dist_func Euclidian -query_path $V/query.i8bin -gt_path $V/msspacev-1M -res_path test.csv -base_path $V/spacev1b_base.i8bin.crop_nb_1000000


# ./neighbors -R 64 -L 128 -a 1.2 -data_type uint8 -dist_func Euclidian -query_path /ssd1/data/bigann/query.public.10K.u8bin -gt_path /ssd1/data/bigann/bigann-1M -res_path test.csv -base_path /ssd1/data/bigann/base.1B.u8bin.crop_nb_1000000

S=/ssd1/data/FB_ssnpp
# make
# ./neighbors -R 128 -L 400 -alpha 1.0 -data_type uint8 -two_pass 1 -dist_func Euclidian -k 10 -query_path $S/ssnpp-1M-nonzero.u8bin -gt_path $S/ssnpp-nn-1M -res_path ssnpp.csv -base_path $S/FB_ssnpp_database.u8bin.crop_nb_1000000
# ./neighbors -R 128 -L 400 -alpha 1.0 -data_type uint8 -two_pass 1 -dist_func Euclidian -k 10 -query_path $S/ssnpp-10M-nonzero.u8bin -gt_path $S/ssnpp-nn-10M -res_path ssnpp.csv -base_path $S/FB_ssnpp_database.u8bin.crop_nb_10000000
# ./neighbors -R 128 -L 400 -alpha 1.0 -data_type uint8 -two_pass 1 -dist_func Euclidian -k 10 -query_path $S/ssnpp-100M-nonzero.u8bin -gt_path $S/ssnpp-nn-100M -res_path ssnpp.csv -base_path $S/FB_ssnpp_database.u8bin.crop_nb_100000000

T=/ssd1/data/gist
./neighbors -R 64 -L 128 -alpha 1.15 -two_pass 0 -data_type float -dist_func Euclidian -query_path $T/gist_query.fbin -gt_path $T/gist-1M -base_path $T/gist_base.fbin
