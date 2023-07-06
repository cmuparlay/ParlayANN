#!/bin/bash
cd ~/ParlayANN/algorithms/HCNNG
make clean all

P=/ssd1/data
G=/home/landrum/outputs
O=/ssd1/results

BP=$P/bigann
BG=$G/bigann
./neighbors -a 1000 -R 3 -L 30 -b 1 -q $BP/query.public.10K.u8bin -c $BP/bigann-1M -res $G/bigann_HCNNG.csv -f bin -t uint8 -D Euclidian $BP/base.1B.u8bin.crop_nb_1000000
# ./neighbors -a 1000 -R 3 -L 30 -b 1 -o $BG/10M_HCNNG_30 -q $BP/query.public.10K.u8bin -c $BP/bigann-10M -res $O/bigann_HCNNG.csv -f bin -t uint8 $BP/base.1B.u8bin.crop_nb_10000000
# ./neighbors -a 1000 -R 3 -L 30 -b 1 -o $BG/100M_HCNNG_30 -q $BP/query.public.10K.u8bin -c $BP/bigann-100M -res $O/bigann_HCNNG.csv -f bin -t uint8 $BP/base.1B.u8bin.crop_nb_100000000
# ./neighbors -a 1000 -R 3 -L 30 -b 1 -o $BG/1B_HCNNG_30 -q $BP/query.public.10K.u8bin -c $BP/bigann-1B -res $O/bigann_HCNNG.csv -f bin -t uint8 $BP/base.1B.u8bin

SP=$P/MSSPACEV1B
SG=$G/MSSPACEV1B
./neighbors -a 1000 -R 3 -L 50 -b 1 -q $SP/query.i8bin -c $SP/msspacev-1M -res $G/spacev_HCNNG.csv -f bin -t int8 -D Euclidian $SP/spacev1b_base.i8bin.crop_nb_1000000
# ./neighbors -a 1000 -R 3 -L 50 -b 1 -o $SG/10M_HCNNG_50 -q $SP/query.i8bin -c $SP/msspacev-10M -res $O/spacev_HCNNG.csv -f bin -t int8 $SP/spacev1b_base.i8bin.crop_nb_10000000
# ./neighbors -a 1000 -R 3 -L 50 -b 1 -o $SG/100M_HCNNG_50 -q $SP/query.i8bin -c $SP/msspacev-100M -res $O/spacev_HCNNG.csv -f bin -t int8 $SP/spacev1b_base.i8bin.crop_nb_100000000
# ./neighbors -a 1000 -R 3 -L 50 -b 1 -o $SG/1B_HCNNG_50 -q $SP/query.i8bin -c $SP/msspacev-1B -res $O/spacev_HCNNG.csv -f bin -t int8 $SP/spacev1b_base.i8bin

TP=$P/text2image1B
TG=$G/text2image1B
./neighbors -a 1000 -R 3 -L 30 -b 1 -o $TG/1M_HCNNG_30 -q $TP/query.public.100K.fbin -c $TP/text2image-1M -res $TG/t2i_HCNNG.csv -f bin -t float -D mips $TP/base.1B.fbin.crop_nb_1000000
# ./neighbors -a 1000 -R 3 -L 30 -b 1 -o $TG/10M_HCNNG_30 -q $TP/query.public.100K.fbin -c $TP/text2image-10M -res $TG/t2i_HCNNG.csv -f bin -t float -D 1 $TP/base.1B.fbin.crop_nb_10000000
# ./neighbors -a 1000 -R 3 -L 30 -b 1 -o $TG/100M_HCNNG_30 -q $TP/query.public.100K.fbin -c $TP/text2image-100M -res $TG/t2i_HCNNG.csv -f bin -t float -D 1 $TP/base.1B.fbin.crop_nb_100000000
# ./neighbors -a 1000 -R 3 -L 30 -b 1 -o $TG/1B_HCNNG_30 -q $TP/query.public.100K.fbin -c $TP/text2image-1B -res $TG/t2i_HCNNG.csv -f bin -t float -D 1 $TP/base.1B.fbin
