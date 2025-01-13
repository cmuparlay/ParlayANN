#/bin/bash
make

#   bigann_early_stopping_rad = 38000;
#   ssnpp_early_stopping_rad = 200000;
#   msturing_early_stopping_rad = 1.1;
#   gist_early_stopping_rad = 2.0; 
#   deep_early_stopping_rad = .4;
# PARLAY_NUM_THREADS=1 ./command

echo "SimSearchNet"
P=/ssd1/data/FB_ssnpp
PARLAY_NUM_THREADS=1 ./range -alpha 1.0 -R 64 -L 128 -r 96237 -early_stopping_radius 200000 -parlay -base_path $P/FB_ssnpp_database.u8bin.crop_nb_1000000 -query_path $P/FB_ssnpp_public_queries.u8bin -gt_path $P/ssnpp-1M -data_type uint8 -dist_func Euclidian > SSNPP_doublebeamsame_sp_one_thread.out
echo ""
echo ""
echo "BIGANN"
R=/ssd1/data/bigann
PARLAY_NUM_THREADS=1 ./range -alpha 1.15 -R 64 -L 128 -r 10000 -early_stopping_radius 30000 -base_path $R/base.1B.u8bin.crop_nb_1000000 -data_type uint8 -dist_func Euclidian -query_path $R/query.public.10K.u8bin  -gt_path $R/range_gt_1M_10000 > bigann_doublebeamsame_sp_one_thread.out
echo ""
echo ""
echo "MSTuring"
S=/ssd1/data/MSTuringANNS
PARLAY_NUM_THREADS=1 ./range -alpha 1.15 -R 64 -L 128 -r 0.3 -early_stopping_radius 1.4 -base_path $S/base1b.fbin.crop_nb_1000000 -data_type float -dist_func Euclidian -query_path $S/query100K.fbin  -gt_path $S/range_gt_1M_100K_.3 > msturing_doublebeamsame_sp_one_thread.out
echo ""
echo ""
echo "GIST"
G=/ssd1/data/gist
PARLAY_NUM_THREADS=1 ./range -alpha 1.15 -R 64 -L 128 -r 0.5 -early_stopping_radius 2.5 -base_path $G/gist_base.fbin -data_type float -dist_func Euclidian -query_path $G/gist_query.fbin -gt_path $G/range_gt_1M_.5 > gist_one_thread.out
echo ""
echo ""
echo "DEEP"
H=/ssd1/data/deep1b 
PARLAY_NUM_THREADS=1 ./range -alpha 1.15 -R 64 -L 128 -r 0.02 -early_stopping_radius 0.6 -base_path $H/base.1B.fbin.crop_nb_1000000 -data_type float -dist_func Euclidian -query_path $H/query.public.10K.fbin -gt_path $H/range_gt_1M_.02 > deep_doublebeamsame_sp_one_thread.out
