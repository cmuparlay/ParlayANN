#/bin/bash

./threshold_histo -graph_path /ssd1/data/bigann/vamana_R64_L128 -base_path /ssd1/data/bigann/base.1B.u8bin.crop_nb_1000000 

./threshold_histo -graph_path /ssd1/data/FB_ssnpp/vamana_64_500  -base_path /ssd1/data/FB_ssnpp/FB_ssnpp_database.u8bin.crop_nb_1000000

./threshold_histo -graph_path /ssd1/data/gist/gist-vamana -base_path /ssd1/data/gist/gist_base.fbin -data_type float -dist_func Euclidian

./threshold_histo -graph_path /ssd1/data/text2image1B/text2image1B-vamana -base_path /ssd1/data/text2image1B/base.1B.fbin.crop_nb_1000000 -data_type float -dist_func mips 

./threshold_histo -graph_path /ssd1/data/MCSPACEV1B/MCSPACE1B-vamana -base_path /ssd1/data/MCSPACEV1B/spacev1b_base.i8bin.crop_nb_1000000 -data_type int8 -dist_func Euclidian 

------------------------------------------------------------------------------------------------------------------------------------------------------------

./neighbors -R 32 -L 64 -alpha 1.2 -graph_outfile /ssd1/data/text2image1B/text2image1B-vamana -data_type float -dist_func mips -base_path /ssd1/data/text2image1B/base.1B.fbin.crop_nb_1000000

./neighbors -R 32 -L 64 -alpha 1.2 -graph_outfile /ssd1/data/MCSPACEV1B/MCSPACE1B-vamana -data_type int8 -dist_func Euclidian -base_path /ssd1/data/MSSPACEV1B/spacev1b_base.i8bin.crop_nb_1000000

./neighbors -R 32 -L 64 -alpha 1.2 -graph_outfile /ssd1/data/vecs/sift-vamana -data_type float -dist_func Euclidian -base_path /ssd1/data/vecs/sift/sift.fbin

--------------------------------------------------------------------------------------------------------------------------------------------------------------
./random_sample /ssd1/data/FB_ssnpp/FB_ssnpp_database.u8bin.crop_nb_10000000 100000 uint8 /ssd1/data/FB_ssnpp/FB_ssnpp_random

----------------------------------------------------------------------------------------------------------------------------------------------------------------

./compute_range_groundtruth -base_path /ssd1/data/FB_ssnpp/FB_ssnpp_database.u8bin.crop_nb_1000000 -query_path ssd1/data/FB_ssnpp/FB_ssnpp_random -data_type uint8 -rad 96237 -dist_func Euclidian -gt_path /ssd1/data/FB_ssnpp/FB_ssnpp_random_gt