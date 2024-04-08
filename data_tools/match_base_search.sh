./match_base_search -R 32 -L 64 -alpha 1.2 -graph_path /ssd1/data/bigann/vamana_R64_L128 -query_path /ssd1/data/bigann/query.public.10K.u8bin -gt_path /ssd1/data/sift/bigann-1M -res_path /ssd1/data/bigann/bigann-1M.csv -data_type uint8  -dist_func Euclidian -base_path /ssd1/data/bigann/base.1B.u8bin.crop_nb_1000000

./neighbors -R 32 -L 64 -alpha 1.2 -graph_outfile /ssd1/data/FB_ssnpp/vamana_64_500_100M -data_type uint8 -dist_func Euclidian -base_path /ssd1/data/FB_ssnpp/FB_ssnpp_database.u8bin.crop_nb_100000000


./neighbors -R 32 -L 64 -alpha 1.2 -graph_path /ssd1/data/FB_ssnpp/vamana_64_500 -query_path /ssd1/data/FB_ssnpp/ssnpp-1M-nonzero.u8bin -gt_path /ssd1/data/FB_ssnpp/ssnpp-nn-1M -res_path /ssd1/data/FB_ssnpp/ssnpp-1M.csv -data_type uint8  -dist_func Euclidian -base_path /ssd1/data/FB_ssnpp/FB_ssnpp_database.u8bin.crop_nb_1000000

./neighbors -R 32 -L 64 -alpha 1.2 -graph_path /ssd1/data/FB_ssnpp/vamana_64_500_10M -query_path /ssd1/data/FB_ssnpp/ssnpp-10M-nonzero.u8bin -gt_path /ssd1/data/FB_ssnpp/ssnpp-nn-10M -res_path /ssd1/data/FB_ssnpp/ssnpp-10M.csv -data_type uint8  -dist_func Euclidian -base_path /ssd1/data/FB_ssnpp/FB_ssnpp_database.u8bin.crop_nb_10000000
