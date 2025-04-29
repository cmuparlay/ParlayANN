#/bin/bash

echo "BIGANN"

P=/ssd1/data/bigann
nohup ./ivfGraph -base_path $P/base.1B.u8bin.crop_nb_1000000 -query_path $P/query.public.10K.u8bin -cluster_outfile $P/kmeans_graph_1M_100  -gt_path $P/bigann-1M -data_type uint8 -num_clusters 100 -dist_func Euclidian -k 10 > graph_bigann_1M.out

