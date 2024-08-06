./rangeGraph -base_path $P/base.1B.u8bin.crop_nb_1000000 -query_path $P/query.public.10K.u8bin -cluster_outfile $P/range_graph_1M_100 -res_path bigann_ivf_range_graph.csv -gt_path $P/range_gt_1M_10000 -data_type uint8 -num_clusters 100 -dist_func Euclidian -k 10 -r 10000

./range -base_path $P/base.1B.u8bin.crop_nb_1000000 -query_path $P/query.public.10K.u8bin -cluster_outfile $P/range_graph_1M_100 -res_path bigann_ivf_range_graph.csv -gt_path $P/range_gt_1M_10000 -data_type uint8 -num_clusters 100 -dist_func Euclidian -k 10 -r 10000

./range -base_path $P/base.1B.u8bin.crop_nb_10000000 -query_path $P/query.public.10K.u8bin -cluster_outfile $P/range_graph_10M_100 -res_path bigann_ivf_range_graph.csv -gt_path $P/range_gt_10M_10000 -data_type uint8 -num_clusters 100 -dist_func Euclidian -k 10 -r 10000

./rangeGraph -base_path $P/base.1B.u8bin.crop_nb_10000000 -query_path $P/query.public.10K.u8bin -cluster_outfile $P/range_graph_10M_100 -res_path bigann_ivf_range_graph.csv -gt_path $P/range_gt_10M_10000 -data_type uint8 -num_clusters 100 -dist_func Euclidian -k 10 -r 10000

./range -base_path $P/base.1B.u8bin.crop_nb_100000000 -query_path $P/query.public.10K.u8bin -cluster_outfile $P/range_graph_100M_100 -res_path bigann_ivf_range_graph_10M.csv -gt_path $P/range_gt_100M_10000 -data_type uint8 -num_clusters 100000 -dist_func Euclidian -k 10 -r 10000

./rangeGraph -base_path $P/base.1B.u8bin.crop_nb_100000000 -query_path $P/query.public.10K.u8bin -cluster_outfile $P/range_graph_100M_100 -res_path bigann_ivf_range_graph_10M.csv -gt_path $P/range_gt_100M_10000 -data_type uint8 -num_clusters 100000 -dist_func Euclidian -k 10 -r 10000

nohup ./rangeGraph -base_path $P/base.1B.u8bin.crop_nb_10000000 -query_path $P/query.public.10K.u8bin -cluster_outfile $P/range_graph_10M_100 -res_path bigann_ivf_range_graph.csv -gt_path $P/range_gt_10M_10000 -data_type uint8 -num_clusters 10000 -dist_func Euclidian -k 10 -r 10000 > rangeGraph_10M_10000_sift.out

./compute_range_groundtruth -base_path $P/base.1B.u8bin.crop_nb_100000000 -query_path $P/query.public.10K.u8bin -data_type uint8 -k 100 -dist_func Euclidian -gt_path $P/range_gt_100M_10000

nohup ./range -base_path $P/base.1B.u8bin.crop_nb_10000000 -query_path $P/query.public.10K.u8bin -cluster_outfile $P/range_graph_10M_100 -res_path bigann_ivf_range_graph.csv -gt_path $P/range_gt_10M_10000 -data_type uint8 -num_clusters 10000 -dist_func Euclidian -k 10 -r 10000 > range_10M_10000_sift.out

nohup ./rangeGraph -base_path $P/base.1B.u8bin.crop_nb_100000000 -query_path $P/query.public.10K.u8bin -cluster_outfile $P/range_graph_10M_100 -res_path bigann_ivf_range_graph.csv -gt_path $P/range_gt_100M_10000 -data_type uint8 -num_clusters 10000 -dist_func Euclidian -k 10 -r 10000 > rangeGraph_100M_sift.out

nohup ./range -base_path $P/base.1B.u8bin.crop_nb_100000000 -query_path $P/query.public.10K.u8bin -cluster_outfile $P/range_graph_10M_100 -res_path bigann_ivf_range_graph.csv -gt_path $P/range_gt_100M_10000 -data_type uint8 -num_clusters 100 -dist_func Euclidian -k 10 -r 10000 > range_100M_sift.out

nohup ./rangeGraph -base_path $P/base.1B.u8bin.crop_nb_1000000 -query_path $P/public_query_gt100.bin -cluster_outfile $P/range_graph_10M_100 -res_path msspacev_ivf_range_graph.csv -gt_path $P/range_gt_100M_10000 -data_type int8 -num_clusters 10000 -dist_func Euclidian -k 10 -r 10000 > rangeGraph_100M_sift.out
