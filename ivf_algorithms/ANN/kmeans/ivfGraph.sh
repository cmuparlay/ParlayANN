nohup ./ivfGraph -base_path $P/base.1B.u8bin.crop_nb_1000000 -query_path $P/query.public.10K.u8bin -cluster_outfile $P/kmeans_graph_1M_100 -res_path bigann_ivf_graph.csv -gt_path $P/bigann-1M -data_type uint8 -num_clusters 100 -dist_func Euclidian -k 10 >ivfGraph_1M_bigann

./ivf -base_path $P/base.1B.u8bin.crop_nb_1000000 -query_path $P/query.public.10K.u8bin -cluster_outfile $P/kmeans_graph_1M_100 -res_path bigann_ivf_comp.csv -gt_path $P/bigann-1M -data_type uint8 -num_clusters 100 -dist_func Euclidian -k 10
    

./ivf -base_path $P/base.1B.u8bin.crop_nb_10000000 -query_path $P/query.public.10K.u8bin -cluster_outfile $P/kmeans_graph_10M_10000 -res_path bigann_ivf_comp_10M.csv -gt_path $P/bigann-10M -data_type uint8 -num_clusters 10000 -dist_func Euclidian -k 10

./ivfGraph -base_path $P/base.1B.u8bin.crop_nb_10000000 -query_path $P/query.public.10K.u8bin -cluster_outfile $P/kmeans_graph_10M_10000 -res_path bigann_ivf_graph_10M.csv -gt_path $P/bigann-10M -data_type uint8 -num_clusters 10000 -dist_func Euclidian -k 10


./ivf -base_path $P/base.1B.u8bin.crop_nb_100000000 -query_path $P/query.public.10K.u8bin -cluster_outfile $P/kmeans_graph_100M_100000 -res_path bigann_ivf_comp_100M.csv -gt_path $P/bigann-100M -data_type uint8 -num_clusters 100000 -dist_func Euclidian -k 10

./ivfGraph -base_path $P/base.1B.u8bin.crop_nb_100000000 -query_path $P/query.public.10K.u8bin -cluster_outfile $P/kmeans_graph_100M_100000 -res_path bigann_ivf_graph_100M.csv -gt_path $P/bigann-100M -data_type uint8 -num_clusters 10000 -dist_func Euclidian -k 10

# 1M
nohup ./ivfGraph -base_path $P/spacev1b_base.i8bin.crop_nb_1000000 -query_path $P/private_query_30k.bin -cluster_outfile $P/kmeans_graph_1M_100 -res_path msspacev1b_ivf_graph.csv -gt_path $P/private-gt-1M -data_type int8 -num_clusters 1000 -dist_func Euclidian -k 10 > ivfGraph_1M_MSSPACEV1B

# 10M

nohup ./ivf -base_path $P/FB_ssnpp_database.u8bin.crop_nb_10000000 -query_path $P/ssnpp-10M-nonzero.u8bin  -cluster_outfile $P/kmeans_graph_1M_100 -res_path msspacev1b_ivf_graph.csv -gt_path $P/ssnpp-nn-10M -data_type uint8 -num_clusters 10000 -dist_func Euclidian -k 10 > ivf_10M_ssnpp

nohup ./ivfGraph -base_path $P/FB_ssnpp_database.u8bin.crop_nb_10000000 -query_path $P/ssnpp-10M-nonzero.u8bin  -cluster_outfile $P/kmeans_graph_1M_100 -res_path msspacev1b_ivf_graph.csv -gt_path $P/ssnpp-nn-10M -data_type uint8 -num_clusters 10000 -dist_func Euclidian -k 10 > ivfGraph_10M_ssnpp
nohup ./ivfGraph -base_path $P/spacev1b_base.i8bin.crop_nb_10000000 -query_path $P/private_query_30k.bin -cluster_outfile $P/kmeans_graph_10M_100 -res_path msspacev1b_ivf_graph.csv -gt_path $P/private-gt-10M -data_type int8 -num_clusters 10000 -dist_func Euclidian -k 10 > ivfGraph_10M_MSSPACEV1B


nohup ./ivfGraph -base_path $P/base.1B.u8bin -query_path $P/query.public.10K.u8bin  -cluster_outfile $P/kmeans_graph_1M_100 -res_path bigann_ivf_graph.csv -gt_path $P/GT.public.1B.ibin -data_type uint8 -num_clusters 1000000 -dist_func Euclidian -k 10 > ivfGraph_1B_bigann

nohup ./ivfGraph -base_path $P/base.1B.u8bin -query_path $P/query.public.10K.u8bin  -cluster_outfile $P/kmeans_graph_1M_100 -res_path bigann_ivf_graph.csv -gt_path $P/GT.public.1B.ibin -data_type uint8 -num_clusters 100000 -dist_func Euclidian -k 10 > ivfGraph_1B_bigann_100000


nohup ./ivfGraph -base_path $P/base.1B.u8bin.crop_nb_100000000 -query_path $P/query.public.10K.u8bin -cluster_outfile $P/kmeans_graph_100M_10000 -res_path bigann_ivf_graph_10M.csv -gt_path $P/bigann-100M -data_type uint8 -num_clusters 100000 -dist_func Euclidian -k 10 > ivfGraph_100M_bigann_100000

nohup ./ivfGraph -base_path $P/base.1B.fbin.crop_nb_100000000 -query_path $P/query.public.10K.fbin  -cluster_outfile $P/kmeans_graph_1M_100 -res_path bigann_ivf_graph.csv -gt_path $P/text2image-100M -data_type float -num_clusters 100000 -dist_func mips -k 10 > ivfGraph_100M_text2Image