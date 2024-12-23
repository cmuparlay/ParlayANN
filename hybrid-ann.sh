cd algorithms/vamana
make
# ./neighbors -R 32 -L 50 -alpha 1.2 -graph_outfile /nvmessd1/fbv4/avarhade/parlay_prec1M_32_50 -query_path /nvmessd1/fbv4/queries.fbin -gt_path /nvmessd1/fbv4/prec1M_gt100.bin -res_path parlay_vamana_res.csv -data_type float -dist_func Euclidian -base_path /nvmessd1/fbv4/prec1M.fbin

./neighbors -R 32 -L 50 -k 25 -alpha 1.2 -graph_path /nvmessd1/fbv4/avarhade/parlay_prec1M_32_50 -query_path /nvmessd1/fbv4/queries.fbin -gt_path /nvmessd1/fbv4/prec1M_gt100.bin -res_path parlay_vamana_results.csv -data_type float -dist_func Euclidian -base_path /nvmessd1/fbv4/prec1M.fbin
