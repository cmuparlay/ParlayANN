#!/bin/bash

cd vamana
make
echo "Vamana:"
./neighbors -R 32 -L 64 -a 1.2 -graph_outfile ../../data/sift/sift_learn_32_64 -query_path ../../data/sift/sift_query.fvecs -gt_path ../../data/sift/sift-100K -res_path tutorial.csv -data_type float -file_type vec -dist_func Euclidian -base_path ../../data/sift/sift_learn.fvecs

echo "" 
echo "" 

cd ../HCNNG
make
echo "HCNNG:"
./neighbors -R 3 -L 10 -a 1000 -memory_flag 1 -graph_outfile ../../data/sift/sift_learn_3_10 -query_path ../../data/sift/sift_query.fvecs -gt_path ../../data/sift/sift-100K -res_path tutorial.csv -data_type float -file_type vec -dist_func Euclidian -base_path ../../data/sift/sift_learn.fvecs

echo ""
echo ""

cd ../pyNNDescent
make
echo "pyNNDescent:"
./neighbors -R 30 -L 100 -a 10 -d 1.2 -graph_outfile ../../data/sift/sift_learn_30 -query_path ../../data/sift/sift_query.fvecs -gt_path ../../data/sift/sift-100K -res_path tutorial.csv -data_type float -file_type vec -dist_func Euclidian -base_path ../../data/sift/sift_learn.fvecs