bazel build bench_run

PARANN_DIR=~/ParlayANN
SAMPLE_DIR=$PARANN_DIR/algorithms/kmeans/bench_files
DATA_DIR=/ssd1/anndata/bigann

#-ns = file containing values of n, one per line, first line contains the number of values that follow
#Ex.
#n_samples.txt
#3
#1000000
#2000000
#4000000
#
#-ds = file containing values of d
#-vs = file containing values of var: how many times we rerun each run (and then take the average of all the runs). This feature not currently implemented, so put
#var_samples.txt
#1
#1
#
#into that file.
#-is = file containing the # of iterations to run the k-means algorithm
#-ks = file containing values of k
#-D = Distance object to use
#-i = data input file
#-rn = algorithm to bench (naive or yy)
#-o = csv file to write benching output to
$PARANN_DIR/bazel-bin/algorithms/kmeans/bench_run -ns $SAMPLE_DIR/n1mil.txt -ds $SAMPLE_DIR/dBigann.txt -vs $SAMPLE_DIR/var1.txt -is $SAMPLE_DIR/iter5.txt -ks $SAMPLE_DIR/kAll.txt -D "fast" -i $DATA_DIR/base.1B.u8bin.crop_nb_1000000 -rn "yy" -o "bench_files/bench_out.csv" -t "uint8"