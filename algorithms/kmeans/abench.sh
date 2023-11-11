bazel build bench_run


PARANN_DIR=~/ParlayANN
SAMPLE_DIR=/ssd1/andrew/kmeans_benchmarks
DATA_DIR=/ssd1/anndata/bigann

$PARANN_DIR/bazel-bin/algorithms/kmeans/bench_run -ns $SAMPLE_DIR/n_samples.txt -ds $SAMPLE_DIR/d_samples.txt -vs $SAMPLE_DIR/var_samples.txt -is $SAMPLE_DIR/iter_samples.txt -ks $SAMPLE_DIR/k_samples.txt -D "fast" -i $DATA_DIR/base.1B.u8bin.crop_nb_1000000