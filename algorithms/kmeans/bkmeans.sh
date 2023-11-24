#!bin/bash
#This bash file for running kmeans, except through the PointRange interface
bazel build kmeans_run

PARANN_DIR=~/ParlayANN

$PARANN_DIR/bazel-bin/algorithms/kmeans/kmeans_run -k 10 -i /ssd1/anndata/bigann/base.1B.u8bin.crop_nb_1000 -f bin -t uint8 -D fast -m 5 -two three
