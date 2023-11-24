#!bin/bash
#This bash file is for running kmeans, through the parlaykmeans-style interface

bazel build kmeans_run


PARANN_DIR=~/ParlayANN

$PARANN_DIR/bazel-bin/algorithms/kmeans/kmeans_run -k 200 -i /ssd1/anndata/bigann/base.1B.u8bin.crop_nb_1000000 -f bin -t uint8 -D fast -m 5 -two yes 
