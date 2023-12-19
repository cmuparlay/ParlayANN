#!/bin/bash

PARANN_DIR=~/ParlayANN

cd $PARANN_DIR/algorithms/kmeans

bazel test distance_gtest
bazel test naive_gtest
bazel test yy_gtest
bazel test naive_yy_compare_gtest