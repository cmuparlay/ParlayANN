#!/bin/bash

# crawl enron gist msong audio sift uqv 
for dataset in $@ ; do
    for method in parlayivf parlayivf-no-material-join parlayivf-no-bitvector parlayivf-no-weight-classes; do
        echo "Running $method on $dataset"
        python memory_and_build.py $method $dataset
    done
    # wait # if using, add & to call
    # sudo chmod 777 -R results
    # python plot.py --neurips23track filter --dataset $dataset
done


