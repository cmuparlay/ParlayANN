#!/bin/bash
cd ../data_tools/

./run_vbase_graph.sh

cd ../misc_python_functions/

python3 vbase_graph.py -p /home/nam/vector_index_rs/ParlayANN/data_tools/data.txt

