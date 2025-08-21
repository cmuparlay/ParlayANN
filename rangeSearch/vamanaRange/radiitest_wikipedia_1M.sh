#!/bin/bash
make

declare -A datasets
datasets["WIKI"]="/ssd1/data/wikipedia_cohere"

search_modes=("greedy" "doubling")
RADIUS_VALUES=(-9.0 -9.5 -10.5 -11.0 -11.5 -12.0 -12.5)
#RADIUS_VALUES=(-10.0)

for radius in "${RADIUS_VALUES[@]}"; do
  early_stopping_radius=$(echo "$radius + 1.0" | bc)

  for mode in "${search_modes[@]}"; do
    #echo "Running mode: $mode with radius: $radius (early stopping radius: $early_stopping_radius)"
    echo "Running mode: $mode with radius: $radius"

    nohup ./range \
      -search_mode "$mode" \
      -alpha 1.0 -R 64 -L 128 \
      -r "$radius" \
      -base_path "${datasets["WIKI"]}/wikipedia_base.bin.crop_nb_1000000" \
      -query_path "${datasets["WIKI"]}/wikipedia_query.bin" \
      -gt_path "${datasets["WIKI"]}/range_gt_1M_radiitest_${radius}" \
      -data_type float -dist_func mips -file_type bin \
      -graph_path "${datasets["WIKI"]}/graph1M" \
      > "radiitest_wikipedia_1M_noearlystop_${mode}_${radius}.out"


    echo ""
  done
done
