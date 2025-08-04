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
    echo "Running mode: $mode with radius: $radius (early stopping radius: $early_stopping_radius)"

    nohup ./range \
      -search_mode "$mode" \
      -alpha 1.0 -R 64 -L 128 \
      -r "$radius" \
      -early_stopping_radius "$early_stopping_radius" \
      -base_path "${datasets["WIKI"]}/wikipedia_base.bin.crop_nb_10000000" \
      -query_path "${datasets["WIKI"]}/wikipedia_query.bin" \
      -gt_path "${datasets["WIKI"]}/range_gt_10M_radiitest_${radius}" \
      -data_type float -dist_func mips -file_type bin \
      -graph_path "${datasets["WIKI"]}/graph_10M" \
      -early_stop -quantize_mode 1 \
      > "radiitest_wikipedia_10M_final_${mode}_${radius}.out"

    echo ""
  done
done
