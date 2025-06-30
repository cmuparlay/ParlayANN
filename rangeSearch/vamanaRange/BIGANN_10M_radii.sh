#!/bin/bash
make

declare -A datasets
datasets["BIGANN"]="/ssd1/data/bigann"

search_modes=("greedy" "doubling")
RADIUS_VALUES=(5000 7500 9000 10000 11000 12500 15000 20000)

for radius in "${RADIUS_VALUES[@]}"; do
  # Calculate early stopping radius
  early_stopping_radius=$(printf "%.0f" "$(echo "$radius * 3" | bc)")

  for mode in "${search_modes[@]}"; do
    echo "Running mode: $mode with radius: $radius (early stopping radius: $early_stopping_radius)"

    nohup ./range \
      -search_mode "$mode" \
      -alpha 1.15 -R 64 -L 128 \
      -r "$radius" \
      -early_stopping_radius "$early_stopping_radius" \
      -base_path "${datasets["BIGANN"]}/base.1B.u8bin.crop_nb_10000000" \
      -query_path "${datasets["BIGANN"]}/query.public.10K.u8bin" \
      -gt_path "${datasets["BIGANN"]}/range_gt_10M_10000_${radius}" \
      -data_type uint8 -dist_func Euclidian -graph_path "${datasets["BIGANN"]}/graph10M"\
       -early_stop true -quantize_mode 1\
      > "radiitest_BIGANN_10M_final_${mode}_${radius}.out"
    
    echo ""
  done
done