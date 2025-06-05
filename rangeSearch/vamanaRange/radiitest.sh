#!/bin/bash

declare -A MATCHES
MATCHES[5000]=10887
MATCHES[7500]=52233
MATCHES[9000]=104045
MATCHES[10000]=152958
MATCHES[11000]=215336
MATCHES[12500]=336039
MATCHES[15000]=621575
MATCHES[20000]=1565281

# Define recall cutoffs
recalls=(0.9 0.95 0.99 0.995 0.999)

for recall_cut in "${recalls[@]}"; do
  CSV_FILE="qps_vs_matches_recall_${recall_cut}.csv"
  echo "Radius,NumMatches,QPS,Beam,Algorithm" > "$CSV_FILE"

  for file in radiitest_BIGANN_10M_*_*.out; do
    [[ "$file" == "radiitest_groundtruth_info.out" ]] && continue

    radius=$(basename "$file" | grep -oE '[0-9]+' | tail -1)

    if [[ "$file" == *doublingSearchEarlyStopping* ]]; then
      algo="doublingSearchEarlyStopping"
    elif [[ "$file" == *earlyStopping* ]]; then
      algo="earlyStopping"
    else
      algo="unknown"
    fi

    while IFS= read -r line; do
      [[ "$line" == For\ Beam:* ]] || continue

      recall=$(echo "$line" | awk -F'Pointwise Recall = ' '{print $2}' | awk -F',' '{print $1}' | tr -d ' ')
      if [[ "$recall" =~ ^[0-9.]+$ ]] && awk "BEGIN {exit !($recall > $recall_cut)}"; then
        beam=$(echo "$line" | awk -F'[:,]' '{print $2}' | tr -d ' ')
        qps=$(echo "$line" | awk -F'QPS = ' '{print $2}' | tr -d ' ')
        if [[ -n "$radius" && -n "${MATCHES[$radius]}" && -n "$qps" && -n "$beam" ]]; then
          echo "$radius,${MATCHES[$radius]},$qps,$beam,$algo" >> "$CSV_FILE"
        else
          echo "Warning: Parsing failed for radius '$radius' in file '$file'" >&2
        fi
        break
      fi
    done < "$file"
  done

  python3 radiitest_pscript.py "$CSV_FILE" "$recall_cut"
done
