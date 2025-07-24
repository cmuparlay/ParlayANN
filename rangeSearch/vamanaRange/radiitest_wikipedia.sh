#!/bin/bash

declare -A MATCHES
MATCHES[-9.0]=43439576
MATCHES[-9.5]=2249656
MATCHES[-10.0]=130056
MATCHES[-10.5]=12705
MATCHES[-11.0]=2200
MATCHES[-11.5]=559
MATCHES[-12.0]=177
MATCHES[-12.5]=65

recalls=(0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 0.99)

for recall_cut in "${recalls[@]}"; do
  CSV_FILE="qps_vs_matches_recall_wikipedia_${recall_cut}.csv"
  echo "Radius,NumMatches,QPS,Beam,Algorithm" > "$CSV_FILE"

  for file in radiitest_wikipedia_10M_final_*_*.out; do
    [[ "$file" == *groundtruth* ]] && continue
    [[ "$file" != *doubling* && "$file" != *greedy* ]] && continue

    # Extract algorithm and radius from filename
    filename=$(basename "$file")
    algo=$(echo "$filename" | awk -F'_' '{print $(NF-1)}')
    radius=$(echo "$filename" | awk -F'_' '{print $NF}' | sed 's/.out//')
    
    while IFS= read -r line; do
      [[ "$line" == For\ Beam:* ]] || continue

      recall=$(echo "$line" | awk -F'Cum Recall=' '{print $2}' | awk -F',' '{print $1}' | tr -d ' ')
      [[ ! "$recall" =~ ^[0-9.]+$ ]] && continue

      if (( $(echo "$recall > $recall_cut" | bc -l) )); then
        beam=$(echo "$line" | awk -F'[:,]' '{print $2}' | tr -d ' ')
        qps=$(echo "$line" | awk -F'QPS=' '{print $2}' | awk -F',' '{print $1}' | tr -d ' ')
        echo "$radius,${MATCHES[$radius]},$qps,$beam,$algo" >> "$CSV_FILE"
        break
      fi
    done < "$file"
  done

  python3 radiitest_pscript_wikipedia.py "$CSV_FILE" "$recall_cut"
done
