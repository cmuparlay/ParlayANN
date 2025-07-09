import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

if len(sys.argv) != 3:
    print("Usage: python radiitest_pscript.py <csv_file> <recall_value>")
    sys.exit(1)

csv_file = sys.argv[1]
recall_val = sys.argv[2]

label_map = {
    "greedy": "Greedy Search with Early Stopping",
    "doubling": "Doubling Search with Early Stopping"
}

df = pd.read_csv(csv_file)

plt.figure(figsize=(10, 6))

for algo, group in df.groupby("Algorithm"):
    print(algo)
    readable_label = label_map.get(algo, algo)
    print(readable_label)
    group = group.sort_values("NumMatches")
    plt.plot(group["NumMatches"], group["QPS"], marker="o", label=readable_label)
    for _, row in group.iterrows():
        plt.text(row["NumMatches"], row["QPS"], f"{row['Beam']}", fontsize=20, ha='right')

plt.xlabel("Number of Matches", fontsize=30)
plt.ylabel("QPS", fontsize=30)
plt.xscale("symlog", linthresh=1000)
plt.tick_params(axis='x', labelsize=30)
plt.tick_params(axis='y', labelsize=30)

plt.legend(fontsize=24)
plt.grid(True)
plt.tight_layout()

output_file = f"qps_vs_matches_symlog_{recall_val}.pdf"
plt.savefig(output_file)
plt.close()
