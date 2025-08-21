import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

if len(sys.argv) != 3:
    print("Usage: python radiitest_pscript.py <csv_file> <recall_value>")
    sys.exit(1)

csv_file = sys.argv[1]
recall_val = sys.argv[2]

total_queries = 5000

label_map = {
    "earlyStopping": "Greedy Search with Early Stopping",
    "doublingSearchEarlyStopping": "Doubling Search with Early Stopping",
    "greedy": "Greedy Search with Early Stopping",
    "doubling": "Doubling Search with Early Stopping"
}

df = pd.read_csv(csv_file)
df["MatchesPerQuery"] = df["NumMatches"] / total_queries

plt.figure(figsize=(10, 6))

for algo, group in df.groupby("Algorithm"):
    readable_label = label_map.get(algo, algo)
    group = group.sort_values("MatchesPerQuery")
    plt.plot(group["MatchesPerQuery"], group["QPS"], marker="o", label=readable_label)
    for _, row in group.iterrows():
        plt.text(row["MatchesPerQuery"], row["QPS"], f"{row['Beam']}", fontsize=15, ha='right')

plt.xlabel("Matches per Query", fontsize=30)
plt.ylabel("QPS", fontsize=30)
plt.xscale("symlog", linthresh=0.2)
plt.yscale("log")
plt.tick_params(axis='x', labelsize=25)
plt.tick_params(axis='y', labelsize=25)
plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=False))

# Save and export legend separately
output_file = f"qps_vs_matches_symlog_noearlystop_wikipedia_1M_{recall_val}.pdf"
plt.grid(True)
plt.tight_layout()
plt.savefig(output_file, bbox_inches='tight')

def export_legend(handles, labels, filename="legend.pdf"):
    fig = plt.figure(figsize=(6, 1))
    ax = fig.add_subplot(111)
    ax.axis('off')
    fig.legend(handles, labels, loc='center', frameon=False, fontsize=20)
    fig.savefig(filename, bbox_inches='tight', transparent=True)
    plt.close(fig)

handles, labels = plt.gca().get_legend_handles_labels()
export_legend(handles, labels, filename=f"qps_vs_matches_symlog_noearlystop_wikipedia_1M_{recall_val}_legend.pdf")

