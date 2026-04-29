"""
Per-agent plot: each fold as its own line, with a bold mean ± std band.
Produces one subplot per agent so fold-level variation is clearly visible.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


results_dir = ".."
output_plot_dir = "../plots" # relative

# ── Load all result CSVs ───────────────────────────────────────────────
all_data = []
for file in sorted(os.listdir(results_dir)):
    if file.endswith(".csv"):
        try:
            df = pd.read_csv(os.path.join(results_dir, file))
            # Require std_revenue alongside other columns
            if all(c in df.columns for c in ["agent", "fold", "total_steps", "mean_revenue", "std_revenue"]):
                all_data.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")

if not all_data:
    print("No valid CSV files found.")
    exit()

combined = pd.concat(all_data, ignore_index=True)
agents = sorted(combined["agent"].unique())

# ── One separate plot per agent ─────────────────────────────────────────
cmap = plt.cm.tab10

for agent in agents:
    agent_df = combined[combined["agent"] == agent]
    folds = sorted(agent_df["fold"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))

    for fold in folds:
        fold_df = agent_df[agent_df["fold"] == fold].sort_values("total_steps")
        ax.plot(
            fold_df["total_steps"], fold_df["mean_revenue"],
            marker="o", markersize=3, lw=1, alpha=0.45,
            color=cmap(fold % 10), label=f"Fold {fold}"
        )

    agg = agent_df.groupby("total_steps")["mean_revenue"].agg(["mean", "std"]).reset_index()
    agg["std"] = agg["std"].fillna(0)

    ax.plot(agg["total_steps"], agg["mean"], color="black", lw=2.2, label="Mean")
    ax.fill_between(
        agg["total_steps"],
        agg["mean"] - agg["std"],
        agg["mean"] + agg["std"],
        color="black", alpha=0.12, label="± 1 std"
    )

    ax.set_title(f"Per-Fold Performance: {agent}", fontsize=13, weight="bold")
    ax.set_xlabel("Total Steps")
    ax.set_ylabel("Revenue")
    ax.legend(fontsize=8, frameon=True)
    ax.grid(True, ls="--", lw=0.5, alpha=0.6)
    fig.tight_layout()

    safe_name = agent.replace(" ", "_").replace("/", "_")
    output = f"{output_plot_dir}/agent_per_fold_{safe_name}.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")

