import os
import pandas as pd
import matplotlib.pyplot as plt

results_dir = ".."
output_plot_dir = "../plots" # relative

all_data = []

# Load data containing standard deviations
for file in sorted(os.listdir(results_dir)):
    if file.endswith(".csv"):
        try:
            df = pd.read_csv(os.path.join(results_dir, file))
            if all(c in df.columns for c in ["agent", "fold", "total_steps", "mean_revenue", "std_revenue"]):
                all_data.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")

if not all_data:
    print("No valid CSV files found.")
    exit()

combined = pd.concat(all_data, ignore_index=True)
agents = sorted(combined["agent"].unique())

for agent in agents:
    agent_df = combined[combined["agent"] == agent]

    # Calculate statistics across folds for each step count
    stats = agent_df.groupby("total_steps").agg(
        cross_fold_std=("mean_revenue", "std"),      # Variance between different folds (Generalization)
        mean_seed_std=("std_revenue", "mean")        # Average variance inside folds (Seed stability)
    ).reset_index()

    # Replace NaNs with 0 if it couldn't calculate std (e.g., only 1 fold exists)
    stats = stats.fillna(0)

    fig, ax = plt.subplots(figsize=(7, 5))

    # Plot 1: Variance between different folds (Generalization variance)
    ax.plot(
        stats["total_steps"], stats["cross_fold_std"],
        marker="o", color="#1f77b4", lw=2.5,
        label="Cross-Fold Std"
    )

    # Plot 2: Average variance inside the folds (Seed variance)
    ax.plot(
        stats["total_steps"], stats["mean_seed_std"],
        marker="s", color="#ff7f0e", lw=2.5, ls="--",
        label="Intra-Fold Std"
    )

    ax.set_title(f"Variance Reduction Over Training Time: {agent}", fontsize=13, weight="bold")
    ax.set_xlabel("Total Steps", fontsize=11)
    ax.set_ylabel("Standard Deviation of Revenue [EUR]", fontsize=11)
    ax.legend(fontsize=10, frameon=True)
    ax.grid(True, ls="--", lw=0.6, alpha=0.6)

    # Optional styling for a cleaner thesis plot
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()

    safe_name = agent.replace(" ", "_").replace("/", "_")
    output = f"{output_plot_dir}/variance_over_time_{safe_name}.png"
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved variance plot: {output}")

