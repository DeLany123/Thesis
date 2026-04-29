import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
results_dir = ".."
plot_dir = "../plots"
all_data = []
# Read all CSV files in the results directory
for file in os.listdir(results_dir):
    if file.endswith(".csv"):
        filepath = os.path.join(results_dir, file)
        try:
            df = pd.read_csv(filepath)
            # Make sure it has the required columns
            if all(col in df.columns for col in ["agent", "total_steps", "mean_revenue"]):
                all_data.append(df)
        except Exception as e:
            print(f"Error reading {file}: {e}")
if all_data:
    combined_df = pd.concat(all_data, ignore_index=True)
    plt.figure(figsize=(10, 6))
    # lineplot automatically calculates the mean and draws a confidence interval (or standard deviation) 
    # shade when there are multiple observations (folds) per x-value (total_steps).
    # errorbar="sd" shows standard deviation over the folds.
    sns.lineplot(
        data=combined_df, 
        x="total_steps", 
        y="mean_revenue", 
        hue="agent", 
        marker="o", 
        errorbar="sd"
    )
    plt.title("Agent Performance over Time Steps (Mean & Variance across Folds)")
    plt.xlabel("Total Steps")
    plt.ylabel("Revenue")
    plt.grid(True)
    plt.tight_layout()
    output_filename = f"{plot_dir}/agent_performance_plot.png"
    plt.savefig(output_filename)
    print(f"Plot successfully saved to {output_filename}")
else:
    print("No valid CSV files found in the results directory.")
