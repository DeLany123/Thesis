import os

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def plot_simulation_results_minute_by_minute(results: pd.DataFrame, start_minute: int = 0, end_minute: int = 1440):
    """
    Plots minute-by-minute results with full date and time labels,
    derived from the 'datetimes' key in the results.
    """
    if end_minute is None:
        end_minute = len(results)
    plot_df = results.iloc[start_minute:end_minute].copy()

    fig, ax1 = plt.subplots(figsize=(18, 7))

    color = 'tab:blue'
    ax1.set_xlabel('Date and Time')
    ax1.set_ylabel('Price (€/MWh)', color=color)
    # Gebruik de integer-index van de geslicede DataFrame voor de x-as
    ax1.plot(plot_df.index, plot_df['prices'], color=color, linestyle='-', label='Price')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Energy (MWh per minute)', color='black')
    width = 0.8
    colors = np.where(plot_df['energy_charged_discharged'] >= 0, 'green', 'red')
    ax2.bar(plot_df.index, plot_df['energy_charged_discharged'], width=width, color=colors, alpha=0.7)
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.axhline(0, color='black', linewidth=0.5)

    # Get the current limits of both axes
    ax1_min, ax1_max = ax1.get_ylim()
    ax2_min, ax2_max = ax2.get_ylim()

    # Calculate the largest absolute value for each axis
    ax1_abs_max = max(abs(ax1_min), abs(ax1_max))
    ax2_abs_max = max(abs(ax2_min), abs(ax2_max))

    # Set the new symmetric limits with 10% padding
    ax1.set_ylim(-ax1_abs_max * 1.1, ax1_abs_max * 1.1)
    ax2.set_ylim(-ax2_abs_max * 1.1, ax2_abs_max * 1.1)

    all_datetimes = pd.to_datetime(results['Datetime'])

    def format_full_datetime_ticks(tick_value, pos):
        if 0 <= tick_value < len(all_datetimes):
            current_datetime = all_datetimes[int(tick_value)]
            return current_datetime.strftime('%d-%m-%Y %H:%M')
        return ""

    # Format string on given every 60 minutes on x-axis
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(format_full_datetime_ticks))
    ax1.xaxis.set_major_locator(mticker.MultipleLocator(60))

    plt.title(f'Minute-by-Minute Simulation Results')
    fig.autofmt_xdate(rotation=30, ha='right')
    plt.tight_layout()

    # Define the directory and filename for the plot
    plots_dir = 'plots'
    filename = 'simulation_results.png'
    full_path = os.path.join(plots_dir, filename)
    os.makedirs(plots_dir, exist_ok=True)

    # Save the figure
    print(f"Saving plot to: {full_path}")
    plt.savefig(full_path, dpi=300)

    plt.show()