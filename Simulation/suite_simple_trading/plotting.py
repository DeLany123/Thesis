import os
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def plot_simulation_results_minute_by_minute(
        results: pd.DataFrame,
        filename: str,
        start_minute: int = 0,
        end_minute: int = 1440
):
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
    plot_duration_minutes = end_minute - start_minute

    # Choose a sensible interval for the x-axis ticks
    if plot_duration_minutes <= 180:  # Up to 3 hours
        locator_interval = 15  # A tick every 15 minutes
    elif plot_duration_minutes <= 24 * 60:  # Up to 1 day
        locator_interval = 120  # A tick every 2 hours
    elif plot_duration_minutes <= 7 * 24 * 60:  # Up to 1 week
        locator_interval = 24 * 60  # A tick every day
    else:  # For longer periods
        locator_interval = 7 * 24 * 60  # A tick every week

    # Apply the formatter and the new dynamic locator
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(format_full_datetime_ticks))
    ax1.xaxis.set_major_locator(mticker.MultipleLocator(base=locator_interval))

    plt.title(f'Minute-by-Minute Simulation Results')
    fig.autofmt_xdate(rotation=30, ha='right')
    plt.tight_layout()

    # Define the directory and filename for the plot
    plots_dir = 'plots'
    full_path = os.path.join(plots_dir, filename)
    os.makedirs(plots_dir, exist_ok=True)

    # Save the figure
    print(f"Saving plot to: {full_path}")
    plt.savefig(full_path, dpi=300)

    try:
        plt.show()
    except Exception as e:
        print(f"Unable to display plot interactively: {e}")


# (Include the moving_average function here)
def moving_average(data: np.ndarray, window_size: int) -> np.ndarray:
    return np.convolve(data, np.ones(window_size), 'valid') / window_size


def plot_episode_rewards(
        results_dict: Dict[str, List[float]],
        title: str = "Training Performance",
        smoothing_window: int = 10,
        x_axis_scale: int = 1
):
    """
    Plots the rewards per episode from one or more training runs.

    Args:
        results_dict: A dictionary where keys are the names of the runs (e.g., "PPO - Daily")
                      and values are lists of the total reward from each episode.
        title: The title of the plot.
        smoothing_window: The number of episodes to average over for a smoother line.
                          Set to 1 for no smoothing.
        x_axis_scale: A factor to scale the x-axis by (e.g., 1000 for "x10³").
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    for label, rewards in results_dict.items():
        rewards_np = np.array(rewards)

        # Apply smoothing
        if smoothing_window > 1 and len(rewards_np) > smoothing_window:
            smoothed_rewards = moving_average(rewards_np, smoothing_window)
        else:
            smoothed_rewards = rewards_np  # No smoothing if not enough data or window is 1

        # The x-axis represents the episode number after smoothing
        # We scale it for readability (e.g., from 10000 episodes to "10" on the axis)
        episodes = np.arange(len(smoothed_rewards)) / x_axis_scale

        ax.plot(episodes, smoothed_rewards, label=label)

    # --- Formatting ---
    ax.set_title(title, fontsize=16)

    # Set x-axis label based on the scaling_comparison factor
    if x_axis_scale == 1:
        ax.set_xlabel("Training Episodes", fontsize=12)
    else:
        # Create a formatted string like "(×10³)"
        exponent = int(np.log10(x_axis_scale))
        x_label = f"Training Episodes (×10³)" if exponent == 3 else f"Training Episodes (×10^{exponent})"
        ax.set_xlabel(x_label, fontsize=12)

    ax.set_ylabel("Total Episode Reward (€)", fontsize=12)
    ax.legend(fontsize=11)

    # Set a tight layout and show the plot
    plt.tight_layout()
    try:
        plt.show()
    except Exception as e:
        print(f"Unable to display plot interactively: {e}")


def plot_learning_curve(
    reward_file_path: str,
    title: str = "Training Performance",
    days_per_episode: int = 1,
    smoothing_window: int = 5,
    full_path: str = 'plots/learning_curve'
):
    """
    Loads episode rewards from a file and plots a smoothed learning curve.

    Args:
        reward_file_path: Path to the .npz file containing the rewards.
        title: The title for the plot.
        days_per_episode: The number of days in each episode, for calculating daily average.
        smoothing_window: The window size for the moving average.
        :param full_path: name of path where the png will be stored.
    """
    # Load the data
    try:
        # reward file but .npz extension needs to be joined
        full_reward_file_path = reward_file_path if reward_file_path.endswith('.npz') else reward_file_path + '.npz'
        data = np.load(full_reward_file_path)
        rewards = data['rewards']
    except FileNotFoundError:
        print(f"Error: Reward file not found at {reward_file_path}")
        return

    # Calculate the average daily reward per episode
    avg_daily_reward = rewards / days_per_episode

    # Smooth the curve
    smoothed_rewards = moving_average(avg_daily_reward, smoothing_window)
    episodes = np.arange(len(smoothed_rewards))

    # Plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(episodes, smoothed_rewards, label="Smoothed Reward", color="dodgerblue")

    ax.set_title(title, fontsize=16)
    ax.set_xlabel(f"Training Episodes (Smoothed over {smoothing_window} episodes)", fontsize=12)
    ax.set_ylabel("Average Daily Reward (€)", fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.savefig(full_path, dpi=300)
    plt.show()


def plot_total_charged_discharged_in_quarter_per_price(
        results: pd.DataFrame,
        price_bins: int = 50,
        full_path: str = 'plots/charged_discharged_decisions'
):
    """
        Plots the mean energy charged/discharged for different final quarter price bins.

        This shows the agent's average "bet size" at various price levels.

        Args:
            results: The evaluation results DataFrame. Must contain 'Datetime',
                        'final_quarter_price', 'total_charged_per_quarter',
                        and 'total_discharged_per_quarter' columns.
            price_bins: The number of price buckets to create on the x-axis.
            :param full_path: name of path where png will be stored
    """
    # Filter only end of quarter data
    end_of_quarter_mask = results['Datetime'].dt.minute % 15 == 14
    eoq_data = results[end_of_quarter_mask]

    # 2. Isolate quarters where there was significant charging or discharging.
    charge_quarters = eoq_data[eoq_data['total_charged_per_quarter'] > 0.01].copy()
    discharge_quarters = eoq_data[eoq_data['total_discharged_per_quarter'] > 0.01].copy()

    charge_quarters['price_bin'] = pd.cut(charge_quarters['prices'], bins=price_bins)
    discharge_quarters['price_bin'] = pd.cut(discharge_quarters['prices'], bins=price_bins)

    # 4. Group by the price bins and calculate the mean traded volume for each bin.
    charge_means = charge_quarters.groupby('price_bin')['total_charged_per_quarter'].mean()
    discharge_means = discharge_quarters.groupby('price_bin')['total_discharged_per_quarter'].mean()

    # 5. Plot the results.
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    fig.suptitle('Agent Trading Strategy: Mean Volume by Price', fontsize=18)

    # Charging Plot
    charge_means.plot(kind='bar', ax=ax[0], color='dodgerblue', width=0.8)
    ax[0].set_title('Charging Behavior')
    ax[0].set_ylabel('Mean Energy Charged (MWh)')
    ax[0].tick_params(axis='x', rotation=45)

    # Discharging Plot
    discharge_means.plot(kind='bar', ax=ax[1], color='crimson', width=0.8)
    ax[1].set_title('Discharging Behavior')
    ax[1].set_ylabel('Mean Energy Discharged (MWh)')
    ax[1].set_xlabel('Final Quarter Price (€)')
    ax[1].tick_params(axis='x', rotation=90)  # Rotate for readability

    plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
    plt.savefig(full_path, dpi=300)
    plt.show()
