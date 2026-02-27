import os
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

def plot_simulation_results_minute_by_minute(
        results: pd.DataFrame,
        filename: str,
        start_minute: int = 0,
        end_minute: int = 1440,
        battery_capacity_mwh: float = None  # Optional: Pass capacity for accurate %
):
    """
    Plots minute-by-minute results including Price, Energy Traded, and SoC %.
    """
    if end_minute is None:
        end_minute = len(results)
    plot_df = results.iloc[start_minute:end_minute].copy()

    # --- Prepare SoC Data ---
    # If capacity is not provided, estimate it from the max observed SoC
    if battery_capacity_mwh is None:
        battery_capacity_mwh = results['soc'].max()
        if battery_capacity_mwh == 0: battery_capacity_mwh = 1.0  # Prevent div/0

    # Calculate SoC Percentage
    soc_percentage = (plot_df['soc'] / battery_capacity_mwh) * 100

    # --- Setup Plot ---
    fig, ax1 = plt.subplots(figsize=(18, 8))  # Slightly taller for 3 axes

    # --- AXIS 1: Price (Line) ---
    color_price = 'tab:blue'
    ax1.set_xlabel('Date and Time', fontsize=12)
    ax1.set_ylabel('Price (€/MWh)', color=color_price, fontsize=12)
    ax1.plot(plot_df.index, plot_df['prices'], color=color_price, linestyle='-', linewidth=2, label='Price', zorder=10)
    ax1.tick_params(axis='y', labelcolor=color_price)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- AXIS 2: Energy Traded (Bars) ---
    ax2 = ax1.twinx()
    ax2.set_ylabel('Energy Traded (MWh)', color='black', fontsize=12)

    # Define colors: Green for Charge (+), Red for Discharge (-)
    colors = np.where(plot_df['energy_charged_discharged'] >= 0, 'forestgreen', 'firebrick')
    ax2.bar(plot_df.index, plot_df['energy_charged_discharged'], width=0.6, color=colors, alpha=0.9,
            label='Energy Traded', zorder=5)
    ax2.tick_params(axis='y', labelcolor='black')

    # --- AXIS 3: SoC % (Background Bars/Area) ---
    ax3 = ax1.twinx()

    # Offset the third axis spine to the right so it doesn't overlap with ax2
    ax3.spines["right"].set_position(("axes", 1.08))

    color_soc = 'gold'
    ax3.set_ylabel('State of Charge (%)', color='goldenrod', fontsize=12)

    # Plot SoC as wide, semi-transparent bars in the background
    ax3.bar(plot_df.index, soc_percentage, width=1.0, color=color_soc, alpha=0.5, label='SoC %', zorder=1)
    ax3.set_ylim(0, 100)  # SoC is always 0-100%
    ax3.tick_params(axis='y', labelcolor='goldenrod')

    # --- Formatting Limits (Symmetric Price/Energy) ---
    # We only symmetrize Price and Energy Traded, SoC stays 0-100
    ax1_min, ax1_max = ax1.get_ylim()
    ax2_min, ax2_max = ax2.get_ylim()

    ax1_abs_max = max(abs(ax1_min), abs(ax1_max))
    ax2_abs_max = max(abs(ax2_min), abs(ax2_max))

    # Apply padding
    ax1.set_ylim(-ax1_abs_max * 1.1, ax1_abs_max * 1.1)
    ax2.set_ylim(-ax2_abs_max * 1.1, ax2_abs_max * 1.1)

    # Draw zero line for energy/price
    ax2.axhline(0, color='black', linewidth=0.8, zorder=6)

    # --- Date Formatting (Your existing logic) ---
    all_datetimes = pd.to_datetime(results['Datetime'])

    def format_full_datetime_ticks(tick_value, pos):
        if 0 <= tick_value < len(all_datetimes):
            return all_datetimes[int(tick_value)].strftime('%d-%m %H:%M')
        return ""

    plot_duration_minutes = end_minute - start_minute
    if plot_duration_minutes <= 180:
        locator_interval = 15
    elif plot_duration_minutes <= 1440:
        locator_interval = 120
    elif plot_duration_minutes <= 10080:
        locator_interval = 1440
    else:
        locator_interval = 10080

    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(format_full_datetime_ticks))
    ax1.xaxis.set_major_locator(mticker.MultipleLocator(base=locator_interval))

    # Title and Layout
    plt.title(f'Simulation Analysis: Price, Trading Action, and SoC', fontsize=14)
    fig.autofmt_xdate(rotation=30, ha='right')

    # Legend (Combining handles from different axes)
    # lines, labels = ax1.get_legend_handles_labels()
    # bars2, labels2 = ax2.get_legend_handles_labels()
    # bars3, labels3 = ax3.get_legend_handles_labels()
    # ax1.legend(lines + bars2 + bars3, labels + labels2 + labels3, loc='upper left')

    # Saving
    plots_dir = 'plots'
    full_path = os.path.join(plots_dir, filename)
    os.makedirs(plots_dir, exist_ok=True)

    print(f"Saving plot to: {full_path}")
    plt.savefig(full_path, dpi=300, bbox_inches='tight')  # bbox_inches='tight' prevents cutting off the 3rd axis

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


def plot_feature_importance(model, feature_names):
    """
    Extracts and plots the importance of features based on the weights
    of the first layer of the network. Supports PPO and DQN.
    """
    # 1. Identify the algorithm and find the first layer
    first_layer_weights = None

    # CASE 1: PPO / MaskablePPO (Actor-Critic)
    if hasattr(model.policy, "mlp_extractor"):
        # We access the Actor's (policy_net) first layer
        first_layer_weights = model.policy.mlp_extractor.policy_net[0].weight
        print("Detected PPO/Actor-Critic architecture.")

    # CASE 2: DQN (Q-Network)
    elif hasattr(model.policy, "q_net"):
        # We access the Q-Network's first layer
        # In SB3 DQN, the MLP is stored in policy.q_net.q_net
        first_layer_weights = model.policy.q_net.q_net[0].weight
        print("Detected DQN architecture.")

    else:
        print("Error: Could not find the first layer weights for this model type.")
        return

    # 2. Convert to numpy
    weights_np = first_layer_weights.detach().cpu().numpy()

    # 3. Calculate importance
    # Average absolute weight for each input feature
    feature_importance = np.mean(np.abs(weights_np), axis=0)

    # 4. Create the plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x=feature_names, y=feature_importance, palette='viridis')
    plt.title(f"Feature Importance ({model.__class__.__name__})")
    plt.ylabel("Importance Score (Avg Abs Weight)")
    plt.xlabel("Features")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_volatility_vs_activity(result_data):
    df = result_data.copy()
    df['quarter_group'] = np.arange(len(df)) // 15

    quarters = []
    for _, group in df.groupby('quarter_group'):
        if len(group) < 15: continue

        # Calculate intra-quarter volatility
        volatility = group['prices'].max() - group['prices'].min()

        # Calculate total activity
        final = group.iloc[-1]
        activity = final['total_charged_per_quarter'] + final['total_discharged_per_quarter']

        quarters.append({'Volatility (€ range)': volatility, 'Traded Volume (MWh)': activity})

    plot_data = pd.DataFrame(quarters)

    plt.figure(figsize=(10, 6))
    sns.regplot(data=plot_data, x='Volatility (€ range)', y='Traded Volume (MWh)', scatter_kws={'alpha': 0.3})
    plt.title("Agent Risk Profile: Does it trade more during chaos?")
    plt.show()