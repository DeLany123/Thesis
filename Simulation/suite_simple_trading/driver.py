import argparse
import os

import numpy as np
import pandas as pd

from .grid_search import perform_grid_search
from .pre_processing import clean_data
from .simulation import run_evaluation
from .model import BatteryTradingEnv
from .policy import QuarterlyTrendDecisionMaker, train_rl_agent
from .plotting import plot_simulation_results_minute_by_minute

def simulate_policy(environment, all_data, theta_buy, theta_sell):
    """
    Simulates the trading policy over the given price data and returns the total profit.

    Args:
        environment: The Gymnasium environment instance.
        all_data: The full DataF/home/lander/Documents/school/Thesis/ThesisGit/plotsrame containing the price data and timestamps.
        theta_buy: The price threshold for buying/charging.
        theta_sell: The price threshold for selling/discharging.
    """
    decision_maker = QuarterlyTrendDecisionMaker(
        theta_buy=theta_buy,
        theta_sell=theta_sell,
        past_prices_needed=environment.number_of_past_prices
    )
    history_df = pd.DataFrame(run_evaluation(environment, decision_maker))
    history_df['prices'] = environment.prices
    history_df['Datetime'] = all_data['Datetime']
    return history_df

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run simulation or grid search for the battery trading model.")
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['gridsearch', 'run'],
        help="The mode to run the script in: 'gridsearch' to find best parameters, or 'run' to execute a single simulation."
    )

    # Optional arguments
    parser.add_argument('--buy', type=float, default=10.0, help="The buy/charge threshold for a single run.")
    parser.add_argument('--sell', type=float, default=120.0, help="The sell/discharge threshold for a single run.")
    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help="Start timestamp for plotting (e.g., '2025-01-01 08:00:00'). Defaults to the beginning."
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help="End timestamp for plotting (e.g., '2025-01-01 17:00:00'). Defaults to the end."
    )
    args = parser.parse_args()

    # Script logic
    raw_data_path = 'data/2025_minute.csv'
    cleaned_data_cache_path = 'data/2025_minute_cleaned.pkl'

    if os.path.exists(cleaned_data_cache_path):
        print(f"Loading cached cleaned data from: {cleaned_data_cache_path}")
        cleaned_df = pd.read_pickle(cleaned_data_cache_path)
    else:
        print("No cached data found. Running the full cleaning process...")
        raw_df = pd.read_csv(raw_data_path, sep=';')
        cleaned_df = clean_data(raw_df)
        print(f"Saving cleaned data to cache: {cleaned_data_cache_path}")
        cleaned_df.to_pickle(cleaned_data_cache_path)

    all_data = cleaned_df[['Datetime','Imbalance Price']]

    env = BatteryTradingEnv(
        battery_capacity_mwh=10.0,
        charge_discharge_rate_mw=5.0,
        all_data=all_data,
        number_of_past_prices=5
    )

    if args.mode == 'gridsearch':
        print("\n--- Starting Grid Search Mode ---")

        # Search ranges for buy and sell thresholds # TODO: should be arguments
        buy_thresholds = np.arange(0, 51, 10)
        sell_thresholds = np.arange(100, 201, 20)

        perform_grid_search(env, buy_thresholds, sell_thresholds)

    elif args.mode == 'run':
        print(f"\n--- Starting Single Run Mode ---")
        print(f"Using parameters: Buy Threshold = {args.buy}, Sell Threshold = {args.sell}")

        history_df = simulate_policy(env, cleaned_df, args.buy, args.sell)

        start_minute_index = 0
        end_minute_index = None

        if args.start_date:
            try:
                start_ts = pd.to_datetime(args.start_date, utc=True)
                start_minute_index = history_df['Datetime'].searchsorted(start_ts, side='left')
            except Exception as e:
                print(
                    f"Warning: Could not parse start-date '{args.start_date}'. Plotting from the beginning. Error: {e}")

        if args.end_date:
            try:
                end_ts = pd.to_datetime(args.end_date)
                end_minute_index = history_df.index.searchsorted(end_ts)
            except Exception as e:
                print(f"Warning: Could not parse end-date '{args.end_date}'. Plotting until the end. Error: {e}")
        else:
            end_minute_index = start_minute_index + 1440 # Default to 1 day (1440 minutes)

        print(f"Plotting from minute {start_minute_index} to {end_minute_index}...")

        plot_simulation_results_minute_by_minute(history_df, start_minute_index, end_minute_index)
        print(f"Total Profit: {history_df['rewards'].sum():.2f} EUR")

    env.close()
    print("\nScript finished.")
