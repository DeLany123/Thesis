import argparse
import os

import numpy as np
import pandas as pd
from stable_baselines3 import DQN

from .dqn_driver import train_dqn_agent
from .grid_search import perform_grid_search
from .pre_processing import clean_data
from .simulation import run_evaluation
from .model import BatteryTradingEnv2
from .policy import QuarterlyTrendDecisionMaker, RLAgentDecisionMaker
from .plotting import plot_simulation_results_minute_by_minute


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run simulation or grid search for the battery trading model.")
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['gridsearch', 'run', 'train'],
        help="The mode to run the script in: 'gridsearch' to find best parameters, or 'run' to execute a single simulation."
    )

    parser.add_argument(
        '--policy',
        type=str,
        default='heuristic',
        choices=['heuristic', 'rl'],
        help="Policy to use for 'run' mode: 'heuristic' or 'rl'. Defaults to 'heuristic'."
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

    # Define the split point for the training data
    split_fraction = 0.8
    split_index = int(len(cleaned_df) * split_fraction)

    # Create the training and testing DataFrames
    train_df = all_data.iloc[:split_index]
    test_df = all_data.iloc[split_index:].reset_index()

    train_env = BatteryTradingEnv2(
        battery_capacity_mwh=10.0,
        charge_discharge_rate_mw=5.0,
        all_data=train_df,
        number_of_past_prices=5
    )

    if args.mode == 'gridsearch':
        print("\n--- Starting Grid Search Mode ---")

        # Search ranges for buy and sell thresholds # TODO: should be arguments
        buy_thresholds = np.arange(0, 51, 10)
        sell_thresholds = np.arange(100, 201, 20)

        perform_grid_search(train_env, buy_thresholds, sell_thresholds)


    elif args.mode == 'run':
        print(f"\n--- Starting Single Run Mode ---")
        decision_maker = None

        test_env = BatteryTradingEnv2(
            battery_capacity_mwh=10.0,
            charge_discharge_rate_mw=5.0,
            all_data=test_df,
            number_of_past_prices=5
        )

        if args.policy == 'heuristic':
            print(f"Policy: Heuristic")
            print(f"Using parameters: Buy Threshold = {args.buy}, Sell Threshold = {args.sell}")

            decision_maker = QuarterlyTrendDecisionMaker(
                theta_buy=args.buy,
                theta_sell=args.sell,
                past_prices_needed=test_env.number_of_past_prices
            )

        elif args.policy == 'rl':
            print(f"Policy: Reinforcement Learning (DQN)")
            rl_model_path = 'models/dqn_battery_trading_model.zip'
            try:
                print(f"Loading trained model from: {rl_model_path}")
                rl_model = DQN.load(rl_model_path)
                decision_maker = RLAgentDecisionMaker(rl_model)

            except FileNotFoundError:
                print(f"Error: Trained model not found at '{rl_model_path}'.")
                print("Please run the script with '--mode train' first.")
                exit()

        history_df = pd.DataFrame(run_evaluation(test_env, decision_maker))
        history_df['Datetime'] = test_df['Datetime']

        start_minute_index = history_df['Datetime'].searchsorted(history_df['Datetime'].iloc[0], side='left')
        end_minute_index = start_minute_index + 10_080
        if args.start_date:
            try:
                start_ts = pd.to_datetime(args.start_date, utc=True)
                start_minute_index = history_df['Datetime'].searchsorted(start_ts, side='left')
            except Exception as e:
                print(
                    f"Warning: Could not parse start-date '{args.start_date}'. Plotting from the beginning. Error: {e}")

        if args.end_date:
            try:
                end_ts = pd.to_datetime(args.end_date, utc=True)
                end_minute_index = history_df['Datetime'].searchsorted(end_ts, side='right')
            except Exception as e:
                print(f"Warning: Could not parse end-date '{args.end_date}'. Plotting until the end. Error: {e}")

        # If no end date is given but a start date is, default to plotting one day
        elif args.start_date:
            end_minute_index = start_minute_index + 1440

        print(f"Plotting from minute {start_minute_index} to {end_minute_index}...")
        plot_simulation_results_minute_by_minute(history_df, start_minute_index, end_minute_index)
        print(f"Total Profit: {history_df['rewards'].sum():.2f} EUR")
        test_env.close()

    elif args.mode == 'train':
        print(f"\n--- Starting RL Agent Training Mode ---")
        train_dqn_agent(
            env=train_env,
            model_save_path='models/dqn_battery_trading_model',
            total_timesteps=100_000
        )

    train_env.close()
    print("\nScript finished.")
