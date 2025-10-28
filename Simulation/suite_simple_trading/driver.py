import argparse
import os

import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sklearn.model_selection import TimeSeriesSplit
from stable_baselines3 import DQN

from .agent_trainer import train_dqn_agent, train_ppo_agent
from .grid_search import perform_grid_search
from .pre_processing import clean_data
from .simulation import run_evaluation
from .model import BatteryTradingEnv2, BatteryTradingEnvMasking
from .policy import QuarterlyTrendDecisionMaker, RLAgentDecisionMaker
from .plotting import plot_simulation_results_minute_by_minute


TRAIN_TEST_SPLIT_FRACTION = 0.8


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run simulation or grid search for the battery trading model.")
    subparsers = parser.add_subparsers(dest='command', required=True, help='Mode of execution')

    rl_mode = subparsers.add_parser('rl', help='Start the program in RL mode')
    rl_mode.add_argument(
        '--method',
        default='dqn',
        choices=['dqn', 'ppo'],
        help='Reinforcement learning method to use'
    )
    rl_mode.add_argument(
        '--mode',
        default='run',
        choices=['run', 'train', 'test'],
        help='Mode of execution'
    )
    rl_mode.add_argument(
        '--start-date',
        type=str,
        default=None,
        help="Start timestamp for plotting (e.g., '2025-01-01 08:00:00'). Defaults to the beginning."
    )
    rl_mode.add_argument(
        '--end-date',
        type=str,
        default=None,
        help="End timestamp for plotting (e.g., '2025-01-01 17:00:00'). Defaults to the end."
    )

    heuristic_mode = subparsers.add_parser('heuristic', help='Start the program in heuristic mode')
    heuristic_mode.add_argument(
        '--mode',
        default='run',
        choices=['run', 'gridsearch'],
        help='Mode of execution'
    )

    # Optional arguments
    heuristic_mode.add_argument('--buy', type=float, default=10.0, help="The buy/charge threshold for a single run.")
    heuristic_mode.add_argument('--sell', type=float, default=120.0, help="The sell/discharge threshold for a single run.")
    heuristic_mode.add_argument(
        '--start-date',
        type=str,
        default=None,
        help="Start timestamp for plotting (e.g., '2025-01-01 08:00:00'). Defaults to the beginning."
    )
    heuristic_mode.add_argument(
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
    split_index = int(len(cleaned_df) * TRAIN_TEST_SPLIT_FRACTION)
    train_df = all_data.iloc[:split_index]
    test_df = all_data.iloc[split_index:].reset_index()

    train_env = BatteryTradingEnv2(
        battery_capacity_mwh=10.0,
        charge_discharge_rate_mw=5.0,
        all_data=train_df,
        number_of_past_prices=5
    )

    test_env = BatteryTradingEnv2(
        battery_capacity_mwh=10.0,
        charge_discharge_rate_mw=5.0,
        all_data=test_df,
        number_of_past_prices=5
    )

    decision_maker = None

    if args.command == 'rl':
        if args.method == 'dqn':
            if args.mode == 'run':
                # Any mode in run, runs on test data
                rl_model_path = 'models/dqn_battery_trading_model.zip'

                try:
                    print(f"Loading trained model from: {rl_model_path}")
                    rl_model = DQN.load(rl_model_path)
                    decision_maker = RLAgentDecisionMaker(rl_model)

                except FileNotFoundError:
                    print(f"Error: Trained model not found at '{rl_model_path}'.")
                    print("Please run the script with '--mode train' first.")
                    exit()

            elif args.mode == 'train':
                print(f"\n--- Starting RL Agent Training Mode ---")
                train_dqn_agent(
                    env=train_env,
                    model_save_path='models/dqn_battery_trading_model',
                    total_timesteps=3*len(train_df), # TODO should be argument
                )
                print("--- Training Finished ---")
                train_env.close()
                test_env.close()
                exit()

            elif args.mode == 'test':
                print("\n--- Starting Time-Series Cross-Validation Mode ---")

                n_splits = 3
                gap_size = 1440

                tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap_size)
                fold_results = []

                for fold, (train_index, test_index) in enumerate(tscv.split(all_data)):
                    print(f"\n----- FOLD {fold + 1}/{n_splits} -----")

                    train_df = all_data.iloc[train_index]
                    test_df = all_data.iloc[test_index]

                    print(f"Training on {len(train_df)} samples, testing on {len(test_df)} samples.")

                    train_env = BatteryTradingEnv2(
                        battery_capacity_mwh=10.0,
                        charge_discharge_rate_mw=5.0,
                        all_data=train_df,
                        number_of_past_prices=5
                    )

                    model_path_for_fold = f"models/dqn_model_fold_{fold + 1}"
                    train_dqn_agent(
                        env=train_env,
                        model_save_path=model_path_for_fold,
                        total_timesteps=100_000
                    )

                    test_env = BatteryTradingEnv2(
                        battery_capacity_mwh=10.0,
                        charge_discharge_rate_mw=5.0,
                        all_data=test_df,
                        number_of_past_prices=5
                    )

                    rl_model = DQN.load(f"{model_path_for_fold}.zip")
                    decision_maker = RLAgentDecisionMaker(rl_model)

                    history_df = run_evaluation(test_env, decision_maker)
                    fold_profit = history_df['rewards'].sum()
                    fold_results.append(fold_profit)
                    print(f"Profit for Fold {fold + 1}: {fold_profit:.2f} EUR")

                print("\n--- Cross-Validation Summary ---")
                mean_profit = np.mean(fold_results)
                std_profit = np.std(fold_results)
                print(f"Average Profit across {n_splits} folds: {mean_profit:.2f} EUR")
                print(f"Standard Deviation of Profit: {std_profit:.2f} EUR")

        elif args.method == 'ppo':
            if args.mode == 'train':
                print(f"\n--- Starting PPO RL Agent Training Mode ---")
                train_env = BatteryTradingEnvMasking(
                    battery_capacity_mwh=10.0,
                    charge_discharge_rate_mw=5.0,
                    all_data=train_df,
                    number_of_past_prices=5
                )
                train_ppo_agent(
                    env=train_env,
                    model_save_path='models/ppo_battery_trading_model',
                    total_timesteps=3*len(train_df),
                )

            elif args.mode == 'run':
                print(f"\n--- Starting PPO RL Agent Run Mode ---")
                test_env = BatteryTradingEnvMasking(
                    battery_capacity_mwh=10.0,
                    charge_discharge_rate_mw=5.0,
                    all_data=test_df,
                    number_of_past_prices=5
                )

                model_path = 'models/ppo_battery_model.zip'
                try:
                    rl_model = MaskablePPO.load(model_path)
                    decision_maker = RLAgentDecisionMaker(rl_model)
                    history_df = run_evaluation(test_env, decision_maker)
                except FileNotFoundError:
                    print(f"Error: Trained PPO model not found at '{model_path}'.")
                    exit()


    elif args.command == 'heuristic':
        if args.mode == 'run':

            decision_maker = QuarterlyTrendDecisionMaker(
                theta_buy=args.buy,
                theta_sell=args.sell,
                past_prices_needed=test_env.number_of_past_prices
            )

        elif args.mode == 'gridsearch':
            print("\n--- Starting Grid Search Mode ---")

            # Search ranges for buy and sell thresholds # TODO: should be arguments
            buy_thresholds = np.arange(0, 51, 10)
            sell_thresholds = np.arange(100, 201, 20)

            perform_grid_search(train_env, buy_thresholds, sell_thresholds)
            train_env.close()
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
    train_env.close()
    print("\nScript finished.")
