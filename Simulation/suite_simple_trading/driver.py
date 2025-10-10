import os

import numpy as np
import pandas
import pandas as pd
from model import BatteryTradingEnv
from policy import QuarterlyTrendDecisionMaker, train_rl_agent
from plotting import plot_simulation_results_minute_by_minute

def run_evaluation(env: BatteryTradingEnv, decision_maker) -> dict:
    """
    Runs a single evaluation and returns a detailed history of the simulation.
    """
    obs, info = env.reset()
    decision_maker.reset()

    prices_history = []
    soc_history = []
    action_history = []
    reward_history = []
    energy_charged_discharged_history = []

    done = False
    while not done:
        action = decision_maker.get_action(obs, env.current_step)
        obs, reward, terminated, truncated, info = env.step(action)

        energy_charged_discharged = info.get('energy_charged_discharged', 0)

        prices_history.append(obs[-1])
        soc_history.append(env.soc_mwh)  # Get current SoC from the env
        action_history.append(action)
        reward_history.append(reward)
        energy_charged_discharged_history.append(energy_charged_discharged)

        done = terminated or truncated

    # Return all collected data in a dictionary
    return {
        "prices": None,
        "soc": soc_history,
        "actions": action_history,
        "rewards": reward_history,
        "energy_charged_discharged": energy_charged_discharged_history,
    }


def perform_grid_search(env: BatteryTradingEnv, buy_range: np.ndarray, sell_range: np.ndarray):
    """
    Performs a grid search to find the optimal theta_buy and theta_sell values.

    Args:
        env: The Gymnasium environment instance.
        buy_range: A NumPy array of theta_buy values to test.
        sell_range: A NumPy array of theta_sell values to test.
    """
    best_reward = -np.inf
    best_params = {'buy': None, 'sell': None}

    total_combinations = len(buy_range) * len(sell_range)
    current_run = 0

    print("--- Starting Grid Search ---")

    # Iterate through all combinations of theta values
    for theta_buy in buy_range:
        for theta_sell in sell_range:
            current_run += 1

            # A logical constraint: selling price should be higher than buying price
            if theta_sell <= theta_buy:
                continue

            print(f"[{current_run}/{total_combinations}] Testing: Buy <= {theta_buy:.2f}, Sell >= {theta_sell:.2f} ...")

            # Run a full simulation with the current set of parameters
            decision_maker = QuarterlyTrendDecisionMaker(
                theta_buy=theta_buy,
                theta_sell=theta_sell,
                past_prices_needed=env.number_of_past_prices
            )
            reward_history = run_evaluation(env, decision_maker)['rewards']
            total_reward = sum(reward_history)
            print(f"  -> Total Reward: {total_reward:.2f} EUR")

            # Check if this combination is the best one found so far
            if total_reward > best_reward:
                best_reward = total_reward
                best_params['buy'] = theta_buy
                best_params['sell'] = theta_sell
                print(f"  -> New best found!")

    print("\n--- Grid Search Complete ---")
    if best_params['buy'] is not None:
        print(f"Optimal Buy Threshold: {best_params['buy']:.2f} EUR/MWh")
        print(f"Optimal Sell Threshold: {best_params['sell']:.2f} EUR/MWh")
        print(f"Resulting Best Reward: {best_reward:.2f} EUR")
    else:
        print("No valid parameter combinations found.")


def clean_data(data: pandas.DataFrame) -> pd.DataFrame:
    """
    Checks if data is in the right order,
    if it starts on a quarter hour
    if there are any NaN values, if so, it removes the whole quarter

    Args:
        data: A NumPy array of price data.

    Returns:
        A cleaned NumPy array with NaN values removed.
    """
    data = data.iloc[::-1].reset_index(drop=True)

    # Check data start on quarter
    try:
        data['Datetime'] = pd.to_datetime(data['Datetime'], utc=True)
        start_minute = data.loc[0, 'Datetime'].minute
        if start_minute % 15 != 0:
            raise ValueError(
                f"Data does not start on a quarter hour (minute 0, 15, 30, 45). Started at minute: {start_minute}")
        print("Data successfully verified to start on a quarter hour.")
    except:
        raise ValueError("Something went wrong with the datetime conversion.")

    # Check for nan values
    nan_rows = data[data['Imbalance Price'].isnull()]
    if len(nan_rows) > 0:
        # Removing all rows in a quarter where a nan value is found
        faulty_quarters = nan_rows["Datetime"].dt.floor('15T').unique()
        full_data_quarters = data['Datetime'].dt.floor('15T')
        keep_mask = ~full_data_quarters.isin(faulty_quarters)
        data = data[keep_mask]
    return data.reset_index(drop=True)


def simulate_policy(env, all_data, theta_buy, theta_sell):
    """
    Simulates the trading policy over the given price data and returns the total profit.

    Args:
        prices: A NumPy array of price data.
        theta_buy: The price threshold for buying/charging.
        theta_sell: The price threshold for selling/discharging.
        battery_capacity_mwh: The maximum capacity of the battery in MWh.
        charge_discharge_rate_mw: The charge/discharge rate in MW.
    """
    decision_maker = QuarterlyTrendDecisionMaker(
        theta_buy=theta_buy,
        theta_sell=theta_sell,
        past_prices_needed=env.number_of_past_prices
    )
    history_df = pd.DataFrame(run_evaluation(env, decision_maker))
    history_df['prices'] = env.prices
    history_df['Datetime'] = all_data['Datetime']
    plot_simulation_results_minute_by_minute(history_df, 0, 1440)

if __name__ == '__main__':
    # TODO parse args for grid search ranges, or running a single evaluation.
    # --- SETUP ---
    # Load data once
    raw_data_path = '../../data/2025_minute.csv'
    cleaned_data_cache_path = '../../data/2025_minute_cleaned.pkl'  # .pkl for pickle format
    rl_model_path = 'dqn_battery_trading_model'

    # Check if the cleaned data file already exists
    if os.path.exists(cleaned_data_cache_path):
        # If it exists, load it directly (this is very fast)
        print(f"Loading cached cleaned data from: {cleaned_data_cache_path}")
        cleaned_df = pd.read_pickle(cleaned_data_cache_path)
    else:
        # If it does not exist, perform the slow cleaning process
        print("No cached data found. Running the full cleaning process...")
        raw_df = pd.read_csv(raw_data_path, sep=';')
        cleaned_df = clean_data(raw_df)

        # Save the cleaned DataFrame to the cache file for next time
        print(f"Saving cleaned data to cache: {cleaned_data_cache_path}")
        cleaned_df.to_pickle(cleaned_data_cache_path)

    if not os.path.exists(f"{rl_model_path}.zip"):
        print("--- Geen getraind RL-model gevonden. Starten van de training... ---")
        train_rl_agent(
            data_path=cleaned_data_cache_path,
            model_save_path=rl_model_path,
            total_timesteps=200000  # Zet dit hoger voor betere resultaten
        )
    else:
        print(f"--- Getraind RL-model gevonden op: {rl_model_path}.zip ---")

    # From this point on, you can use 'cleaned_df'
    price_data = cleaned_df['Imbalance Price']
    # Create the environment instance once
    # Make sure your env class accepts and stores 'number_of_past_prices'
    env = BatteryTradingEnv(
        battery_capacity_mwh=10.0,
        charge_discharge_rate_mw=5.0,
        prices=price_data,
        number_of_past_prices=5
    )

    # --- DEFINE THE GRID SEARCH SPACE ---
    buy_thresholds = np.arange(0, 51, 10)
    sell_thresholds = np.arange(100, 201, 20)

    # --- EXECUTE THE GRID SEARCH ---
    # perform_grid_search(env, buy_thresholds, sell_thresholds)

    # --- EXECUTE HEURISTIC's SINGLE EVALUATION WITH BEST PARAMETERS ---
    simulate_policy(env, cleaned_df,0, 120)
    # ---
    env.close()
