from Simulation.suite_simple_trading.model import BatteryTradingEnv1, BaseBatteryEnv
import numpy as np

from Simulation.suite_simple_trading.policy import QuarterlyTrendDecisionMaker
from Simulation.suite_simple_trading.simulation import run_evaluation


def perform_grid_search(env: BaseBatteryEnv, buy_range: np.ndarray, sell_range: np.ndarray):
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