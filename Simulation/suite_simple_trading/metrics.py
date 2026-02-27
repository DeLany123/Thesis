import pandas as pd
import numpy as np

from Simulation.suite_simple_trading.simulation import EvaluationResult


def amount_of_good_quarters(result_data: pd.DataFrame) -> int:
    """
    Gives the number of quarters where the reward was strictly positive.
    We assume the reward is only given at the last minute of the quarter (minute 14).
    """
    # Filter for the end of the quarter where rewards are typically calculated
    # Assuming rows are minutes and sequential: 14, 29, 44, 59...
    end_of_quarter_rows = result_data.iloc[14::15]

    # Count rows where reward is positive
    count = (end_of_quarter_rows['real_rewards'] > 0).sum()
    return int(count)


def amount_of_bad_quarters(result_data: pd.DataFrame) -> int:
    """
    Gives the number of quarters where the reward was strictly negative.
    """
    end_of_quarter_rows = result_data.iloc[14::15]
    count = (end_of_quarter_rows['real_rewards'] < 0).sum()
    return int(count)


def good_quarter_ratio(result_data: pd.DataFrame) -> float:
    """
    Calculates the ratio of profitable quarters out of all quarters where
    a non-zero reward (profit or loss) occurred.

    Uses previously defined functions for consistency.

    Returns:
        float: A value between 0.0 and 1.0. Returns 0.0 if no active quarters exist.
    """
    n_good = amount_of_good_quarters(result_data)
    n_bad = amount_of_bad_quarters(result_data)

    total_active = n_good + n_bad

    if total_active == 0:
        return 0.0

    return float(n_good / total_active)


def mean_reward_per_quarter(result_data: pd.DataFrame) -> float:
    """
    Calculates the average reward over ALL quarters (including idle ones).
    """
    end_of_quarter_rows = result_data.iloc[14::15]
    amount_of_non_zero_quarters = end_of_quarter_rows[end_of_quarter_rows != 0]

    if len(end_of_quarter_rows) == 0:
        return 0.0

    return float(amount_of_non_zero_quarters['real_rewards'].mean())


def bad_quarter_started_good(result_data: pd.DataFrame) -> int:
    """
    Quarters where the outcome was negative (loss), but the initial price signals 
    suggested a profit (e.g., started with negative price for charging, 
    but ended with positive price).

    Logic:
    1. Identify quarters with negative final reward.
    2. Check the price at the START (minute 0) of those quarters.
    3. Count if:
       - We Charged, and Start Price was < 0 (Looked like we would get paid to charge)
       - We Discharged, and Start Price was > 0 (Looked like we would get paid to sell)
    """
    # Create a grouping key for every 15 minutes (0-14 is group 0, 15-29 is group 1, etc.)
    result_data = result_data.copy()
    result_data['quarter_group'] = np.arange(len(result_data)) // 15

    bad_luck_count = 0

    # Iterate through quarters (this is reasonably fast for evaluation data)
    # Grouping lets us look at the start (min 0) and end (min 14) together
    for _, group in result_data.groupby('quarter_group'):

        final_step = group.iloc[-1]
        start_step = group.iloc[0]

        # Only look at quarters that ended in a loss
        if final_step['real_rewards'] < 0:

            # Did we take action? (Sum of actions > 0 implies we did something)
            # Or check if energy traded is non-zero
            total_charged = final_step['total_charged_per_quarter']
            total_discharged = final_step['total_discharged_per_quarter']

            start_price = start_step['prices']

            # Case 1: We charged mostly, and price started negative (looked good!)
            if total_charged > total_discharged and start_price < 0:
                bad_luck_count += 1

            # Case 2: We discharged mostly, and price started positive (looked good!)
            elif total_discharged > total_charged and start_price > 0:
                bad_luck_count += 1

    return bad_luck_count


def mean_daily_reward(result_data: pd.DataFrame) -> float:
    """
    Calculates the average reward per 24-hour period (Day).
    This includes idle time, making it a good metric for overall profitability.
    """
    total_reward = result_data['real_rewards'].sum()

    # Assuming 1 row = 1 minute. 1 Day = 1440 minutes.
    total_minutes = len(result_data)
    total_days = total_minutes / 1440.0

    if total_days == 0:
        return 0.0

    return float(total_reward / total_days)


def profit_per_battery_cycle(result_data: pd.DataFrame, battery_capacity_mwh: float = 10.0) -> float:
    """
    Calculates the profit earned per complete battery cycle.

    A battery cycle is defined as the equivalent of charging the battery from
    empty to full capacity once. This metric helps evaluate profitability
    relative to battery wear/degradation.

    Args:
        result_data: DataFrame containing simulation results with 'energy_charged_discharged' and 'real_rewards'
        battery_capacity_mwh: Battery capacity in MWh (default: 10.0 MWh)

    Returns:
        float: Profit (€) per complete battery cycle. Returns 0.0 if no cycles occurred.

    Note:
        - Total cycles = Total energy throughput / (2 * battery_capacity)
        - We divide by 2 because one cycle involves both charging AND discharging
        - Alternatively: cycles = total_charged / battery_capacity
    """
    # Calculate total profit
    total_profit = result_data['real_rewards'].sum()

    # Calculate total energy charged (absolute value since energy_charged_discharged can be negative)
    # Positive values = charging, Negative values = discharging
    total_charged = result_data[result_data['energy_charged_discharged'] > 0]['energy_charged_discharged'].sum()

    # Calculate number of complete cycles
    # One cycle = charging the battery capacity once (then discharging it)
    num_cycles = total_charged / battery_capacity_mwh

    if num_cycles == 0:
        return 0.0

    return float(total_profit / num_cycles)


def battery_cycle_count(result_data: pd.DataFrame, battery_capacity_mwh: float = 10.0) -> float:
    """
    Calculates the total number of complete battery cycles.

    This is useful for understanding battery degradation and wear.

    Args:
        result_data: DataFrame containing simulation results with 'energy_charged_discharged'
        battery_capacity_mwh: Battery capacity in MWh (default: 10.0 MWh)

    Returns:
        float: Number of complete battery cycles
    """
    # Calculate total energy charged
    total_charged = result_data[result_data['energy_charged_discharged'] > 0]['energy_charged_discharged'].sum()

    # Calculate number of complete cycles
    num_cycles = total_charged / battery_capacity_mwh

    return float(num_cycles)


def print_agent_performance(df: pd.DataFrame, agent_name: str = "Agent", battery_capacity_mwh: float = 10.0):
    """
    Calculates and prints a formatted performance report using the defined metrics.

    Args:
        df: DataFrame containing simulation results
        agent_name: Name of the agent for display
        battery_capacity_mwh: Battery capacity in MWh (default: 10.0)
    """
    # 0. Data Cleaning
    df = df.copy()

    # 1. Calculate Metrics
    n_good = amount_of_good_quarters(df)
    n_bad = amount_of_bad_quarters(df)
    win_ratio = good_quarter_ratio(df)
    avg_reward = mean_reward_per_quarter(df)
    trap_quarters = bad_quarter_started_good(df)
    avg_daily_reward = mean_daily_reward(df)
    profit_per_cycle = profit_per_battery_cycle(df, battery_capacity_mwh)
    total_cycles = battery_cycle_count(df, battery_capacity_mwh)

    # 2. Print Report
    print(f"========================================")
    print(f"  PERFORMANCE REPORT: {agent_name}")
    print(f"========================================")
    print(f"  Pos. Profit Quarters : {n_good}")
    print(f"  Neg. Profit Quarters : {n_bad}")
    print(f"----------------------------------------")
    print(f"  Success Rate (Active): {win_ratio:.2%}")
    print(f"  Mean Reward / active Quarter: €{avg_reward:.4f}")
    print(f"  Mean Daily Reward    : €{avg_daily_reward:.4f}")
    print(f"----------------------------------------")
    print(f"  Battery Cycles       : {total_cycles:.2f}")
    print(f"  Profit per Cycle     : €{profit_per_cycle:.2f}")
    print(f"----------------------------------------")
    print(f"  'Trap' Quarters      : {trap_quarters}")
    print(f"  (Started good -> Ended bad)")
    print(f"========================================\n")