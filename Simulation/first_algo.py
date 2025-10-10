import numpy as np
import pandas as pd


def simulate_policy(prices, theta_buy, theta_sell, battery_capacity_mwh, charge_discharge_rate_mw):
    """
    Simuleert een heuristische policy die per minuut werkt maar beslissingen per kwartier neemt.
    """
    soc_mwh = 0.0
    total_profit = 0.0

    current_quarter_decision = 0

    # Amount of datapoints to look in the past
    confirmation_period_minutes = 5

    for i, price in enumerate(prices):

        # Only take a decision at the start of each quarter (every 15 minutes)
        if i % 15 == 0:
            current_quarter_decision = 0

            if i >= confirmation_period_minutes:
                trend_check_prices = prices[i - confirmation_period_minutes: i]

                # Check if charging can start
                is_charge_signal = all(p <= theta_buy for p in trend_check_prices)
                if is_charge_signal:
                    current_quarter_decision = 1

                is_discharge_signal = all(p >= theta_sell for p in trend_check_prices)
                if is_discharge_signal:
                    current_quarter_decision = -1

        # Logic after taking a decision
        decision_this_minute = current_quarter_decision

        # Battery constraints
        if decision_this_minute == 1 and soc_mwh >= battery_capacity_mwh:
            decision_this_minute = 0 # Battery is full
        elif decision_this_minute == -1 and soc_mwh <= 0:
            decision_this_minute = 0  # Battery is empty

        # Update battery
        if decision_this_minute == 1:
            energy_to_charge = charge_discharge_rate_mw * (1 / 60)
            actual_charge = min(energy_to_charge, battery_capacity_mwh - soc_mwh)
            soc_mwh += actual_charge

            cost = actual_charge * price
            total_profit -= cost

        elif decision_this_minute == -1:
            energy_to_discharge = charge_discharge_rate_mw * (1 / 60)
            actual_discharge = min(energy_to_discharge, soc_mwh)
            soc_mwh -= actual_discharge
            revenue = actual_discharge * price
            total_profit += revenue

    return total_profit


def grid_search_tuning(prices, buy_thresholds, sell_thresholds, battery_capacity_mwh, charge_discharge_rate_mw):
    """
    Performs a grid search to find the optimal buy/sell thresholds.
    """
    best_profit = -np.inf
    best_params = {'buy': None, 'sell': None}
    print("Amount of loops to perform:", len(buy_thresholds))
    # Loop through every combination of parameters
    for theta_buy in buy_thresholds:
        for theta_sell in sell_thresholds:

            # Ensure sell threshold is higher than buy threshold to be logical
            if theta_sell <= theta_buy:
                continue

            # Simulate the policy with the current parameter set
            profit = simulate_policy(prices, theta_buy, theta_sell, battery_capacity_mwh, charge_discharge_rate_mw)

            print(f"Testing: Buy at {theta_buy:.2f}, Sell at {theta_sell:.2f} -> Profit: {profit:.2f} EUR")

            # Check if this is the best result so far
            if profit > best_profit:
                best_profit = profit
                best_params['buy'] = theta_buy
                best_params['sell'] = theta_sell

    return best_params, best_profit


# --- 3. Run the Example ---

if __name__ == '__main__':
    data = pd.read_csv('../data/2025_minute.csv', sep=';')
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

    prices = data['Imbalance Price'].to_numpy()
    # Define battery characteristics
    CAPACITY_MWH = 10.0
    RATE_MW = 2.0

    # Define the grid of parameters to search
    buy_range = np.arange(80, 101, 5)  # Search from 80 to 100 in steps of 5
    sell_range = np.arange(105, 126, 5)  # Search from 105 to 125 in steps of 5

    print("--- Starting Grid Search ---")

    optimal_params, optimal_profit = grid_search_tuning(prices, buy_range, sell_range, CAPACITY_MWH, RATE_MW)

    print("\n--- Grid Search Complete ---")
    print(f"Optimal Buy Threshold: {optimal_params['buy']:.2f} EUR/MWh")
    print(f"Optimal Sell Threshold: {optimal_params['sell']:.2f} EUR/MWh")
    print(f"Resulting Best Profit: {optimal_profit:.2f} EUR")