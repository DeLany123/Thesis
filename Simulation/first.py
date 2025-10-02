import numpy as np


def simulate_policy(prices, theta_buy, theta_sell, battery_capacity_mwh, charge_discharge_rate_mw):
    """
    Simuleert een heuristische policy die per minuut werkt maar beslissingen per kwartier neemt.
    """
    soc_mwh = 0.0
    total_profit = 0.0

    current_quarter_decision = 0

    # Amount of datapoints to look in the past
    confirmation_period_minutes = 3

    # Loop door elke minuut van de prijsdata. 'i' is de index van de minuut.
    for i, price in enumerate(prices):

        # --- BESLISSINGSLOGICA: Alleen aan het begin van een kwartier ---
        # De modulo operator (%) checkt of 'i' een veelvoud van 15 is.
        if i % 15 == 0:
            # We staan aan het begin van een nieuw kwartier. Reset de beslissing.
            current_quarter_decision = 0

            # We kunnen alleen een beslissing nemen als we genoeg geschiedenis hebben.
            if i >= confirmation_period_minutes:
                # Pak de prijzen van de laatste paar minuten van het VORIGE kwartier.
                trend_check_prices = prices[i - confirmation_period_minutes: i]

                # Check 1: Is dit een goed moment om te beginnen met LADEN?
                # Voorwaarde: alle recente prijzen waren onder de koopdrempel.
                is_charge_signal = all(p <= theta_buy for p in trend_check_prices)
                if is_charge_signal:
                    current_quarter_decision = 1

                # Check 2: Is dit een goed moment om te beginnen met ONTLADEN?
                # Voorwaarde: alle recente prijzen waren boven de verkoopdrempel.
                is_discharge_signal = all(p >= theta_sell for p in trend_check_prices)
                if is_discharge_signal:
                    current_quarter_decision = -1

        # --- ACTIE LOGICA: Voer de beslissing voor dit kwartier uit ---
        # Start met de beslissing die aan het begin van het kwartier is genomen.
        decision_this_minute = current_quarter_decision

        # Controleer de fysieke limieten van de batterij.
        if decision_this_minute == 1 and soc_mwh >= battery_capacity_mwh:
            decision_this_minute = 0  # Stop met laden, batterij is vol.
        elif decision_this_minute == -1 and soc_mwh <= 0:
            decision_this_minute = 0  # Stop met ontladen, batterij is leeg.

        # --- UPDATE VAN BATTERIJ EN WINST ---
        if decision_this_minute == 1:  # Opladen
            # Energieberekening voor 1 minuut (1/60 van een uur).
            energy_to_charge = charge_discharge_rate_mw * (1 / 60)
            actual_charge = min(energy_to_charge, battery_capacity_mwh - soc_mwh)
            soc_mwh += actual_charge
            # De kost wordt berekend met de prijs-schatting van DEZE minuut.
            cost = actual_charge * price
            total_profit -= cost

        elif decision_this_minute == -1:  # Ontladen
            # Energieberekening voor 1 minuut.
            energy_to_discharge = charge_discharge_rate_mw * (1 / 60)
            actual_discharge = min(energy_to_discharge, soc_mwh)
            soc_mwh -= actual_discharge
            # De opbrengst wordt berekend met de prijs-schatting van DEZE minuut.
            revenue = actual_discharge * price
            total_profit += revenue

    return total_profit


# --- 2. Implement the Grid Search ---

def grid_search_tuning(prices, buy_thresholds, sell_thresholds, battery_capacity_mwh, charge_discharge_rate_mw):
    """
    Performs a grid search to find the optimal buy/sell thresholds.
    """
    best_profit = -np.inf
    best_params = {'buy': None, 'sell': None}

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
    # Create a dummy historical price path (replace with your actual data)
    np.random.seed(42)
    dummy_prices = 100 + np.random.randn(500).cumsum()

    # Define battery characteristics
    CAPACITY_MWH = 10.0
    RATE_MW = 2.0

    # Define the grid of parameters to search
    buy_range = np.arange(80, 101, 5)  # Search from 80 to 100 in steps of 5
    sell_range = np.arange(105, 126, 5)  # Search from 105 to 125 in steps of 5

    print("--- Starting Grid Search ---")

    optimal_params, optimal_profit = grid_search_tuning(dummy_prices, buy_range, sell_range, CAPACITY_MWH, RATE_MW)

    print("\n--- Grid Search Complete ---")
    print(f"Optimal Buy Threshold: {optimal_params['buy']:.2f} EUR/MWh")
    print(f"Optimal Sell Threshold: {optimal_params['sell']:.2f} EUR/MWh")
    print(f"Resulting Best Profit: {optimal_profit:.2f} EUR")