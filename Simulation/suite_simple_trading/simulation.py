from Simulation.suite_simple_trading.model import BatteryTradingEnv1, BaseBatteryEnv


def run_evaluation(env: BaseBatteryEnv, decision_maker) -> dict:
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
    # Simulate steps required for the history required by the state
    if env.number_of_past_prices > 0:
        for _ in range(env.number_of_past_prices):
            action = env.get_idle_action()
            obs, reward, terminated, truncated, info = env.step(action)
            energy_charged_discharged = info.get('energy_charged_discharged', 0)

            prices_history.append(obs[1])
            soc_history.append(env.soc_mwh)  # Get current SoC from the env
            action_history.append(action)
            reward_history.append(reward)
            energy_charged_discharged_history.append(energy_charged_discharged)

    while not done:
        action = decision_maker.get_action(obs, env.current_step)
        obs, reward, terminated, truncated, info = env.step(action)

        energy_charged_discharged = info.get('energy_charged_discharged', 0)

        prices_history.append(obs[1])
        soc_history.append(env.soc_mwh)  # Get current SoC from the env
        action_history.append(action)
        reward_history.append(reward)
        energy_charged_discharged_history.append(energy_charged_discharged)

        done = terminated or truncated

    # Return all collected data in a dictionary
    return {
        "prices": prices_history,
        "soc": soc_history,
        "actions": action_history,
        "rewards": reward_history,
        "energy_charged_discharged": energy_charged_discharged_history,
    }
