from Simulation.suite_simple_trading.model import BaseBatteryEnv
from Simulation.suite_simple_trading.policy import DecisionMaker


def run_evaluation(
        env: BaseBatteryEnv,
        decision_maker: DecisionMaker,
        number_of_episodes: int = 1
) -> dict:
    """
    Runs a single evaluation and returns a detailed history of the simulation.
    """
    prices_history = []
    soc_history = []
    action_history = []
    reward_history = []
    energy_charged_discharged_history = []
    episodic_rewards = []

    for episode_num in range(number_of_episodes):
        print(f"Starting episode {episode_num + 1}/{number_of_episodes}")
        obs, info = env.reset()
        decision_maker.reset()


        done = False
        # Simulate steps required for the history required by the state
        # TODO add number_of_past_prices to base env if needed again
        # if env.number_of_past_prices > 0:
        #    run_paste_evaluations(env, env.number_of_past_prices)
        reward_per_episode = 0
        while not done:
            action = decision_maker.get_action(obs, env.current_step)
            obs, reward, terminated, truncated, info = env.step(action)

            energy_charged_discharged = info.get('energy_charged_discharged', 0)

            prices_history.append(obs[1])
            soc_history.append(env.soc_mwh)  # Get current SoC from the env
            action_history.append(action)
            reward_history.append(reward)
            reward_per_episode += reward
            energy_charged_discharged_history.append(energy_charged_discharged)

            done = terminated or truncated

        episodic_rewards.append(reward_per_episode)
    # Return all collected data in a dictionary
    return {
        "prices": prices_history,
        "soc": soc_history,
        "actions": action_history,
        "rewards": reward_history,
        "energy_charged_discharged": energy_charged_discharged_history,
        "episodic_rewards": episodic_rewards
    }


def run_paste_evaluations(
        env: BaseBatteryEnv,
        history_needed: int = 10
) -> dict:
    """
    Runs multiple step to get the amount of history that is needed before running the actual evaluation with rewards.
    """
    prices_history = []
    soc_history = []
    action_history = []
    reward_history = []
    energy_charged_discharged_history = []

    for _ in range(history_needed):
            action = env.get_idle_action()
            obs, reward, terminated, truncated, info = env.step(action)
            energy_charged_discharged = info.get('energy_charged_discharged', 0)

            prices_history.append(obs[1])
            soc_history.append(env.soc_mwh)  # Get current SoC from the env
            action_history.append(action)
            reward_history.append(reward)
            energy_charged_discharged_history.append(energy_charged_discharged)

    return {
        "prices": prices_history,
        "soc": soc_history,
        "actions": action_history,
        "rewards": reward_history,
        "energy_charged_discharged": energy_charged_discharged_history,
    }