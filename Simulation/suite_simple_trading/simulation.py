import gymnasium as gym

from Simulation.suite_simple_trading.model import BaseBatteryEnv
from Simulation.suite_simple_trading.policy import DecisionMaker


def run_evaluation(
        scaled_env: gym.Env,
        decision_maker: DecisionMaker,
        number_of_episodes: int = 1
) -> dict:
    """
    Runs an evaluation using a (potentially wrapped) environment and returns a detailed history.
    """
    unwrapped_env = scaled_env.env

    # Initialize lists to store the history of UN-SCALED, human-readable data
    prices_history = []
    soc_history = []
    total_charged_per_quarter_history = []
    total_discharged_per_quarter_history = []
    action_history = []
    reward_history = []
    energy_charged_discharged_history = []
    episodic_rewards = []

    for episode_num in range(number_of_episodes):
        print(f"Starting episode {episode_num + 1}/{number_of_episodes}")

        # --- INTERACT WITH THE WRAPPER ---
        # Call .reset() on the scaled_env. It returns a SCALED observation.
        obs, info = scaled_env.reset()

        if hasattr(decision_maker, 'reset'):
            decision_maker.reset()

        # --- ACCESS STATE FROM THE UNWRAPPED ENV ---
        # For logging, we use the original env to get the correct start/end times.
        start_time = unwrapped_env.all_data.iloc[unwrapped_env.current_step]['Datetime']
        end_time = unwrapped_env.all_data.iloc[unwrapped_env.current_episode_end_step]['Datetime']
        print(f"From {start_time} to {end_time}")

        done = False
        reward_per_episode = 0
        while not done:
            action = decision_maker.get_action(obs, unwrapped_env.current_step)

            obs, reward, terminated, truncated, info = scaled_env.step(action)

            energy_charged_discharged = info.get('energy_charged_discharged', 0)

            # Get the TRUE price, not the scaled one from obs[1]
            prices_history.append(unwrapped_env.prices[unwrapped_env.current_step - 1])
            soc_history.append(unwrapped_env.soc_mwh)
            total_charged_per_quarter_history.append(unwrapped_env.total_charged_in_quarter)
            total_discharged_per_quarter_history.append(unwrapped_env.total_discharged_in_quarter)

            action_history.append(action)
            reward_history.append(reward)
            reward_per_episode += reward
            energy_charged_discharged_history.append(energy_charged_discharged)

            done = terminated or truncated

        episodic_rewards.append(reward_per_episode)
        print(f"Finished with total reward: {reward_per_episode:.2f}")

    return {
        "prices": prices_history,
        "soc": soc_history,
        "total_charged_per_quarter": total_charged_per_quarter_history,
        "total_discharged_per_quarter": total_discharged_per_quarter_history,
        "actions": action_history,
        "rewards": reward_history,
        "energy_charged_discharged": energy_charged_discharged_history,
        "episodic_rewards": episodic_rewards
    }

# NOTE: The run_paste_evaluations function should also be updated if you use it,
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