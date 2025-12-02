import gymnasium as gym

from Simulation.suite_simple_trading.model import BaseBatteryEnv
from Simulation.suite_simple_trading.policy import DecisionMaker

import gymnasium as gym
import pandas as pd
from stable_baselines3.common.base_class import BaseAlgorithm

# Import your custom environment class for type hinting
from Simulation.suite_simple_trading.model import BaseBatteryEnv


def run_evaluation(
        scaled_env: gym.Env,
        model: BaseAlgorithm,
        is_masked: bool = True,
        number_of_episodes: int = 1
) -> dict:
    """
    Runs an evaluation using a fully wrapped environment and a trained SB3 model.
    """
    # Use the .unwrapped attribute to get the original BaseBatteryEnv for logging
    unwrapped_env: BaseBatteryEnv = scaled_env.unwrapped

    # Initialize lists to store the history of UN-SCALED, human-readable data
    prices_history = []
    soc_history = []
    total_charged_per_quarter_history = []
    total_discharged_per_quarter_history = []
    action_history = []
    scaled_reward_history = []
    real_reward_history = []
    energy_charged_discharged_history = []
    episodic_rewards = []

    for episode_num in range(number_of_episodes):
        print(f"Starting episode {episode_num + 1}/{number_of_episodes}")

        # --- Interact with the SCALED environment ---
        obs, info = scaled_env.reset()

        # --- Use the unwrapped environment for logging ---
        start_time = unwrapped_env.all_data.iloc[unwrapped_env.current_step]['Datetime']
        end_time = unwrapped_env.all_data.iloc[unwrapped_env.current_episode_end_step]['Datetime']
        print(f"From {start_time} to {end_time}")

        done = False
        reward_per_episode = 0
        while not done:
            # --- Get the action directly from the MODEL ---
            # We get the action mask from the unwrapped env because wrappers hide custom methods
            action_mask = unwrapped_env.action_masks()

            # The model predicts based on the SCALED observation
            if is_masked:
                action, _states = model.predict(
                    obs,
                    deterministic=True,  # Use deterministic mode for evaluation
                    action_masks=action_mask
                )
            else:
                action, _states = model.predict(obs)
            action = int(action)

            # --- Step the SCALED environment ---
            obs, reward, terminated, truncated, info = scaled_env.step(action)

            # --- Log the UN-SCALED data from the unwrapped environment ---
            energy_charged_discharged = info.get('energy_charged_discharged', 0)

            prices_history.append(unwrapped_env.prices[unwrapped_env.current_step - 1])
            soc_history.append(unwrapped_env.soc_mwh)
            total_charged_per_quarter_history.append(unwrapped_env.total_charged_in_quarter)
            total_discharged_per_quarter_history.append(unwrapped_env.total_discharged_in_quarter)

            action_history.append(action)
            scaled_reward_history.append(reward)
            real_reward_history.append(info.get('real_reward', 0))
            reward_per_episode += reward
            energy_charged_discharged_history.append(energy_charged_discharged)

            done = terminated or truncated

        episodic_rewards.append(reward_per_episode)
        print(f"Finished with total (scaled) reward: {reward_per_episode:.2f}")

    return {
        "prices": prices_history,
        "soc": soc_history,
        "total_charged_per_quarter": total_charged_per_quarter_history,
        "total_discharged_per_quarter": total_discharged_per_quarter_history,
        "actions": action_history,
        "scaled_rewards": scaled_reward_history,
        "real_rewards": real_reward_history,
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