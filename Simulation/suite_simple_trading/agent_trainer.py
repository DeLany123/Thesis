import os
import pandas as pd
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3 import DQN
from .model import BaseBatteryEnv


def train_dqn_agent(
        env: BaseBatteryEnv,
        model_save_path: str,
        total_timesteps: int = 200000,
        dqn_params: dict = None
):
    """
    Initializes a BatteryTradingEnv, trains a DQN agent, and saves the model.

    Args:
        env: The trading environment instance.
        model_save_path: Path to save the trained model.
        total_timesteps: The number of training steps.
        dqn_params: Dictionary of hyperparameters for the DQN agent.
    """
    if dqn_params is None:
        dqn_params = {
            'learning_rate':5 * 1e-4,  # How big are the update steps for the neural network.
            'buffer_size': 100_000,
            # **EXPERIENCE REPLAY**: How many (state, action, reward, next_state) transitions to store.
            'learning_starts': 1000,  # How many random steps to take before starting to learn from the buffer.
            'batch_size': 512,
            # **EXPERIENCE REPLAY**: How many transitions to sample from the buffer for each training update.
            'gamma': 0.99,  # Discount factor for future rewards.
            'tau': 0.1
        }

    print("Creating the DQN agent...")
    model = DQN("MlpPolicy", env, **dqn_params)
    print("Agent created.")

    print(f"\n--- Starting Training for {total_timesteps} Timesteps ---")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    print("--- Training Complete ---")

    model.save(model_save_path)
    print(f"Trained model saved to: {model_save_path}.zip")


def train_ppo_agent(
        env: BaseBatteryEnv,
        model_save_path: str,
        total_timesteps: int = 200000,
        ppo_params: dict = None
):
    """
    Initializes a masking-compatible environment, trains a MaskablePPO agent, and saves the model.
    """
    # if ppo_params is None:
    #     ppo_params = {
    #         'n_steps': 2048,           # Number of steps to collect per update
    #         'batch_size': 64,          # Minibatch size for the update
    #         'n_epochs': 10,            # Number of times to iterate over the collected data
    #         'gamma': 0.99,
    #         'learning_rate': 0.0003,
    #         'verbose': 1,
    #         'tensorboard_log': "./ppo_tensorboard_logs/"
    #     }

    # 3. Create the MaskablePPO agent
    print("Creating the MaskablePPO agent...")
    model = MaskablePPO("MlpPolicy", env)
    print("Agent created.")

    # 4. Train the agent
    print(f"\n--- Starting PPO Training for {total_timesteps} Timesteps ---")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    print("--- Training Complete ---")

    # 5. Save the trained model
    model.save(model_save_path)
    print(f"Trained model saved to: {model_save_path}.zip")