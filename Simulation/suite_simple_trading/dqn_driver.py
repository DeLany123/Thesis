import os
import pandas as pd
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
            'learning_rate': 1e-4,  # How big are the update steps for the neural network.
            'buffer_size': 100_000,
            # **EXPERIENCE REPLAY**: How many (state, action, reward, next_state) transitions to store.
            'learning_starts': 5_000,  # How many random steps to take before starting to learn from the buffer.
            'batch_size': 32,
            # **EXPERIENCE REPLAY**: How many transitions to sample from the buffer for each training update.
            'gamma': 0.99,  # Discount factor for future rewards.
            'target_update_interval': 500,  # How often to update the 'fixed' target network.
            'exploration_fraction': 0.1,  # Fraction of training to spend decreasing the exploration rate.
            'exploration_final_eps': 0.05,  # The minimum exploration rate.
            'verbose': 1,
            'tensorboard_log': "./dqn_tensorboard_logs/"
        }

    print("Creating the DQN agent...")
    model = DQN("MlpPolicy", env, **dqn_params)
    print("Agent created.")

    print(f"\n--- Starting Training for {total_timesteps} Timesteps ---")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    print("--- Training Complete ---")

    model.save(model_save_path)
    print(f"Trained model saved to: {model_save_path}.zip")

