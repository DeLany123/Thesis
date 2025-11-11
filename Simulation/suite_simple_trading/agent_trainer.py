import os
import pandas as pd
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

from .model import BaseBatteryEnv
from .observation_wrappers import RobustScalingWrapper


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
        reward_save_path: str,
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
    train_env_scaled = RobustScalingWrapper(env)
    model = MaskablePPO(MaskableActorCriticPolicy, train_env_scaled)
    print("Agent created.")

    reward_callback = EpisodeRewardCallback(save_path=reward_save_path, verbose=1)

    # 4. Train the agent
    print(f"\n--- Starting PPO Training for {total_timesteps} Timesteps ---")
    model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=reward_callback)
    print("--- Training Complete ---")

    # 5. Save the trained model
    model.save(model_save_path)
    print(f"Trained model saved to: {model_save_path}.zip")


class EpisodeRewardCallback(BaseCallback):
    """
    A custom callback to log the total reward of each episode and save them at the end of training.
    """

    def __init__(self, save_path: str, verbose: int = 0):
        super(EpisodeRewardCallback, self).__init__(verbose)
        self.save_path = save_path
        self.episode_rewards = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        """
        This method is called after each step in the environment.
        It checks if an episode has just finished and logs the reward.
        """
        # In a non-vectorized env, `dones` is a boolean array of size 1.
        # It's True when an episode has terminated.
        if self.locals['dones'][0]:
            # The 'infos' dict contains the 'episode' key when an episode ends.
            # 'r' is the total reward for that episode.
            episode_reward = self.locals['infos'][0]['episode']['r']
            self.episode_rewards.append(episode_reward)
            self.episode_count += 1

            # Optional: Print progress every 10 episodes
            if self.verbose > 0 and self.episode_count % 10 == 0:
                print(f"Episode {self.episode_count} finished. Total Reward: {episode_reward:.2f}")

        return True  # Must return True to continue training

    def _on_training_end(self) -> None:
        """
        This method is called at the end of the training process.
        It saves the collected episode rewards to a file.
        """
        print("\n--- Training finished. Saving episode rewards. ---")
        # Save the rewards as a numpy archive.
        np.savez(self.save_path, rewards=np.array(self.episode_rewards))
        print(f"Episode rewards saved to: {self.save_path}")