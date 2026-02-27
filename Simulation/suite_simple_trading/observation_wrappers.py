import gymnasium as gym
import numpy as np
import pandas as pd
from typing import Literal

from Simulation.suite_simple_trading.model import BaseBatteryEnv


class RobustScalingWrapper(gym.ObservationWrapper):
    """
    A flexible observation scaling_comparison wrapper that supports different scaling_comparison methods.
    It is designed to be robust to outliers in the data.
    """

    def __init__(
            self,
            env: BaseBatteryEnv,
            method: Literal['standard', 'robust', 'minmax_clipped'] = 'robust'
    ):
        super().__init__(env)
        print(f"--- Initializing Robust Observation Scaling Wrapper (Method: {method}) ---")

        self.method = method

        # --- For Min-Max Scaling (SoC and Trades - their bounds are known and fixed) ---
        self.soc_min = 0.0
        self.soc_max = env.battery_capacity_mwh
        self.trade_min = 0.0
        self.trade_max = env.charge_discharge_rate * (15 / 60)

        # --- Fit the scaler for the PRICE feature using the chosen method ---
        price_series = env.prices

        if self.method == 'standard':
            self.price_param1 = price_series.mean()
            self.price_param2 = price_series.std()
            print(f"Price (Standard): Mean={self.price_param1:.2f}, Std={self.price_param2:.2f}")

        elif self.method == 'robust':
            self.price_param1 = np.median(price_series)
            q1 = np.quantile(price_series, 0.25)
            q3 = np.quantile(price_series, 0.75)
            self.price_param2 = q3 - q1  # This is the IQR
            print(f"Price (Robust): Median={self.price_param1:.2f}, IQR={self.price_param2:.2f}")

        elif self.method == 'minmax_clipped':
            # This method clips extreme outliers before scaling_comparison
            p01 = np.quantile(price_series, 0.01)
            p99 = np.quantile(price_series, 0.99)
            self.price_param1 = p01
            self.price_param2 = p99 - p01  # This is the range after clipping
            print(f"Price (Clipped MinMax): Min(1%)={self.price_param1:.2f}, Range={self.price_param2:.2f}")

        else:
            raise ValueError("Method must be one of 'standard', 'robust', or 'minmax_clipped'")

        # Add a small epsilon to denominators to prevent division by zero
        if self.price_param2 == 0:
            self.price_param2 = 1e-8

        print("--- Wrapper ready. ---")

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """Applies the chosen scaling_comparison method to the observation vector."""
        # Assuming observation is [soc, price, total_charged, total_discharged]
        scaled_obs = np.copy(obs)

        # --- Apply Min-Max scaling_comparison to features with fixed, known bounds ---
        scaled_obs[0] = (obs[0] - self.soc_min) / (self.soc_max - self.soc_min + 1e-8)
        scaled_obs[2] = (obs[2] - self.trade_min) / (self.trade_max - self.trade_min + 1e-8)
        scaled_obs[3] = (obs[3] - self.trade_min) / (self.trade_max - self.trade_min + 1e-8)

        # --- Apply the chosen scaling_comparison method to the PRICE feature (index 1) ---
        price = obs[1]
        if self.method == 'standard':
            scaled_obs[1] = (price - self.price_param1) / self.price_param2
        elif self.method == 'robust':
            scaled_obs[1] = (price - self.price_param1) / self.price_param2
        elif self.method == 'minmax_clipped':
            clipped_price = np.clip(price, self.price_param1, self.price_param1 + self.price_param2)
            scaled_obs[1] = (clipped_price - self.price_param1) / self.price_param2

        # Final clip on SoC and trades to ensure they are perfectly in [0,1]
        scaled_obs[[0, 2, 3]] = np.clip(scaled_obs[[0, 2, 3]], 0.0, 1.0)

        return scaled_obs.astype(np.float32)

    def action_masks(self):
        return self.env.action_masks()


class SoCPenaltyWrapper(gym.Wrapper):
    """
    Adds a penalty if the battery is not fully charged, as described in the paper.
    Formula: Penalty = weight * (SoC_max - SoC_current)

    This encourages the agent to keep the battery charged to be ready for price peaks.
    """
    def __init__(self, env: gym.Env, weight: float = 0.1):
        """
        Args:
            env: The environment to wrap.
            weight: The 'omega' parameter from the paper.
                    Controls the strength of the penalty.
                    Start small (e.g., 0.01 or 0.1).
        """
        super().__init__(env)
        self.weight = weight
        # We need to access the max capacity from the base environment
        # Assuming your BaseBatteryEnv has 'battery_capacity_mwh'
        self.soc_max = self.env.unwrapped.battery_capacity_mwh

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 1. Get the current SoC
        # We access the unwrapped environment to get the true physical value
        current_soc = self.env.unwrapped.soc_mwh

        # 2. Calculate the Penalty (Equation 31)
        # penalty is positive number representing the "cost"
        penalty_value = self.weight * (self.soc_max - current_soc)

        # 3. Subtract penalty from the reward
        new_reward = reward - penalty_value

        # 4. Optional: Log for debugging
        info['soc_penalty'] = penalty_value
        info['original_reward'] = reward

        return obs, new_reward, terminated, truncated, info

    def action_masks(self):
        return self.env.action_masks()


class FlexibleSoCPenaltyWrapper(gym.Wrapper):
    """
    Applies a penalty based on how much the battery is empty (Missing Charge).

    The penalty is calculated based on the 'Deficit Ratio' (0.0 = Full, 1.0 = Empty).

    Modes:
    - 'linear': Penalty grows evenly. 50% empty = 0.5 * weight.
    - 'quadratic': Penalty is low at start, high at end. 50% empty = 0.25 * weight.
                   (Good for allowing small trades but preventing empty batteries).
    - 'root': Penalty is high immediately. 50% empty = 0.71 * weight.
              (Good for forcing the agent to keep it topped up strictly).
    """

    def __init__(
            self,
            env: gym.Env,
            weight: float = 1.0,
            mode: Literal['linear', 'quadratic', 'root'] = 'linear'
    ):
        super().__init__(env)
        self.weight = weight
        self.mode = mode

        # Access max capacity to calculate ratios
        self.soc_max = self.env.unwrapped.battery_capacity_mwh

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # 1. Calculate Deficit Ratio (0.0 = Full, 1.0 = Empty)
        current_soc = self.env.unwrapped.soc_mwh
        # Clip to ensure numerical stability (0 to 1)
        deficit_ratio = np.clip((self.soc_max - current_soc) / self.soc_max, 0.0, 1.0)

        # 2. Apply the Shape Function
        if self.mode == 'linear':
            # f(x) = x
            shape_val = deficit_ratio

        elif self.mode == 'quadratic':
            # f(x) = x^2
            # Penalizes 10% empty very little (0.01), but 100% empty fully (1.0).
            shape_val = deficit_ratio ** 2

        elif self.mode == 'root':
            # f(x) = sqrt(x)
            # Penalizes 10% empty heavily (0.31).
            shape_val = np.sqrt(deficit_ratio)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # 3. Calculate Final Penalty
        penalty_value = self.weight * shape_val

        # 4. Subtract from Reward
        new_reward = reward - penalty_value

        # Log for debugging
        info['soc_penalty'] = penalty_value

        return obs, new_reward, terminated, truncated, info

    def action_masks(self):
        return self.env.action_masks()