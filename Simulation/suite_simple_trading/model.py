import gymnasium as gym
import numpy as np
import pandas as pd


class BaseBatteryEnv(gym.Env):
    """
    A BASE class for battery trading environments.
    Contains all the shared simulation logic for state transitions and rewards.
    Subclasses must implement _get_observation() and define observation_space.
    """

    def __init__(
            self,
            battery_capacity_mwh: float,
            charge_discharge_rate_mw: float,
            all_data: pd.DataFrame,
    ):
        super().__init__()

        # --- Core Simulation Parameters ---
        self.battery_capacity_mwh = battery_capacity_mwh
        self.charge_discharge_rate = charge_discharge_rate_mw
        self.all_data = all_data
        self.prices = all_data['Imbalance Price'].to_numpy()
        self.time_interval = 1 / 60
        self.max_steps = len(self.prices)

        # --- Fixed Action Space ---
        self.action_space = gym.spaces.Discrete(3)  # 0: Idle, 1: Charge, 2: Discharge

        # --- Initialize all possible state variables ---
        # All variables subclasses need to define their state space.
        self.current_step = 0
        self.soc_mwh = 0.0
        self.total_energy_traded_per_quarter = 0.0
        self.total_charged_in_quarter = 0.0
        self.total_discharged_in_quarter = 0.0

    def _get_observation(self) -> np.ndarray:
        """Abstract method: Subclasses MUST implement this."""
        raise NotImplementedError("This method must be implemented by the subclass.")

    def _get_power_rate_from_action(self, action: int) -> float:
        """Translates a discrete action into a power rate in MW."""
        if action == 0:
            return 0.0
        elif action == 1:
            return self.charge_discharge_rate
        elif action == 2:
            return -self.charge_discharge_rate
        else:
            raise ValueError(f"Invalid action {action}")

    def _calculate_delayed_reward(self) -> float:
        """Calculates reward at the end of each 15-minute interval."""
        if self.all_data['Datetime'].iloc[self.current_step].minute % 15 == 14:
            return -self.prices[self.current_step] * self.total_energy_traded_per_quarter
        return 0.0

    def reset(self, seed=None, options=None):
        """Resets the environment to its initial state."""
        super().reset(seed=seed)
        self.current_step = 0
        self.soc_mwh = 0.0
        self.total_energy_traded_per_quarter = 0.0
        self.total_charged_in_quarter = 0.0
        self.total_discharged_in_quarter = 0.0
        return self._get_observation(), {}

    def step(self, action: int):
        """Executes one time step within the environment. This logic is shared."""
        # Reset tracking variables at the beginning of a new quarter
        if self.all_data['Datetime'].iloc[self.current_step].minute % 15 == 0:
            self.total_energy_traded_per_quarter = 0.0
            self.total_charged_in_quarter = 0.0
            self.total_discharged_in_quarter = 0.0

        power_rate = self._get_power_rate_from_action(action)
        intended_energy_trade = power_rate * self.time_interval

        actual_energy_traded = 0.0
        if intended_energy_trade > 0:
            actual_energy_traded = min(intended_energy_trade, self.battery_capacity_mwh - self.soc_mwh)
        elif intended_energy_trade < 0:
            actual_energy_traded = max(intended_energy_trade, -self.soc_mwh)

        self.soc_mwh += actual_energy_traded
        self.total_energy_traded_per_quarter += actual_energy_traded
        if actual_energy_traded > 0:
            self.total_charged_in_quarter += actual_energy_traded
        elif actual_energy_traded < 0:
            self.total_discharged_in_quarter += abs(actual_energy_traded)

        reward = self._calculate_delayed_reward()
        terminated = self.current_step >= self.max_steps - 1
        obs = self._get_observation()

        self.current_step += 1
        info = {'energy_charged_discharged': actual_energy_traded}
        return obs, reward, terminated, False, info

    def action_masks(self) -> np.ndarray:
        """Returns a binary mask of valid actions."""
        mask = [1, 1, 1]
        epsilon = 1e-6
        if self.soc_mwh >= self.battery_capacity_mwh - epsilon: mask[1] = 0
        if self.soc_mwh <= epsilon: mask[2] = 0
        return np.array(mask, dtype=np.int8)


class BasicBatteryEnv(BaseBatteryEnv):
    """
    A basic battery environment.
    Observation space: [SoC, Current Price]
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # --- BASIC OBSERVATION SPACE DEFINITION ---
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, -np.inf], dtype=np.float32),
            high=np.array([self.battery_capacity_mwh, np.inf], dtype=np.float32),
            shape=(2,), dtype=np.float32
        )

    def _get_observation(self) -> np.ndarray:
        """Constructs the basic observation array."""
        return np.array([
            self.soc_mwh,
            self.prices[self.current_step]
        ], dtype=np.float32)


class ExtendedBatteryEnv(BaseBatteryEnv):
    """
    An extended battery environment.
    Observation space: [SoC, Current Price, Total Charged, Total Discharged]
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # --- EXTENDED OBSERVATION SPACE DEFINITION ---
        low_bounds = np.array([0.0, -np.inf, 0.0, 0.0], dtype=np.float32)
        high_bounds = np.array([self.battery_capacity_mwh, np.inf, np.inf, np.inf], dtype=np.float32)

        self.observation_space = gym.spaces.Box(
            low=low_bounds, high=high_bounds, shape=(4,), dtype=np.float32
        )

    def _get_observation(self) -> np.ndarray:
        """Constructs the extended observation array."""
        return np.array([
            self.soc_mwh,
            self.prices[self.current_step],
            self.total_charged_in_quarter,
            self.total_discharged_in_quarter
        ], dtype=np.float32)