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
            days_per_episode: int = 1,
            cycle_cost_eur: float = 6.25
    ):
        super().__init__()

        # --- Core Simulation Parameters ---
        self.battery_capacity_mwh = battery_capacity_mwh
        self.charge_discharge_rate = charge_discharge_rate_mw
        self.all_data = all_data
        self.prices = all_data['Imbalance Price'].to_numpy()
        self.time_interval = 1 / 60
        self.max_steps = len(self.prices)
        self.days_per_episode = days_per_episode
        self.cycle_cost_eur = cycle_cost_eur

        throughput_per_cycle = 2 * self.battery_capacity_mwh
        self.marginal_cost_per_mwh = self.cycle_cost_eur / throughput_per_cycle

        # Pre-calculate the starting index of each day.
        self.daily_start_indices = self.all_data.groupby(
            self.all_data['Datetime'].dt.date
        ).head(1).index.tolist()
        # This counter now tracks which *day* we start the episode on.
        self.start_day_counter = 0
        # This will store the calculated end step for the current episode.
        self.current_episode_end_step = 0

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
            revenue = -self.prices[self.current_step] * self.total_energy_traded_per_quarter

            # Degradation Cost
            throughput = self.total_charged_in_quarter + self.total_discharged_in_quarter
            degradation_cost = throughput * self.marginal_cost_per_mwh

            return revenue - degradation_cost

        return 0.0

    def reset(self, seed=None, options=None):
        """Resets the environment to its initial state."""
        super().reset(seed=seed)

        # If the counter is at the end of the available days, wrap around to 0.
        if self.start_day_counter >= len(self.daily_start_indices):
            self.start_day_counter = 0

        # Start at the beginning of a day
        self.current_step = self.daily_start_indices[self.start_day_counter]
        # Determine the end step for this multi-day episode.
        end_day_index = self.start_day_counter + self.days_per_episode

        # Calculate the end step based on the start of the next day or the end of the data
        if end_day_index >= len(self.daily_start_indices):
            self.current_episode_end_step = self.max_steps - 1
        else:
            self.current_episode_end_step = self.daily_start_indices[end_day_index] - 1

        # Advance the counter for the NEXT time reset() is called
        self.start_day_counter += self.days_per_episode

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

        # Check if the episode is done
        episode_done = self.current_step >= self.current_episode_end_step
        # Check if we are at the end of the dataframe
        data_done = self.current_step >= self.max_steps - 1
        terminated = episode_done or data_done
        obs = self._get_observation()

        self.current_step += 1
        info = {
            'energy_charged_discharged': actual_energy_traded,
            'real_reward': reward
        }
        return obs, reward, terminated, False, info

    def action_masks(self) -> np.ndarray:
        """Returns a binary mask of valid actions."""
        mask = [1, 1, 1]
        epsilon = 1e-6
        if self.soc_mwh >= self.battery_capacity_mwh - epsilon: mask[1] = 0
        if self.soc_mwh <= epsilon: mask[2] = 0
        return np.array(mask, dtype=np.int8)

    def get_idle_action(self) -> int:
        """Returns the action index for 'Idle'."""
        return 0


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
    Observation space: [SoC, Current Price, Total Charged in Quarter, Total Discharged in Quarter]
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


class TimeAwareBatteryEnv(BaseBatteryEnv):
    """
    A new environment that extends the battery logic with time-aware features.

    Observation Space (11 dimensions):
    [0] SoC (State of Charge)
    [1] Current Imbalance Price
    [2] Total Charged in Quarter
    [3] Total Discharged in Quarter
    [4] Minute in Quarter (Normalized 0-1) - Represents urgency/progress to reward
    [5] Hour Sine   (Cyclic)
    [6] Hour Cosine (Cyclic)
    [7] Week Sine   (Cyclic)
    [8] Week Cosine (Cyclic)
    [9] Month Sine  (Cyclic)
    [10] Month Cosine (Cyclic)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # --- TIME FEATURE ENGINEERING ---
        # We pre-calculate these for the entire dataset in __init__ for max performance.

        dt_series = self.all_data['Datetime'].dt

        # 1. Minute in Quarter (0-14) -> Normalized to [0, 1]
        minutes = dt_series.minute
        self.minute_in_quarter = (minutes % 15 / 14.0).to_numpy(dtype=np.float32)

        # 2. Hours (0-23) -> Cyclic encoding
        hours = dt_series.hour
        self.h_sin = np.sin(2 * np.pi * hours / 23.0).to_numpy(dtype=np.float32)
        self.h_cos = np.cos(2 * np.pi * hours / 23.0).to_numpy(dtype=np.float32)

        # # 3. Weeks (1-53) -> Cyclic encoding
        # weeks = dt_series.isocalendar().week
        # self.w_sin = np.sin(2 * np.pi * weeks / 53.0).to_numpy(dtype=np.float32)
        # self.w_cos = np.cos(2 * np.pi * weeks / 53.0).to_numpy(dtype=np.float32)
        #
        # # 4. Months (1-12) -> Cyclic encoding
        # months = dt_series.month
        # self.m_sin = np.sin(2 * np.pi * months / 12.0).to_numpy(dtype=np.float32)
        # self.m_cos = np.cos(2 * np.pi * months / 12.0).to_numpy(dtype=np.float32)

        # --- OBSERVATION SPACE DEFINITION ---
        # 4 Base + 1 Minute + 6 Cyclic = 11 dimensions

        # Bounds for: [SoC, Price, Charged, Discharged]
        base_low = np.array([0.0, -np.inf, 0.0, 0.0], dtype=np.float32)
        base_high = np.array([self.battery_capacity_mwh, np.inf, np.inf, np.inf], dtype=np.float32)

        # Bounds for: [Minute in Quarter] (0 to 1)
        minute_low = np.array([0.0], dtype=np.float32)
        minute_high = np.array([1.0], dtype=np.float32)

        # Bounds for: [h_sin, h_cos, w_sin, w_cos, m_sin, m_cos] (-1 to 1)
        # cyclical_low = np.array([-1.0] * 6, dtype=np.float32)
        # cyclical_high = np.array([1.0] * 6, dtype=np.float32)
        cyclical_low = np.array([-1.0] * 2, dtype=np.float32)
        cyclical_high = np.array([1.0] * 2, dtype=np.float32)

        self.observation_space = gym.spaces.Box(
            low=np.concatenate([base_low, minute_low, cyclical_low]),
            high=np.concatenate([base_high, minute_high, cyclical_high]),
            # shape=(11,),
            shape=(7,),
            dtype=np.float32
        )

    def _get_observation(self) -> np.ndarray:
        """
        Constructs the full observation array for the current step.
        """
        i = self.current_step

        obs = np.array([
            self.soc_mwh,
            self.prices[i],
            self.total_charged_in_quarter,
            self.total_discharged_in_quarter,
            self.minute_in_quarter[i],
            self.h_sin[i],
            self.h_cos[i],
            # self.w_sin[i],
            # self.w_cos[i],
            # self.m_sin[i],
            # self.m_cos[i]
        ], dtype=np.float32)

        return obs