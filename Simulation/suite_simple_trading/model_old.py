import gymnasium as gym
import numpy as np
import pandas as pd


class TestBatteryEnv(gym.Env):
    """
    A standalone battery trading environment that incorporates the professor's feedback.
    The state includes the total energy charged and discharged within the current 15-minute interval.
    """

    def __init__(
            self,
            battery_capacity_mwh: float,
            charge_discharge_rate_mw: float,
            all_data: pd.DataFrame,
    ):
        super().__init__()

        self.battery_capacity_mwh = battery_capacity_mwh
        self.charge_discharge_rate = charge_discharge_rate_mw
        self.all_data = all_data
        self.prices = all_data['Imbalance Price'].to_numpy()
        self.time_interval = 1 / 60  # 1 minute expressed in hours
        self.number_of_past_prices = 0

        # Environment configuration
        self.max_steps = len(self.prices)
        self.action_space = gym.spaces.Discrete(3)  # 0: Idle, 1: Charge, 2: Discharge

        # --- NEW OBSERVATION SPACE DEFINITION ---
        # The state now includes:
        # 1. State of Charge (SoC)
        # 2. Current Imbalance Price
        # 3. Total energy charged in the current quarter
        # 4. Total energy discharged in the current quarter
        num_of_dim = 2

        low_bounds = np.array([
            0.0,  # SoC lower bound
            -np.inf,  # Price lower bound
            #0.0,  # Total charged lower bound
            #0.0  # Total discharged lower bound
        ], dtype=np.float32)

        high_bounds = np.array([
            self.battery_capacity_mwh,  # SoC upper bound (capacity)
            np.inf,  # Price upper bound
            #np.inf,  # Total charged upper bound
            #np.inf  # Total discharged upper bound
        ], dtype=np.float32)

        self.observation_space = gym.spaces.Box(
            low=low_bounds, high=high_bounds, shape=(num_of_dim,), dtype=np.float32
        )
        # --- END NEW OBSERVATION SPACE DEFINITION ---

        # Initialize state variables
        self.current_step = 0
        self.soc_mwh = 0.0
        self.total_energy_traded_per_quarter = 0.0
        #self.total_charged_in_quarter = 0.0
        #self.total_discharged_in_quarter = 0.0

    def _get_power_rate_from_action(self, action: int) -> float:
        """Translates a discrete action into a power rate in MW."""
        if action == 0:  # Idle
            return 0.0
        elif action == 1:  # Charge
            return self.charge_discharge_rate
        elif action == 2:  # Discharge
            return -self.charge_discharge_rate
        else:
            raise ValueError(f"Invalid action {action}")

    def _get_observation(self) -> np.ndarray:
        """Constructs the observation array from the current state."""
        return np.array([
            self.soc_mwh,
            self.prices[self.current_step],
            # self.total_charged_in_quarter,
            # self.total_discharged_in_quarter
        ], dtype=np.float32)

    def _calculate_delayed_reward(self) -> float:
        """Calculates the reward at the end of each 15-minute interval."""
        # The reward is only given at the 14th minute of every quarter (0-indexed)
        if self.all_data['Datetime'].iloc[self.current_step].minute % 15 == 14:
            return -self.prices[self.current_step] * self.total_energy_traded_per_quarter
        return 0.0

    def reset(self, seed=None, options=None):
        """Resets the environment to its initial state."""
        super().reset(seed=seed)

        self.current_step = 0
        self.soc_mwh = 0.0
        self.total_energy_traded_per_quarter = 0.0
        #self.total_charged_in_quarter = 0.0
        #self.total_discharged_in_quarter = 0.0

        obs = self._get_observation()
        info = {}
        return obs, info

    def step(self, action: int):
        """Executes one time step within the environment."""
        # Reset tracking variables at the beginning of a new quarter
        if self.all_data['Datetime'].iloc[self.current_step].minute % 15 == 0:
            self.total_energy_traded_per_quarter = 0.0
            #self.total_charged_in_quarter = 0.0
            #self.total_discharged_in_quarter = 0.0

        # 1. Determine the intended power rate from the action
        power_rate = self._get_power_rate_from_action(action)
        intended_energy_trade = power_rate * self.time_interval

        # 2. Apply physical battery constraints
        actual_energy_traded = 0.0
        if intended_energy_trade > 0:  # Charging
            # Cannot charge more than the available capacity
            actual_energy_traded = min(intended_energy_trade, self.battery_capacity_mwh - self.soc_mwh)
        elif intended_energy_trade < 0:  # Discharging
            # Cannot discharge more than the current state of charge
            actual_energy_traded = max(intended_energy_trade, -self.soc_mwh)

        # 3. Update the battery's state of charge
        self.soc_mwh += actual_energy_traded

        # 4. Update the quarter-based tracking variables
        self.total_energy_traded_per_quarter += actual_energy_traded
        # if actual_energy_traded > 0:
        #     self.total_charged_in_quarter += actual_energy_traded
        # elif actual_energy_traded < 0:
        #     self.total_discharged_in_quarter += abs(actual_energy_traded)

        # 5. Calculate the reward
        reward = self._calculate_delayed_reward()

        # 6. Prepare for the next step
        terminated = self.current_step + 1 >= self.max_steps

        # 7. Get the next observation
        obs = self._get_observation() if not terminated else self.observation_space.sample()
        info = {'energy_charged_discharged': actual_energy_traded}

        self.current_step += 1

        return obs, reward, terminated, False, info

    def action_masks(self) -> np.ndarray:
        """
        Returns binary mask indicating what actions are valid in the current state.
        This method is automatically called by MaskablePPO.
        """
        mask = [1, 1, 1]
        if self.soc_mwh >= self.battery_capacity_mwh: mask[1] = 0
        if self.soc_mwh <= 0: mask[2] = 0
        return np.array(mask, dtype=np.int8)

class BaseBatteryEnv(gym.Env):
    """
    A base class for battery trading environments.
    Contains all the shared logic for state transitions, rewards, and observation creation.
    Subclasses must implement _get_power_rate_from_action().
    """

    def __init__(
            self,
            battery_capacity_mwh: float,
            charge_discharge_rate_mw: float,
            all_data: pd.DataFrame,
            number_of_past_prices: int = 5
    ):
        super().__init__()

        self.battery_capacity_mwh = battery_capacity_mwh
        self.time_interval = 1 / 60
        self.charge_discharge_rate = charge_discharge_rate_mw
        self.all_data = all_data
        self.prices = all_data['Imbalance Price'].to_numpy()
        self.number_of_past_prices = number_of_past_prices
        self.total_energy_traded_per_quarter = 0

        self.max_steps = len(self.prices)

        # --- SHARED OBSERVATION SPACE ---
        num_of_dim = 1 + 1 + self.number_of_past_prices
        low_bounds = np.concatenate([
            np.array([0.0], dtype=np.float32),
            np.array([-np.inf] * (1 + self.number_of_past_prices), dtype=np.float32)
        ])
        high_bounds = np.concatenate([
            np.array([battery_capacity_mwh], dtype=np.float32),
            np.array([np.inf] * (1 + self.number_of_past_prices), dtype=np.float32)
        ])
        self.observation_space = gym.spaces.Box(
            low=low_bounds, high=high_bounds, shape=(num_of_dim,), dtype=np.float32
        )

        # --- STATE INITIALIZATION ---
        self.current_step = 0
        self.soc_mwh = 0.0

    def _get_power_rate_from_action(self, action: int) -> float:
        """
        Abstract method: Subclasses MUST implement this to translate an action
        into a power rate (in MW).
        """
        raise NotImplementedError("This method must be implemented by the subclass.")

    def get_idle_action(self) -> int:
        """
        Abstract method: Subclasses MUST implement this to return the action
        corresponding to 'Idle' (no charge/discharge).
        """
        raise NotImplementedError("This method must be implemented by the subclass.")

    def _get_observation(self) -> np.ndarray:
        """Helper function to create the observation array."""
        start_index = max(0, self.current_step - self.number_of_past_prices)
        price_history = self.prices[start_index: self.current_step]
        padded_history = np.pad(
            price_history,
            (self.number_of_past_prices - len(price_history), 0),
            'constant', constant_values=0
        )
        obs = np.concatenate([
            np.array([self.soc_mwh], dtype=np.float32),
            np.array([self.prices[self.current_step]], dtype=np.float32),
            padded_history
        ])
        return obs

    def _calculate_direct_reward(self, actual_energy_traded: float) -> float:
        return -self.prices[self.current_step] * actual_energy_traded

    def _calculate_delayed_reward(self):
        # Reward based on price at the end of a quarter, this is also what businesses have to pay
        return -self.prices[self.current_step] * self.total_energy_traded_per_quarter if self.all_data['Datetime'][self.current_step].minute % 15 == 14 else 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.soc_mwh = 0.0

        obs = self._get_observation()  # Use helper to create initial observation
        info = {}
        return obs, info

    def step(self, action: int):
        if self.all_data['Datetime'][self.current_step].minute % 15 == 0:
            self.total_energy_traded_per_quarter = 0.0

        # Get power rate from the specific policy implementation
        x_t = self._get_power_rate_from_action(action)

        energy_traded = x_t * self.time_interval

        if energy_traded > 0:
            actual_energy_traded = min(energy_traded, self.battery_capacity_mwh - self.soc_mwh)
        elif energy_traded < 0:
            actual_energy_traded = max(energy_traded, -self.soc_mwh)
        else:
            actual_energy_traded = 0.0

        self.soc_mwh += actual_energy_traded
        self.total_energy_traded_per_quarter += actual_energy_traded

        reward = self._calculate_delayed_reward()

        terminated = self.current_step + 1 >= self.max_steps
        obs = self._get_observation() if not terminated else self.observation_space.sample()
        info = {'energy_charged_discharged': actual_energy_traded}

        self.current_step += 1


        return obs, reward, terminated, False, info


# --- CONCRETE IMPLEMENTATIONS ---

class BatteryTradingEnv1(BaseBatteryEnv):
    """
    The simple environment with 3 discrete actions: Idle, Full Charge, Full Discharge.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_space = gym.spaces.Discrete(3)

    def _get_power_rate_from_action(self, action: int) -> float:
        if action == 0:  # Idle
            return 0.0
        elif action == 1:  # Charge
            return self.charge_discharge_rate
        elif action == 2:  # Discharge
            return -self.charge_discharge_rate

    def get_idle_action(self) -> int:
        return 0


class BatteryTradingEnv2(BaseBatteryEnv):
    """
    The granular environment with 11 discrete actions from -100% to +100%.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_space = gym.spaces.Discrete(11)
        self.action_to_percentage = {
            0: -1.0, 1: -0.8, 2: -0.6, 3: -0.4, 4: -0.2,
            5: 0.0,
            6: 0.2, 7: 0.4, 8: 0.6, 9: 0.8, 10: 1.0
        }

    def _get_power_rate_from_action(self, action: int) -> float:
        percentage = self.action_to_percentage[action]
        return self.charge_discharge_rate * percentage

    def get_idle_action(self) -> int:
        return 5

class BatteryTradingEnvMasking(BatteryTradingEnv1):
    """
    An environment that extends BatteryTradingEnv1 by adding action masking.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def action_masks(self) -> np.ndarray:
        """
        Returns binary mask indicating what actions are valid in the current state.
        This method is automatically called by MaskablePPO.
        """
        mask = [1, 1, 1]
        if self.soc_mwh >= self.battery_capacity_mwh: mask[1] = 0
        if self.soc_mwh <= 0: mask[2] = 0
        return np.array(mask, dtype=np.int8)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        action_mask = self.action_masks()
        info['action_mask'] = action_mask
        return obs, info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = super().step(action)
        action_mask = self.action_masks()
        info['action_mask'] = action_mask
        return obs, reward, terminated, truncated, info
