import gymnasium as gym
import numpy as np

class BatteryTradingEnv(gym.Env):
    """

    """

    def __init__(
            self,
            battery_capacity_mwh: float,
            charge_discharge_rate_mw: float,
            prices,
            number_of_past_prices: int = 5
    ):
        self.battery_capacity_mwh = battery_capacity_mwh
        self.time_interval = 1/60  # 1 minute in hours
        self.charge_discharge_rate = charge_discharge_rate_mw
        self.prices = prices
        self.number_of_past_prices = number_of_past_prices

        self.max_steps = len(prices)
        # State space, what the agent can observe
        num_of_dim = 2 + self.number_of_past_prices # [SoC, Price, past prices]

        low_bounds = np.concatenate([
            np.array([0.0], dtype=np.float32),
            np.array([-np.inf] * (1 + self.number_of_past_prices), dtype=np.float32)
        ])

        high_bounds = np.concatenate([
            np.array([battery_capacity_mwh], dtype=np.float32),  # SoC max
            np.array([np.inf] * (1 + self.number_of_past_prices), dtype=np.float32)
        ])

        # Definieer de observation space met de nieuwe shape
        self.observation_space = gym.spaces.Box(
            low=low_bounds,
            high=high_bounds,
            shape=(num_of_dim,),
            dtype=np.float32
        )

        # Action space, what actions the agent can take
        self.action_space = gym.spaces.Discrete(3)

        # Initialize state
        self.current_step = 0
        self.soc_mwh = 0.0  # State of Charge in MWh

    def _get_obs(self):
        return np.array([self.soc_mwh, self.prices[self.current_step]], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.soc_mwh = 0.0
        # Initialize whole history with zeros
        price_history = np.zeros(self.number_of_past_prices + 1, dtype=np.float32)

        obs = np.concatenate([
            np.array([self.soc_mwh], dtype=np.float32),
            price_history
        ])
        info = {}
        return obs, info

    def step(self, action):
        x_t = 0.0 # Energy traded
        is_charged = None
        if action == 1: # Charge
            x_t = self.charge_discharge_rate
            is_charged = True
        elif action == 2: # Discharge
            x_t = -self.charge_discharge_rate
            is_charged = False

        # Converse to energy in MWh
        energy_traded = x_t * self.time_interval

        # Check constraints
        if energy_traded > 0:
            actual_energy_traded = min(energy_traded, self.battery_capacity_mwh - self.soc_mwh)  # Can't charge beyond capacity
        elif energy_traded < 0:
            actual_energy_traded = max(energy_traded, -self.soc_mwh)  # Can't discharge below 0
        else:
            actual_energy_traded = 0.0

        self.soc_mwh += actual_energy_traded

        terminated = self.current_step >= self.max_steps - 1
        reward = -self.prices[self.current_step] * actual_energy_traded
        start_index = max(0, self.current_step - self.number_of_past_prices)

        # Prepare the observation with price history, pad is added if history is shorter than required
        price_history = self.prices[start_index: self.current_step]
        padded_history = np.pad(
            price_history,
            (self.number_of_past_prices - len(price_history), 0),
            'constant',
            constant_values=0
        )

        obs = np.concatenate([
            np.array([self.soc_mwh], dtype=np.float32),
            np.array([self.prices[self.current_step]], dtype=np.float32),
            padded_history
        ])
        info = {
            'energy_charged_discharged': actual_energy_traded,
        }
        self.current_step += 1
        return obs, reward, terminated, False, info

