from typing import Optional
import gymnasium as gym
import numpy as np

class BatteryEnv(gym.Env):
    """

    """

    def __init__(
            self,
            battery_capacity_mwh: float,
            charge_discharge_rate_mw: float,
            energy_demand,
            prices
    ):
        self.battery_capacity_mwh = battery_capacity_mwh
        self.time_interval = 1/60  # 1 minute in hours
        self.charge_discharge_rate = charge_discharge_rate_mw
        self.energy_demand = energy_demand
        self.prices = prices

        self.max_steps = len(energy_demand)
        # State space, what the agent can observe
        num_of_dim = 3 # [SoC, Price, Demand]

        low_bounds = np.array([
            0,                      # SoC min
            -np.inf,                # Price min
            0                       # Demand min
        ], dtype=np.float32)
        high_bounds = np.array([
            battery_capacity_mwh,   # SoC max
            np.inf,                 # Price max
            np.inf                  # Demand max
        ], dtype=np.float32)

        self.observation_space = gym.spaces.Box(
            low=low_bounds,
            high=high_bounds,
            shape=(num_of_dim,),
            dtype=np.float32
        )

        # Action space, what actions the agent can take
        self.action_space = gym.spaces.Discrete(5)

        # Initialize state
        self.current_step = 0
        self.soc_mwh = 0.0  # State of Charge in MWh


    def step(self, action):
        # Execute one time step within the environment
        x_gb, x_bg, x_gd, x_bd = 0, 0, 0, 0
        if action == 0: # Idle
            x_gd = self.energy_demand[self.current_step]
        elif action == 1: # Charge from grid
            x_gb = min(self.charge_discharge_rate * self.time_interval, self.battery_capacity_mwh - self.soc_mwh)
            x_gd = self.energy_demand[self.current_step]
        elif action == 2: # Discharge to grid and demand from battery
            max_battery_power = min(self.charge_discharge_rate * self.time_interval, self.soc_mwh)
            # If battery can cover demand, sell rest to grid
            if max_battery_power >= self.energy_demand[self.current_step]:
                x_bd = self.energy_demand[self.current_step]
                x_bg = max_battery_power - self.energy_demand[self.current_step]
            else:
                x_bd = max_battery_power
                x_gd = self.energy_demand[self.current_step] - max_battery_power
        elif action == 3: # Discharge to demand
            max_battery_power = min(self.charge_discharge_rate * self.time_interval, self.soc_mwh)
            x_bd = min(max_battery_power, self.energy_demand[self.current_step])
            if x_bd < self.energy_demand[self.current_step]:
                x_gd = self.energy_demand[self.current_step] - x_bd

        self.soc_mwh += x_gb - (x_bd + x_bg)

        reward = self.prices[self.current_step] * (-x_bg + x_bd + x_bg)
        #x_bg * - self.prices[self.current_step] + (x_gd + x_gb) * self

        self.current_step += 1
        pass