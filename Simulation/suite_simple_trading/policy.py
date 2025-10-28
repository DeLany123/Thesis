# In a new file, e.g., decision_maker.py
import numpy as np
import os
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.base_class import BaseAlgorithm


class QuarterlyTrendDecisionMaker:
    """
    A decision maker that implements the quarterly trend-based heuristic policy.
    It waits for a signal within a quarter and then locks in the decision for the
    remainder of that quarter.
    """

    def __init__(self, theta_buy: float, theta_sell: float, past_prices_needed: int):
        """
        Initializes the decision maker with its specific parameters.

        Args:
            theta_buy: The price threshold for buying/charging.
            theta_sell: The price threshold for selling/discharging.
            past_prices_needed: The number of historical prices required for the trend analysis.
        """
        self.theta_buy = theta_buy
        self.theta_sell = theta_sell
        self.past_prices_needed = past_prices_needed

        # Internal state for the manager logic
        self.quarterly_decision = 0  # 0: Idle, 1: Charge, 2: Discharge
        self.decision_made_this_quarter = False

    def reset(self):
        """
        Resets the internal state of the decision maker for a new episode.
        """
        self.quarterly_decision = 0
        self.decision_made_this_quarter = False

    def get_action(self, observation: np.ndarray, current_step: int) -> int:
        """
        The main method called by the runner to get the next action.
        It contains the full quarterly manager logic.

        Args:
            observation: The current state observation from the environment.
            current_step: The current time step in the simulation.

        Returns:
            The integer action to be taken.
        """
        current_minute_of_quarter = current_step % 15

        # --- Quarterly Reset Logic ---
        if current_minute_of_quarter == 0:
            self.quarterly_decision = 0
            self.decision_made_this_quarter = False

        # --- Decision Making Logic ---
        if current_step >= self.past_prices_needed and not self.decision_made_this_quarter:

            # This is the actual heuristic policy logic
            policy_action = self._trend_based_heuristic(observation)

            if policy_action != 0:  # 0 is the Idle action
                self.quarterly_decision = policy_action
                self.decision_made_this_quarter = True

        return self.quarterly_decision

    def _trend_based_heuristic(self, observation: np.ndarray) -> int:
        """
        Private method containing the core trend-based policy.
        (This is your old 'trend_based_heuristic_policy' function, now inside the class)
        """
        current_price = observation[1]  # Assumes price is the second element
        price_history = observation[-self.past_prices_needed:]  # Assumes history is at the end

        ACTION_IDLE = 0
        ACTION_CHARGE = 1
        ACTION_DISCHARGE = 2

        # Example trend check: are the last 3 prices negative?
        is_buy_trend = all(p < 0 for p in price_history[-3:])
        if current_price <= self.theta_buy and is_buy_trend:
            return ACTION_CHARGE

        # Example trend check: are the last 3 prices positive?
        is_sell_trend = all(p > 0 for p in price_history[-3:])
        if current_price >= self.theta_sell and is_sell_trend:
            return ACTION_DISCHARGE

        return ACTION_IDLE


class RLAgentDecisionMaker:
    """
    A wrapper class to make a trained Stable-Baselines3 agent compatible
    with the generic evaluation loop (run_evaluation).
    """

    def __init__(self, model: BaseAlgorithm):
        """
        Initializes the decision maker with a pre-trained RL model.

        Args:
            model: The trained Stable-Baselines3 model object (e.g., loaded via DQN.load()).
        """
        # Store the trained model internally
        self.model = model

    def get_action(self, observation: np.ndarray, current_step: int) -> int:
        """
        Uses the trained RL model to predict the best action for a given observation.

        Args:
            observation: The current state observation from the environment.
            current_step: The current time step (not used by the RL agent, but included for interface consistency).

        Returns:
            The integer action predicted by the model.
        """
        action, _states = self.model.predict(observation, deterministic=True)

        # The action is returned as a NumPy array, so we convert it to a standard Python int.
        return int(action)

    def reset(self):
        """
        Resets the decision maker's state. For a stateless RL agent wrapper,
        this method doesn't need to do anything. It's included to maintain a
        consistent interface with other stateful decision makers.
        """
        # A trained SB3 model is stateless between episodes, so nothing to do here.
        pass


# In policy.py or decision_maker.py
class PPOAgentDecisionMaker:
    """
    Wrapper for a trained Stable-Baselines3 agent.
    Now handles action masks passed via the info dictionary.
    """

    def __init__(self, model):
        self.model = model

    def get_action(self, observation: np.ndarray, info: dict) -> int:
        """
        Predicts the best action, using an action mask if available.

        Args:
            observation: The current state observation.
            info: The info dictionary from the environment, which may contain an 'action_mask'.
        """
        # Get the action mask from the info dictionary provided by the env
        action_mask = info.get('action_mask')

        # Pass the mask to the predict method
        action, _states = self.model.predict(
            observation,
            deterministic=True,
            action_masks=action_mask  # This is the key change
        )
        return int(action)

    def reset(self):
        pass