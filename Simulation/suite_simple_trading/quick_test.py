from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from Simulation.suite_simple_trading.model import ExtendedBatteryEnv
from Simulation.suite_simple_trading.observation_wrappers import RobustScalingWrapper, SoCPenaltyWrapper
from Simulation.suite_simple_trading.pre_processing import clean_data
import pandas as pd
import os
from Simulation.suite_simple_trading.data_splitting import get_or_create_train_test_split

from Simulation.DRL_implementations.Masked_DQN import MaskedDoubleDQN

if __name__ == '__main__':
    ### Global Variables
    DATA_SAVE_PATH = "../../models/used_data"
    DAYS_PER_EPISODE = 3
    TRAIN_TEST_SPLIT_FRACTION = 0.2  # Amount of data you reserve for testing.
    BUFFER_DAYS = 3  # Amount of days that needs to be between the test episodes.
    ### Getting data

    raw_data_path = '../../data/2025_minute.csv'
    cleaned_data_cache_path = '../../data/2025_minute_cleaned.pkl'

    cleaned_df = clean_data(raw_path=raw_data_path, cache_path=cleaned_data_cache_path)

    all_data = cleaned_df[['Datetime', 'Imbalance Price']]
    train_df, test_df, nr_of_episodes = get_or_create_train_test_split(
        all_data=all_data,
        save_path=DATA_SAVE_PATH,
        days_per_episode=DAYS_PER_EPISODE,
        test_fraction=TRAIN_TEST_SPLIT_FRACTION,
        buffer_days=BUFFER_DAYS
    )

    train_env_dqn = ExtendedBatteryEnv(
        battery_capacity_mwh=10.0,
        charge_discharge_rate_mw=5.0,
        all_data=train_df[0:10],
        days_per_episode=DAYS_PER_EPISODE
    )
    train_env_scaled_obs = RobustScalingWrapper(train_env_dqn)
    train_env_scaled_obs_SoC_pen = SoCPenaltyWrapper(train_env_scaled_obs)
    dqn_agent = MaskablePPO(MaskableActorCriticPolicy, train_env_scaled_obs_SoC_pen)
    dqn_agent.learn(11)
