import os

import numpy as np
import pandas as pd
from typing import Tuple


def split_data_for_episodic_rl(
        all_data: pd.DataFrame,
        days_per_episode: int,
        test_fraction: float = 0.2,
        buffer_days: int = 7
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Splits time-series data into training and testing sets for episodic RL
    based on a fractional target for the test set size.

    It calculates the number of episodes that corresponds to the test_fraction of
    total days, randomly selects them, and uses the rest for training. A buffer
    ensures test episodes are not adjacent.

    Args:
        all_data: The complete DataFrame with a 'Datetime' column.
        days_per_episode: The number of consecutive days that form one episode.
        test_fraction: The fraction of total days to be allocated to the test set (e.g., 0.2 for 20%).
        buffer_days: The number of days to exclude on either side of a chosen
                     test episode to prevent data leakage.

    Returns:
        A tuple containing two DataFrames: (train_df, test_df).
    """
    if not (0.0 < test_fraction < 1.0):
        raise ValueError("test_fraction must be between 0.0 and 1.0.")

    print("--- Starting Episodic Train-Test Split ---")

    # 1. Identify the start index of every day in the dataset.
    daily_start_indices = all_data.groupby(
        all_data['Datetime'].dt.date
    ).head(1).index.tolist()

    num_total_days = len(daily_start_indices)
    print(f"Found {num_total_days} unique days in the dataset.")

    # 2. --- NEW: Calculate the target number of test episodes ---
    target_test_days = num_total_days * test_fraction
    # Convert the number of days to the number of full episodes, rounding to the nearest whole number.
    n_test_episodes = int(round(target_test_days / days_per_episode))

    if n_test_episodes == 0:
        raise ValueError(
            f"Test fraction of {test_fraction} is too small to create even one "
            f"test episode of {days_per_episode} days."
        )
    print(f"Targeting {test_fraction:.0%} of data for testing.")
    print(f"This corresponds to {n_test_episodes} episodes of {days_per_episode} days each.")

    # 3. Identify all possible N-day episodes.
    possible_episode_start_days = list(range(num_total_days - days_per_episode + 1))

    if len(possible_episode_start_days) < n_test_episodes:
        raise ValueError(
            f"Cannot select {n_test_episodes} test episodes. "
            f"Only {len(possible_episode_start_days)} possible episodes exist in the data."
        )

    # 4. Randomly select test episodes, ensuring they are separated by a buffer.
    test_episode_start_days = []
    candidate_pool = possible_episode_start_days.copy()
    np.random.shuffle(candidate_pool)

    while len(test_episode_start_days) < n_test_episodes and candidate_pool:
        chosen_start_day = candidate_pool.pop(0)
        test_episode_start_days.append(chosen_start_day)

        # Define the buffer zone around the chosen episode
        # This zone is where the START of another episode cannot be
        buffer_start = chosen_start_day - (days_per_episode + buffer_days)
        buffer_end = chosen_start_day + (days_per_episode + buffer_days)

        candidate_pool = [
            day for day in candidate_pool
            if not (buffer_start < day < buffer_end)
        ]

    if len(test_episode_start_days) < n_test_episodes:
        print(
            f"\nWarning: Could only select {len(test_episode_start_days)} test episodes "
            f"due to buffer constraints. The actual test size will be smaller than {test_fraction:.0%}.\n"
            "Consider reducing the buffer_days or days_per_episode."
        )

    print(f"Successfully selected {len(test_episode_start_days)} test episodes.")

    # 5. Construct the train and test DataFrames using a boolean mask.
    is_train_mask = pd.Series(True, index=all_data.index)

    for start_day_index in test_episode_start_days:
        start_df_index = daily_start_indices[start_day_index]

        end_day_of_episode = start_day_index + days_per_episode
        if end_day_of_episode >= len(daily_start_indices):
            end_df_index = len(all_data) - 1
        else:
            end_df_index = daily_start_indices[end_day_of_episode] - 1

        is_train_mask.loc[start_df_index:end_df_index] = False

    train_df = all_data[is_train_mask].reset_index(drop=True)
    test_df = all_data[~is_train_mask].reset_index(drop=True)

    actual_test_fraction = len(test_df) / len(all_data)
    print("--- Split Complete ---")
    print(f"Training data length: {len(train_df)} rows")
    print(f"Testing data length:  {len(test_df)} rows")
    print(f"Actual fraction of data in test set: {actual_test_fraction:.2%}")

    return train_df, test_df, n_test_episodes


def get_or_create_train_test_split(
        all_data: pd.DataFrame,
        save_path: str,
        days_per_episode: int,
        test_fraction: float,
        buffer_days: int
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """
    Loads a train/test split from disk if it exists.
    If not, it creates a new split and saves it for future use.

    Args:
        all_data: The complete DataFrame.
        save_path: The directory to save/load the data from (e.g., "model/used_data").
        days_per_episode: Number of days that constitute one episode.
        test_fraction: The fraction of data to allocate to the test set.
        buffer_days: Number of buffer days to place around test episodes.

    Returns:
        A tuple containing (train_df, test_df, number_of_test_episodes).
    """
    train_file_path = os.path.join(save_path, "train_data.pkl")
    test_file_path = os.path.join(save_path, "test_data.pkl")

    # Check if both data files already exist
    if os.path.exists(train_file_path) and os.path.exists(test_file_path):
        print(f"--- Loading existing train/test data from '{save_path}' ---")
        train_df = pd.read_pickle(train_file_path)
        test_df = pd.read_pickle(test_file_path)

        # Count the number of unique days in the test dataframe
        num_days_in_test = test_df['Datetime'].dt.date.nunique()
        # Calculate the number of full episodes this corresponds to
        number_of_episodes = num_days_in_test // days_per_episode
        test_to_train_ratio = len(test_df) / len(train_df)


        print(f"--- Data loaded successfully. Found {number_of_episodes} test episodes. ---")
        print(f"Test Data Size / Train Data Size: {test_to_train_ratio:.2f}")
    else:
        print(f"--- No existing data found. Creating new train/test split. ---")

        # Call the splitting function, which now returns the episode count
        train_df, test_df, number_of_episodes = split_data_for_episodic_rl(
            all_data=all_data,
            days_per_episode=days_per_episode,
            test_fraction=test_fraction,
            buffer_days=buffer_days
        )

        # Ensure the target directory exists
        os.makedirs(save_path, exist_ok=True)

        # Save the newly created dataframes using pickle for efficiency
        train_df.to_pickle(train_file_path)
        test_df.to_pickle(test_file_path)
        print(f"--- New data split saved to '{save_path}' ---")

    return train_df, test_df, number_of_episodes