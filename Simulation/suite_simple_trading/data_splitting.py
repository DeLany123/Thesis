import os

import numpy as np
import pandas as pd
from typing import Tuple, List

from Simulation.suite_simple_trading.pre_processing import clean_data


def split_data_for_episodic_rl(
        all_data: pd.DataFrame,
        days_per_episode: int,
        test_fraction: float = 0.2,
        val_fraction: float = 0.1,
        min_spacing_days: int = 3,
        exclusion_hours: int = 4
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame]]:
    """
     Splits time-series data into Train, Validation, and Test sets using episodic sampling.

     This function identifies potential episodes within the continuous data and randomly selects
     a subset for validation and testing. It ensures a minimum spacing between selected evaluation
     episodes to guarantee diversity. Additionally, it removes a small safety buffer of hours
     around evaluation episodes from the training set to prevent data leakage. The remaining
     training data is returned as a list of continuous chunks (shards).

     Args:
         all_data (pd.DataFrame): The complete dataset containing a 'Datetime' column.
         days_per_episode (int): The length of a single episode in days.
         test_fraction (float, optional): The fraction of the total dataset duration to allocate
             to the Test set. Defaults to 0.2 (20%).
         val_fraction (float, optional): The fraction of the total dataset duration to allocate
             to the Validation set. Defaults to 0.1 (10%).
         min_spacing_days (int, optional): The minimum number of days required between any two
             selected evaluation (Test/Val) episodes. Defaults to 3.
         exclusion_hours (int, optional): The number of hours immediately preceding and following
             a selected evaluation episode to exclude from the Training set. Defaults to 4.

     Returns:
         Tuple[List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame]]: A tuple containing:
             - train_df pd.DataFrame: A DataFrame, of training data.
             - val_episodes (List[pd.DataFrame]): A list of DataFrames, where each DataFrame is a
               single episode selected for validation.
             - test_episodes (List[pd.DataFrame]): A list of DataFrames, where each DataFrame is a
               single episode selected for testing.
     """
    print("--- Starting Episodic Train-Val-Test Split ---")

    # 1. Identify start indices
    daily_start_indices = all_data.groupby(all_data['Datetime'].dt.date).head(1).index.tolist()
    num_total_days = len(daily_start_indices)

    # 2. Calculate Targets
    target_test_days = num_total_days * test_fraction
    target_val_days = num_total_days * val_fraction
    n_test_episodes = int(round(target_test_days / days_per_episode))
    n_val_episodes = int(round(target_val_days / days_per_episode))

    # 3. Selection Logic
    possible_starts = list(range(num_total_days - days_per_episode + 1))
    candidate_pool = possible_starts.copy()
    np.random.shuffle(candidate_pool)

    def select_episodes(target_count, pool):
        selected = []
        while len(selected) < target_count and pool:
            chosen = pool.pop(0)
            selected.append(chosen)
            block_start = chosen - min_spacing_days
            block_end = chosen + days_per_episode + min_spacing_days
            pool = [d for d in pool if not (block_start < d < block_end)]
        return selected, pool

    test_starts, candidate_pool = select_episodes(n_test_episodes, candidate_pool)
    val_starts, candidate_pool = select_episodes(n_val_episodes, candidate_pool)

    print(f"Selected: {len(test_starts)} Test, {len(val_starts)} Val episodes.")

    # 4. Constructing the Lists and Mask
    # Mask to define what is NOT training data
    is_excluded_mask = pd.Series(False, index=all_data.index)
    exclusion_steps = exclusion_hours * 60

    def extract_episodes_and_mask(start_days):
        episode_list = []
        for start in start_days:
            # A. Get indices for the episode
            idx_start = daily_start_indices[start]
            day_end = start + days_per_episode
            if day_end >= len(daily_start_indices):
                idx_end = len(all_data) - 1
            else:
                idx_end = daily_start_indices[day_end] - 1

            # B. Extract and append
            episode_slice = all_data.loc[idx_start:idx_end].copy().reset_index(drop=True)
            episode_list.append(episode_slice)

            # C. Update exclusion mask (Episode + Safety Buffer)
            unsafe_start = max(0, idx_start - exclusion_steps)
            unsafe_end = min(len(all_data) - 1, idx_end + exclusion_steps)
            is_excluded_mask.loc[unsafe_start:unsafe_end] = True

        return episode_list

    test_episodes_list = extract_episodes_and_mask(test_starts)
    val_episodes_list = extract_episodes_and_mask(val_starts)

    # 5. Define Training Data (One Big DataFrame)
    # Train = Everything that is NOT marked as excluded
    train_df = all_data[~is_excluded_mask].reset_index(drop=True)

    print(
        f"Split Complete. Train rows: {len(train_df)}. Val Episodes: {len(val_episodes_list)}. Test Episodes: {len(test_episodes_list)}.")

    return train_df, val_episodes_list, test_episodes_list


def get_or_create_train_val_test_split(
        raw_data_path: str,
        save_path: str,
        days_per_episode: int,
        test_fraction: float = 0.2,
        val_fraction: float = 0.1,
        min_spacing_days: int = 3,
        exclusion_hours: int = 4
) -> Tuple[pd.DataFrame, List[pd.DataFrame], List[pd.DataFrame]]:
    """
    1. Loads or creates cleaned data from the raw CSV.
    2. Loads or creates episodic train/val/test splits.

    Args:
        raw_data_path (str): Path to the raw .csv file.
        save_path (str): Directory where split pickles are stored.
        days_per_episode (int): Episode length in days.
        test_fraction (float): Fraction of data for testing.
        val_fraction (float): Fraction of data for validation.
        min_spacing_days (int): Minimum days between selected eval episodes.
        exclusion_hours (int): Safety buffer hours around eval episodes.

    Returns:
        Tuple[pd.DataFrame, List[pd.DataFrame], List[pd.DataFrame]]:
            (train_df, val_episodes_list, test_episodes_list)
    """
    # Define paths
    cleaned_data_path = raw_data_path.replace(".csv", "_cleaned.pkl")
    train_path = os.path.join(save_path, "train_data.pkl")
    val_path = os.path.join(save_path, "val_episodes_list.pkl")
    test_path = os.path.join(save_path, "test_episodes_list.pkl")

    # Step 1: Manage Cleaned Data
    if os.path.exists(cleaned_data_path):
        print(f"--- Loading cached cleaned data: {cleaned_data_path} ---")
        all_data = pd.read_pickle(cleaned_data_path)
    else:
        print(f"--- No cleaned data found. Processing: {raw_data_path} ---")
        # Ensure raw data exists
        if not os.path.exists(raw_data_path):
            raise FileNotFoundError(f"Raw data file not found at {raw_data_path}")

        raw_df = pd.read_csv(raw_data_path, sep=';')
        all_data = clean_data(raw_df)
        all_data.to_pickle(cleaned_data_path)
        print(f"--- Cleaned data saved to: {cleaned_data_path} ---")

    # Filter columns to essential ones
    all_data = all_data[['Datetime', 'Imbalance Price']]

    # Step 2: Manage Splits
    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        print(f"--- Loading existing split from '{save_path}' ---")
        train_df = pd.read_pickle(train_path)
        val_episodes = pd.read_pickle(val_path)
        test_episodes = pd.read_pickle(test_path)

        total_rows = len(train_df) + sum(len(df) for df in val_episodes) + sum(len(df) for df in test_episodes)
        print(f"Split loaded. Total Test Episodes: {len(test_episodes)}")
    else:
        print(f"--- Creating new episodic Train/Val/Test split ---")
        # split_data_for_episodic_rl() must be imported in the scope
        train_df, val_episodes, test_episodes = split_data_for_episodic_rl(
            all_data, days_per_episode, test_fraction, val_fraction,
            min_spacing_days, exclusion_hours
        )

        os.makedirs(save_path, exist_ok=True)
        train_df.to_pickle(train_path)
        pd.to_pickle(val_episodes, val_path)
        pd.to_pickle(test_episodes, test_path)
        print(f"--- New split saved to '{save_path}' ---")

    return train_df, val_episodes, test_episodes