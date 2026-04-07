import os

import numpy as np
import pandas as pd
import math
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
    train_path = os.path.join(save_path, "train_data.pkl")
    val_path = os.path.join(save_path, "val_episodes_list.pkl")
    test_path = os.path.join(save_path, "test_episodes_list.pkl")

    # Step 1: Load or create cleaned data (caching is handled internally)
    cleaned_data_path = raw_data_path.replace(".csv", "_cleaned.pkl")
    all_data = clean_data(raw_path=raw_data_path, cache_path=cleaned_data_path)

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


def generate_hv_block_k_folds(
        all_data: pd.DataFrame,
        k_folds: int = 5,
        days_per_episode: int = 3,
        exclusion_hours: int = 4,
        test_fraction: float = 0.15,
        val_fraction: float = 0.15,
        save_path: str = None
) -> List[Tuple[pd.DataFrame, List[pd.DataFrame], List[pd.DataFrame]]]:
    """
    Generates K iterations of Train/Val/Test splits using hv-Block Cross-Validation.

    This version groups Validation and Test episodes adjacently into "Evaluation Blocks".
    These blocks are spaced evenly across the year, maximizing the length of the
    continuous training periods between them.
    """
    if save_path is not None:
        from Simulation.suite_simple_trading.data_splitting import _load_folds, _save_folds
        folds_output = _load_folds(save_path, k_folds)
        if folds_output is not None:
            return folds_output

    # Check validity
    group_fraction = 1.0 / k_folds
    if test_fraction > group_fraction + 1e-9:
        raise ValueError(
            f"test_fraction ({test_fraction}) cannot exceed 1/k_folds ({group_fraction}) to ensure disjoint test sets.")

    print(f"--- Generating {k_folds}-Fold hv-Block Splits (Val+Test Grouped) ---")

    daily_start_indices = all_data.groupby(all_data['Datetime'].dt.date).head(1).index.tolist()
    num_total_days = len(daily_start_indices)
    exclusion_steps = exclusion_hours * 60

    # 1. Identify all non-overlapping episodes
    all_episodes_start_days = []
    current_day = 0
    while current_day + days_per_episode <= num_total_days:
        all_episodes_start_days.append(current_day)
        current_day += days_per_episode

    total_episodes = len(all_episodes_start_days)
    print(f"Found {total_episodes} discrete {days_per_episode}-day episodes.")

    # 2. Calculate Block Composition
    num_test_total = int(round(total_episodes * test_fraction))
    num_val_total = int(round(total_episodes * val_fraction))

    # We want to distribute these evenly. We find the greatest common divisor
    # to create identical "Macro-Blocks" containing both Test and Val.
    # If test=15 and val=15, gcd is 15. We make 15 blocks, each has 1 Test, 1 Val.
    block_gcd = math.gcd(num_test_total, num_val_total)

    if block_gcd == 0:
        raise ValueError("Fractions result in 0 episodes.")

    test_per_block = num_test_total // block_gcd
    val_per_block = num_val_total // block_gcd

    num_eval_blocks = block_gcd
    stride = total_episodes / num_eval_blocks

    folds_output = []

    # 3. Build each fold
    for iteration in range(k_folds):
        test_starts_indices = []
        val_starts_indices = []

        # Calculate the shift for this fold to ensure disjoint Test sets
        # We shift by a fraction of the stride.
        fold_offset = (iteration / k_folds) * stride

        # Place the Evaluation Blocks
        for b in range(num_eval_blocks):
            base_idx = int(round(fold_offset + b * stride))

            # A) Append Test Episodes
            for t in range(test_per_block):
                test_starts_indices.append((base_idx + t) % total_episodes)

            # B) Append Val Episodes IMMEDIATELY after Test
            for v in range(val_per_block):
                val_starts_indices.append((base_idx + test_per_block + v) % total_episodes)

        test_starts = [all_episodes_start_days[i] for i in test_starts_indices]
        val_starts = [all_episodes_start_days[i] for i in val_starts_indices]

        # Build masks
        is_test_mask = pd.Series(False, index=all_data.index)
        is_val_mask = pd.Series(False, index=all_data.index)
        is_buffer_mask = pd.Series(False, index=all_data.index)

        def extract_episodes_and_mark(start_days, target_mask):
            episodes_list = []
            for start in start_days:
                idx_start = daily_start_indices[start]
                day_end = start + days_per_episode

                if day_end >= len(daily_start_indices):
                    idx_end = len(all_data) - 1
                else:
                    idx_end = daily_start_indices[day_end] - 1

                episode_slice = all_data.loc[idx_start:idx_end].copy().reset_index(drop=True)
                episodes_list.append(episode_slice)

                target_mask.loc[idx_start:idx_end] = True

                # Buffer zone (h-block)
                unsafe_start = max(0, idx_start - exclusion_steps)
                unsafe_end = min(len(all_data) - 1, idx_end + exclusion_steps)
                is_buffer_mask.loc[unsafe_start:unsafe_end] = True

            return episodes_list, target_mask

        test_episodes_list, is_test_mask = extract_episodes_and_mark(test_starts, is_test_mask)
        val_episodes_list, is_val_mask = extract_episodes_and_mark(val_starts, is_val_mask)

        # Train data is everything that is NOT Test, NOT Val, and NOT a Buffer
        is_train_mask = ~(is_test_mask | is_val_mask | is_buffer_mask)
        train_df = all_data[is_train_mask].reset_index(drop=True)

        folds_output.append((train_df, val_episodes_list, test_episodes_list))
        print(f"  Fold {iteration + 1}/{k_folds} -> "
              f"Train: {len(train_df)} rows, "
              f"Val: {len(val_episodes_list)} eps, "
              f"Test: {len(test_episodes_list)} eps")

    if save_path is not None:
        _save_folds(save_path, folds_output)

    return folds_output

# ── Private helpers for fold caching ────────────────────────────────────

def _fold_paths(save_path: str, k: int) -> List[Tuple[str, str, str]]:
    """Returns a list of (train_path, val_path, test_path) for each fold."""
    return [
        (
            os.path.join(save_path, f"fold_{i}_train.pkl"),
            os.path.join(save_path, f"fold_{i}_val.pkl"),
            os.path.join(save_path, f"fold_{i}_test.pkl"),
        )
        for i in range(k)
    ]


def _load_folds(
        save_path: str, k: int
) -> "List[Tuple[pd.DataFrame, List[pd.DataFrame], List[pd.DataFrame]]] | None":
    """Load all K folds from *save_path*. Returns ``None`` if any file is missing."""
    paths = _fold_paths(save_path, k)
    all_exist = all(
        os.path.exists(p) for triple in paths for p in triple
    )
    if not all_exist:
        return None

    print(f"--- Loading cached {k}-fold splits from '{save_path}' ---")
    folds = []
    for i, (train_p, val_p, test_p) in enumerate(paths):
        train_df = pd.read_pickle(train_p)
        val_eps = pd.read_pickle(val_p)
        test_eps = pd.read_pickle(test_p)
        folds.append((train_df, val_eps, test_eps))
        print(
            f"  Fold {i + 1}/{k} -> Train rows: {len(train_df)}, "
            f"Val episodes: {len(val_eps)}, Test episodes: {len(test_eps)}"
        )
    return folds


def _save_folds(
        save_path: str,
        folds: List[Tuple[pd.DataFrame, List[pd.DataFrame], List[pd.DataFrame]]]
) -> None:
    """Persist every fold to *save_path* as individual pickles."""
    os.makedirs(save_path, exist_ok=True)
    paths = _fold_paths(save_path, len(folds))
    for i, ((train_df, val_eps, test_eps), (train_p, val_p, test_p)) in enumerate(
            zip(folds, paths)
    ):
        train_df.to_pickle(train_p)
        pd.to_pickle(val_eps, val_p)
        pd.to_pickle(test_eps, test_p)
    print(f"--- Saved {len(folds)} fold(s) to '{save_path}' ---")
