import logging
import os
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def clean_data(
    data: Optional[pd.DataFrame] = None,
    raw_path: Optional[str] = None,
    cache_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Cleans raw Elia imbalance price data for use in episodic RL, with
    optional built-in caching so the cleaning only runs once.

    Usage examples::

        # Basic — pass a DataFrame, no caching:
        df = clean_data(data=raw_df)

        # With caching — returns cached result on subsequent calls:
        df = clean_data(raw_path="data/2025_minute.csv",
                        cache_path="data/2025_minute_cleaned.pkl")

        # DataFrame + caching:
        df = clean_data(data=raw_df, cache_path="data/cleaned.pkl")

    Steps performed (when cleaning is needed):
        1. Validates that required columns ('Datetime', 'Imbalance Price')
           exist.
        2. Parses the 'Datetime' column to UTC-aware timestamps.
        3. Sorts by datetime in ascending order (only reverses if needed).
        4. Removes duplicate timestamps, keeping the first occurrence.
        5. Validates the first record starts on a quarter-hour boundary.
        6. Drops every 15-min quarter that contains at least one NaN in
           'Imbalance Price'.
        7. Reports gaps (missing minutes) in the resulting time series.

    Args:
        data: Raw DataFrame (e.g. from ``pd.read_csv(..., sep=';')``).
            Must contain at least 'Datetime' and 'Imbalance Price'.
            Either *data* or *raw_path* (or both) must be provided.
        raw_path: Path to the raw CSV file (semicolon-separated). Used to
            load the data when *data* is not supplied, and to derive a
            default *cache_path* when none is given explicitly.
        cache_path: Path to a pickle file for caching the cleaned
            DataFrame. If the file already exists the cached version is
            loaded and returned immediately (no cleaning is performed).
            If ``None`` and *raw_path* is given, defaults to
            ``<raw_path stem>_cleaned.pkl``.

    Returns:
        A cleaned DataFrame sorted by ascending datetime with a fresh
        integer index.

    Raises:
        ValueError: If required columns are missing, the datetime column
            cannot be parsed, the data does not start on a quarter-hour
            boundary, or neither *data* nor *raw_path* is provided.
    """
    # ── Resolve cache path ──────────────────────────────────────────────
    if cache_path is None and raw_path is not None:
        base, _ = os.path.splitext(raw_path)
        cache_path = f"{base}_cleaned.pkl"

    # ── Return cached result if available ───────────────────────────────
    if cache_path and os.path.exists(cache_path):
        logger.info("Loading cached cleaned data from: %s", cache_path)
        return pd.read_pickle(cache_path)

    # ── Obtain raw data ─────────────────────────────────────────────────
    if data is None and raw_path is not None:
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"Raw data file not found: {raw_path}")
        logger.info("Reading raw data from: %s", raw_path)
        data = pd.read_csv(raw_path, sep=";")
    elif data is None:
        raise ValueError("Provide at least one of 'data' or 'raw_path'.")

    # ── Validate required columns ───────────────────────────────────────
    required_cols = {"Datetime", "Imbalance Price"}
    missing = required_cols - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    rows_before = len(data)
    data = data.copy()

    # ── 1. Parse datetimes ──────────────────────────────────────────────
    try:
        data["Datetime"] = pd.to_datetime(data["Datetime"], utc=True)
    except Exception as exc:
        raise ValueError(f"Failed to parse 'Datetime' column: {exc}") from exc

    # ── 2. Sort ascending (only reverse when necessary) ─────────────────
    if not data["Datetime"].is_monotonic_increasing:
        data = data.sort_values("Datetime").reset_index(drop=True)
        logger.info("Data was not in ascending order — sorted by Datetime.")

    # ── 3. Remove duplicate timestamps ──────────────────────────────────
    dup_mask = data["Datetime"].duplicated(keep="first")
    n_dups = dup_mask.sum()
    if n_dups:
        data = data[~dup_mask]
        logger.warning("Removed %d duplicate timestamp(s).", n_dups)

    # ── 4. Validate quarter-hour alignment ──────────────────────────────
    start_minute = data.iloc[0]["Datetime"].minute
    if start_minute % 15 != 0:
        raise ValueError(
            f"Data does not start on a quarter-hour boundary "
            f"(expected minute 0, 15, 30, or 45 — got {start_minute})."
        )
    logger.info("Data starts on a quarter-hour boundary (%02d:%02d).",
                data.iloc[0]["Datetime"].hour, start_minute)

    # ── 5. Resample to continuous 1-minute intervals and handle NaNs ────
    data = data.set_index("Datetime").resample("1min").asfreq().reset_index()
    logger.info("Resampled to strict 1-minute intervals. Generated NaNs for missing minutes.")

    quarter_id = data["Datetime"].dt.floor("15min")

    # We want to fill small gaps (e.g., missing a few minutes).
    # If a whole quarter (15 straight minutes) is missing, forward-filling
    # 15 minutes of prices might be too inaccurate, so we'll drop those quarters instead.

    # First, let's forward-fill *within* limit (e.g. up to 10 missing minutes)
    # Applied to all columns so no other variables leak NaNs into the environment
    data = data.ffill(limit=10)

    # Any NaNs remaining mean there was a gap longer than 10 minutes.
    # Drop the entire 15-min quarter for any remaining NaNs
    nan_mask = data["Imbalance Price"].isna()
    n_nan = nan_mask.sum()

    if n_nan > 0:
        faulty_quarters = quarter_id[nan_mask].unique()
        keep_mask = ~quarter_id.isin(faulty_quarters)
        data = data[keep_mask]
        logger.warning(
            "Found %d unfillable NaN value(s) in 'Imbalance Price' across %d quarter(s) "
            "— dropped %d row(s).",
            n_nan, len(faulty_quarters), (~keep_mask).sum(),
        )

    data = data.reset_index(drop=True)

    # ── 6. Report gaps in the time series ───────────────────────────────
    time_diffs = data["Datetime"].diff()
    expected_freq = pd.Timedelta(minutes=1)
    gaps = time_diffs[time_diffs > expected_freq].dropna()
    if not gaps.empty:
        logger.warning(
            "Detected %d gap(s) in the time series (expected 1-min spacing):",
            len(gaps),
        )
        for idx, delta in gaps.items():
            gap_start = data.loc[idx - 1, "Datetime"]
            gap_end = data.loc[idx, "Datetime"]
            logger.warning("  Gap of %s from %s to %s", delta, gap_start, gap_end)

    rows_after = len(data)
    logger.info(
        "Cleaning complete: %d → %d rows (removed %d).",
        rows_before, rows_after, rows_before - rows_after,
    )

    # ── Save to cache ───────────────────────────────────────────────────
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        data.to_pickle(cache_path)
        logger.info("Saved cleaned data to cache: %s", cache_path)

    return data

if __name__ == "__main__":
    # Temporary main function to test if there are any incomplete quarters
    import warnings
    warnings.filterwarnings("ignore")

    # Using the relative path to the root data folder
    test_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "data/raw_elia_data.csv"))

    if os.path.exists(test_path):
        print(f"Cleaning data from {test_path}...")

        # Test original reading for raw incomplete quarters comparison
        raw_data = pd.read_csv(test_path, sep=";")
        raw_data["Datetime"] = pd.to_datetime(raw_data["Datetime"], utc=True)
        raw_counts = raw_data.groupby(pd.Grouper(key="Datetime", freq="15min")).size()
        raw_inc = raw_counts[raw_counts < 15]
        print(f"\n--- BEFORE CLEANING ---")
        print(f"Total quarters: {len(raw_counts)}")
        print(f"Incomplete quarters: {len(raw_inc)}")
        if len(raw_inc) > 0:
            print("Row counts in incomplete quarters:")
            print(raw_inc.value_counts().sort_index())

        # Test after cleaning
        cleaned_data = clean_data(data=raw_data.copy(), cache_path=None)

        # Group by 15-minute intervals and count the number of 1-minute rows
        quarterly_counts = cleaned_data.groupby(pd.Grouper(key="Datetime", freq="15min")).size()

        # Filter for quarters with less than 15 minutes
        incomplete_quarters = quarterly_counts[quarterly_counts < 15]

        print(f"\n--- AFTER CLEANING ---")
        print(f"Total quarters: {len(quarterly_counts)}")
        print(f"Incomplete quarters: {len(incomplete_quarters)}")
        if len(incomplete_quarters) > 0:
            print("Row counts in incomplete quarters:")
            print(incomplete_quarters.value_counts().sort_index())
        else:
            print("Row counts in incomplete quarters: 0 (All quarters contain exactly 15 minutes!)")
    else:
        print(f"Could not find test data at {test_path}")
