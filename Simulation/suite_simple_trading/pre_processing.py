import pandas as pd


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Checks if data is in the right order,
    if it starts on a quarter hour
    if there are any NaN values, if so, it removes the whole quarter

    Args:
        data: A NumPy array of price data.

    Returns:
        A cleaned NumPy array with NaN values removed.
    """
    data = data.iloc[::-1].reset_index(drop=True)

    # Check data start on quarter
    try:
        data['Datetime'] = pd.to_datetime(data['Datetime'], utc=True)
        start_minute = data.loc[0, 'Datetime'].minute
        if start_minute % 15 != 0:
            raise ValueError(
                f"Data does not start on a quarter hour (minute 0, 15, 30, 45). Started at minute: {start_minute}")
        print("Data successfully verified to start on a quarter hour.")
    except:
        raise ValueError("Something went wrong with the datetime conversion.")

    # Check for nan values
    nan_rows = data[data['Imbalance Price'].isnull()]
    if len(nan_rows) > 0:
        # Removing all rows in a quarter where a nan value is found
        faulty_quarters = nan_rows["Datetime"].dt.floor('15T').unique()
        full_data_quarters = data['Datetime'].dt.floor('15T')
        keep_mask = ~full_data_quarters.isin(faulty_quarters)
        data = data[keep_mask]
    return data.reset_index(drop=True)