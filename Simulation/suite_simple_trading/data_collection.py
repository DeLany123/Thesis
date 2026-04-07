import os
import urllib.parse
import requests
import pandas as pd


def fetch_elia_1min_data(
        start_date: str = None,
        end_date: str = None,
        save_path: str = "data/raw_elia_data.csv"
) -> pd.DataFrame:
    """
    Downloads imbalance data from the Elia API.

    Args:
        start_date: Format 'YYYY-MM-DD'. If None, fetches from the beginning of the dataset.
        end_date: Format 'YYYY-MM-DD'. If None, fetches up to the current date.
        save_path: Path to save the downloaded CSV file.

    Returns:
        pd.DataFrame: The downloaded raw dataset.
    """
    # Base URL for the CSV export endpoint
    base_url = "https://opendata.elia.be/api/explore/v2.1/catalog/datasets/ods133/exports/csv"

    # API parameters
    params = {
        "delimiter": ";",  # Matches your current clean_data requirements
        "timezone": "UTC",
        "use_labels": "true"  # Gives readable column names instead of database IDs
    }

    # Construct the time filter
    where_clauses = []
    # Note: The database column for time in ods133 is usually called 'datetime'
    if start_date:
        where_clauses.append(f'datetime >= "{start_date}"')
    if end_date:
        where_clauses.append(f'datetime <= "{end_date}"')

    if where_clauses:
        params["where"] = " AND ".join(where_clauses)

    # Encode URL
    query_string = urllib.parse.urlencode(params)
    full_url = f"{base_url}?{query_string}"

    print(f"--- Fetching data from Elia API ---")
    if start_date or end_date:
        print(f"Filter: Start={start_date}, End={end_date}")
    else:
        print("Filter: Entire dataset")

    # Ensure the target directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Download the file in chunks (safe for large files)
    response = requests.get(full_url, stream=True)

    if response.status_code != 200:
        raise ConnectionError(f"API request failed with status code {response.status_code}: {response.text}")

    with open(save_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

    print(f"--- Download complete. File saved to: {save_path} ---")

    # Load and return the raw DataFrame
    raw_df = pd.read_csv(save_path, sep=';')
    return raw_df