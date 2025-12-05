"""
Blue Bikes Station Name Mapping

Provides utilities to convert station IDs to human-readable names.
"""

from pathlib import Path
import pandas as pd

# Path to station list Excel file
STATION_LIST_PATH = Path(__file__).parent / "-External-_Bluebikes_Station_List.xlsx"


def load_station_names() -> dict:
    """
    Load station ID to name mapping from Excel file.

    Returns:
        dict: Mapping of station ID (str) to station name (str)
    """
    try:
        df = pd.read_excel(STATION_LIST_PATH, header=1)
        # Create mapping: {station_id: station_name}
        mapping = dict(zip(df['Number'], df['NAME']))
        return mapping
    except Exception as e:
        print(f"Warning: Could not load station names: {e}")
        return {}


def get_station_name(station_id: str, fallback_to_id: bool = True) -> str:
    """
    Get the human-readable name for a station ID.

    Args:
        station_id (str): Station ID (e.g., 'M32006')
        fallback_to_id (bool): If True, return the ID if name not found

    Returns:
        str: Station name or ID
    """
    mapping = load_station_names()

    if station_id in mapping:
        return mapping[station_id]
    elif fallback_to_id:
        return station_id
    else:
        return "Unknown Station"


def add_station_names_to_dataframe(df: pd.DataFrame, id_column: str = 'pickup_location_id') -> pd.DataFrame:
    """
    Add a column with station names to a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with station IDs
        id_column (str): Name of column containing station IDs

    Returns:
        pd.DataFrame: DataFrame with added 'station_name' column
    """
    if id_column not in df.columns:
        return df

    mapping = load_station_names()
    df = df.copy()
    df['station_name'] = df[id_column].map(mapping).fillna(df[id_column])

    return df


# Pre-defined mapping for the main stations used in the project
MAIN_STATIONS = {
    'M32006': 'MIT at Mass Ave / Amherst St',
    'M32011': 'Central Square at Mass Ave / Essex St',
    'M32018': 'Harvard Square at Mass Ave/ Dunster',
    'M32005': 'MIT Stata Center at Vassar St / Main St',
    'M32041': 'MIT Pacific St at Purrington St',
    'M32042': 'MIT Vassar St',
    'M32037': 'Ames St at Main St',
}


def get_main_station_name(station_id: str) -> str:
    """
    Get name for main stations (faster than loading full Excel).

    Args:
        station_id (str): Station ID

    Returns:
        str: Station name
    """
    return MAIN_STATIONS.get(station_id, station_id)
