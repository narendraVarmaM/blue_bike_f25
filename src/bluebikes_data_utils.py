import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

import calendar
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytz
import requests
import holidays

from src.bluebikes_config import RAW_DATA_DIR, WEATHER_DATA_DIR, WEATHER_API_KEY, WEATHER_API_URL, TIME_SHIFT_WEEKS


def fetch_bluebikes_data(year: int, month: int) -> Path:
    """
    Fetch Blue Bikes data for a specific year and month.
    
    Args:
        year (int): Year to fetch data for
        month (int): Month to fetch data for (1-12)
    
    Returns:
        Path: Path to the saved parquet file
    """
    # Blue Bikes data is typically available as monthly CSV files
    # Format: YYYYMM-bluebikes-tripdata.zip or similar
    file_name = f"{year}{month:02}-bluebikes-tripdata.csv"
    zip_name = f"{year}{month:02}-bluebikes-tripdata.zip"
    
    # Try both CSV and ZIP formats
    urls = [
        f"https://s3.amazonaws.com/hubway-data/{file_name}",
        f"https://s3.amazonaws.com/hubway-data/{zip_name}",
    ]
    
    save_path = RAW_DATA_DIR / f"rides_{year}_{month:02}.parquet"
    
    # If file already exists, return the path
    if save_path.exists():
        print(f"File already exists: {save_path}")
        return save_path
    
    for url in urls:
        try:
            print(f"Attempting to download from: {url}")
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                # Save as temporary CSV
                temp_csv = RAW_DATA_DIR / f"temp_{year}_{month:02}.csv"
                
                if url.endswith('.zip'):
                    import zipfile
                    import io
                    # Extract CSV from ZIP
                    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                        csv_files = [f for f in zip_file.namelist() if f.endswith('.csv')]
                        if csv_files:
                            with open(temp_csv, 'wb') as f:
                                f.write(zip_file.read(csv_files[0]))
                else:
                    with open(temp_csv, 'wb') as f:
                        f.write(response.content)
                
                # Convert to parquet
                df = pd.read_csv(temp_csv)
                df.to_parquet(save_path, engine='pyarrow')
                
                # Clean up temp file
                temp_csv.unlink()
                
                print(f"Successfully downloaded and converted to parquet: {save_path}")
                return save_path
                
        except Exception as e:
            print(f"Failed to download from {url}: {str(e)}")
            continue
    
    raise Exception(f"Could not download Blue Bikes data for {year}-{month:02}")


def fetch_weather_data(lat: float, lon: float, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Fetch historical weather data for Boston area.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        start_date (datetime): Start date for weather data
        end_date (datetime): End date for weather data
    
    Returns:
        pd.DataFrame: Weather data with hourly resolution
    """
    if not WEATHER_API_KEY:
        print("Warning: WEATHER_API_KEY not set. Returning dummy weather data.")
        # Return dummy weather data for testing
        hours = pd.date_range(start=start_date, end=end_date, freq='H')
        return pd.DataFrame({
            'hour': hours,
            'temperature': np.random.uniform(0, 30, len(hours)),
            'precipitation': np.random.uniform(0, 5, len(hours)),
            'wind_speed': np.random.uniform(0, 20, len(hours)),
            'humidity': np.random.uniform(30, 90, len(hours)),
            'weather_condition': np.random.choice(['Clear', 'Clouds', 'Rain'], len(hours))
        })
    
    # Implement actual API call here
    # This is a placeholder - adjust based on your weather API
    weather_data = []
    current_date = start_date
    
    while current_date <= end_date:
        try:
            # Example for OpenWeatherMap API (adjust as needed)
            timestamp = int(current_date.timestamp())
            url = f"{WEATHER_API_URL}/onecall/timemachine"
            params = {
                'lat': lat,
                'lon': lon,
                'dt': timestamp,
                'appid': WEATHER_API_KEY,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                # Extract relevant weather info
                hourly = data.get('hourly', [])
                for hour_data in hourly:
                    weather_data.append({
                        'hour': pd.Timestamp.fromtimestamp(hour_data['dt']),
                        'temperature': hour_data.get('temp', 0),
                        'precipitation': hour_data.get('rain', {}).get('1h', 0),
                        'wind_speed': hour_data.get('wind_speed', 0),
                        'humidity': hour_data.get('humidity', 0),
                        'weather_condition': hour_data.get('weather', [{}])[0].get('main', 'Unknown')
                    })
            
            current_date += timedelta(days=1)
            
        except Exception as e:
            print(f"Error fetching weather data for {current_date}: {e}")
            current_date += timedelta(days=1)
    
    return pd.DataFrame(weather_data)


def filter_bluebikes_data(rides: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    """
    Filters Blue Bikes ride data for a specific year and month, removing outliers and invalid records.
    Preserves station coordinate information for weather mapping.
    
    Args:
        rides (pd.DataFrame): DataFrame containing Blue Bikes ride data
        year (int): Year to filter for
        month (int): Month to filter for (1-12)
    
    Returns:
        pd.DataFrame: Filtered DataFrame containing only valid rides with coordinates
    """
    # Validate inputs
    if not (1 <= month <= 12):
        raise ValueError("Month must be between 1 and 12.")
    
    # Blue Bikes column names may vary - handle different naming conventions
    start_time_col = None
    station_id_col = None
    start_lat_col = None
    start_lon_col = None
    
    # Common column name variations
    for col in rides.columns:
        col_lower = col.lower()
        # Look for timestamp column: started_at, start_time, etc.
        if 'start' in col_lower and ('at' in col_lower or 'time' in col_lower):
            start_time_col = col
        # Look for station ID column
        if 'start' in col_lower and 'station' in col_lower and 'id' in col_lower:
            station_id_col = col
        # Look for latitude
        if 'start' in col_lower and ('lat' in col_lower or 'latitude' in col_lower):
            start_lat_col = col
        # Look for longitude
        if 'start' in col_lower and ('lon' in col_lower or 'lng' in col_lower or 'longitude' in col_lower):
            start_lon_col = col
    
    if start_time_col is None or station_id_col is None:
        raise ValueError(f"Could not identify required columns. Available columns: {rides.columns.tolist()}")
    
    print(f"Using columns: start_time={start_time_col}, station_id={station_id_col}")
    if start_lat_col and start_lon_col:
        print(f"  Coordinates: lat={start_lat_col}, lon={start_lon_col}")
    else:
        print(f"  Warning: Station coordinates not found, weather will use default Boston location")
    
    # Standardize column names
    rename_dict = {
        start_time_col: 'started_at',
        station_id_col: 'start_station_id'
    }
    
    if start_lat_col:
        rename_dict[start_lat_col] = 'start_lat'
    if start_lon_col:
        rename_dict[start_lon_col] = 'start_lng'
    
    rides = rides.rename(columns=rename_dict)
    
    # Convert timestamps
    rides['started_at'] = pd.to_datetime(rides['started_at'], errors='coerce')
    
    # Calculate start and end dates for the specified month
    start_date = pd.Timestamp(year=year, month=month, day=1)
    end_date = pd.Timestamp(year=year + (month // 12), month=(month % 12) + 1, day=1)
    
    # Define filters
    date_range_filter = (rides['started_at'] >= start_date) & (rides['started_at'] < end_date)
    valid_station_filter = rides['start_station_id'].notna()
    valid_time_filter = rides['started_at'].notna()
    
    top_locations = ["M32006", "M32011", "M32018", "M32005", "M32006", "M32041", "M32042", "M32037"]
    location_filter = rides["start_station_id"].isin(top_locations)
    # Combine all filters
    final_filter = date_range_filter & valid_station_filter & valid_time_filter & location_filter
    
    # Calculate dropped records
    total_records = len(rides)
    valid_records = final_filter.sum()
    records_dropped = total_records - valid_records
    percent_dropped = (records_dropped / total_records) * 100 if total_records > 0 else 0
    
    print(f"Total records: {total_records:,}")
    print(f"Valid records: {valid_records:,}")
    print(f"Records dropped: {records_dropped:,} ({percent_dropped:.2f}%)")
    
    # Filter the DataFrame
    validated_rides = rides[final_filter].copy()
    
    # Select columns to keep
    columns_to_keep = ['started_at', 'start_station_id']
    if 'start_lat' in validated_rides.columns:
        columns_to_keep.append('start_lat')
    if 'start_lng' in validated_rides.columns:
        columns_to_keep.append('start_lng')
    
    validated_rides = validated_rides[columns_to_keep]
    
    validated_rides.rename(
        columns={
            'started_at': 'pickup_datetime',
            'start_station_id': 'pickup_location_id',
        },
        inplace=True,
    )
    
    # Verify we have data
    if validated_rides.empty:
        raise ValueError(f"No valid rides found for {year}-{month:02} after filtering.")
    
    return validated_rides


def is_us_holiday(date):
    """Check if a date is a US holiday"""
    us_holidays = holidays.UnitedStates()
    return date in us_holidays


def fill_missing_rides_full_range(df, hour_col, location_col, rides_col):
    """
    Fills in missing rides for all hours in the range and all unique locations.
    
    Parameters:
    - df: DataFrame with columns [hour_col, location_col, rides_col]
    - hour_col: Name of the column containing hourly timestamps
    - location_col: Name of the column containing location IDs
    - rides_col: Name of the column containing ride counts
    
    Returns:
    - DataFrame with missing hours and locations filled in with 0 rides
    """
    # Ensure the hour column is in datetime format
    df[hour_col] = pd.to_datetime(df[hour_col])
    
    # Get the full range of hours (from min to max) with hourly frequency
    full_hours = pd.date_range(
        start=df[hour_col].min(),
        end=df[hour_col].max(),
        freq="h"
    )
    
    # Get all unique location IDs
    all_locations = df[location_col].unique()
    
    # Create a DataFrame with all combinations of hours and locations
    full_combinations = pd.DataFrame(
        [(hour, location) for hour in full_hours for location in all_locations],
        columns=[hour_col, location_col]
    )
    
    # Merge the original DataFrame with the full combinations DataFrame
    merged_df = pd.merge(full_combinations, df, on=[hour_col, location_col], how='left')
    
    # Fill missing rides with 0
    merged_df[rides_col] = merged_df[rides_col].fillna(0).astype(int)
    
    return merged_df


def load_and_process_bluebikes_data(
    year: int, months: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Load and process Blue Bikes ride data for a specified year and list of months.
    
    Args:
        year (int): Year to load data for
        months (Optional[List[int]]): List of months to load. If None, loads all months (1-12)
    
    Returns:
        pd.DataFrame: Combined and processed ride data
    """
    if months is None:
        months = list(range(1, 13))
    
    monthly_rides = []
    
    for month in months:
        file_path = RAW_DATA_DIR / f"rides_{year}_{month:02}.parquet"
        
        try:
            # Download the file if it doesn't exist
            if not file_path.exists():
                print(f"Downloading data for {year}-{month:02}...")
                fetch_bluebikes_data(year, month)
                print(f"Successfully downloaded data for {year}-{month:02}.")
            else:
                print(f"File already exists for {year}-{month:02}.")
            
            # Load the data
            print(f"Loading data for {year}-{month:02}...")
            rides = pd.read_parquet(file_path, engine="pyarrow")
            
            # Filter and process the data
            rides = filter_bluebikes_data(rides, year, month)
            print(f"Successfully processed data for {year}-{month:02}.")
            
            monthly_rides.append(rides)
            
        except FileNotFoundError:
            print(f"File not found for {year}-{month:02}. Skipping...")
        except Exception as e:
            print(f"Error processing data for {year}-{month:02}: {str(e)}")
            continue
    
    if not monthly_rides:
        raise Exception(
            f"No data could be loaded for the year {year} and specified months: {months}"
        )
    
    print("Combining all monthly data...")
    combined_rides = pd.concat(monthly_rides, ignore_index=True)
    print("Data loading and processing complete!")
    
    return combined_rides


def transform_raw_data_into_ts_data(rides: pd.DataFrame, include_weather: bool = False) -> pd.DataFrame:
    """
    Transform raw Blue Bikes data into time series format with optional weather features.
    
    Args:
        rides (pd.DataFrame): Raw ride data
        include_weather (bool): Whether to include weather features
    
    Returns:
        pd.DataFrame: Time series data aggregated by hour and location
    """
    # Round to hour
    rides['pickup_hour'] = rides['pickup_datetime'].dt.floor('H')
    
    # Aggregate rides by hour and location
    ts_data = rides.groupby(['pickup_hour', 'pickup_location_id']).size().reset_index(name='rides')
    
    # Fill missing hours for all locations
    ts_data = fill_missing_rides_full_range(ts_data, 'pickup_hour', 'pickup_location_id', 'rides')
    
    if include_weather:
        # Fetch weather data
        min_date = ts_data['pickup_hour'].min()
        max_date = ts_data['pickup_hour'].max()
        
        print(f"Fetching weather data from {min_date} to {max_date}...")
        weather_df = fetch_weather_data(
            lat=42.3601,  # Boston latitude
            lon=-71.0589,  # Boston longitude
            start_date=min_date,
            end_date=max_date
        )
        
        # Merge weather data
        if not weather_df.empty:
            ts_data = ts_data.merge(weather_df, left_on='pickup_hour', right_on='hour', how='left')
            ts_data = ts_data.drop(columns=['hour'], errors='ignore')
            
            # Fill missing weather values with forward fill then backward fill
            weather_cols = ['temperature', 'precipitation', 'wind_speed', 'humidity']
            for col in weather_cols:
                if col in ts_data.columns:
                    ts_data[col] = ts_data[col].fillna(method='ffill').fillna(method='bfill')
            
            print("Weather data merged successfully.")
        else:
            print("Warning: No weather data available.")
    
    # Add temporal features
    ts_data['hour_of_day'] = ts_data['pickup_hour'].dt.hour
    ts_data['day_of_week'] = ts_data['pickup_hour'].dt.dayofweek
    ts_data['is_weekend'] = ts_data['day_of_week'].isin([5, 6]).astype(int)
    ts_data['month'] = ts_data['pickup_hour'].dt.month
    ts_data['is_holiday'] = ts_data['pickup_hour'].dt.date.apply(is_us_holiday).astype(int)
    
    # Sort by station and hour
    ts_data = ts_data.sort_values(['pickup_location_id', 'pickup_hour']).reset_index(drop=True)
    
    return ts_data


def fetch_batch_raw_data(
    from_date: Union[datetime, str], to_date: Union[datetime, str]
) -> pd.DataFrame:
    """
    Simulate production data by sampling historical data from 8 weeks ago (2 months).
    
    Args:
        from_date (datetime or str): The start date for the data batch
        to_date (datetime or str): The end date for the data batch
    
    Returns:
        pd.DataFrame: A DataFrame containing the simulated production data
    """
    # Convert string inputs to datetime if necessary
    if isinstance(from_date, str):
        from_date = datetime.fromisoformat(from_date)
    if isinstance(to_date, str):
        to_date = datetime.fromisoformat(to_date)
    
    # Validate input dates
    if not isinstance(from_date, datetime) or not isinstance(to_date, datetime):
        raise ValueError(
            "Both 'from_date' and 'to_date' must be datetime objects or valid ISO format strings."
        )
    if from_date >= to_date:
        raise ValueError("'from_date' must be earlier than 'to_date'.")
    
    # Shift dates back by TIME_SHIFT_WEEKS (8 weeks = 2 months)
    historical_from_date = from_date - timedelta(weeks=TIME_SHIFT_WEEKS)
    historical_to_date = to_date - timedelta(weeks=TIME_SHIFT_WEEKS)
    
    print(f"Fetching historical data from {historical_from_date} to {historical_to_date}")
    
    # Load and filter data for the historical period
    rides_from = load_and_process_bluebikes_data(
        year=historical_from_date.year, months=[historical_from_date.month]
    )
    rides_from = rides_from[
        rides_from.pickup_datetime >= historical_from_date
    ]
    
    if historical_to_date.month != historical_from_date.month:
        rides_to = load_and_process_bluebikes_data(
            year=historical_to_date.year, months=[historical_to_date.month]
        )
        rides_to = rides_to[
            rides_to.pickup_datetime < historical_to_date
        ]
        # Combine the filtered data
        rides = pd.concat([rides_from, rides_to], ignore_index=True)
    else:
        rides = rides_from
    
    # Shift the data forward by TIME_SHIFT_WEEKS to simulate recent data
    rides['pickup_datetime'] += timedelta(weeks=TIME_SHIFT_WEEKS)
    
    # Sort the data for consistency
    rides.sort_values(by=['pickup_location_id', 'pickup_datetime'], inplace=True)
    
    return rides


def transform_ts_data_into_features(
    df: pd.DataFrame,
    window_size: int = 672,  # 28 days * 24 hours
    step_size: int = 23,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Transforms time series data into features for machine learning.
    
    Args:
        df (pd.DataFrame): Time series data
        window_size (int): Number of hours to use as features (default: 28 days)
        step_size (int): Step size for sliding window
    
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features and targets
    """
    location_ids = df['pickup_location_id'].unique()
    transformed_data = []
    
    for location_id in location_ids:
        try:
            location_data = df[df['pickup_location_id'] == location_id].reset_index(drop=True)
            
            values = location_data['rides'].values
            times = location_data['pickup_hour'].values
            
            if len(values) <= window_size:
                print(f"Skipping location {location_id}: Not enough data")
                continue
            
            rows = []
            for i in range(0, len(values) - window_size, step_size):
                features = values[i:i + window_size]
                target = values[i + window_size]
                target_time = times[i + window_size]
                
                row = np.append(features, [target, location_id, target_time])
                rows.append(row)
            
            feature_columns = [f"rides_t-{window_size - i}" for i in range(window_size)]
            all_columns = feature_columns + ['target', 'pickup_location_id', 'pickup_hour']
            transformed_df = pd.DataFrame(rows, columns=all_columns)
            
            transformed_data.append(transformed_df)
            
        except Exception as e:
            print(f"Error processing location {location_id}: {str(e)}")
    
    if not transformed_data:
        raise ValueError("No data could be transformed.")
    
    final_df = pd.concat(transformed_data, ignore_index=True)
    
    features = final_df[feature_columns + ['pickup_hour', 'pickup_location_id']]
    targets = final_df['target']
    
    return features, targets


def transform_ts_data_info_features(
    df, feature_col="rides", window_size=672, step_size=23
):
    """
    Transforms time series data for inference (no target).
    
    Parameters:
        df (pd.DataFrame): The input DataFrame containing time series data
        feature_col (str): The column name containing the values to use as features
        window_size (int): The number of hours to use as features (default: 28 days)
        step_size (int): The number of rows to slide the window by
    
    Returns:
        pd.DataFrame: Features DataFrame with pickup_hour and location_id
    """
    location_ids = df['pickup_location_id'].unique()
    transformed_data = []
    
    for location_id in location_ids:
        try:
            location_data = df[df['pickup_location_id'] == location_id].reset_index(drop=True)
            
            values = location_data[feature_col].values
            times = location_data['pickup_hour'].values
            
            if len(values) <= window_size:
                continue
            
            rows = []
            for i in range(0, len(values) - window_size, step_size):
                features = values[i:i + window_size]
                target_time = times[i + window_size]
                row = np.append(features, [location_id, target_time])
                rows.append(row)
            
            feature_columns = [f"{feature_col}_t-{window_size - i}" for i in range(window_size)]
            all_columns = feature_columns + ['pickup_location_id', 'pickup_hour']
            transformed_df = pd.DataFrame(rows, columns=all_columns)
            
            transformed_data.append(transformed_df)
            
        except Exception as e:
            print(f"Skipping location_id {location_id}: {str(e)}")
    
    if not transformed_data:
        raise ValueError("No data could be transformed.")
    
    final_df = pd.concat(transformed_data, ignore_index=True)
    
    return final_df


def split_time_series_data(
    df: pd.DataFrame,
    cutoff_date: datetime,
    target_column: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Splits a time series DataFrame into training and testing sets based on a cutoff date.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing the time series data
        cutoff_date (datetime): The date used to split the data
        target_column (str): The name of the target column
    
    Returns:
        Tuple: X_train, y_train, X_test, y_test
    """
    train_data = df[df['pickup_hour'] < cutoff_date].reset_index(drop=True)
    test_data = df[df['pickup_hour'] >= cutoff_date].reset_index(drop=True)
    
    X_train = train_data.drop(columns=[target_column])
    y_train = train_data[target_column]
    X_test = test_data.drop(columns=[target_column])
    y_test = test_data[target_column]
    
    return X_train, y_train, X_test, y_test