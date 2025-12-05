from datetime import datetime, timedelta, timezone

import hopsworks
import numpy as np
import pandas as pd
from hsfs.feature_store import FeatureStore

import src.bluebikes_config as config
from src.bluebikes_data_utils import transform_ts_data_info_features


def get_hopsworks_project() -> hopsworks.project.Project:
    return hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME, api_key_value=config.HOPSWORKS_API_KEY
    )


def get_feature_store() -> FeatureStore:
    project = get_hopsworks_project()
    return project.get_feature_store()


def get_model_predictions(model, features: pd.DataFrame) -> pd.DataFrame:
    """
    Generate predictions for Blue Bikes demand.
    
    Args:
        model: Trained model
        features (pd.DataFrame): Feature data
    
    Returns:
        pd.DataFrame: Predictions with station IDs
    """
    predictions = model.predict(features)
    
    results = pd.DataFrame()
    results["pickup_location_id"] = features["pickup_location_id"].values
    results["predicted_demand"] = predictions.round(0).astype(int)
    
    return results


def load_batch_of_features_from_store(
    current_date: datetime,
) -> pd.DataFrame:
    """
    Load features for inference from the feature store.
    Uses 2-month shifted data for real-time simulation.
    
    Args:
        current_date (datetime): Current datetime
    
    Returns:
        pd.DataFrame: Features for prediction
    """
    feature_store = get_feature_store()
    
    # Read time-series data from the feature store
    # Using 28 days of history + 1 hour buffer
    fetch_data_to = current_date - timedelta(hours=1)
    fetch_data_from = current_date - timedelta(days=29)
    
    print(f"Fetching data from {fetch_data_from} to {fetch_data_to}")
    
    feature_view = feature_store.get_feature_view(
        name=config.FEATURE_VIEW_NAME, version=config.FEATURE_VIEW_VERSION
    )
    
    ts_data = feature_view.get_batch_data(
        start_time=(fetch_data_from - timedelta(days=1)),
        end_time=(fetch_data_to + timedelta(days=1)),
    )
    ts_data = ts_data[ts_data.pickup_hour.between(fetch_data_from, fetch_data_to)]
    
    # Sort data by location and time
    ts_data.sort_values(by=["pickup_location_id", "pickup_hour"], inplace=True)
    
    # Transform into features (28 days * 24 hours = 672 hours window)
    features = transform_ts_data_info_features(
        ts_data, window_size=24 * 28, step_size=23
    )
    
    return features


def load_model_from_registry(version=None):
    """
    Load the trained Blue Bikes demand prediction model from registry.
    
    Args:
        version: Model version (optional)
    
    Returns:
        Trained model object
    """
    from pathlib import Path
    import joblib
    
    from src.pipeline_utils import (
        TemporalFeatureEngineer,
        average_rides_last_4_weeks,
    )
    
    project = get_hopsworks_project()
    model_registry = project.get_model_registry()
    
    models = model_registry.get_models(name=config.MODEL_NAME)
    
    if not models:
        raise ValueError(f"No models found with name: {config.MODEL_NAME}")
    
    # Get the latest version if not specified
    model = max(models, key=lambda m: m.version) if version is None else [m for m in models if m.version == version][0]
    
    model_dir = model.download()
    model = joblib.load(Path(model_dir) / "lgb_model.pkl")
    
    return model


def load_metrics_from_registry(version=None):
    """
    Load training metrics from the model registry.
    
    Args:
        version: Model version (optional)
    
    Returns:
        dict: Training metrics
    """
    project = get_hopsworks_project()
    model_registry = project.get_model_registry()
    
    models = model_registry.get_models(name=config.MODEL_NAME)
    
    if not models:
        raise ValueError(f"No models found with name: {config.MODEL_NAME}")
    
    model = max(models, key=lambda m: m.version) if version is None else [m for m in models if m.version == version][0]
    
    return model.training_metrics


def fetch_next_hour_predictions():
    """
    Fetch predictions for the next hour from the feature store.
    
    Returns:
        pd.DataFrame: Predictions for the next hour
    """
    # Get current UTC time and round up to next hour
    now = datetime.now(timezone.utc)
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    
    fs = get_feature_store()
    fg = fs.get_feature_group(name=config.FEATURE_GROUP_MODEL_PREDICTION, version=1)
    df = fg.read()
    
    # Filter for next hour
    df = df[df["pickup_hour"] == next_hour]
    
    print(f"Current UTC time: {now}")
    print(f"Next hour: {next_hour}")
    print(f"Found {len(df)} records")
    
    return df


def fetch_predictions(hours):
    """
    Fetch predictions for the last N hours.
    
    Args:
        hours (int): Number of hours to fetch
    
    Returns:
        pd.DataFrame: Historical predictions
    """
    current_hour = (pd.Timestamp.now(tz="Etc/UTC") - timedelta(hours=hours)).floor("h")
    
    fs = get_feature_store()
    fg = fs.get_feature_group(name=config.FEATURE_GROUP_MODEL_PREDICTION, version=1)
    
    df = fg.filter((fg.pickup_hour >= current_hour)).read()
    
    return df


def fetch_hourly_rides(hours):
    """
    Fetch actual ride data for the last N hours.
    
    Args:
        hours (int): Number of hours to fetch
    
    Returns:
        pd.DataFrame: Actual ride data
    """
    current_hour = (pd.Timestamp.now(tz="Etc/UTC") - timedelta(hours=hours)).floor("h")
    
    fs = get_feature_store()
    fg = fs.get_feature_group(name=config.FEATURE_GROUP_NAME, version=1)
    
    query = fg.select_all()
    query = query.filter(fg.pickup_hour >= current_hour)
    
    return query.read()


def fetch_days_data(days):
    """
    Fetch historical data for analysis (from 2 months ago).
    
    Args:
        days (int): Number of days to fetch
    
    Returns:
        pd.DataFrame: Historical ride data
    """
    current_date = pd.to_datetime(datetime.now(timezone.utc))
    
    # Shift by TIME_SHIFT_WEEKS (8 weeks = 2 months) to get historical data
    fetch_data_from = current_date - timedelta(days=(config.TIME_SHIFT_WEEKS * 7 + days))
    fetch_data_to = current_date - timedelta(days=config.TIME_SHIFT_WEEKS * 7)
    
    print(f"Fetching data from {fetch_data_from} to {fetch_data_to}")
    
    fs = get_feature_store()
    fg = fs.get_feature_group(name=config.FEATURE_GROUP_NAME, version=1)
    
    query = fg.select_all()
    df = query.read()
    
    cond = (df["pickup_hour"] >= fetch_data_from) & (df["pickup_hour"] <= fetch_data_to)
    
    return df[cond]


def save_predictions_to_store(predictions: pd.DataFrame, prediction_hour: datetime):
    """
    Save model predictions to the feature store.
    
    Args:
        predictions (pd.DataFrame): Model predictions
        prediction_hour (datetime): Hour for which predictions were made
    """
    predictions['pickup_hour'] = prediction_hour
    
    fs = get_feature_store()
    
    try:
        fg = fs.get_feature_group(
            name=config.FEATURE_GROUP_MODEL_PREDICTION,
            version=1
        )
    except:
        # Create feature group if it doesn't exist
        fg = fs.create_feature_group(
            name=config.FEATURE_GROUP_MODEL_PREDICTION,
            version=1,
            primary_key=['pickup_hour', 'pickup_location_id'],
            event_time='pickup_hour',
            online_enabled=True,
        )
    
    fg.insert(predictions, write_options={"wait_for_job": False})
    print(f"Predictions saved for {prediction_hour}")
