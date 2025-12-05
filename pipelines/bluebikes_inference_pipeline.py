"""
Blue Bikes Boston - Inference Pipeline

This script runs hourly to:
1. Fetch features from the feature store
2. Load the trained model
3. Make predictions for the next hour
4. Save predictions back to the feature store

Can be scheduled with cron: 0 * * * * python bluebikes_inference_pipeline.py
"""

from datetime import datetime, timedelta

import pandas as pd

import backup.config as config
from src.bluebikes_inference import (
    get_feature_store,
    get_model_predictions,
    load_model_from_registry,
)

# Get the current datetime for Blue Bikes
# Uncomment below to run for past hours for testing
# for number in range(22, 24 * 29):
#     current_date = pd.Timestamp.now(tz="Etc/UTC") - timedelta(hours=number)

current_date = pd.Timestamp.now(tz="Etc/UTC")
feature_store = get_feature_store()

# Read time-series data from the feature store
# Blue Bikes uses same 28-day window as NYC Taxi
fetch_data_to = current_date - timedelta(hours=1)
fetch_data_from = current_date - timedelta(days=1 * 29)

print(f"[Blue Bikes Inference Pipeline]")
print(f"Current time: {current_date}")
print(f"Fetching data from {fetch_data_from} to {fetch_data_to}")

feature_view = feature_store.get_feature_view(
    name=config.FEATURE_VIEW_NAME, version=config.FEATURE_VIEW_VERSION
)

ts_data = feature_view.get_batch_data(
    start_time=(fetch_data_from - timedelta(days=1)),
    end_time=(fetch_data_to + timedelta(days=1)),
)

# Filter to exact time range
ts_data = ts_data[ts_data.pickup_hour.between(fetch_data_from, fetch_data_to)]
ts_data = ts_data.sort_values(["pickup_location_id", "pickup_hour"]).reset_index(drop=True)
ts_data["pickup_hour"] = ts_data["pickup_hour"].dt.tz_localize(None)

print(f"Loaded {len(ts_data)} time series records")
print(f"Unique stations: {ts_data['pickup_location_id'].nunique()}")

# Transform time series data into features
from src.bluebikes_data_utils import transform_ts_data_info_features

print("Transforming data into features (28-day window)...")
features = transform_ts_data_info_features(ts_data, window_size=24 * 28, step_size=23)

print(f"Generated features for {len(features)} stations")

# Load model from registry
print("Loading model from Hopsworks...")
model = load_model_from_registry()

# Make predictions
print("Making predictions for next hour...")
predictions = get_model_predictions(model, features)
predictions["pickup_hour"] = current_date.ceil("h")

print(f"\nPredictions Summary:")
print(f"  Total stations: {len(predictions)}")
print(f"  Average demand: {predictions['predicted_demand'].mean():.1f}")
print(f"  Max demand: {predictions['predicted_demand'].max():.0f}")
print(f"  Min demand: {predictions['predicted_demand'].min():.0f}")

# Top 5 stations
print(f"\nTop 5 stations by predicted demand:")
top5 = predictions.nlargest(5, 'predicted_demand')
for idx, row in top5.iterrows():
    print(f"  Station {row['pickup_location_id']}: {row['predicted_demand']:.0f} rides")

# Save predictions to feature store
print(f"\nSaving predictions to feature store...")
feature_group = get_feature_store().get_or_create_feature_group(
    name=config.FEATURE_GROUP_MODEL_PREDICTION,
    version=1,
    description="Blue Bikes Boston - Hourly Demand Predictions from LightGBM Model",
    primary_key=["pickup_location_id", "pickup_hour"],
    event_time="pickup_hour",
)

feature_group.insert(predictions, write_options={"wait_for_job": False})

print(f"✓ Predictions saved successfully!")
print(f"✓ Pipeline completed at {pd.Timestamp.now(tz='Etc/UTC')}")
