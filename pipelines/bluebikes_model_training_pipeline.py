"""
Blue Bikes Boston - Model Training Pipeline

This script:
1. Fetches historical data from the feature store
2. Transforms into features and targets
3. Trains a new LightGBM model
4. Evaluates against the current production model
5. Registers new model if it performs better

Can be run manually or scheduled weekly/monthly.
"""

import joblib
from hsml.model_schema import ModelSchema
from hsml.schema import Schema
from sklearn.metrics import mean_absolute_error

import src.bluebikes_config as config
from src.bluebikes_data_utils import transform_ts_data_into_features
from src.bluebikes_inference import (
    fetch_days_data,
    get_hopsworks_project,
    load_metrics_from_registry,
)
from src.bluebikes_pipeline_utils import get_pipeline_with_weather

print("=" * 70)
print("BLUE BIKES BOSTON - MODEL TRAINING PIPELINE")
print("=" * 70)

# Fetch historical data from feature store
print(f"\n[1/5] Fetching data from feature store...")
print(f"Looking back: 180 days (with 2-month time shift)")
ts_data = fetch_days_data(180)

print(f"Data fetched with {len(ts_data)} records")

print(f"Loaded {len(ts_data):,} records")
print(f"Date range: {ts_data['pickup_hour'].min()} to {ts_data['pickup_hour'].max()}")
print(f"Unique stations: {ts_data['pickup_location_id'].nunique()}")

# Transform to features and targets
print(f"\n[2/5] Transforming time series data...")
print(f"Creating 28-day (672 hours) sliding windows")

features, targets = transform_ts_data_into_features(
    ts_data, window_size=24 * 28, step_size=23
)

print(f"Generated {len(features):,} training samples")
print(f"Features shape: {features.shape}")
print(f"Targets shape: {targets.shape}")

# Create pipeline with weather features
print(f"\n[3/5] Training LightGBM model with weather features...")

pipeline = get_pipeline_with_weather(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
)

print("Training in progress...")
pipeline.fit(features, targets)
print("✓ Training completed!")

# Make predictions on training data
print(f"\n[4/5] Evaluating model performance...")
predictions = pipeline.predict(features)

# Calculate metrics
train_mae = mean_absolute_error(targets, predictions)
print(f"\nNew model MAE: {train_mae:.4f}")

# Compare with current production model
try:
    current_metrics = load_metrics_from_registry()
    current_mae = current_metrics.get('test_mae', float('inf'))
    print(f"Current production model MAE: {current_mae:.4f}")
    
    improvement = ((current_mae - train_mae) / current_mae) * 100 if current_mae > 0 else 0
    print(f"Improvement: {improvement:.2f}%")
except Exception as e:
    print(f"No existing model found or error loading metrics: {e}")
    current_mae = float('inf')
    improvement = 0

# Register new model if it's better
print(f"\n[5/5] Model registration decision...")

if train_mae < current_mae:
    print(f"✓ New model performs better! Registering to Hopsworks...")
    
    # Save model locally first
    model_path = config.MODELS_DIR / "lgb_model.pkl"
    joblib.dump(pipeline, model_path)
    print(f"  Saved model to: {model_path}")
    
    # Create model schema
    input_schema = Schema(features)
    output_schema = Schema(targets)
    model_schema = ModelSchema(input_schema=input_schema, output_schema=output_schema)
    
    # Connect to Hopsworks
    project = get_hopsworks_project()
    model_registry = project.get_model_registry()
    
    # Register model
    model = model_registry.sklearn.create_model(
        name=config.MODEL_NAME,
        metrics={"test_mae": train_mae},
        description=f"Blue Bikes Boston demand predictor - MAE: {train_mae:.4f}",
        input_example=features.sample(),
        model_schema=model_schema,
    )
    model.save(str(model_path))
    
    print(f"✓ Model registered successfully!")
    print(f"  Model name: {config.MODEL_NAME}")
    print(f"  MAE: {train_mae:.4f}")
    print(f"  Improvement over previous: {improvement:.2f}%")
    
else:
    print(f"✗ New model does not perform better than current model")
    print(f"  Current MAE: {current_mae:.4f}")
    print(f"  New MAE: {train_mae:.4f}")
    print(f"  Skipping model registration")

print("\n" + "=" * 70)
print("PIPELINE COMPLETED")
print("=" * 70)

# Summary statistics
print(f"\nModel Performance Summary:")
print(f"  Mean Absolute Error: {train_mae:.4f} rides/hour")
print(f"  Target mean: {targets.mean():.2f} rides/hour")
print(f"  Target std: {targets.std():.2f} rides/hour")
print(f"  MAE as % of mean: {(train_mae / targets.mean() * 100):.2f}%")
print(f"\nTraining Data:")
print(f"  Samples: {len(features):,}")
print(f"  Stations: {features['pickup_location_id'].nunique()}")
print(f"  Features: {features.shape[1]}")
