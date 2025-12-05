import lightgbm as lgb
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
import numpy as np


# Function to calculate the average rides over the last 4 weeks
def average_rides_last_4_weeks(X: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate average rides over the last 4 weeks for Blue Bikes.
    
    Args:
        X (pd.DataFrame): Feature dataframe
    
    Returns:
        pd.DataFrame: DataFrame with new average feature
    """
    last_4_weeks_columns = [
        f"rides_t-{7*24}",  # 1 week ago
        f"rides_t-{14*24}",  # 2 weeks ago
        f"rides_t-{21*24}",  # 3 weeks ago
        f"rides_t-{28*24}",  # 4 weeks ago
    ]
    
    # Ensure the required columns exist in the DataFrame
    for col in last_4_weeks_columns:
        if col not in X.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Calculate the average of the last 4 weeks
    X["average_rides_last_4_weeks"] = X[last_4_weeks_columns].mean(axis=1)
    
    return X


# Function to add weather-based features
def add_weather_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Add weather-based features if available.
    
    Args:
        X (pd.DataFrame): Feature dataframe
    
    Returns:
        pd.DataFrame: DataFrame with weather features
    """
    # Check if weather features exist
    weather_cols = ['temperature', 'precipitation', 'wind_speed', 'humidity']
    
    for col in weather_cols:
        if col not in X.columns:
            # If weather data not available, create dummy features
            X[col] = 0
    
    # Create derived weather features
    if 'temperature' in X.columns:
        # Temperature bins for bike riding comfort
        X['temp_comfort'] = pd.cut(
            X['temperature'],
            bins=[-np.inf, 0, 10, 20, 30, np.inf],
            labels=[0, 1, 2, 3, 4]
        ).astype(int)
    
    if 'precipitation' in X.columns:
        # Binary feature for rain
        X['is_raining'] = (X['precipitation'] > 0).astype(int)
    
    if 'wind_speed' in X.columns:
        # Binary feature for high wind
        X['high_wind'] = (X['wind_speed'] > 15).astype(int)
    
    return X


# FunctionTransformer to add the average rides feature
add_feature_average_rides_last_4_weeks = FunctionTransformer(
    average_rides_last_4_weeks, validate=False
)

# FunctionTransformer to add weather features
# add_feature_weather = FunctionTransformer(
#     add_weather_features, validate=False
# )


# Custom transformer to add temporal features
class TemporalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Add temporal features for Blue Bikes demand prediction.
    """
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_ = X.copy()
        
        # Extract temporal features from pickup_hour
        X_["hour"] = X_["pickup_hour"].dt.hour
        X_["day_of_week"] = X_["pickup_hour"].dt.dayofweek
        X_["month"] = X_["pickup_hour"].dt.month
        X_["is_weekend"] = (X_["day_of_week"] >= 5).astype(int)
        
        # Rush hour features (7-9 AM and 5-7 PM)
        X_["is_morning_rush"] = X_["hour"].isin([7, 8, 9]).astype(int)
        X_["is_evening_rush"] = X_["hour"].isin([17, 18, 19]).astype(int)
        
        # Day part features
        X_["is_night"] = X_["hour"].isin(range(0, 6)).astype(int)
        X_["is_morning"] = X_["hour"].isin(range(6, 12)).astype(int)
        X_["is_afternoon"] = X_["hour"].isin(range(12, 18)).astype(int)
        X_["is_evening"] = X_["hour"].isin(range(18, 24)).astype(int)
        
        # Season features (Boston seasons)
        # Winter: Dec-Feb (12, 1, 2)
        # Spring: Mar-May (3, 4, 5)
        # Summer: Jun-Aug (6, 7, 8)
        # Fall: Sep-Nov (9, 10, 11)
        X_["is_winter"] = X_["month"].isin([12, 1, 2]).astype(int)
        X_["is_spring"] = X_["month"].isin([3, 4, 5]).astype(int)
        X_["is_summer"] = X_["month"].isin([6, 7, 8]).astype(int)
        X_["is_fall"] = X_["month"].isin([9, 10, 11]).astype(int)
        
        # Drop original datetime and location ID columns
        columns_to_drop = ["pickup_hour", "pickup_location_id"]
        X_ = X_.drop(columns=[col for col in columns_to_drop if col in X_.columns])
        
        return X_


# Instantiate the temporal feature engineer
add_temporal_features = TemporalFeatureEngineer()


# Custom transformer for station-specific features
class StationFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Add station-specific features.
    """
    
    def __init__(self):
        self.station_avg = {}
    
    def fit(self, X, y=None):
        # Calculate average rides per station during training
        if 'pickup_location_id' in X.columns and y is not None:
            station_rides = pd.DataFrame({
                'station': X['pickup_location_id'],
                'rides': y
            })
            self.station_avg = station_rides.groupby('station')['rides'].mean().to_dict()
        return self
    
    def transform(self, X, y=None):
        X_ = X.copy()
        
        # Add station average feature if available
        if 'pickup_location_id' in X_.columns and self.station_avg:
            X_['station_avg_rides'] = X_['pickup_location_id'].map(self.station_avg).fillna(0)
        
        return X_


# Function to return the pipeline
def get_pipeline(**hyper_params):
    """
    Returns a pipeline with feature engineering and LGBMRegressor for Blue Bikes.
    
    Parameters:
    ----------
    **hyper_params : dict
        Optional parameters to pass to the LGBMRegressor.
    
    Returns:
    -------
    pipeline : sklearn.pipeline.Pipeline
        A pipeline with feature engineering and LGBMRegressor.
    """
    # Default hyperparameters optimized for Blue Bikes
    default_params = {
        'n_estimators': 100,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
    }
    
    # Update with user-provided hyperparameters
    default_params.update(hyper_params)
    
    pipeline = make_pipeline(
        add_feature_average_rides_last_4_weeks,
        add_temporal_features,
        lgb.LGBMRegressor(**default_params),
    )
    
    return pipeline


# Alternative pipeline with weather features
def get_pipeline_with_weather(**hyper_params):
    """
    Returns a pipeline with weather features included.
    
    Parameters:
    ----------
    **hyper_params : dict
        Optional parameters to pass to the LGBMRegressor.
    
    Returns:
    -------
    pipeline : sklearn.pipeline.Pipeline
        A pipeline with weather and temporal feature engineering.
    """
    default_params = {
        'n_estimators': 100,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1,
    }
    
    default_params.update(hyper_params)
    
    pipeline = make_pipeline(
        add_feature_average_rides_last_4_weeks,
        add_temporal_features,
        lgb.LGBMRegressor(**default_params),
    )
    
    return pipeline




# Make sure numpy is imported for the weather features function
