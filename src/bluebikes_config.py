import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Define directories
PARENT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PARENT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRANSFORMED_DATA_DIR = DATA_DIR / "transformed"
MODELS_DIR = PARENT_DIR / "models"
WEATHER_DATA_DIR = DATA_DIR / "weather"

# Create directories if they don't exist
for directory in [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    TRANSFORMED_DATA_DIR,
    MODELS_DIR,
    WEATHER_DATA_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# Hopsworks Configuration
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")

# Feature Store Configuration
FEATURE_GROUP_NAME = "bluebikes_hourly_feature_group"
FEATURE_GROUP_VERSION = 1

FEATURE_VIEW_NAME = "bluebikes_feature_view_hourly"
FEATURE_VIEW_VERSION = 1

# Model Configuration
MODEL_NAME = "bluebikes_demand_predictor_next_hour"
MODEL_VERSION = 1

FEATURE_GROUP_MODEL_PREDICTION = "bluebikes_hourly_model_prediction"

# Weather API Configuration
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")  # OpenWeatherMap or similar
WEATHER_API_URL = "https://api.openweathermap.org/data/2.5"

# Time shift configuration (2 months = 8 weeks for production simulation)
TIME_SHIFT_WEEKS = 8

# Boston coordinates for weather data
BOSTON_LAT = 42.3601
BOSTON_LON = -71.0589

# Blue Bikes data source
BLUEBIKES_DATA_URL = "https://s3.amazonaws.com/hubway-data"
