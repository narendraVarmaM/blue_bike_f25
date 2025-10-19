from datetime import timedelta
from typing import Optional

import pandas as pd
import plotly.express as px



def plot_aggregated_time_series(
    features: pd.DataFrame,
    targets: pd.Series,
    row_id: int,
    predictions: Optional[pd.Series] = None,
):
    """
    Plots the time series data for a specific location from NYC taxi data.

    Args:
        features (pd.DataFrame): DataFrame containing feature data, including historical ride counts and metadata.
        targets (pd.Series): Series containing the target values (e.g., actual ride counts).
        row_id (int): Index of the row to plot.
        predictions (Optional[pd.Series]): Series containing predicted values (optional).

    Returns:
        plotly.graph_objects.Figure: A Plotly figure object showing the time series plot.
    """
    # Extract the specific location's features and target
    location_features = features.iloc[row_id]
    actual_target = targets.iloc[row_id]

    # Identify time series columns (e.g., historical ride counts)
    time_series_columns = [
        col for col in features.columns if col.startswith("rides_t-")
    ]
    time_series_values = [location_features[col] for col in time_series_columns] + [
        actual_target
    ]

    # Generate corresponding timestamps for the time series
    time_series_dates = pd.date_range(
        start=location_features["hour"]
        - timedelta(hours=4*len(time_series_columns)),
        end=location_features["hour"],
        freq="4h",
    )

    # Create the plot title with relevant metadata
    title = f"Pickup Hour: {location_features['hour']}, Location ID: {location_features['start_station_id']}"

    # Create the base line plot
    fig = px.line(
        x=time_series_dates,
        y=time_series_values,
        template="plotly_white",
        markers=True,
        title=title,
        labels={"x": "Time", "y": "Ride Counts"},
    )

    # Add the actual target value as a green marker
    fig.add_scatter(
        x=time_series_dates[-1:],  # Last timestamp
        y=[actual_target],  # Actual target value
        line_color="green",
        mode="markers",
        marker_size=10,
        name="Actual Value",
    )

    # Optionally add the prediction as a red marker
    if predictions is not None:
        predicted_value = predictions[row_id]
        fig.add_scatter(
            x=time_series_dates[-1:],  # Last timestamp
            y=[predicted_value],  # Predicted value
            line_color="red",
            mode="markers",
            marker_symbol="x",
            marker_size=15,
            name="Prediction",
        )

    # if predictions is not None:
    #     fig.add_scatter(
    #         x=time_series_dates[-1:],  # Last timestamp
    #         y=predictions[
    #             predictions["pickup_location_id" == row_id]
    #         ],  # Predicted value
    #         line_color="red",
    #         mode="markers",
    #         marker_symbol="x",
    #         marker_size=15,
    #         name="Prediction",
    #     )

    return fig
