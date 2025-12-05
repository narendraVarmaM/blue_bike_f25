from datetime import timedelta
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_aggregated_time_series(
    features: pd.DataFrame,
    targets: pd.Series,
    row_id: int,
    predictions: Optional[pd.Series] = None,
):
    """
    Plots the time series data for a specific Blue Bikes station.
    
    Args:
        features (pd.DataFrame): DataFrame containing feature data
        targets (pd.Series): Series containing the target values (actual ride counts)
        row_id (int): Index of the row to plot
        predictions (Optional[pd.Series]): Series containing predicted values
    
    Returns:
        plotly.graph_objects.Figure: A Plotly figure object showing the time series plot
    """
    # Extract the specific station's features and target
    location_features = features.iloc[row_id]
    actual_target = targets.iloc[row_id]
    
    # Identify time series columns (historical ride counts)
    time_series_columns = [
        col for col in features.columns if col.startswith("rides_t-")
    ]
    time_series_values = [location_features[col] for col in time_series_columns] + [
        actual_target
    ]
    
    # Generate corresponding timestamps for the time series
    time_series_dates = pd.date_range(
        start=location_features["pickup_hour"]
        - timedelta(hours=len(time_series_columns)),
        end=location_features["pickup_hour"],
        freq="h",
    )
    
    # Create the plot title with relevant metadata
    title = f"Blue Bikes - Hour: {location_features['pickup_hour']}, Station ID: {location_features['pickup_location_id']}"
    
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
        x=time_series_dates[-1:],
        y=[actual_target],
        line_color="green",
        mode="markers",
        marker_size=10,
        name="Actual Value",
    )
    
    # Optionally add the prediction as a red marker
    if predictions is not None:
        predicted_value = predictions[row_id]
        fig.add_scatter(
            x=time_series_dates[-1:],
            y=[predicted_value],
            line_color="red",
            mode="markers",
            marker_symbol="x",
            marker_size=15,
            name="Prediction",
        )
    
    return fig


def plot_prediction(features: pd.DataFrame, prediction: pd.DataFrame):
    """
    Plot predicted Blue Bikes demand.
    
    Args:
        features (pd.DataFrame): Feature data
        prediction (pd.DataFrame): Prediction data with 'predicted_demand' column
    
    Returns:
        plotly.graph_objects.Figure: Plot of predictions
    """
    # Identify time series columns
    time_series_columns = [
        col for col in features.columns if col.startswith("rides_t-")
    ]
    time_series_values = [
        features[col].iloc[0] for col in time_series_columns
    ] + prediction["predicted_demand"].to_list()
    
    # Convert pickup_hour to single timestamp
    pickup_hour = pd.Timestamp(features["pickup_hour"].iloc[0])
    
    # Generate timestamps
    time_series_dates = pd.date_range(
        start=pickup_hour - timedelta(hours=len(time_series_columns)),
        end=pickup_hour,
        freq="h",
    )
    
    # Create DataFrame for historical data
    historical_df = pd.DataFrame(
        {"datetime": time_series_dates, "rides": time_series_values}
    )
    
    # Create the plot title
    title = f"Blue Bikes - Hour: {pickup_hour}, Station ID: {features['pickup_location_id'].iloc[0]}"
    
    # Create the base line plot
    fig = px.line(
        historical_df,
        x="datetime",
        y="rides",
        template="plotly_white",
        markers=True,
        title=title,
        labels={"datetime": "Time", "rides": "Ride Counts"},
    )
    
    # Add prediction point
    fig.add_scatter(
        x=[pickup_hour],
        y=prediction["predicted_demand"].to_list(),
        line_color="red",
        mode="markers",
        marker_symbol="x",
        marker_size=10,
        name="Prediction",
    )
    
    return fig


def plot_station_comparison(
    predictions: pd.DataFrame,
    actuals: Optional[pd.DataFrame] = None,
    top_n: int = 10
):
    """
    Plot comparison of top N stations by predicted demand.
    
    Args:
        predictions (pd.DataFrame): Predictions with 'pickup_location_id' and 'predicted_demand'
        actuals (Optional[pd.DataFrame]): Actual rides if available
        top_n (int): Number of top stations to display
    
    Returns:
        plotly.graph_objects.Figure: Bar chart comparison
    """
    # Get top N stations by predicted demand
    top_stations = predictions.nlargest(top_n, 'predicted_demand')
    
    if actuals is not None:
        # Merge with actuals
        comparison = top_stations.merge(
            actuals,
            on='pickup_location_id',
            how='left',
            suffixes=('_pred', '_actual')
        )
        
        fig = go.Figure(data=[
            go.Bar(
                name='Predicted',
                x=comparison['pickup_location_id'].astype(str),
                y=comparison['predicted_demand'],
                marker_color='indianred'
            ),
            go.Bar(
                name='Actual',
                x=comparison['pickup_location_id'].astype(str),
                y=comparison['rides'],
                marker_color='lightsalmon'
            )
        ])
        
        fig.update_layout(
            title=f'Top {top_n} Blue Bikes Stations: Predicted vs Actual Demand',
            xaxis_title='Station ID',
            yaxis_title='Ride Count',
            barmode='group',
            template='plotly_white'
        )
    else:
        fig = px.bar(
            top_stations,
            x='pickup_location_id',
            y='predicted_demand',
            title=f'Top {top_n} Blue Bikes Stations by Predicted Demand',
            labels={'pickup_location_id': 'Station ID', 'predicted_demand': 'Predicted Rides'},
            template='plotly_white'
        )
    
    return fig


def plot_hourly_pattern(
    data: pd.DataFrame,
    date: Optional[str] = None
):
    """
    Plot hourly demand pattern for Blue Bikes.
    
    Args:
        data (pd.DataFrame): Data with 'pickup_hour' and 'rides' or 'predicted_demand'
        date (Optional[str]): Specific date to filter for
    
    Returns:
        plotly.graph_objects.Figure: Line plot of hourly patterns
    """
    df = data.copy()
    
    if date:
        df['date'] = pd.to_datetime(df['pickup_hour']).dt.date
        df = df[df['date'] == pd.to_datetime(date).date()]
    
    df['hour'] = pd.to_datetime(df['pickup_hour']).dt.hour
    
    # Aggregate by hour
    value_col = 'predicted_demand' if 'predicted_demand' in df.columns else 'rides'
    hourly_agg = df.groupby('hour')[value_col].sum().reset_index()
    
    fig = px.line(
        hourly_agg,
        x='hour',
        y=value_col,
        title='Blue Bikes Hourly Demand Pattern',
        labels={'hour': 'Hour of Day', value_col: 'Total Rides'},
        template='plotly_white',
        markers=True
    )
    
    # Add vertical lines for rush hours
    fig.add_vline(x=8, line_dash="dash", line_color="gray", annotation_text="Morning Rush")
    fig.add_vline(x=17, line_dash="dash", line_color="gray", annotation_text="Evening Rush")
    
    return fig


def plot_weather_impact(
    data: pd.DataFrame,
    weather_col: str = 'temperature'
):
    """
    Plot the impact of weather on Blue Bikes demand.
    
    Args:
        data (pd.DataFrame): Data with weather and ride information
        weather_col (str): Weather column to analyze
    
    Returns:
        plotly.graph_objects.Figure: Scatter plot showing weather impact
    """
    if weather_col not in data.columns:
        raise ValueError(f"Column '{weather_col}' not found in data")
    
    value_col = 'predicted_demand' if 'predicted_demand' in data.columns else 'rides'
    
    if value_col not in data.columns:
        raise ValueError(f"No ride count column found in data")
    
    fig = px.scatter(
        data,
        x=weather_col,
        y=value_col,
        title=f'Blue Bikes Demand vs {weather_col.title()}',
        labels={weather_col: weather_col.title(), value_col: 'Ride Count'},
        template='plotly_white',
        trendline='ols'
    )
    
    return fig


def plot_daily_totals(
    data: pd.DataFrame,
    days: int = 7
):
    """
    Plot daily total rides for the last N days.
    
    Args:
        data (pd.DataFrame): Data with 'pickup_hour' and ride counts
        days (int): Number of days to display
    
    Returns:
        plotly.graph_objects.Figure: Bar chart of daily totals
    """
    df = data.copy()
    df['date'] = pd.to_datetime(df['pickup_hour']).dt.date
    
    value_col = 'predicted_demand' if 'predicted_demand' in df.columns else 'rides'
    
    # Get last N days
    latest_date = df['date'].max()
    earliest_date = latest_date - timedelta(days=days)
    df = df[df['date'] > earliest_date]
    
    # Aggregate by date
    daily_totals = df.groupby('date')[value_col].sum().reset_index()
    daily_totals['date'] = pd.to_datetime(daily_totals['date'])
    
    fig = px.bar(
        daily_totals,
        x='date',
        y=value_col,
        title=f'Blue Bikes Daily Totals (Last {days} Days)',
        labels={'date': 'Date', value_col: 'Total Rides'},
        template='plotly_white'
    )
    
    return fig
