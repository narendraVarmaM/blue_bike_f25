"""
Blue Bikes Boston - Model Performance Monitor

Dashboard to monitor model accuracy over time by comparing:
- Predicted demand vs actual rides
- Mean Absolute Error (MAE) by hour
- Performance trends

Helps identify:
- When model needs retraining
- Time periods with poor predictions
- Overall model health
"""

import sys
from pathlib import Path

parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.bluebikes_inference import fetch_hourly_rides, fetch_predictions
from src.bluebikes_station_names import add_station_names_to_dataframe

# Configure page
st.set_page_config(
    page_title="Blue Bikes Model Monitor",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Blue Bikes Boston - Model Performance Monitor")
st.markdown("Monitor prediction accuracy and identify when model retraining is needed")

# Sidebar for settings
st.sidebar.header("⚙️ Settings")
past_hours = st.sidebar.slider(
    "Number of Past Hours to Analyze",
    min_value=12,
    max_value=24 * 28,  # 28 days max
    value=24 * 7,  # Default: 1 week
    step=12,
    help="Select how many hours of historical data to analyze"
)

# Display info
st.sidebar.info(f"Analyzing last **{past_hours} hours** ({past_hours // 24} days)")

# Fetch data
with st.spinner(f"📥 Fetching data for the past {past_hours} hours..."):
    try:
        # Fetch actual rides
        df_actual = fetch_hourly_rides(past_hours)
        df_actual = add_station_names_to_dataframe(df_actual)
        st.sidebar.write(f"✓ Loaded {len(df_actual):,} actual ride records")

        # Fetch predictions
        df_predictions = fetch_predictions(past_hours)
        df_predictions = add_station_names_to_dataframe(df_predictions)
        st.sidebar.write(f"✓ Loaded {len(df_predictions):,} prediction records")

    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()

# Merge the DataFrames
with st.spinner("🔄 Processing data..."):
    merged_df = pd.merge(
        df_actual,
        df_predictions,
        on=["pickup_location_id", "pickup_hour"],
        how="inner",
        suffixes=('_actual', '_pred')
    )

    if merged_df.empty:
        st.error("No matching data found between predictions and actual rides. Please check the time range.")
        st.stop()

    # Keep station_name from either source (they should be the same)
    if 'station_name_actual' in merged_df.columns:
        merged_df['station_name'] = merged_df['station_name_actual']
        merged_df = merged_df.drop(columns=['station_name_actual', 'station_name_pred'], errors='ignore')
    elif 'station_name_pred' in merged_df.columns:
        merged_df['station_name'] = merged_df['station_name_pred']
        merged_df = merged_df.drop(columns=['station_name_pred'], errors='ignore')

    # Calculate error metrics
    merged_df["absolute_error"] = abs(merged_df["predicted_demand"] - merged_df["rides"])
    merged_df["squared_error"] = (merged_df["predicted_demand"] - merged_df["rides"]) ** 2
    merged_df["percentage_error"] = (merged_df["absolute_error"] / merged_df["rides"].replace(0, 1)) * 100

# Overall metrics
st.header("📈 Overall Performance Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    mae = merged_df["absolute_error"].mean()
    st.metric("Mean Absolute Error (MAE)", f"{mae:.2f}", help="Average prediction error in rides")

with col2:
    rmse = (merged_df["squared_error"].mean() ** 0.5)
    st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}", help="Penalizes larger errors more")

with col3:
    mape = merged_df["percentage_error"].mean()
    st.metric("Mean Abs. % Error (MAPE)", f"{mape:.1f}%", help="Error as percentage of actual")

with col4:
    r2 = 1 - (merged_df["squared_error"].sum() / 
              ((merged_df["rides"] - merged_df["rides"].mean()) ** 2).sum())
    st.metric("R² Score", f"{r2:.3f}", help="How well predictions match actuals (1.0 = perfect)")

# MAE over time
st.header("📊 MAE by Pickup Hour")

mae_by_hour = merged_df.groupby("pickup_hour")["absolute_error"].mean().reset_index()
mae_by_hour.rename(columns={"absolute_error": "MAE"}, inplace=True)

fig_mae = px.line(
    mae_by_hour,
    x="pickup_hour",
    y="MAE",
    title=f"Mean Absolute Error Over Time ({past_hours} hours)",
    labels={"pickup_hour": "Pickup Hour", "MAE": "Mean Absolute Error (rides)"},
    markers=True,
)

fig_mae.add_hline(
    y=mae_by_hour["MAE"].mean(),
    line_dash="dash",
    line_color="red",
    annotation_text=f"Average MAE: {mae_by_hour['MAE'].mean():.2f}",
    annotation_position="top right"
)

fig_mae.update_layout(
    hovermode='x unified',
    template='plotly_white',
    height=400
)

st.plotly_chart(fig_mae, use_container_width=True)

# Additional visualizations
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 Predictions vs Actuals")
    
    # Sample data for scatter plot (all points would be too many)
    sample_size = min(1000, len(merged_df))
    sample_df = merged_df.sample(n=sample_size)

    # Try to create scatter plot with trendline, fallback without if statsmodels unavailable
    try:
        fig_scatter = px.scatter(
            sample_df,
            x="rides",
            y="predicted_demand",
            title=f"Predicted vs Actual Rides (sample of {sample_size})",
            labels={"rides": "Actual Rides", "predicted_demand": "Predicted Rides"},
            opacity=0.5,
            trendline="ols"
        )
    except ImportError:
        # Fallback without trendline if statsmodels is not available
        fig_scatter = px.scatter(
            sample_df,
            x="rides",
            y="predicted_demand",
            title=f"Predicted vs Actual Rides (sample of {sample_size})",
            labels={"rides": "Actual Rides", "predicted_demand": "Predicted Rides"},
            opacity=0.5
        )
    
    # Add perfect prediction line
    max_val = max(sample_df["rides"].max(), sample_df["predicted_demand"].max())
    fig_scatter.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode='lines',
            name='Perfect Predictions',
            line=dict(color='red', dash='dash')
        )
    )
    
    fig_scatter.update_layout(template='plotly_white', height=400)
    st.plotly_chart(fig_scatter, use_container_width=True)

with col2:
    st.subheader("📉 Error Distribution")
    
    fig_hist = px.histogram(
        merged_df,
        x="absolute_error",
        nbins=50,
        title="Distribution of Absolute Errors",
        labels={"absolute_error": "Absolute Error (rides)"},
    )
    
    fig_hist.add_vline(
        x=mae,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mae:.2f}",
    )
    
    fig_hist.update_layout(template='plotly_white', height=400)
    st.plotly_chart(fig_hist, use_container_width=True)

# Performance by station
st.header("🚉 Station-Level Performance")

station_performance = merged_df.groupby(["pickup_location_id", "station_name"]).agg({
    "absolute_error": "mean",
    "rides": "sum",
    "predicted_demand": "sum"
}).reset_index()

station_performance.columns = ["Station ID", "Station Name", "Avg MAE", "Total Actual", "Total Predicted"]
station_performance["Error %"] = (station_performance["Avg MAE"] / 
                                   station_performance["Total Actual"].replace(0, 1) * 100)

# Show worst performing stations
st.subheader("⚠️ Stations with Highest MAE")
worst_stations = station_performance.nlargest(10, "Avg MAE")
st.dataframe(
    worst_stations[["Station Name", "Avg MAE", "Total Actual", "Total Predicted", "Error %"]].style.format({
        "Avg MAE": "{:.2f}",
        "Total Actual": "{:.0f}",
        "Total Predicted": "{:.0f}",
        "Error %": "{:.1f}%"
    }),
    use_container_width=True
)

# Show best performing stations
st.subheader("✅ Stations with Lowest MAE")
best_stations = station_performance.nsmallest(10, "Avg MAE")
st.dataframe(
    best_stations[["Station Name", "Avg MAE", "Total Actual", "Total Predicted", "Error %"]].style.format({
        "Avg MAE": "{:.2f}",
        "Total Actual": "{:.0f}",
        "Total Predicted": "{:.0f}",
        "Error %": "{:.1f}%"
    }),
    use_container_width=True
)

# Time of day analysis
st.header("🕐 Performance by Hour of Day")

merged_df['hour_of_day'] = pd.to_datetime(merged_df['pickup_hour']).dt.hour

hourly_performance = merged_df.groupby('hour_of_day')['absolute_error'].mean().reset_index()
hourly_performance.columns = ['Hour', 'MAE']

fig_hourly = px.bar(
    hourly_performance,
    x='Hour',
    y='MAE',
    title='Average MAE by Hour of Day',
    labels={'Hour': 'Hour of Day (0-23)', 'MAE': 'Mean Absolute Error'},
)

fig_hourly.update_layout(template='plotly_white', height=400)
st.plotly_chart(fig_hourly, use_container_width=True)

# Model health indicator
st.header("🏥 Model Health Status")

# Define thresholds (adjust based on your requirements)
MAE_THRESHOLD_GOOD = 2.0
MAE_THRESHOLD_WARNING = 4.0

if mae < MAE_THRESHOLD_GOOD:
    st.success(f"✅ Model is performing well! MAE: {mae:.2f}")
elif mae < MAE_THRESHOLD_WARNING:
    st.warning(f"⚠️ Model performance is acceptable but could be improved. MAE: {mae:.2f}")
else:
    st.error(f"❌ Model performance is poor. Consider retraining! MAE: {mae:.2f}")

# Recommendations
st.subheader("💡 Recommendations")

recommendations = []

if mae > MAE_THRESHOLD_WARNING:
    recommendations.append("🔴 **High MAE detected** - Model retraining recommended")

if mape > 30:
    recommendations.append("🔴 **High percentage error** - Check for systematic bias")

if r2 < 0.7:
    recommendations.append("🟡 **Low R² score** - Consider adding more features or tuning hyperparameters")

# Check for stations with consistently high errors
high_error_stations = (station_performance["Avg MAE"] > mae * 1.5).sum()
if high_error_stations > len(station_performance) * 0.2:
    recommendations.append(f"🟡 **{high_error_stations} stations** have significantly higher errors - Investigate specific station patterns")

# Check time-based patterns
if hourly_performance['MAE'].std() > mae * 0.3:
    recommendations.append("🟡 **Large variation in MAE by hour** - Consider adding more time-based features")

if not recommendations:
    st.info("✅ No major issues detected. Model is performing as expected.")
else:
    for rec in recommendations:
        st.markdown(rec)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    """
    This dashboard monitors the Blue Bikes prediction model performance by comparing 
    predictions against actual ride data.
    
    **Key Metrics:**
    - **MAE**: Average prediction error
    - **RMSE**: Error with penalty for large deviations
    - **MAPE**: Error as percentage
    - **R²**: Goodness of fit (0-1)
    """
)

refresh = st.sidebar.button("🔄 Refresh Data")
if refresh:
    st.rerun()
