"""
Blue Bikes Boston - Interactive Dashboard v2

Real-time dashboard showing:
- Interactive map of predicted demand by station
- Statistics and metrics
- Top stations visualization
- Time series plots for selected stations
"""

import sys
from pathlib import Path
import pytz

parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)

import zipfile

import folium
import pandas as pd
import requests
import streamlit as st
from branca.colormap import LinearColormap
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

from src.bluebikes_config import DATA_DIR
from src.bluebikes_inference import fetch_next_hour_predictions, load_batch_of_features_from_store
from src.bluebikes_plot_utils import plot_prediction

# Initialize session state for the map
if "map_created" not in st.session_state:
    st.session_state.map_created = False


def create_bluebikes_map(prediction_data):
    """
    Create an interactive map of Blue Bikes Boston with predicted demand.
    Uses markers with color coding based on demand.
    """
    # Create base map centered on Boston
    m = folium.Map(
        location=[42.3601, -71.0589],  # Boston center
        zoom_start=13,
        tiles="cartodbpositron"
    )
    
    # Create color map based on predicted demand
    min_demand = prediction_data['predicted_demand'].min()
    max_demand = prediction_data['predicted_demand'].max()
    
    colormap = LinearColormap(
        colors=['#FFEDA0', '#FED976', '#FEB24C', '#FD8D3C', '#FC4E2A', '#E31A1C', '#BD0026'],
        vmin=min_demand,
        vmax=max_demand,
        caption='Predicted Demand (rides/hour)'
    )
    
    colormap.add_to(m)
    
    # If we have station coordinates, use them
    if 'lat_rounded' in prediction_data.columns and 'lon_rounded' in prediction_data.columns:
        # Use actual station locations
        for idx, row in prediction_data.iterrows():
            demand = row['predicted_demand']
            color = colormap(demand)
            
            folium.CircleMarker(
                location=[row['lat_rounded'], row['lon_rounded']],
                radius=8 + (demand / max_demand * 10),  # Size based on demand
                popup=f"Station {row['pickup_location_id']}<br>Demand: {demand:.0f}",
                tooltip=f"{row['pickup_location_id']}: {demand:.0f} rides",
                color='black',
                weight=1,
                fill=True,
                fillColor=color,
                fillOpacity=0.7
            ).add_to(m)
    else:
        # Fallback: Create a general Boston area visualization
        # This creates a simple heatmap-style visualization
        st.warning("Station coordinates not available. Using simplified visualization.")
        
        # Create a marker cluster
        marker_cluster = MarkerCluster().add_to(m)
        
        # Add markers for top stations at random Boston locations
        # (In production, you'd load actual station coordinates)
        import numpy as np
        for idx, row in prediction_data.nlargest(50, 'predicted_demand').iterrows():
            # Random location in Boston area for demo
            lat = 42.3601 + np.random.uniform(-0.05, 0.05)
            lon = -71.0589 + np.random.uniform(-0.05, 0.05)
            
            folium.Marker(
                location=[lat, lon],
                popup=f"Station {row['pickup_location_id']}<br>Demand: {row['predicted_demand']:.0f}",
                icon=folium.Icon(color='red' if row['predicted_demand'] > prediction_data['predicted_demand'].median() else 'blue')
            ).add_to(marker_cluster)
    
    # Store the map in session state
    st.session_state.map_obj = m
    st.session_state.map_created = True
    return m


# Configure page
st.set_page_config(page_title="Blue Bikes Boston Demand", layout="wide")

# Get current time in Boston timezone
boston_tz = pytz.timezone("America/New_York")
current_date = pd.Timestamp.now(tz="UTC").tz_convert(boston_tz)

# Header
st.title("🚴 Blue Bikes Boston - Demand Prediction Dashboard")
st.header(f'{current_date.strftime("%Y-%m-%d %H:%M:%S %Z")}')

# Sidebar for progress
st.sidebar.header("⚙️ Working Progress")
progress_bar = st.sidebar.progress(0)
N_STEPS = 3

# Step 1: Fetch features
with st.spinner(text="📊 Fetching inference data..."):
    try:
        features = load_batch_of_features_from_store(current_date)
        st.sidebar.write("✓ Features loaded from feature store")
        progress_bar.progress(1 / N_STEPS)
    except Exception as e:
        st.error(f"Error loading features: {e}")
        st.stop()

# Step 2: Fetch predictions
with st.spinner(text="🤖 Loading predictions..."):
    try:
        predictions = fetch_next_hour_predictions()
        st.sidebar.write("✓ Predictions loaded")
        progress_bar.progress(2 / N_STEPS)
    except Exception as e:
        st.error(f"Error loading predictions: {e}")
        st.stop()

# # Step 3: Create visualization
# with st.spinner(text="🗺️ Creating interactive map..."):
#     st.subheader("Predicted Demand by Station")
#     map_obj = create_bluebikes_map(predictions)
    
#     # Display the map
#     if st.session_state.map_created:
#         st_folium(st.session_state.map_obj, width=1200, height=600, returned_objects=[])
    
#     st.sidebar.write("✓ Map created")
#     progress_bar.progress(3 / N_STEPS)

# Display statistics
st.subheader("📈 Prediction Statistics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Stations",
        f"{len(predictions):,}",
    )

with col2:
    st.metric(
        "Average Demand",
        f"{predictions['predicted_demand'].mean():.1f}",
    )

with col3:
    st.metric(
        "Maximum Demand",
        f"{predictions['predicted_demand'].max():.0f}",
    )

with col4:
    st.metric(
        "Total Predicted Rides",
        f"{predictions['predicted_demand'].sum():.0f}",
    )

# Top stations table
st.subheader("🔝 Top 10 Stations by Predicted Demand")
top10_df = predictions.sort_values("predicted_demand", ascending=False).head(10)
top10_df_display = top10_df[['pickup_location_id', 'predicted_demand']].copy()
top10_df_display.columns = ['Station ID', 'Predicted Demand']
top10_df_display['Predicted Demand'] = top10_df_display['Predicted Demand'].round(0).astype(int)
st.dataframe(top10_df_display, use_container_width=True)

# Time series plots for top stations
st.subheader("📊 Time Series - Top 10 Stations")

top10_ids = top10_df['pickup_location_id'].tolist()

# Create tabs for each station
tabs = st.tabs([f"Station {sid}" for sid in top10_ids])

for tab, location_id in zip(tabs, top10_ids):
    with tab:
        try:
            station_features = features[features["pickup_location_id"] == location_id]
            station_prediction = predictions[predictions["pickup_location_id"] == location_id]
            
            if not station_features.empty and not station_prediction.empty:
                fig = plot_prediction(
                    features=station_features,
                    prediction=station_prediction,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"No data available for station {location_id}")
        except Exception as e:
            st.error(f"Error plotting station {location_id}: {e}")

# Station selector
st.subheader("🔍 Search Specific Station")
all_station_ids = sorted(predictions['pickup_location_id'].unique())
selected_station = st.selectbox(
    "Select a station to view its prediction:",
    ["-- Select Station --"] + all_station_ids
)

if selected_station != "-- Select Station --":
    try:
        station_features = features[features["pickup_location_id"] == selected_station]
        station_prediction = predictions[predictions["pickup_location_id"] == selected_station]
        
        if not station_features.empty and not station_prediction.empty:
            st.write(f"**Predicted demand:** {station_prediction['predicted_demand'].iloc[0]:.0f} rides")
            
            fig = plot_prediction(
                features=station_features,
                prediction=station_prediction,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"No data available for station {selected_station}")
    except Exception as e:
        st.error(f"Error: {e}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    """
    This dashboard shows real-time Blue Bikes Boston demand predictions 
    using machine learning with weather integration.
    
    **Features:**
    - 28-day historical patterns
    - Weather data integration
    - Station-specific predictions
    - 2-month time shift for real-time simulation
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"Last updated: {current_date.strftime('%H:%M:%S')}")
