#!/usr/bin/env python3
"""
Blue Bikes Boston - MCP Server

This MCP (Model Context Protocol) server provides tools to fetch and query
Blue Bikes prediction data from Hopsworks feature store.

Available Tools:
- get_next_hour_predictions: Get predictions for the next hour
- fetch_predictions: Fetch historical predictions
- fetch_actual_rides: Fetch actual ride data
- get_model_metrics: Get model performance metrics
- get_station_data: Get data for specific station
"""

import sys
import os
import logging
from pathlib import Path

# Suppress logging from dependencies to avoid interfering with MCP JSON communication
logging.basicConfig(level=logging.ERROR)
for logger_name in ['hopsworks', 'hsfs', 'urllib3', 'requests']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)

import io

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Optional

import pandas as pd
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from src.bluebikes_inference import (
    fetch_hourly_rides,
    fetch_next_hour_predictions,
    fetch_predictions,
    get_feature_store,
    load_model_from_registry,
)

# Initialize MCP server
app = Server("bluebikes-hopsworks")


# Context manager to suppress stderr during tool calls (keep stdout for MCP communication)
class SuppressOutput:
    def __enter__(self):
        self._stderr = sys.stderr
        sys.stderr = io.StringIO()  # Only suppress stderr (logs), keep stdout for MCP
        return self

    def __exit__(self, *args):
        sys.stderr = self._stderr


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List all available tools in the MCP server."""
    return [
        Tool(
            name="get_next_hour_predictions",
            description=(
                "Fetch Blue Bikes demand predictions for the next hour across all stations. "
                "Returns station IDs and their predicted demand values."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="fetch_predictions",
            description=(
                "Fetch historical Blue Bikes predictions for the specified number of past hours. "
                "Useful for analyzing prediction trends and model performance over time."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "hours": {
                        "type": "integer",
                        "description": "Number of past hours to fetch predictions for (1-672, default: 24)",
                        "default": 24,
                        "minimum": 1,
                        "maximum": 672,
                    }
                },
                "required": [],
            },
        ),
        Tool(
            name="fetch_actual_rides",
            description=(
                "Fetch actual Blue Bikes ride data for the specified number of past hours. "
                "Returns real ride counts by station and hour for comparison with predictions."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "hours": {
                        "type": "integer",
                        "description": "Number of past hours to fetch actual ride data for (1-672, default: 24)",
                        "default": 24,
                        "minimum": 1,
                        "maximum": 672,
                    }
                },
                "required": [],
            },
        ),
        Tool(
            name="get_model_metrics",
            description=(
                "Calculate and return model performance metrics by comparing predictions vs actual rides. "
                "Includes MAE, RMSE, MAPE, and R² score for the specified time period."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "hours": {
                        "type": "integer",
                        "description": "Number of past hours to analyze (1-672, default: 168 = 1 week)",
                        "default": 168,
                        "minimum": 1,
                        "maximum": 672,
                    }
                },
                "required": [],
            },
        ),
        Tool(
            name="get_station_data",
            description=(
                "Get detailed data for a specific Blue Bikes station, including predictions, "
                "actual rides, and performance metrics."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "station_id": {
                        "type": "string",
                        "description": "Station ID (e.g., 'M32006', 'M32011')",
                    },
                    "hours": {
                        "type": "integer",
                        "description": "Number of past hours to fetch (1-672, default: 24)",
                        "default": 24,
                        "minimum": 1,
                        "maximum": 672,
                    },
                },
                "required": ["station_id"],
            },
        ),
        Tool(
            name="get_top_stations",
            description=(
                "Get the top N stations ranked by predicted demand for the next hour. "
                "Useful for identifying high-demand locations."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "n": {
                        "type": "integer",
                        "description": "Number of top stations to return (default: 10)",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 100,
                    }
                },
                "required": [],
            },
        ),
        Tool(
            name="get_summary_stats",
            description=(
                "Get summary statistics for Blue Bikes predictions including total stations, "
                "average demand, max demand, and total predicted rides."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls from MCP clients."""

    try:
        if name == "get_next_hour_predictions":
            with SuppressOutput():
                predictions = fetch_next_hour_predictions()

            # Convert to JSON-serializable format
            prediction_hour = None
            if len(predictions) > 0 and "pickup_hour" in predictions.columns:
                first_hour = predictions["pickup_hour"].iloc[0]
                prediction_hour = first_hour.isoformat() if hasattr(first_hour, 'isoformat') else str(first_hour)

            result = {
                "timestamp": datetime.now().isoformat(),
                "prediction_hour": prediction_hour,
                "total_stations": len(predictions),
                "predictions": predictions[["pickup_location_id", "predicted_demand"]].to_dict(orient="records"),
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "fetch_predictions":
            hours = arguments.get("hours", 24)
            with SuppressOutput():
                predictions = fetch_predictions(hours)

            # Convert timestamps to strings for JSON serialization
            if "pickup_hour" in predictions.columns:
                predictions["pickup_hour"] = predictions["pickup_hour"].astype(str)

            result = {
                "timestamp": datetime.now().isoformat(),
                "hours_fetched": hours,
                "total_records": len(predictions),
                "data": predictions.to_dict(orient="records"),
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "fetch_actual_rides":
            hours = arguments.get("hours", 24)
            with SuppressOutput():
                rides = fetch_hourly_rides(hours)

            # Convert timestamps to strings for JSON serialization
            if "pickup_hour" in rides.columns:
                rides["pickup_hour"] = rides["pickup_hour"].astype(str)

            result = {
                "timestamp": datetime.now().isoformat(),
                "hours_fetched": hours,
                "total_records": len(rides),
                "data": rides.to_dict(orient="records"),
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "get_model_metrics":
            hours = arguments.get("hours", 168)

            # Fetch both predictions and actual rides
            with SuppressOutput():
                predictions = fetch_predictions(hours)
                actual_rides = fetch_hourly_rides(hours)

            # Merge the data
            merged = pd.merge(
                actual_rides,
                predictions,
                on=["pickup_location_id", "pickup_hour"],
                how="inner"
            )

            if merged.empty:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": "No matching data found for the specified time range"})
                )]

            # Calculate metrics
            merged["absolute_error"] = abs(merged["predicted_demand"] - merged["rides"])
            merged["squared_error"] = (merged["predicted_demand"] - merged["rides"]) ** 2
            merged["percentage_error"] = (merged["absolute_error"] / merged["rides"].replace(0, 1)) * 100

            mae = float(merged["absolute_error"].mean())
            rmse = float((merged["squared_error"].mean() ** 0.5))
            mape = float(merged["percentage_error"].mean())

            # R² score
            ss_res = merged["squared_error"].sum()
            ss_tot = ((merged["rides"] - merged["rides"].mean()) ** 2).sum()
            r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0

            result = {
                "timestamp": datetime.now().isoformat(),
                "hours_analyzed": hours,
                "total_comparisons": len(merged),
                "metrics": {
                    "mae": round(mae, 2),
                    "rmse": round(rmse, 2),
                    "mape": round(mape, 2),
                    "r2_score": round(r2, 3),
                },
                "summary": {
                    "avg_actual_rides": round(float(merged["rides"].mean()), 2),
                    "avg_predicted_rides": round(float(merged["predicted_demand"].mean()), 2),
                    "max_error": round(float(merged["absolute_error"].max()), 2),
                    "min_error": round(float(merged["absolute_error"].min()), 2),
                }
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "get_station_data":
            station_id = arguments["station_id"]
            hours = arguments.get("hours", 24)

            # Fetch predictions and actual rides
            with SuppressOutput():
                predictions = fetch_predictions(hours)
                actual_rides = fetch_hourly_rides(hours)

            # Filter for the specific station
            station_predictions = predictions[predictions["pickup_location_id"] == station_id]
            station_rides = actual_rides[actual_rides["pickup_location_id"] == station_id]

            # Convert timestamps to strings for JSON serialization
            if "pickup_hour" in station_predictions.columns:
                station_predictions["pickup_hour"] = station_predictions["pickup_hour"].astype(str)
            if "pickup_hour" in station_rides.columns:
                station_rides["pickup_hour"] = station_rides["pickup_hour"].astype(str)

            # Merge the data
            merged = pd.merge(
                station_rides,
                station_predictions,
                on=["pickup_location_id", "pickup_hour"],
                how="inner"
            )

            if not merged.empty:
                merged["absolute_error"] = abs(merged["predicted_demand"] - merged["rides"])
                mae = float(merged["absolute_error"].mean())
            else:
                mae = None

            result = {
                "timestamp": datetime.now().isoformat(),
                "station_id": station_id,
                "hours_fetched": hours,
                "predictions_count": len(station_predictions),
                "actual_rides_count": len(station_rides),
                "mae": round(mae, 2) if mae is not None else None,
                "predictions": station_predictions.to_dict(orient="records"),
                "actual_rides": station_rides.to_dict(orient="records"),
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "get_top_stations":
            n = arguments.get("n", 10)
            with SuppressOutput():
                predictions = fetch_next_hour_predictions()

            # Get top N stations by predicted demand
            top_stations = predictions.nlargest(n, "predicted_demand")

            result = {
                "timestamp": datetime.now().isoformat(),
                "top_n": n,
                "stations": top_stations[["pickup_location_id", "predicted_demand"]].to_dict(orient="records"),
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        elif name == "get_summary_stats":
            with SuppressOutput():
                predictions = fetch_next_hour_predictions()

            result = {
                "timestamp": datetime.now().isoformat(),
                "total_stations": len(predictions),
                "total_predicted_rides": round(float(predictions["predicted_demand"].sum()), 0),
                "average_demand": round(float(predictions["predicted_demand"].mean()), 2),
                "max_demand": round(float(predictions["predicted_demand"].max()), 0),
                "min_demand": round(float(predictions["predicted_demand"].min()), 0),
                "median_demand": round(float(predictions["predicted_demand"].median()), 2),
                "std_demand": round(float(predictions["predicted_demand"].std()), 2),
            }

            return [TextContent(type="text", text=json.dumps(result, indent=2))]

        else:
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"Unknown tool: {name}"})
            )]

    except Exception as e:
        return [TextContent(
            type="text",
            text=json.dumps({"error": str(e), "type": type(e).__name__})
        )]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
